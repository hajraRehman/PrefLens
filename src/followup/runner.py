"""Study 2 runner.

    python -m src.followup.runner --phase pilot
    python -m src.followup.runner --phase main --yes
    python -m src.followup.runner --phase main --dry-run

Writes ONLY to data/raw/followup_gpt_oss/. Study 1's records under
data/raw/main/, data/raw/pilot/ and data/raw/manipulation_check/ are never
opened for writing by this module (D-26).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..methods import PreferenceItem
from ..parsing import parse_choice
from ..providers import ModelConfig, SamplingConfig, availability_report, get_provider
from .design import build_trials, budget, displayed_to_semantic, verify_counterbalance

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
STUDY2_RAW = ROOT / "data" / "raw" / "followup_gpt_oss"

# Guard against ever writing into Study 1's directories.
STUDY1_DIRS = {"main", "pilot", "manipulation_check"}


def load_cfg() -> tuple[dict, list[PreferenceItem], list[PreferenceItem]]:
    cfg = yaml.safe_load((CONFIGS / "followup.yaml").read_text(encoding="utf-8"))
    prefs = yaml.safe_load((CONFIGS / "preferences.yaml").read_text(encoding="utf-8"))
    items = [PreferenceItem.from_dict(d) for d in prefs["items"]]
    analysis = [i for i in items if not i.sanity_control]
    controls = [i for i in items if i.sanity_control]
    return cfg, analysis, controls


def plan(cfg: dict, phase: str) -> list:
    analysis_items, controls = load_cfg()[1], load_cfg()[2]
    reps = cfg["design"]["repetitions_per_position"]
    ids = cfg.get("preference_ids")
    if phase == "pilot":
        reps = cfg["pilot"]["repetitions_per_position"]
        ids = cfg["pilot"]["preference_ids"]
    if ids:
        analysis_items = [i for i in analysis_items if i.id in ids]
        controls = [] if phase == "pilot" else controls
    elif not cfg.get("include_controls", True):
        controls = []

    experiment_id = f"{cfg['study_id']}_{phase}"
    trials = []
    for m in cfg["models"]:
        for item in analysis_items:
            trials += build_trials(item, m["key"], reps, cfg["study_id"], experiment_id,
                                   cfg["system_prompt"], cfg["user_template"], False)
        for item in controls:
            trials += build_trials(item, m["key"], reps, cfg["study_id"], experiment_id,
                                   cfg["system_prompt"], cfg["user_template"], True)
    return trials


class RawWriter:
    def __init__(self, path: Path):
        assert path.parent.name not in STUDY1_DIRS, \
            f"refusing to write into a Study 1 directory: {path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")   # append-only
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def completed_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("call_ok") and r.get("parse_success"):
                done.add(r["trial_id"])
    return done


def run(phase: str, assume_yes: bool, dry_run: bool) -> None:
    cfg, _, _ = load_cfg()
    trials = plan(cfg, phase)
    model_map = {m["key"]: m for m in cfg["models"]}

    cb = verify_counterbalance(trials)
    bud = budget(trials, cfg["sampling"]["max_tokens"])

    print("=" * 68)
    print(f"STUDY 2 — position-bias follow-up | phase={phase}")
    print(f"study_id={cfg['study_id']}  models={list(model_map)}")
    print("-" * 68)
    print(f"COUNTERBALANCE: {'EXACT (verified)' if cb['balanced'] else 'BROKEN'}"
          f"  cells={cb['n_cells']}")
    if not cb["balanced"]:
        raise SystemExit(f"counterbalance broken: {cb['unbalanced_cells']}")
    print(f"BUDGET  total calls: {bud['total_calls']}")
    for k, v in sorted(bud["calls_by_model"].items()):
        print(f"          {k:<14}: {v}")
    print(f"        approx prompt tokens: ~{bud['approx_prompt_tokens']:,}")
    print(f"        max completion tokens: {bud['max_completion_tokens']:,}")
    print("=" * 68)

    if dry_run:
        print("--dry-run: no calls made.")
        return

    avail = availability_report()
    if not avail.get("openrouter"):
        raise SystemExit("openrouter unavailable — set OPENROUTER_API_KEY in .env")

    raw_path = STUDY2_RAW / f"{phase}_raw_observations.jsonl"
    already = completed_ids(raw_path) if cfg["runner"]["checkpoint"] else set()
    todo = [t for t in trials if t.trial_id not in already]
    if already:
        print(f"resuming: {len(already)} completed, {len(todo)} remaining")

    if not todo:
        print("nothing to do — all trials already completed.")
        return
    if not assume_yes:
        if input(f"Proceed with {len(todo)} calls? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return

    rcfg = cfg["runner"]
    scfg = cfg["sampling"]
    writer = RawWriter(raw_path)
    counters = {"done": 0, "total": len(todo), "call_failed": 0, "parse_failed": 0}
    started = time.time()

    def make_exec(mc: ModelConfig, provider):
        def execute(t) -> dict:
            res = provider.generate(
                mc, t.messages,
                SamplingConfig(temperature=scfg["temperature"], top_p=scfg["top_p"],
                               max_tokens=scfg["max_tokens"]),
                max_retries=rcfg["max_retries"], base_delay_s=rcfg["retry_base_delay_s"])
            parsed = parse_choice(res.text) if res.ok else None
            disp = parsed.choice_displayed if parsed else None
            rec = t.as_record()
            rec.pop("messages", None)          # prompt is stored verbatim already
            rec.update({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "provider": mc.provider,
                "model_id": mc.model_id,
                "model_family": mc.family,
                "served_model": res.meta.get("served_model"),
                "raw_response": res.text,
                "parsed_display_choice": disp,
                "parsed_semantic_choice": displayed_to_semantic(disp, t.semantic_x_position),
                "parse_success": bool(parsed and parsed.success),
                "parse_stage": parsed.parse_stage if parsed else "call_failed",
                "call_ok": res.ok,
                "temperature": scfg["temperature"],
                "top_p": scfg["top_p"],
                "max_output_tokens": scfg["max_tokens"],
                "latency_s": round(res.latency_s, 3),
                "retry_count": res.attempts - 1,
                "error": res.error,
                "usage": res.meta.get("usage"),
                "finish_reason": res.meta.get("finish_reason"),
                "seed": cfg["design"]["seed"],
            })
            writer.write(rec)
            counters["done"] += 1
            if not res.ok:
                counters["call_failed"] += 1
            elif not rec["parse_success"]:
                counters["parse_failed"] += 1
            if counters["done"] % 25 == 0:
                print(f"  {counters['done']}/{counters['total']} "
                      f"| call-fail {counters['call_failed']} "
                      f"| parse-fail {counters['parse_failed']}", flush=True)
            return rec
        return execute

    try:
        for mk, md in model_map.items():
            mine = [t for t in todo if t.model_key == mk]
            if not mine:
                continue
            mc = ModelConfig.from_dict(md)
            provider = get_provider(mc.provider, timeout_s=rcfg["request_timeout_s"])
            print(f"[{mk}] {len(mine)} calls -> {mc.model_id}", flush=True)
            with cf.ThreadPoolExecutor(max_workers=rcfg["max_workers"]) as pool:
                list(pool.map(make_exec(mc, provider), mine))
    finally:
        writer.close()

    manifest = {
        "study_id": cfg["study_id"], "phase": phase,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - started, 1),
        "python": sys.version, "platform": platform.platform(),
        "config": cfg, "budget": bud, "counterbalance": cb, "counters": counters,
        "resumed_skipped": len(already),
    }
    (STUDY2_RAW / f"{phase}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 68)
    print(f"done in {manifest['elapsed_s']}s | calls={counters['done']} "
          f"| call-fail={counters['call_failed']} | parse-fail={counters['parse_failed']}")
    print(f"raw -> {raw_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Study 2 — position-bias follow-up")
    ap.add_argument("--phase", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.phase, a.yes, a.dry_run)


if __name__ == "__main__":
    main()

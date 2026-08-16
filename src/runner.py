"""Experiment runner: plan -> budget -> execute -> raw JSONL + manifest.

Usage
-----
    python -m src.runner --phase pilot
    python -m src.runner --phase main --yes
    python -m src.runner --phase main --dry-run     # budget only, no calls

Guarantees
----------
* Raw observations are only ever appended. An existing raw file is never
  truncated, so an interrupted run resumes without losing completed calls.
* Every completed trial_id is read back from the raw file on start-up and skipped.
* A manifest recording configs, git-free provenance, timestamps and sampling
  parameters is written next to the raw data.
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

from .methods import PreferenceItem, pairwise, self_report, sequential, tradeoff
from .parsing import parse_choice
from .providers import ModelConfig, SamplingConfig, availability_report, get_provider

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# Methods C and D are run at the neutral framing only. The framing sweep is a
# controlled perturbation of the *question sentence*, which Methods C and D
# embed in a longer protocol; sweeping them too would quadruple the call budget
# for the two most expensive methods (D-07).
FRAMING_SWEEP_METHODS = {"self_report", "pairwise"}


# --------------------------------------------------------------------------- config


def load_configs() -> tuple[dict, dict, list[PreferenceItem]]:
    exp = yaml.safe_load((CONFIGS / "experiment.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text(encoding="utf-8"))
    prefs = yaml.safe_load((CONFIGS / "preferences.yaml").read_text(encoding="utf-8"))
    items = [PreferenceItem.from_dict(d) for d in prefs["items"]]
    model_map = {m["key"]: m for m in models["models"]}
    return exp, model_map, items


def select_items(items: list[PreferenceItem], ids: list[str] | None) -> list[PreferenceItem]:
    if not ids:
        return items
    by_id = {i.id: i for i in items}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown preference_ids in experiment.yaml: {missing}")
    return [by_id[i] for i in ids]


# ----------------------------------------------------------------------------- plan


def plan_trials(phase_cfg: dict, exp: dict, items: list[PreferenceItem]):
    """Return (independent_trials, episode_specs)."""
    sys_prompt = exp["system_prompt"]
    seed = phase_cfg["seed"]
    eid = phase_cfg["experiment_id"]
    reps = phase_cfg["repetitions"]
    tcfg = exp["methods"]["tradeoff"]
    scfg = exp["methods"]["sequential"]

    trials, episodes = [], []
    for mk in phase_cfg["model_keys"]:
        for item in items:
            for method in phase_cfg["methods"]:
                framings = (
                    phase_cfg["framing_variants"]
                    if method in FRAMING_SWEEP_METHODS
                    else ["neutral"]
                )
                for fr in framings:
                    if method == "self_report":
                        trials += self_report.build_trials(
                            item, mk, fr, reps, eid, sys_prompt, seed)
                    elif method == "pairwise":
                        trials += pairwise.build_trials(
                            item, mk, fr, reps, eid, sys_prompt, seed)
                    elif method == "tradeoff":
                        trials += tradeoff.build_trials(
                            item, mk, fr, tcfg["cost_levels"],
                            tcfg["repetitions_per_level"], eid, sys_prompt, seed)
                    elif method == "sequential":
                        for rep in range(scfg["repetitions"]):
                            episodes.append(
                                {"item": item, "model_key": mk, "framing": fr,
                                 "rep": rep, "stages": scfg["stages"]}
                            )
                    else:
                        raise SystemExit(f"unknown method {method!r}")
    return trials, episodes


def budget_report(trials, episodes, exp) -> dict:
    stages = exp["methods"]["sequential"]["stages"]
    # Each episode: 1 initial choice + (stages-1) * (perform + choose).
    per_episode = 1 + 2 * (stages - 1)
    seq_calls = len(episodes) * per_episode

    by_model: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for t in trials:
        by_model[t.model_key] = by_model.get(t.model_key, 0) + 1
        by_method[t.method] = by_method.get(t.method, 0) + 1
    for e in episodes:
        by_model[e["model_key"]] = by_model.get(e["model_key"], 0) + per_episode
    by_method["sequential"] = seq_calls

    total = len(trials) + seq_calls
    # Rough envelope only: ~180 prompt + up to max_tokens completion per call.
    approx_tokens = total * (180 + exp["sampling"]["max_tokens"] * 0.5)
    return {
        "total_calls": total,
        "calls_by_model": by_model,
        "calls_by_method": by_method,
        "sequential_episodes": len(episodes),
        "calls_per_episode": per_episode,
        "approx_total_tokens": int(approx_tokens),
    }


# -------------------------------------------------------------------------- execute


class RawWriter:
    """Append-only, thread-safe JSONL writer."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def completed_trial_ids(path: Path) -> set[str]:
    """trial_ids already present AND successfully called (failures are retried)."""
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
                continue  # a partial final line from an interrupted run
            if r.get("call_ok"):
                done.add(r["trial_id"])
    return done


def make_executor(model_cfg: ModelConfig, provider, sampling: SamplingConfig,
                  runner_cfg: dict, writer: RawWriter, counters: dict):
    def execute(trial) -> dict:
        res = provider.generate(
            model_cfg, trial.messages, sampling,
            max_retries=runner_cfg["max_retries"],
            base_delay_s=runner_cfg["retry_base_delay_s"],
        )
        rec = trial.as_record()
        parsed = parse_choice(res.text) if res.ok else None
        # "perform_task" turns are free-text by design and are not parsed for a choice.
        is_choice_turn = trial.extra.get("stage_kind") != "perform_task"

        rec.update({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provider": model_cfg.provider,
            "model_id": model_cfg.model_id,
            "model_family": model_cfg.family,
            "served_model": res.meta.get("served_model"),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "raw_response": res.text,
            "call_ok": res.ok,
            "error": res.error,
            "attempts": res.attempts,
            "latency_s": round(res.latency_s, 3),
            "usage": res.meta.get("usage"),
            "finish_reason": res.meta.get("finish_reason"),
            "parsed_choice": parsed.choice_displayed if (parsed and is_choice_turn) else None,
            "strength_self_report": parsed.strength_self_report if (parsed and is_choice_turn) else None,
            "parse_success": bool(parsed and parsed.success) if is_choice_turn else None,
            "parse_stage": parsed.parse_stage if (parsed and is_choice_turn) else "not_applicable",
            "parse_note": parsed.parse_note if (parsed and is_choice_turn) else "",
        })
        writer.write(rec)

        counters["done"] += 1
        if not res.ok:
            counters["call_failed"] += 1
        elif is_choice_turn and not rec["parse_success"]:
            counters["parse_failed"] += 1
        if counters["done"] % 25 == 0:
            print(
                f"  {counters['done']}/{counters['total']} calls "
                f"| call-fail {counters['call_failed']} "
                f"| parse-fail {counters['parse_failed']}",
                flush=True,
            )
        return rec

    return execute


def run_phase(phase: str, assume_yes: bool, dry_run: bool, override_models: list[str] | None):
    exp, model_map, all_items = load_configs()
    if phase not in exp:
        raise SystemExit(f"no phase {phase!r} in experiment.yaml")
    phase_cfg = dict(exp[phase])
    if override_models:
        phase_cfg["model_keys"] = override_models

    items = select_items(all_items, phase_cfg.get("preference_ids"))
    trials, episodes = plan_trials(phase_cfg, exp, items)
    budget = budget_report(trials, episodes, exp)

    print("=" * 68)
    print(f"PHASE {phase}  |  experiment_id={phase_cfg['experiment_id']}")
    print(f"items={len(items)}  models={phase_cfg['model_keys']}  "
          f"methods={phase_cfg['methods']}  framings={phase_cfg['framing_variants']}")
    print(f"reps={phase_cfg['repetitions']}  seed={phase_cfg['seed']}")
    print("-" * 68)
    print(f"BUDGET  total API calls : {budget['total_calls']}")
    for k, v in sorted(budget["calls_by_method"].items()):
        print(f"          by method {k:<12}: {v}")
    for k, v in sorted(budget["calls_by_model"].items()):
        print(f"          by model  {k:<12}: {v}")
    print(f"        approx tokens   : ~{budget['approx_total_tokens']:,}")
    print("=" * 68)

    out_dir = ROOT / "data" / "raw" / phase_cfg["experiment_id"]
    raw_path = out_dir / "raw_observations.jsonl"
    already = completed_trial_ids(raw_path) if exp["runner"]["checkpoint"] else set()
    if already:
        print(f"resuming: {len(already)} completed calls already on disk, will be skipped")

    if dry_run:
        print("--dry-run: no calls made.")
        return

    missing = [k for k in phase_cfg["model_keys"] if k not in model_map]
    if missing:
        raise SystemExit(f"unknown model_keys: {missing}")

    avail = availability_report()
    needed = {model_map[k]["provider"] for k in phase_cfg["model_keys"]}
    unusable = [p for p in needed if not avail.get(p)]
    if unusable:
        raise SystemExit(
            f"provider(s) not usable (missing credentials or service down): {unusable}\n"
            f"availability: {avail}\n"
            "Set the relevant key in .env / the environment, or enable a different "
            "model in configs/models.yaml."
        )

    if not assume_yes:
        if input(f"Proceed with up to {budget['total_calls'] - len(already)} calls? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return

    sampling_cfg = exp["sampling"]
    runner_cfg = exp["runner"]
    writer = RawWriter(raw_path)
    started = time.time()
    counters = {"done": 0, "total": budget["total_calls"] - len(already),
                "call_failed": 0, "parse_failed": 0}

    try:
        # --- independent trials, parallel per model ---------------------------------
        for mk in phase_cfg["model_keys"]:
            mc = ModelConfig.from_dict(model_map[mk])
            provider = get_provider(mc.provider, timeout_s=runner_cfg["request_timeout_s"])
            todo = [t for t in trials if t.model_key == mk and t.trial_id not in already]
            if not todo:
                continue
            print(f"[{mk}] {len(todo)} independent calls ...", flush=True)
            with cf.ThreadPoolExecutor(max_workers=runner_cfg["max_workers"]) as pool:
                ex = make_executor(
                    mc, provider,
                    SamplingConfig(
                        temperature=sampling_cfg["temperature"],
                        top_p=sampling_cfg["top_p"],
                        max_tokens=sampling_cfg["max_tokens"],
                        seed=sampling_cfg.get("seed"),
                    ),
                    runner_cfg, writer, counters,
                )
                list(pool.map(ex, todo))

        # --- sequential episodes, strictly serial within an episode -----------------
        eps_by_model: dict[str, list[dict]] = {}
        for e in episodes:
            eps_by_model.setdefault(e["model_key"], []).append(e)

        summaries: list[dict] = []
        for mk, eps in eps_by_model.items():
            mc = ModelConfig.from_dict(model_map[mk])
            provider = get_provider(mc.provider, timeout_s=runner_cfg["request_timeout_s"])
            sampling = SamplingConfig(
                temperature=sampling_cfg["temperature"],
                top_p=sampling_cfg["top_p"],
                max_tokens=sampling_cfg["max_tokens"],
                seed=sampling_cfg.get("seed"),
            )
            ex = make_executor(mc, provider, sampling, runner_cfg, writer, counters)
            print(f"[{mk}] {len(eps)} sequential episodes ...", flush=True)
            for e in eps:
                # Episodes are conversational; a partially-completed episode cannot be
                # resumed mid-way, so an episode is re-run unless all its turns landed.
                recs, summary = sequential.run_episode(
                    item=e["item"], model_key=mk, model_cfg=mc, provider=provider,
                    sampling=sampling, framing=e["framing"], rep=e["rep"],
                    stages=e["stages"], experiment_id=phase_cfg["experiment_id"],
                    system_prompt=exp["system_prompt"], seed=phase_cfg["seed"],
                    max_retries=runner_cfg["max_retries"],
                    retry_base_delay_s=runner_cfg["retry_base_delay_s"],
                    execute=ex,
                )
                summaries.append(summary)

        if summaries:
            (out_dir / "sequential_episodes.jsonl").write_text(
                "\n".join(json.dumps(s) for s in summaries) + "\n", encoding="utf-8"
            )
    finally:
        writer.close()

    manifest = {
        "experiment_id": phase_cfg["experiment_id"],
        "phase": phase,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - started, 1),
        "python": sys.version,
        "platform": platform.platform(),
        "phase_config": phase_cfg,
        "sampling": sampling_cfg,
        "runner": runner_cfg,
        "system_prompt": exp["system_prompt"],
        "methods_config": exp["methods"],
        "framing_sweep_methods": sorted(FRAMING_SWEEP_METHODS),
        "models": [model_map[k] for k in phase_cfg["model_keys"]],
        "preference_items": [i.__dict__ for i in items],
        "budget": budget,
        "counters": counters,
        "resumed_calls_skipped": len(already),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("-" * 68)
    print(f"done in {manifest['elapsed_s']}s | calls={counters['done']} "
          f"| call-failures={counters['call_failed']} "
          f"| parse-failures={counters['parse_failed']}")
    print(f"raw     -> {raw_path}")
    print(f"manifest-> {out_dir / 'manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a preference-elicitation phase.")
    ap.add_argument("--phase", default="pilot", help="phase key in configs/experiment.yaml")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true", help="print the budget and exit")
    ap.add_argument("--models", nargs="*", default=None, help="override model_keys")
    a = ap.parse_args()
    run_phase(a.phase, a.yes, a.dry_run, a.models)


if __name__ == "__main__":
    main()

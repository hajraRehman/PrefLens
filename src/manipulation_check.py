"""Manipulation check for Method C's cost clause.

    python -m src.manipulation_check --model llama31-8b --reps 6

Method C assumes that attaching a surcharge to an option makes the model less
likely to choose it. That assumption is testable directly, and it must be tested
before any Method C score is interpreted.

Design: both options are the SAME activity, and one carries the surcharge. Any
systematic avoidance of the surcharged side is therefore attributable to the cost
clause alone — there is no item preference for it to be confounded with. Which
semantic side carries the cost, and the display order, are randomised per trial.

Two quantities matter:

  avoid rate    P(choosing the option WITHOUT the surcharge). 0.5 = the clause
                does nothing; higher = it deters, as Method C requires.
  gradedness    Spearman rho between surcharge magnitude and avoid rate. Method C
                originally assumed this was positive and reliable; the pilot
                showed it is not, which is why the Method C score averages over
                the whole ladder instead of locating an indifference point (D-20).

Results are written to results/tables/manipulation_check_cost.csv and the raw
calls to data/raw/manipulation_check/raw_observations.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats

from .methods.common import build_messages, render_block
from .parsing import parse_choice
from .providers import ModelConfig, SamplingConfig, get_provider
from .runner import load_configs

ROOT = Path(__file__).resolve().parents[1]

# A deliberately neutral filler activity: the check is about the clause, not the task.
TASK = "write a condensed summary of a long technical document"

PHRASINGS = {
    "v1_parenthetical": lambda n: f" (and then repeat the whole task {n} more times)",
    "v2_explicit": lambda n: f", and then do that same task over again another {n} times",
    "v3_turns": lambda n: f", which will take {n} extra rounds of work to finish",
}

COSTS = [1, 2, 4, 8, 16]
SCHEMA = ('Respond with exactly this JSON object and nothing else:\n'
          '{"choice": "A" or "B"}')


def run(model_key: str, reps: int, phrasings: list[str], seed: int) -> pd.DataFrame:
    exp, model_map, _ = load_configs()
    mc = ModelConfig.from_dict(model_map[model_key])
    prov = get_provider(mc.provider, timeout_s=exp["runner"]["request_timeout_s"])
    sampling = SamplingConfig(
        temperature=exp["sampling"]["temperature"],
        top_p=exp["sampling"]["top_p"],
        max_tokens=40,
    )

    out_dir = ROOT / "data" / "raw" / "manipulation_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_fh = (out_dir / "raw_observations.jsonl").open("a", encoding="utf-8")

    rng = random.Random(seed)
    rows = []
    total = len(phrasings) * len(COSTS) * reps
    print(f"manipulation check: {total} calls on {mc.model_id}")

    try:
        for i, (name, cost, rep) in enumerate(
            itertools.product(phrasings, COSTS, range(reps)), start=1
        ):
            costly_is_a = rng.random() < 0.5      # which semantic side is surcharged
            clause = PHRASINGS[name](cost)
            shown_a = TASK + (clause if costly_is_a else "")
            shown_b = TASK + ("" if costly_is_a else clause)
            prompt = render_block(shown_a, shown_b, "Which would you select?", SCHEMA)

            res = prov.generate(mc, build_messages(exp["system_prompt"], prompt),
                                sampling, max_retries=exp["runner"]["max_retries"],
                                base_delay_s=exp["runner"]["retry_base_delay_s"])
            p = parse_choice(res.text) if res.ok else None
            ok = bool(p and p.success)
            avoided = None
            if ok:
                picked_costly = (p.choice_displayed == "A") == costly_is_a
                avoided = not picked_costly

            raw_fh.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_key": model_key, "model_id": mc.model_id,
                "served_model": res.meta.get("served_model"),
                "phrasing": name, "cost": cost, "repetition_index": rep,
                "costly_side_semantic": "A" if costly_is_a else "B",
                "prompt": prompt, "raw_response": res.text,
                "call_ok": res.ok, "parsed_choice": p.choice_displayed if p else None,
                "parse_success": ok, "avoided_surcharge": avoided,
                "temperature": sampling.temperature,
            }, ensure_ascii=False) + "\n")
            raw_fh.flush()

            rows.append({"phrasing": name, "cost": cost, "parse_ok": ok,
                         "avoided": avoided})
            if i % 25 == 0:
                print(f"  {i}/{total}", flush=True)
    finally:
        raw_fh.close()

    print(f"\nparse failures: {int((~pd.DataFrame(rows).parse_ok).sum())}/{len(rows)}")
    # Summarise from the append-only raw log rather than from this run's rows, so
    # that every model checked so far is tabulated and no earlier run is lost.
    return summarise_from_raw()


def summarise_from_raw() -> pd.DataFrame:
    """Rebuild the manipulation-check tables from the raw JSONL, keyed by model."""
    raw_path = ROOT / "data" / "raw" / "manipulation_check" / "raw_observations.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"no raw manipulation-check data at {raw_path}")

    recs = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    ok = df[df["parse_success"] == True].copy()  # noqa: E712

    by_cost = (ok.groupby(["model_key", "phrasing", "cost"])["avoided_surcharge"]
                 .agg(avoid_rate="mean", n="size").reset_index())

    rows = []
    for (model, name), d in ok.groupby(["model_key", "phrasing"]):
        curve = (d.groupby("cost")["avoided_surcharge"].mean().sort_index())
        rho = (stats.spearmanr(curve.index, curve.to_numpy()).statistic
               if curve.nunique() > 1 else float("nan"))
        n, k = len(d), int(d["avoided_surcharge"].sum())
        rows.append({
            "model_key": model, "phrasing": name,
            "overall_avoid_rate": k / n, "n": n,
            # One-sided: the assumption Method C needs is deterrence, not any change.
            "p_vs_no_effect": stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
            "gradedness_spearman_rho": rho,
        })
    summary = pd.DataFrame(rows).sort_values(["model_key", "phrasing"])

    tbl = ROOT / "results" / "tables"
    tbl.mkdir(parents=True, exist_ok=True)
    by_cost.to_csv(tbl / "manipulation_check_by_cost.csv", index=False)
    summary.to_csv(tbl / "manipulation_check_summary.csv", index=False)

    print("\navoid rate by model, phrasing and surcharge "
          "(0.50 = the clause does nothing):")
    print(by_cost.to_string(index=False))
    print("\nper-model / per-phrasing summary:")
    print(summary.to_string(index=False))
    print(f"\nwrote -> {tbl / 'manipulation_check_summary.csv'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31-8b")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--phrasings", nargs="*", default=list(PHRASINGS))
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild tables from the existing raw log; make no calls")
    a = ap.parse_args()
    if a.summarise_only:
        summarise_from_raw()
    else:
        run(a.model, a.reps, a.phrasings, a.seed)


if __name__ == "__main__":
    main()

"""Study 2 analysis: raw -> per-item decomposition -> model comparison.

    python -m src.followup.analysis

Runs in the order mandated by the brief: completeness, parse rates, duplicates and
exact counterbalance are all verified BEFORE any statistic is computed. If a
verification fails the run aborts rather than reporting a number built on bad data.

Writes only under results/followup/ and data/processed/followup_gpt_oss/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import metrics as FM

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "followup_gpt_oss"
PROC = ROOT / "data" / "processed" / "followup_gpt_oss"
RES = ROOT / "results" / "followup"
TABLES = RES / "tables"
STATS = RES / "statistics"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "followup.yaml").read_text(encoding="utf-8"))


def load_raw(phase: str = "main") -> pd.DataFrame:
    p = RAW / f"{phase}_raw_observations.jsonl"
    if not p.exists():
        raise SystemExit(f"no Study 2 data at {p}")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df["_ok"] = df["call_ok"].astype(int)
    df = (df.sort_values(["trial_id", "_ok"]).drop_duplicates("trial_id", keep="last")
            .drop(columns="_ok").reset_index(drop=True))
    return df


# --------------------------------------------------------------------- verification


def verify(df: pd.DataFrame, cfg: dict) -> dict:
    """All integrity checks. Aborts the run on failure."""
    reps = cfg["design"]["repetitions_per_position"]
    report: dict = {"checks": {}, "ok": True}

    def check(name, ok, detail=""):
        report["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            report["ok"] = False

    raw_lines = sum(1 for l in (RAW / "main_raw_observations.jsonl")
                    .read_text(encoding="utf-8").splitlines() if l.strip())
    check("no_duplicate_trial_ids", len(df) == raw_lines,
          f"{raw_lines} lines -> {len(df)} unique")
    check("no_call_failures", bool((df["call_ok"]).all()),
          f"{int((~df['call_ok'].astype(bool)).sum())} failures")
    check("no_parse_failures", bool(df["parse_success"].all()),
          f"{int((~df['parse_success'].astype(bool)).sum())} failures")
    check("served_model_matches_request",
          bool((df["served_model"].fillna("") == df["model_id"]).all()),
          f"served={sorted(df['served_model'].dropna().unique())}")

    # Exact counterbalance, per (model, item).
    g = df.groupby(["model_key", "preference_id", "semantic_x_position"]).size().unstack(fill_value=0)
    bal = (g.get("x_first", 0) == g.get("x_second", 0))
    check("exact_counterbalance", bool(bal.all()),
          f"{int((~bal).sum())} unbalanced cells of {len(g)}")
    check("full_repetitions",
          bool((g.get("x_first", 0) == reps).all() and (g.get("x_second", 0) == reps).all()),
          f"expected {reps} per position")

    # Semantic mapping re-derived independently of the runner.
    def remap(r):
        if r["parsed_display_choice"] not in ("A", "B"):
            return None
        if r["semantic_x_position"] == "x_first":
            return "X" if r["parsed_display_choice"] == "A" else "Y"
        return "Y" if r["parsed_display_choice"] == "A" else "X"

    check("semantic_mapping_reproducible",
          bool((df.apply(remap, axis=1) == df["parsed_semantic_choice"]).all()))
    # Displayed text must match the declared position.
    want_a = np.where(df["semantic_x_position"] == "x_first", df["semantic_x"], df["semantic_y"])
    check("displayed_A_matches_position", bool((df["displayed_A"] == want_a).all()))
    return report


# ------------------------------------------------------------------- decomposition


def per_item_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mk, pid), d in df.groupby(["model_key", "preference_id"]):
        first = d[d.semantic_x_position == "x_first"]
        second = d[d.semantic_x_position == "x_second"]
        dec = FM.decompose(
            int((first.parsed_semantic_choice == "X").sum()), len(first),
            int((second.parsed_semantic_choice == "X").sum()), len(second))
        rows.append({
            "model_key": mk, "preference_id": pid,
            "is_control": bool(d["is_control"].iloc[0]),
            "preference_category": d["preference_category"].iloc[0],
            **dec,
            "abs_position_effect": abs(dec["position_effect"]),
            "abs_content_signal": abs(dec["content_signal"]),
        })
    return pd.DataFrame(rows).sort_values(["model_key", "preference_id"])


def model_summary(item_tbl: pd.DataFrame, cfg: dict) -> dict:
    reps = cfg["design"]["repetitions_per_position"]
    main = item_tbl[~item_tbl.is_control]
    out: dict = {"per_model": {}}
    for mk, d in main.groupby("model_key"):
        out["per_model"][mk] = {
            "n_items": int(len(d)),
            "mean_abs_position_effect": FM.bootstrap_mean(d["abs_position_effect"]),
            "mean_abs_content_signal": FM.bootstrap_mean(d["abs_content_signal"]),
            "mean_signed_position_effect": float(d["position_effect"].mean()),
            "n_items_position_gt_content": int(
                (d["abs_position_effect"] > d["abs_content_signal"]).sum()),
        }
    out["random_baseline"] = FM.random_baseline(
        n_items=int(main.preference_id.nunique()), reps_per_position=reps)
    return out


def paired_differences(item_tbl: pd.DataFrame, small: str, large: str) -> dict:
    """H1 and H2, on items present for BOTH models, in matching order."""
    main = item_tbl[~item_tbl.is_control]
    a = main[main.model_key == small].set_index("preference_id").sort_index()
    b = main[main.model_key == large].set_index("preference_id").sort_index()
    shared = sorted(set(a.index) & set(b.index))
    a, b = a.loc[shared], b.loc[shared]
    return {
        "items": shared,
        "n_items": len(shared),
        # H1: predicted positive (smaller model more position-susceptible)
        "H1_delta_position_small_minus_large": FM.bootstrap_difference(
            a["abs_position_effect"].to_numpy(), b["abs_position_effect"].to_numpy()),
        # H2: predicted positive (larger model more content signal)
        "H2_delta_content_large_minus_small": FM.bootstrap_difference(
            b["abs_content_signal"].to_numpy(), a["abs_content_signal"].to_numpy()),
    }


def control_accuracy(item_tbl: pd.DataFrame, df: pd.DataFrame) -> dict:
    """On controls, semantic X is the coherent option; X-selection rate is accuracy."""
    out = {}
    ctl = df[df.is_control]
    for mk, d in ctl.groupby("model_key"):
        by_pos = {p: float((g.parsed_semantic_choice == "X").mean())
                  for p, g in d.groupby("semantic_x_position")}
        out[mk] = {
            "accuracy": float((d.parsed_semantic_choice == "X").mean()),
            "accuracy_when_x_first": by_pos.get("x_first", np.nan),
            "accuracy_when_x_second": by_pos.get("x_second", np.nan),
            "n": int(len(d)),
        }
        it = item_tbl[(item_tbl.model_key == mk) & (item_tbl.is_control)]
        out[mk]["mean_abs_position_effect"] = float(it["abs_position_effect"].mean())
        out[mk]["mean_abs_content_signal"] = float(it["abs_content_signal"].mean())
    return out


def exploratory_indifference(item_tbl: pd.DataFrame) -> dict:
    """EXPLORATORY (D-27): do items with weaker content signal show more position effect?"""
    main = item_tbl[~item_tbl.is_control]
    out = {"status": "EXPLORATORY — not pre-specified as confirmatory", "per_model": {}}
    for mk, d in main.groupby("model_key"):
        x, y = d["abs_content_signal"].to_numpy(), d["abs_position_effect"].to_numpy()
        if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1:
            r = stats.spearmanr(x, y)
            out["per_model"][mk] = {"spearman_rho": float(r.statistic),
                                    "p_value": float(r.pvalue), "n": int(len(x))}
        else:
            out["per_model"][mk] = {"spearman_rho": np.nan, "p_value": np.nan, "n": int(len(x))}
    x = main["abs_content_signal"].to_numpy()
    y = main["abs_position_effect"].to_numpy()
    r = stats.spearmanr(x, y)
    out["pooled"] = {"spearman_rho": float(r.statistic), "p_value": float(r.pvalue),
                     "n": int(len(x))}
    return out


# -------------------------------------------------------------------------- driver


def run(phase: str = "main") -> dict:
    cfg = load_cfg()
    for p in (PROC, RES, TABLES, STATS):
        p.mkdir(parents=True, exist_ok=True)

    df = load_raw(phase)
    print(f"Study 2 records: {len(df)}\n")

    print("=== VERIFICATION (before any statistic) ===")
    v = verify(df, cfg)
    for name, r in v["checks"].items():
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {name}"
              + (f"  -- {r['detail']}" if r["detail"] else ""))
    if not v["ok"]:
        raise SystemExit("\nVerification failed. No statistics computed.")
    print("  all checks passed\n")

    item_tbl = per_item_table(df)
    item_tbl.to_csv(TABLES / "gpt_oss_position_decomposition.csv", index=False)
    df.to_csv(PROC / "observations.csv", index=False)

    keys = [m["key"] for m in cfg["models"]]
    small = min(cfg["models"], key=lambda m: m["scale"])["key"]
    large = max(cfg["models"], key=lambda m: m["scale"])["key"]

    summary = {
        "study_id": cfg["study_id"], "phase": phase,
        "n_records": int(len(df)),
        "models": {m["key"]: {"model_id": m["model_id"], "scale_b": m["scale"]}
                   for m in cfg["models"]},
        "verification": v,
        **model_summary(item_tbl, cfg),
        "hypotheses": paired_differences(item_tbl, small, large),
        "sanity_controls": control_accuracy(item_tbl, df),
        "exploratory_indifference": exploratory_indifference(item_tbl),
    }
    (STATS / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- principal comparison table ----
    rows = []
    for mk in keys:
        s = summary["per_model"][mk]
        c = summary["sanity_controls"].get(mk, {})
        rows.append({
            "model": cfg_model_id(cfg, mk),
            "key": mk,
            "mean_abs_position_effect": round(s["mean_abs_position_effect"]["mean"], 4),
            "position_ci_lo": round(s["mean_abs_position_effect"]["lo"], 4),
            "position_ci_hi": round(s["mean_abs_position_effect"]["hi"], 4),
            "mean_abs_content_signal": round(s["mean_abs_content_signal"]["mean"], 4),
            "content_ci_lo": round(s["mean_abs_content_signal"]["lo"], 4),
            "content_ci_hi": round(s["mean_abs_content_signal"]["hi"], 4),
            "sanity_control_accuracy": round(c.get("accuracy", np.nan), 4),
        })
    principal = pd.DataFrame(rows)
    principal.to_csv(TABLES / "gpt_oss_principal_comparison.csv", index=False)

    # ---- console report ----
    b = summary["random_baseline"]
    print("=== PRINCIPAL COMPARISON ===")
    print(principal.to_string(index=False))
    print(f"\nchance (coin-flip, same design): mean|position| "
          f"{b['mean_abs_position']:.3f} (p95 {b['abs_position_p95']:.3f}) | "
          f"mean|content| {b['mean_abs_content']:.3f} (p95 {b['abs_content_p95']:.3f})")

    h = summary["hypotheses"]
    print(f"\n=== PRE-SPECIFIED HYPOTHESES (n={h['n_items']} paired items) ===")
    h1 = h["H1_delta_position_small_minus_large"]
    h2 = h["H2_delta_content_large_minus_small"]
    print(f"  H1  |position| {small} - {large} = {h1['diff']:+.4f} "
          f"[{h1['lo']:+.4f}, {h1['hi']:+.4f}]  -> "
          f"{'SUPPORTED' if h1['lo'] > 0 else 'NOT SUPPORTED'} (predicted > 0)")
    print(f"  H2  |content|  {large} - {small} = {h2['diff']:+.4f} "
          f"[{h2['lo']:+.4f}, {h2['hi']:+.4f}]  -> "
          f"{'SUPPORTED' if h2['lo'] > 0 else 'NOT SUPPORTED'} (predicted > 0)")

    print("\n=== SANITY CONTROLS (held out of all principal statistics) ===")
    for mk, c in summary["sanity_controls"].items():
        print(f"  {mk:<14} accuracy {c['accuracy']:.3f} "
              f"(X first {c['accuracy_when_x_first']:.2f} / "
              f"X second {c['accuracy_when_x_second']:.2f})  "
              f"|position| {c['mean_abs_position_effect']:.3f}")

    e = summary["exploratory_indifference"]
    print(f"\n=== EXPLORATORY: weak content <-> strong position ===")
    print(f"  pooled rho={e['pooled']['spearman_rho']:+.3f} p={e['pooled']['p_value']:.4f} "
          f"n={e['pooled']['n']}")
    for mk, r in e["per_model"].items():
        print(f"  {mk:<14} rho={r['spearman_rho']:+.3f} p={r['p_value']:.4f} n={r['n']}")

    print(f"\nwrote -> {STATS / 'followup_summary.json'}")
    return summary


def cfg_model_id(cfg: dict, key: str) -> str:
    for m in cfg["models"]:
        if m["key"] == key:
            return m["model_id"]
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main")
    run(ap.parse_args().phase)


if __name__ == "__main__":
    main()

"""Analysis: raw JSONL -> normalised scores -> convergence metrics -> tables.

    python -m src.analysis --phase main

Writes processed tables to data/processed/<experiment_id>/ and result tables to
results/<phase>/ + results/tables/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics as M
from .methods import METHOD_ORDER, PreferenceItem, pairwise, self_report, sequential, tradeoff
from .methods.common import displayed_to_semantic

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- loading


def load_raw(experiment_id: str) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / experiment_id / "raw_observations.jsonl"
    if not path.exists():
        raise SystemExit(f"no raw data at {path}. Run: python -m src.runner --phase ...")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # truncated final line of an interrupted run
    df = pd.DataFrame(rows)
    # A resumed run can legitimately contain a failed attempt and a later success
    # for the same trial_id. Keep the last successful record per trial_id.
    df["_rank"] = df["call_ok"].astype(int)
    df = (df.sort_values(["trial_id", "_rank"])
            .drop_duplicates("trial_id", keep="last")
            .drop(columns="_rank")
            .reset_index(drop=True))
    return df


def load_items() -> dict[str, PreferenceItem]:
    prefs = yaml.safe_load((ROOT / "configs" / "preferences.yaml").read_text(encoding="utf-8"))
    return {d["id"]: PreferenceItem.from_dict(d) for d in prefs["items"]}


def load_experiment_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ diagnostics


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Call-failure and parse-failure rates, per model and method."""
    choice_turns = df[df["parse_stage"] != "not_applicable"]
    g = choice_turns.groupby(["model_key", "method"], dropna=False)
    out = g.apply(
        lambda d: pd.Series({
            "n_calls": len(d),
            "call_failure_rate": float((~d["call_ok"].astype(bool)).mean()),
            "parse_failure_rate": float((~d["parse_success"].fillna(False).astype(bool)).mean()),
            "pct_strict_json": float((d["parse_stage"] == "strict_json").mean()),
            "pct_recovered_from_prose": float(
                d["parse_stage"].isin(["labelled_text", "bare_token", "embedded_json"]).mean()),
            "mean_latency_s": float(d["latency_s"].mean()),
            "mean_attempts": float(d["attempts"].mean()),
        }),
        include_groups=False,
    ).reset_index()
    return out


def position_bias_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = df[(df["parse_success"] == True) & (df["method"] != "sequential")]  # noqa: E712
    for (model, method), d in ok.groupby(["model_key", "method"]):
        # Only the zero-cost rung of the trade-off ladder is position-comparable;
        # at non-zero cost the two sides are not symmetric.
        if method == "tradeoff":
            d = d[d["extra"].apply(lambda e: (e or {}).get("signed_cost") == 0)]
            if d.empty:
                continue
        r = M.position_bias(d.to_dict("records"))
        rows.append({"model_key": model, "method": method, **r})
    # Pooled across methods, per model.
    for model, d in ok.groupby("model_key"):
        d = d[~((d["method"] == "tradeoff") &
                (d["extra"].apply(lambda e: (e or {}).get("signed_cost", 0) != 0)))]
        rows.append({"model_key": model, "method": "ALL", **M.position_bias(d.to_dict("records"))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ score construction


def build_scores(df: pd.DataFrame, exp_cfg: dict) -> pd.DataFrame:
    """One row per (model_key, preference_id, framing_variant, method)."""
    cost_levels = exp_cfg["methods"]["tradeoff"]["cost_levels"]
    rows = []

    # Median self-reported strength per model, used to impute missing strengths.
    sr = df[(df["method"] == "self_report") & (df["parse_success"] == True)]  # noqa: E712
    med_strength = (sr.dropna(subset=["strength_self_report"])
                      .groupby("model_key")["strength_self_report"].median().to_dict())

    ok = df[df["call_ok"] == True]  # noqa: E712

    for (model, pid, framing, method), d in ok.groupby(
        ["model_key", "preference_id", "framing_variant", "method"]
    ):
        recs = d.to_dict("records")
        if method == "self_report":
            res = self_report.score(recs, med_strength.get(model, 0.5))
        elif method == "pairwise":
            res = pairwise.score(recs)
        elif method == "tradeoff":
            res = tradeoff.score(recs, cost_levels)
        elif method == "sequential":
            continue  # handled from the episode summaries below
        else:
            continue
        rows.append({
            "model_key": model, "preference_id": pid, "framing_variant": framing,
            "method": method, "score": res["score"], "n_used": res.get("n_used", 0),
            "sd": res.get("sd", np.nan),
            "detail": {k: v for k, v in res.items() if k not in ("score", "raw_values")},
        })

    return pd.DataFrame(rows)


def add_sequential_scores(scores: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / experiment_id / "sequential_episodes.jsonl"
    if not path.exists():
        return scores
    eps = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # episode_id = "<exp>|sequential|<model>|<pid>|<framing>|e<rep>"
    buckets: dict[tuple, list[dict]] = {}
    for e in eps:
        parts = e["episode_id"].split("|")
        if len(parts) < 6:
            continue
        buckets.setdefault((parts[2], parts[3], parts[4]), []).append(e)

    rows = []
    for (model, pid, framing), group in buckets.items():
        res = sequential.score(group)
        rows.append({
            "model_key": model, "preference_id": pid, "framing_variant": framing,
            "method": "sequential", "score": res["score"], "n_used": res.get("n_used", 0),
            "sd": res.get("sd", np.nan),
            "detail": {k: v for k, v in res.items() if k not in ("score", "raw_values")},
        })
    return pd.concat([scores, pd.DataFrame(rows)], ignore_index=True) if rows else scores


def score_matrix(scores: pd.DataFrame, model: str, framing: str,
                 items: dict[str, PreferenceItem], include_controls: bool = False) -> pd.DataFrame:
    """items x methods matrix of normalised scores for one (model, framing) cell."""
    d = scores[(scores.model_key == model) & (scores.framing_variant == framing)]
    if not include_controls:
        keep = [p for p in d.preference_id.unique() if not items[p].sanity_control]
        d = d[d.preference_id.isin(keep)]
    mat = d.pivot_table(index="preference_id", columns="method", values="score")
    cols = [m for m in METHOD_ORDER if m in mat.columns]
    return mat[cols].sort_index()


# --------------------------------------------------------------------- convergence


def convergence_table(mat: pd.DataFrame) -> pd.DataFrame:
    """All method-pair convergence metrics for one score matrix."""
    rows = []
    cols = list(mat.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = mat[a].to_numpy(), mat[b].to_numpy()
            da = M.direction_agreement(x, y)
            sp = M.bootstrap_spearman(x, y)
            rows.append({
                "method_a": a, "method_b": b,
                "direction_agreement": da["agreement"],
                "n_directed_items": da["n_compared"],
                "n_undirected_items": da["n_undirected"],
                "sign_flip_rate": M.sign_flip_rate(x, y),
                "spearman_rho": sp["rho"], "spearman_lo": sp["lo"], "spearman_hi": sp["hi"],
                "spearman_p": M.spearman(x, y)["p"],
                "pearson_r": M.pearson(x, y)["r"],
                "mean_abs_disagreement": M.mean_absolute_disagreement(x, y)["mad"],
                "n_items": int(np.isfinite(x * y).sum()),
            })
    return pd.DataFrame(rows)


def per_item_table(mat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, row in mat.iterrows():
        vals = list(row.to_numpy())
        rows.append({
            "preference_id": pid,
            **{m: row.get(m, np.nan) for m in mat.columns},
            "mean_score": float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan,
            "dispersion": M.dispersion(vals),
            "cmcs": M.cmcs(vals),
            "n_methods": int(np.sum(np.isfinite(vals))),
            "abs_pairwise_strength": abs(row.get("pairwise", np.nan)),
            "all_same_direction": len({M.direction(v) for v in vals if M.direction(v) != 0}) <= 1,
        })
    return pd.DataFrame(rows)


def strength_vs_stability(item_tbl: pd.DataFrame) -> dict:
    """Secondary hypothesis: do larger pairwise margins go with lower cross-method
    disagreement? Single pre-specified test, Spearman, no metric shopping (D-14)."""
    x = item_tbl["abs_pairwise_strength"].to_numpy()
    y = item_tbl["dispersion"].to_numpy()
    sp = M.bootstrap_spearman(x, y)
    return {
        "spearman_rho": sp["rho"], "ci_lo": sp["lo"], "ci_hi": sp["hi"],
        "p_value": M.spearman(x, y)["p"], "n_items": sp["n"],
        "note": "Negative rho supports the hypothesis (stronger pairwise margin "
                "-> less cross-method disagreement).",
    }


def framing_sensitivity(scores: pd.DataFrame, items: dict, model: str) -> dict:
    """How much do the three question framings move the results?"""
    d = scores[(scores.model_key == model)
               & (scores.method.isin(["self_report", "pairwise"]))]
    d = d[~d.preference_id.map(lambda p: items[p].sanity_control)]
    framings = sorted(d.framing_variant.unique())
    if len(framings) < 2:
        return {"available": False}

    per_method = {}
    for method in ["self_report", "pairwise"]:
        piv = d[d.method == method].pivot_table(
            index="preference_id", columns="framing_variant", values="score")
        pairs = []
        for i, f1 in enumerate(framings):
            for f2 in framings[i + 1:]:
                if f1 not in piv or f2 not in piv:
                    continue
                x, y = piv[f1].to_numpy(), piv[f2].to_numpy()
                da = M.direction_agreement(x, y)
                pairs.append({
                    "framing_a": f1, "framing_b": f2,
                    "choice_flip_rate": M.sign_flip_rate(x, y),
                    "direction_agreement": da["agreement"],
                    "mean_abs_score_change": M.mean_absolute_disagreement(x, y)["mad"],
                    "spearman_rho": M.spearman(x, y)["rho"],
                })
        per_method[method] = pairs
    return {"available": True, "framings": framings, "per_method": per_method}


# ---------------------------------------------------------------------------- driver


def run(phase: str) -> dict:
    exp_cfg = load_experiment_cfg()
    experiment_id = exp_cfg[phase]["experiment_id"]
    items = load_items()

    df = load_raw(experiment_id)
    scores = build_scores(df, exp_cfg)
    scores = add_sequential_scores(scores, experiment_id)

    proc = ROOT / "data" / "processed" / experiment_id
    res = ROOT / "results" / phase
    tbl = ROOT / "results" / "tables"
    for p in (proc, res, tbl):
        p.mkdir(parents=True, exist_ok=True)

    df.drop(columns=["messages"], errors="ignore").to_csv(
        proc / "observations.csv", index=False)
    scores.assign(detail=scores["detail"].apply(json.dumps)).to_csv(
        proc / "method_scores.csv", index=False)

    quality = quality_report(df)
    pos_bias = position_bias_report(df)
    quality.to_csv(tbl / f"{phase}_quality.csv", index=False)
    pos_bias.to_csv(tbl / f"{phase}_position_bias.csv", index=False)

    summary: dict = {
        "phase": phase, "experiment_id": experiment_id,
        "n_raw_records": int(len(df)),
        "models": sorted(scores.model_key.unique().tolist()),
        "methods": [m for m in METHOD_ORDER if m in scores.method.unique()],
        "n_items_analysed": int(
            len([p for p in scores.preference_id.unique() if not items[p].sanity_control])),
        "per_model": {},
    }

    # --- sanity controls, reported separately and excluded from the main analysis ---
    ctrl_ids = [p for p in scores.preference_id.unique() if items[p].sanity_control]
    if ctrl_ids:
        c = scores[scores.preference_id.isin(ctrl_ids)]
        summary["sanity_controls"] = {
            "item_ids": sorted(ctrl_ids),
            "mean_score_by_model_method": (
                c.groupby(["model_key", "method"])["score"].mean().round(3)
                 .reset_index().to_dict("records")),
            "note": "Semantic option A is the coherent option in every control item, "
                    "so scores near +1 indicate the model discriminates a degenerate "
                    "alternative. Controls are excluded from all convergence metrics.",
        }

    for model in summary["models"]:
        mat = score_matrix(scores, model, "neutral", items)
        conv = convergence_table(mat)
        per_item = per_item_table(mat)

        mat.to_csv(tbl / f"{phase}_{model}_score_matrix.csv")
        conv.to_csv(tbl / f"{phase}_{model}_convergence.csv", index=False)
        per_item.to_csv(tbl / f"{phase}_{model}_per_item.csv", index=False)

        n_reps = exp_cfg[phase]["repetitions"]
        baseline = M.random_baseline(len(mat), max(len(mat.columns), 2), n_reps)

        summary["per_model"][model] = {
            "n_items": int(len(mat)),
            "methods": list(mat.columns),
            "convergence_pairs": conv.round(4).to_dict("records"),
            "mean_direction_agreement": float(np.nanmean(conv["direction_agreement"])),
            "mean_spearman_rho": float(np.nanmean(conv["spearman_rho"])),
            "mean_abs_disagreement": float(np.nanmean(conv["mean_abs_disagreement"])),
            "mean_sign_flip_rate": float(np.nanmean(conv["sign_flip_rate"])),
            "cmcs": M.bootstrap_ci(per_item["cmcs"].to_numpy()),
            "n_items_all_methods_same_direction": int(per_item["all_same_direction"].sum()),
            "strength_vs_stability": strength_vs_stability(per_item),
            "random_baseline": baseline,
            "framing": framing_sensitivity(scores, items, model),
        }

    # Cross-model replication (RQ5): do the two models rank the items alike?
    if len(summary["models"]) >= 2:
        m1, m2 = summary["models"][:2]
        a = score_matrix(scores, m1, "neutral", items)
        b = score_matrix(scores, m2, "neutral", items)
        shared = [c for c in a.columns if c in b.columns]
        summary["cross_model"] = {
            "model_a": m1, "model_b": m2,
            "per_method_spearman": {
                c: M.bootstrap_spearman(a[c].reindex(b.index).to_numpy(), b[c].to_numpy())
                for c in shared
            },
        }

    (res / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_model"},
                     indent=2, default=str))
    for model, s in summary["per_model"].items():
        print(f"\n[{model}] mean direction agreement "
              f"{s['mean_direction_agreement']:.3f} "
              f"(chance {s['random_baseline']['direction_agreement_mean']:.3f}) | "
              f"mean rho {s['mean_spearman_rho']:.3f} | "
              f"mean CMCS {s['cmcs']['point']:.3f} "
              f"[{s['cmcs']['lo']:.3f}, {s['cmcs']['hi']:.3f}]")
    print(f"\nwrote -> {res / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main")
    run(ap.parse_args().phase)


if __name__ == "__main__":
    main()

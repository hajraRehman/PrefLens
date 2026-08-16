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
    """Call-failure and parse-failure rates, per model and method.

    A failed call carries `parse_stage == "not_applicable"` (there was no text to
    parse), exactly like a free-text `perform_task` turn. Filtering on that field
    alone would therefore hide every call failure and force the failure rate to
    zero. Choice turns are identified from `extra.stage_kind` instead, so failures
    stay visible; parse rates are then conditioned on calls that actually
    returned text.
    """
    is_choice = df["extra"].apply(lambda e: (e or {}).get("stage_kind") != "perform_task")
    ct = df[is_choice]

    rows = []
    for (mk, meth), d in ct.groupby(["model_key", "method"], dropna=False):
        ok = d[d["call_ok"].astype(bool)]
        rows.append({
            "model_key": mk,
            "method": meth,
            "n_choice_calls": len(d),
            "n_call_failures": int((~d["call_ok"].astype(bool)).sum()),
            "call_failure_rate": float((~d["call_ok"].astype(bool)).mean()),
            "n_parse_failures": int((~ok["parse_success"].fillna(False).astype(bool)).sum()),
            # Conditioned on a successful call: a 429 is not a parsing problem.
            "parse_failure_rate": (
                float((~ok["parse_success"].fillna(False).astype(bool)).mean())
                if len(ok) else float("nan")),
            "pct_strict_json": (float((ok["parse_stage"] == "strict_json").mean())
                                if len(ok) else float("nan")),
            "pct_recovered_from_prose": (
                float(ok["parse_stage"].isin(
                    ["labelled_text", "bare_token", "embedded_json"]).mean())
                if len(ok) else float("nan")),
            "mean_latency_s": float(d["latency_s"].mean()),
            "mean_attempts": float(d["attempts"].mean()),
        })
    return pd.DataFrame(rows)


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


def signal_vs_position_table(df: pd.DataFrame, items: dict) -> pd.DataFrame:
    """Per (model, method, item) split of behaviour into position vs content.

    This is the diagnostic that tells a genuine method disagreement apart from a
    measure that carries no preference information at all. Sanity controls are
    excluded; only the neutral framing is used, so it matches the convergence
    analysis exactly.
    """
    ok = df[(df["parse_success"] == True) & (df["framing_variant"] == "neutral")]  # noqa: E712
    ok = ok[~ok["preference_id"].map(lambda p: items[p].sanity_control)]

    rows = []
    for (mk, meth, pid), d in ok.groupby(["model_key", "method", "preference_id"]):
        if meth == "tradeoff":      # only the zero-cost rung is order-symmetric
            d = d[d["extra"].apply(lambda e: (e or {}).get("signed_cost") == 0)]
        elif meth == "sequential":  # only the initial choice is a clean A/B pair
            d = d[d["extra"].apply(lambda e: (e or {}).get("stage_kind") == "initial_choice")]
        if len(d) < 4:
            continue
        r = M.position_bias(d.to_dict("records"))
        if not np.isfinite(r.get("content_effect", np.nan)):
            continue
        rows.append({"model_key": mk, "method": meth, "preference_id": pid,
                     "p_a_when_first": r["p_a_when_first"],
                     "p_a_when_second": r["p_a_when_second"],
                     "position_effect": r["position_effect"],
                     "content_effect": r["content_effect"],
                     "n": r["n_first"] + r["n_second"]})
    return pd.DataFrame(rows)


def signal_summary(sv: pd.DataFrame) -> dict:
    """Aggregate signal_share per (model, method), flagging degenerate measures."""
    out: dict = {}
    for (mk, meth), d in sv.groupby(["model_key", "method"]):
        out.setdefault(mk, {})[meth] = M.signal_share(
            d["content_effect"].tolist(), d["position_effect"].tolist())
    return out


MIN_COVERAGE = 0.8
"""A cell must have at least this fraction of its planned observations to be
scored. Partially observed cells are not comparable with complete ones: a
trade-off score built from 3 of 9 cost rungs is a different estimator from one
built from all 9, and averaging them silently would let an interrupted run
(D-24) inflate agreement. Under-covered cells are set to NaN and reported."""


def _sufficient(method: str, res: dict, exp_cfg: dict, reps: int) -> tuple[bool, str]:
    """Does this cell have enough data to be scored?"""
    if method == "tradeoff":
        want = len(tradeoff.signed_levels(exp_cfg["methods"]["tradeoff"]["cost_levels"]))
        got = res.get("n_levels", 0)
        # Every rung must be present: the score averages P(A) across the ladder,
        # so a missing rung changes what is being averaged.
        return got >= want, f"levels {got}/{want}"
    if method == "sequential":
        want = exp_cfg["methods"]["sequential"]["repetitions"]
        got = res.get("n_used", 0)
        return got >= MIN_COVERAGE * want, f"episodes {got}/{want}"
    got = res.get("n_used", 0)
    return got >= MIN_COVERAGE * reps, f"reps {got}/{reps}"


def build_scores(df: pd.DataFrame, exp_cfg: dict, reps: int | None = None) -> pd.DataFrame:
    """One row per (model_key, preference_id, framing_variant, method)."""
    cost_levels = exp_cfg["methods"]["tradeoff"]["cost_levels"]
    if reps is None:
        reps = exp_cfg["main"]["repetitions"]
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
        enough, cov = _sufficient(method, res, exp_cfg, reps)
        rows.append({
            "model_key": model, "preference_id": pid, "framing_variant": framing,
            "method": method,
            "score": res["score"] if enough else np.nan,
            "score_raw": res["score"],       # retained for auditing, never analysed
            "sufficient_coverage": enough,
            "coverage": cov,
            "n_used": res.get("n_used", 0),
            "sd": res.get("sd", np.nan),
            "detail": {k: v for k, v in res.items() if k not in ("score", "raw_values")},
        })

    return pd.DataFrame(rows)


def reconstruct_episode_summaries(df: pd.DataFrame, stages: int) -> list[dict]:
    """Rebuild sequential-episode occupancy from the raw records.

    The runner also writes a `sequential_episodes.jsonl` side file, but that file
    is rewritten per run, so a second phase sharing an experiment_id (e.g. a model
    family added later) would truncate the earlier phase's summaries. The raw log
    is append-only and therefore the only trustworthy source. Deriving from it
    here removes the dependency entirely.

    An episode counts as complete only if every one of its `stages` choice turns
    parsed; otherwise it is dropped from scoring and counted as incomplete, which
    matches `sequential.run_episode`'s own abandonment rule.
    """
    seq = df[(df["method"] == "sequential") & (df["parse_stage"] != "not_applicable")]
    out = []
    for episode_id, d in seq.groupby(seq["extra"].apply(lambda e: (e or {}).get("episode_id"))):
        if not episode_id:
            continue
        slots: list[str] = []
        broken = False
        turns = sorted(d.to_dict("records"),
                       key=lambda r: (r["extra"] or {}).get("stage", 0))
        for r in turns:
            kind = (r["extra"] or {}).get("stage_kind")
            if kind not in ("initial_choice", "continue_or_switch"):
                continue
            sem = displayed_to_semantic(r.get("parsed_choice"), r["display_order"])
            if sem is None:
                broken = True
                break
            slots.append(sem)
        complete = (not broken) and len(slots) == stages
        out.append({
            "episode_id": episode_id,
            "complete": complete,
            "occupancy": (sum(1 for s in slots if s == "A") / len(slots)) if complete else None,
            "slots": slots,
        })
    return out


def add_sequential_scores(scores: pd.DataFrame, experiment_id: str,
                          df: pd.DataFrame, stages: int,
                          episodes_planned: int = 5) -> pd.DataFrame:
    eps = reconstruct_episode_summaries(df, stages)
    if not eps:
        return scores
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
        enough = res.get("n_used", 0) >= MIN_COVERAGE * episodes_planned
        rows.append({
            "model_key": model, "preference_id": pid, "framing_variant": framing,
            "method": "sequential",
            "score": res["score"] if enough else np.nan,
            "score_raw": res["score"],
            "sufficient_coverage": enough,
            "coverage": f"episodes {res.get('n_used', 0)}/{episodes_planned}",
            "n_used": res.get("n_used", 0),
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


def matched_subset(scores: pd.DataFrame, items: dict, models: list[str]) -> dict:
    """Convergence recomputed on the methods and items common to every model.

    Cross-model comparison is otherwise invalid. Agreement statistics depend
    mechanically on how many methods are averaged over and on which items are
    included, so a model measured with 3 methods on 10 items cannot be compared
    against one measured with 4 methods on 12. The Gemini arm is a partial run
    (free-tier quota exhausted, D-24), which makes this mandatory rather than
    optional.

    Returns the shared basis and per-model convergence restricted to it.
    """
    per_model_mat = {m: score_matrix(scores, m, "neutral", items) for m in models}
    per_model_mat = {m: mat for m, mat in per_model_mat.items() if not mat.empty}
    if len(per_model_mat) < 2:
        return {"available": False}

    common_methods = set.intersection(
        *[{c for c in mat.columns if mat[c].notna().any()} for mat in per_model_mat.values()])
    if not common_methods:
        return {"available": False, "reason": "no method common to all models"}
    ordered = [m for m in METHOD_ORDER if m in common_methods]

    # An item qualifies only if every model has a finite score for it under
    # every shared method.
    common_items = set.intersection(*[
        {i for i in mat.index if mat.loc[i, ordered].notna().all()}
        for mat in per_model_mat.values()
    ])
    if len(common_items) < 3:
        return {"available": False, "reason": f"only {len(common_items)} shared items"}
    idx = sorted(common_items)

    out = {"available": True, "methods": ordered, "items": idx,
           "n_methods": len(ordered), "n_items": len(idx), "per_model": {}}
    for m, mat in per_model_mat.items():
        sub = mat.loc[idx, ordered]
        conv = convergence_table(sub)
        per_item = per_item_table(sub)
        out["per_model"][m] = {
            "mean_direction_agreement": float(np.nanmean(conv["direction_agreement"])),
            "mean_spearman_rho": float(np.nanmean(conv["spearman_rho"])),
            "mean_abs_disagreement": float(np.nanmean(conv["mean_abs_disagreement"])),
            "mean_sign_flip_rate": float(np.nanmean(conv["sign_flip_rate"])),
            "cmcs": M.bootstrap_ci(per_item["cmcs"].to_numpy()),
            "n_items_all_methods_same_direction": int(per_item["all_same_direction"].sum()),
            "convergence_pairs": conv.round(4).to_dict("records"),
        }
    return out


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
    scores = add_sequential_scores(
        scores, experiment_id, df, exp_cfg["methods"]["sequential"]["stages"],
        exp_cfg["methods"]["sequential"]["repetitions"])

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

    sv = signal_vs_position_table(df, items)
    sv.to_csv(tbl / f"{phase}_signal_vs_position.csv", index=False)
    sig = signal_summary(sv)

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

        degenerate = [m for m, v in sig.get(model, {}).items() if v.get("degenerate")]

        summary["per_model"][model] = {
            "n_items": int(len(mat)),
            "methods": list(mat.columns),
            "signal_vs_position": sig.get(model, {}),
            # A degenerate measure encodes only the random display-order draw.
            # Convergence involving it is uninterpretable as method disagreement.
            "degenerate_methods": degenerate,
            "convergence_interpretable": not degenerate,
            "convergence_pairs": conv.round(4).to_dict("records"),
            "mean_direction_agreement": float(np.nanmean(conv["direction_agreement"])),
            "mean_spearman_rho": float(np.nanmean(conv["spearman_rho"])),
            "mean_abs_disagreement": float(np.nanmean(conv["mean_abs_disagreement"])),
            "mean_sign_flip_rate": float(np.nanmean(conv["sign_flip_rate"])),
            "cmcs": M.bootstrap_ci(per_item["cmcs"].to_numpy()),
            "n_items_all_methods_same_direction": int(per_item["all_same_direction"].sum()),
            # H2 regresses cross-method disagreement on |pairwise strength|. If the
            # pairwise measure is degenerate, that x-axis is the random display-order
            # draw and the test is meaningless — flagged, not silently reported.
            "strength_vs_stability": {
                **strength_vs_stability(per_item),
                "valid": "pairwise" not in degenerate,
                "invalid_reason": ("pairwise measure is degenerate (pure position "
                                   "responding); |pairwise strength| carries no "
                                   "preference information"
                                   if "pairwise" in degenerate else None),
            },
            "random_baseline": baseline,
            "framing": framing_sensitivity(scores, items, model),
        }

    # Apples-to-apples cross-model comparison on a shared basis (D-24).
    summary["matched_subset"] = matched_subset(scores, items, summary["models"])

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
        for meth, v in sorted(s.get("signal_vs_position", {}).items()):
            flag = "  <-- DEGENERATE: pure position responding" if v["degenerate"] else ""
            print(f"          {meth:<12} |content|={v['mean_abs_content']:.3f} "
                  f"position={v['mean_position']:+.3f} "
                  f"items with signal {v['n_items_with_content']}/{v['n_items']}{flag}")
        if s["degenerate_methods"]:
            print(f"          !! convergence for {model} is NOT interpretable as method "
                  f"disagreement: {s['degenerate_methods']} carry no signal")
    ms = summary.get("matched_subset", {})
    if ms.get("available"):
        print(f"\n--- MATCHED SUBSET: {ms['n_methods']} methods x {ms['n_items']} items "
              f"common to all models ---")
        print(f"    methods: {', '.join(ms['methods'])}")
        for m, v in ms["per_model"].items():
            deg = summary["per_model"][m]["degenerate_methods"]
            flag = f"   [degenerate: {deg}]" if deg else ""
            print(f"    {m:<22} dir.agree {v['mean_direction_agreement']:.3f}  "
                  f"rho {v['mean_spearman_rho']:+.3f}  "
                  f"all-agree {v['n_items_all_methods_same_direction']}/{ms['n_items']}{flag}")
    else:
        print(f"\nmatched subset unavailable: {ms.get('reason', 'n/a')}")

    print(f"\nwrote -> {res / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main")
    run(ap.parse_args().phase)


if __name__ == "__main__":
    main()

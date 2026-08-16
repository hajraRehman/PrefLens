"""Figures.

    python -m src.plotting --phase main

Figure 1  method x item score heatmap
Figure 2  method-pair Spearman correlation matrix
Figure 3  pairwise strength vs cross-method disagreement
Figure 4  framing sensitivity (only if >1 framing was run)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import analysis as A
from . import metrics as M
from .methods import METHOD_LABELS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
DPI = 300

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

SHORT = {"self_report": "A\nself-report", "pairwise": "B\npairwise",
         "tradeoff": "C\ntrade-off", "sequential": "D\nsequential"}


def _stamp(fig, model: str, phase: str, n_items: int) -> None:
    # Placed below the axes (negative figure coords) so it cannot collide with
    # tick labels; savefig's tight bbox expands to include it.
    fig.text(0.0, -0.045,
             f"model: {model}  |  phase: {phase}  |  {n_items} preference items  |  "
             f"scores in [-1,+1], + favours semantic option A",
             fontsize=6.5, color="0.35", ha="left", va="top")


def fig1_heatmap(mat, model: str, phase: str) -> Path:
    fig, ax = plt.subplots(figsize=(1.5 + 1.15 * len(mat.columns), 0.42 * len(mat) + 2.0))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([SHORT.get(c, c) for c in mat.columns])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    ax.set_ylabel("preference item")
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 else "black")

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("normalised score\n(-1 = option B, +1 = option A)", fontsize=8)
    ax.set_title("Fig 1. Elicited preference score by item and elicitation method")
    _stamp(fig, model, phase, len(mat))
    p = FIG / f"fig1_heatmap_{phase}_{model}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2_corr(mat, model: str, phase: str) -> Path:
    cols = list(mat.columns)
    k = len(cols)
    R = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            R[i, j] = 1.0 if i == j else M.spearman(mat[cols[i]], mat[cols[j]])["rho"]

    fig, ax = plt.subplots(figsize=(1.0 + 1.1 * k, 1.0 + 1.0 * k))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels([SHORT.get(c, c) for c in cols])
    ax.set_yticks(range(k)); ax.set_yticklabels([SHORT.get(c, c) for c in cols])
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            t = "n/a" if not np.isfinite(R[i, j]) else f"{R[i, j]:+.2f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(R[i, j]) and abs(R[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03).set_label("Spearman rho", fontsize=8)
    ax.set_title("Fig 2. Rank correlation of item scores between methods")
    _stamp(fig, model, phase, len(mat))
    p = FIG / f"fig2_method_corr_{phase}_{model}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3_strength_vs_disagreement(item_tbl, stats_d: dict, model: str, phase: str,
                                  degenerate: bool = False) -> Path:
    """H2: |pairwise strength| vs cross-method disagreement.

    When the pairwise measure is degenerate (pure position responding) the x-axis
    is the random display-order draw, so the statistic is meaningless. The figure
    is still produced for transparency, but the statistic is suppressed from the
    title and the panel is stamped INVALID — a figure must never display a number
    the report withdraws.
    """
    x = item_tbl["abs_pairwise_strength"].to_numpy(dtype=float)
    y = item_tbl["dispersion"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colour = "#a0aec0" if degenerate else "#2b6cb0"
    ax.scatter(x[m], y[m], s=42, color=colour, edgecolor="white", zorder=3)
    for xi, yi, pid in zip(x[m], y[m], item_tbl["preference_id"].to_numpy()[m]):
        ax.annotate(pid, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=6.5,
                    color="0.35")
    # No trend line on a degenerate x-axis: it would imply a relationship exists.
    if not degenerate and m.sum() >= 3 and len(np.unique(x[m])) > 1:
        b, a = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, a + b * xs, color="#c05621", lw=1.4, ls="--", zorder=2,
                label="OLS trend (visual aid only)")
        ax.legend(fontsize=7, loc="best")

    ax.set_xlabel("|pairwise preference strength|  (Method B, 0 = indifferent)")
    ax.set_ylabel("cross-method disagreement\n(mean abs. deviation of method scores)")

    if degenerate:
        ax.set_title(
            "Fig 3. Do stronger pairwise preferences agree better across methods?\n"
            "STATISTIC WITHDRAWN — the pairwise measure for this model is pure\n"
            "position artefact, so the x-axis carries no preference information",
            fontsize=8.5, color="#9b2c2c",
        )
        ax.text(0.5, 0.5, "INVALID", transform=ax.transAxes, fontsize=44,
                color="#9b2c2c", alpha=0.16, ha="center", va="center",
                rotation=22, zorder=1, fontweight="bold")
    else:
        rho, lo, hi = stats_d["spearman_rho"], stats_d["ci_lo"], stats_d["ci_hi"]
        ax.set_title(
            "Fig 3. Do stronger pairwise preferences agree better across methods?\n"
            f"Spearman rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
            f"p = {stats_d['p_value']:.3f}  n = {stats_d['n_items']}\n"
            "positive rho = the OPPOSITE of the hypothesis",
            fontsize=8.5,
        )
    _stamp(fig, model, phase, int(m.sum()))
    p = FIG / f"fig3_strength_vs_disagreement_{phase}_{model}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig4_framing(framing: dict, model: str, phase: str) -> Path | None:
    if not framing.get("available"):
        return None
    methods = [m for m in ("self_report", "pairwise") if framing["per_method"].get(m)]
    if not methods:
        return None

    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 3.6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        rows = framing["per_method"][method]
        labels = [f"{r['framing_a']}\nvs {r['framing_b']}" for r in rows]
        xpos = np.arange(len(rows))
        w = 0.38
        ax.bar(xpos - w / 2, [r["choice_flip_rate"] for r in rows], w,
               label="direction-flip rate", color="#c05621")
        ax.bar(xpos + w / 2, [r["mean_abs_score_change"] for r in rows], w,
               label="mean |score change|", color="#2b6cb0")
        ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, max(0.5, max(
            [r["mean_abs_score_change"] for r in rows] +
            [r["choice_flip_rate"] for r in rows if np.isfinite(r["choice_flip_rate"])] + [0]) * 1.25))
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=9)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("effect of changing the question wording")
    fig.suptitle("Fig 4. Sensitivity of elicited preferences to question framing", fontsize=10)
    _stamp(fig, model, phase, 0)
    p = FIG / f"fig4_framing_{phase}_{model}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig5_signal_vs_convergence(summary: dict, phase: str) -> Path | None:
    """The study's central diagnostic: convergence plotted against how much
    position-independent preference signal each model actually produced."""
    ms = (summary or {}).get("matched_subset", {})
    if not ms.get("available"):
        return None

    rows = []
    for model, v in ms["per_model"].items():
        sig = summary["per_model"][model].get("signal_vs_position", {})
        # Average |content| over the methods included in the matched subset.
        vals = [sig[m]["mean_abs_content"] for m in ms["methods"]
                if m in sig and np.isfinite(sig[m]["mean_abs_content"])]
        if not vals:
            continue
        rows.append({
            "model": model,
            "content": float(np.mean(vals)),
            "rho": v["mean_spearman_rho"],
            "agree": v["mean_direction_agreement"],
            "degenerate": bool(summary["per_model"][model]["degenerate_methods"]),
        })
    if len(rows) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, key, lab in (
        (axes[0], "rho", "mean Spearman rho between methods"),
        (axes[1], "agree", "mean direction agreement"),
    ):
        for r in rows:
            colour = "#c05621" if r["degenerate"] else "#2b6cb0"
            ax.scatter(r["content"], r[key], s=110, color=colour,
                       edgecolor="white", zorder=3)
            ax.annotate(r["model"].replace("-31-flash-lite", "-3.1-fl"),
                        (r["content"], r[key]), textcoords="offset points",
                        xytext=(7, -3), fontsize=7.5)
        ax.set_xlabel("mean |content effect|\n(order-free preference signal)")
        ax.set_ylabel(lab)
        ax.set_xlim(-0.02, max(r["content"] for r in rows) * 1.35)

    # Matched permutation null (mean across models), not the deprecated
    # parametric baseline (D-32).
    nulls = [summary["per_model"][r["model"]]["permutation_null"]
             ["mean_direction_agreement"]["null_mean"] for r in rows]
    axes[1].axhline(float(np.mean(nulls)), ls=":", color="0.5", lw=1,
                    label="permutation null (mean)")
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Fig 5. Cross-method convergence tracks position-independent signal\n"
        f"matched subset: {ms['n_methods']} methods x {ms['n_items']} items; "
        "orange = measure degenerate (pure position responding)",
        fontsize=9.5,
    )
    _stamp(fig, "all models", phase, ms["n_items"])
    p = FIG / f"fig5_signal_vs_convergence_{phase}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def run(phase: str) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    exp_cfg = A.load_experiment_cfg()
    experiment_id = exp_cfg[phase]["experiment_id"]
    items = A.load_items()

    df = A.load_raw(experiment_id)
    scores = A.add_sequential_scores(
        A.build_scores(df, exp_cfg), experiment_id, df,
        exp_cfg["methods"]["sequential"]["stages"],
        exp_cfg["methods"]["sequential"]["repetitions"])

    summary_path = ROOT / "results" / phase / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    made = []
    for model in sorted(scores.model_key.unique()):
        mat = A.score_matrix(scores, model, "neutral", items)
        if mat.empty:
            continue
        per_item = A.per_item_table(mat)
        made.append(fig1_heatmap(mat, model, phase))
        if len(mat.columns) >= 2:
            made.append(fig2_corr(mat, model, phase))
        degen = bool((summary or {}).get("per_model", {})
                     .get(model, {}).get("degenerate_methods"))
        made.append(fig3_strength_vs_disagreement(
            per_item, A.strength_vs_stability(per_item), model, phase, degenerate=degen))
        fr = (summary or {}).get("per_model", {}).get(model, {}).get("framing") \
            or A.framing_sensitivity(scores, items, model)
        f4 = fig4_framing(fr, model, phase)
        if f4:
            made.append(f4)

    f5 = fig5_signal_vs_convergence(summary, phase)
    if f5:
        made.append(f5)

    for p in made:
        print(f"wrote {p}")
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main")
    run(ap.parse_args().phase)


if __name__ == "__main__":
    main()

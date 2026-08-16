"""Study 2 figures.

    python -m src.followup.plotting

A  position vs content plane, per item, both models
B  model-level comparison with bootstrap CIs
C  per-item absolute position effect (is the aggregate driven by a few items?)
D  sanity controls vs balanced items
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "followup" / "figures"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

COLOURS = {"gpt-oss-20b": "#2b6cb0", "gpt-oss-120b": "#c05621"}
LABEL = {"gpt-oss-20b": "GPT-OSS 20B", "gpt-oss-120b": "GPT-OSS 120B"}


def _stamp(fig, extra: str = "") -> None:
    fig.text(0.0, -0.04,
             "Study 2 — within-family position-bias follow-up | both models via OpenRouter, "
             "identical prompts and sampling | exact order counterbalancing, 10 trials per "
             "position per item" + (f" | {extra}" if extra else ""),
             fontsize=6.3, color="0.35", ha="left", va="top")


def fig_a_plane(item_tbl: pd.DataFrame, baseline: dict) -> Path:
    d = item_tbl[~item_tbl.is_control]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    # Region where coin-flipping alone lands (95% of simulated items).
    ax.axhspan(0, baseline["abs_position_p95"], color="0.85", alpha=0.5, zorder=0,
               label=f"chance zone (|position| ≤ {baseline['abs_position_p95']:.2f})")

    # Many items land on exactly the same coordinates (notably 120B at 0, 1.0),
    # which hides both the markers and their labels. A small deterministic jitter
    # separates them; magnitude is stated in the caption so no reader mistakes it
    # for measured spread.
    rng = np.random.default_rng(0)
    for mk, g in d.groupby("model_key"):
        jx = rng.uniform(-0.012, 0.012, len(g))
        jy = rng.uniform(-0.012, 0.012, len(g))
        ax.scatter(g["abs_content_signal"] + jx, g["abs_position_effect"] + jy, s=70,
                   color=COLOURS.get(mk, "#555"), edgecolor="white", zorder=3,
                   label=LABEL.get(mk, mk))
        for (_, r), dx, dy in zip(g.iterrows(), jx, jy):
            ax.annotate(r["preference_id"],
                        (r["abs_content_signal"] + dx, r["abs_position_effect"] + dy),
                        textcoords="offset points", xytext=(5, 3), fontsize=6,
                        color="0.4")

    n_pinned = int(((d.model_key == "gpt-oss-120b")
                    & (d.abs_position_effect >= 1.0)
                    & (d.abs_content_signal <= 1e-9)).sum())
    if n_pinned:
        ax.annotate(f"{n_pinned}/12 GPT-OSS 120B items sit exactly at\n"
                    f"|position| = 1.0, |content| = 0.0",
                    xy=(0.02, 1.0), xytext=(0.30, 0.94), fontsize=7, color="#9b2c2c",
                    arrowprops=dict(arrowstyle="->", color="#9b2c2c", lw=0.9))

    lim = 1.05
    ax.plot([0, lim], [0, lim], ls=":", color="0.55", lw=1)
    ax.text(0.72, 0.66, "equal influence", rotation=39, fontsize=6.5, color="0.45")
    ax.text(0.06, 0.95, "position-dominated", fontsize=8, color="#9b2c2c", fontweight="bold")
    ax.text(0.55, 0.06, "content-dominated", fontsize=8, color="#1a4731", fontweight="bold")

    ax.set_xlim(-0.03, lim); ax.set_ylim(-0.03, lim)
    ax.set_xlabel("|order-invariant content-associated signal|")
    ax.set_ylabel("|position effect|")
    ax.set_title("Fig A. Position vs content plane, one point per preference item")
    ax.legend(fontsize=7.5, loc="center right")
    _stamp(fig, "markers jittered by <=0.012 to separate exact ties")
    p = FIG / "figA_position_content_plane.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_b_model_comparison(summary: dict) -> Path:
    keys = list(summary["per_model"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    width = 0.35
    xs = np.arange(2)   # position, content

    for i, mk in enumerate(keys):
        s = summary["per_model"][mk]
        means = [s["mean_abs_position_effect"]["mean"], s["mean_abs_content_signal"]["mean"]]
        los = [s["mean_abs_position_effect"]["lo"], s["mean_abs_content_signal"]["lo"]]
        his = [s["mean_abs_position_effect"]["hi"], s["mean_abs_content_signal"]["hi"]]
        err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
        ax.bar(xs + (i - 0.5) * width, means, width, yerr=err, capsize=4,
               color=COLOURS.get(mk, "#555"), label=LABEL.get(mk, mk))

    b = summary["random_baseline"]
    ax.axhline(b["mean_abs_position"], ls="--", color="0.4", lw=1.2,
               label=f"chance ({b['mean_abs_position']:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(["mean |position effect|", "mean |content signal|"])
    ax.set_ylabel("value (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig B. Model-level comparison, 95% bootstrap CIs over 12 items")
    ax.legend(fontsize=7.5)
    _stamp(fig, "error bars = percentile bootstrap over items")
    p = FIG / "figB_model_comparison.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_c_per_item(item_tbl: pd.DataFrame, baseline: dict) -> Path:
    d = item_tbl[~item_tbl.is_control]
    piv = d.pivot_table(index="preference_id", columns="model_key",
                        values="abs_position_effect").sort_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = np.arange(len(piv))
    width = 0.38
    for i, mk in enumerate(piv.columns):
        ax.bar(xs + (i - 0.5) * width, piv[mk], width,
               color=COLOURS.get(mk, "#555"), label=LABEL.get(mk, mk))
    ax.axhline(baseline["abs_position_p95"], ls="--", color="0.4", lw=1.2,
               label=f"chance 95th pct ({baseline['abs_position_p95']:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(piv.index, rotation=0, fontsize=8)
    ax.set_xlabel("preference item")
    ax.set_ylabel("|position effect|")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig C. Per-item position effect — is the aggregate driven by a few items?")
    ax.legend(fontsize=7.5)
    _stamp(fig)
    p = FIG / "figC_per_item_position.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_d_controls(item_tbl: pd.DataFrame, summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    keys = list(summary["per_model"])
    xs = np.arange(len(keys))
    width = 0.35

    bal = [item_tbl[(item_tbl.model_key == k) & (~item_tbl.is_control)]
           ["abs_position_effect"].mean() for k in keys]
    ctl = [item_tbl[(item_tbl.model_key == k) & (item_tbl.is_control)]
           ["abs_position_effect"].mean() for k in keys]

    ax.bar(xs - width / 2, bal, width, color="#c05621", label="balanced preference items")
    ax.bar(xs + width / 2, ctl, width, color="#2f855a", label="sanity controls")
    for i, k in enumerate(keys):
        acc = summary["sanity_controls"][k]["accuracy"]
        ax.annotate(f"control accuracy\n{acc:.0%}", (i + width / 2, ctl[i]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    ax.set_xticks(xs); ax.set_xticklabels([LABEL.get(k, k) for k in keys])
    ax.set_ylabel("mean |position effect|")
    ax.set_ylim(0, 1.12)
    ax.set_title("Fig D. Position dominance disappears when one option is clearly invalid")
    ax.legend(fontsize=7.5)
    _stamp(fig, "controls are excluded from all principal statistics")
    p = FIG / "figD_sanity_controls.png"
    fig.savefig(p); plt.close(fig)
    return p


def run() -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (ROOT / "results" / "followup" / "statistics" / "followup_summary.json")
        .read_text(encoding="utf-8"))
    item_tbl = pd.read_csv(
        ROOT / "results" / "followup" / "tables" / "gpt_oss_position_decomposition.csv")
    b = summary["random_baseline"]

    made = [fig_a_plane(item_tbl, b), fig_b_model_comparison(summary),
            fig_c_per_item(item_tbl, b), fig_d_controls(item_tbl, summary)]
    for p in made:
        print(f"wrote {p}")
    return made


if __name__ == "__main__":
    run()

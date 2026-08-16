"""Robustness checks for Study 1 analytic choices.

    python -m src.robustness

Currently: dead-zone sensitivity. Direction agreement and sign-flip rate depend
on a threshold below which a score counts as "no direction expressed". The value
0.05 was chosen on judgement, not derived, so the conclusions must be shown not
to depend on it (D-35).

Costs no API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from . import metrics as M

ROOT = Path(__file__).resolve().parents[1]
OUT_TABLE = ROOT / "results" / "tables" / "main_dead_zone_sensitivity.csv"
OUT_JSON = ROOT / "results" / "main" / "dead_zone_sensitivity.json"

DEAD_ZONES = [0.00, 0.05, 0.10, 0.20]


def _stats_at(mat: np.ndarray, dz: float) -> dict:
    """Mean direction agreement / sign-flip over all method pairs at dead zone dz."""
    a = np.asarray(mat, dtype=float)
    finite = np.isfinite(a)
    dirs = np.where(~finite | (np.abs(a) <= dz), 0, np.sign(a)).astype(int)
    das, flips, undirected = [], [], []
    k = a.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            di, dj = dirs[:, i], dirs[:, j]
            both = (di != 0) & (dj != 0)
            n = int(both.sum())
            undirected.append(a.shape[0] - n)
            if n:
                agree = int((di[both] == dj[both]).sum())
                das.append(agree / n)
                flips.append((n - agree) / n)
            else:
                das.append(np.nan)
                flips.append(np.nan)
    with np.errstate(invalid="ignore"):
        return {
            "mean_direction_agreement": float(np.nanmean(das)) if das else np.nan,
            "mean_sign_flip_rate": float(np.nanmean(flips)) if flips else np.nan,
            "mean_undirected_items_per_pair": float(np.mean(undirected)),
        }


def run(phase: str = "main") -> pd.DataFrame:
    exp = A.load_experiment_cfg()
    items = A.load_items()
    df = A.load_raw(exp[phase]["experiment_id"])
    scores = A.add_sequential_scores(
        A.build_scores(df, exp), exp[phase]["experiment_id"], df,
        exp["methods"]["sequential"]["stages"],
        exp["methods"]["sequential"]["repetitions"])

    models = sorted(scores.model_key.unique())
    ms = A.matched_subset(scores, items, models, n_perm=0)
    if not ms.get("available"):
        raise SystemExit("matched subset unavailable")
    methods, idx = ms["methods"], ms["items"]

    rows = []
    for dz in DEAD_ZONES:
        for mk in models:
            mat = A.score_matrix(scores, mk, "neutral", items)
            sub = mat.loc[idx, methods].to_numpy(dtype=float)
            rows.append({"dead_zone": dz, "model_key": mk, **_stats_at(sub, dz)})
    tbl = pd.DataFrame(rows)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT_TABLE, index=False)

    piv = tbl.pivot(index="dead_zone", columns="model_key",
                    values="mean_direction_agreement").round(3)
    # Is the model ordering the same at every threshold?
    orders = [tuple(piv.loc[dz].sort_values(ascending=False).index) for dz in DEAD_ZONES]
    stable = len(set(orders)) == 1
    # A sorted order hides ties. Report them: at some thresholds two models can be
    # numerically equal, and calling that a stable "ordering" would overstate it.
    ties = {str(dz): [list(g) for _, g in piv.loc[dz].groupby(piv.loc[dz]) if len(g) > 1]
            for dz in DEAD_ZONES}
    ties = {k: v for k, v in ties.items() if v}
    gemini_top = all(piv.loc[dz].idxmax() == "gemini-31-flash-lite" for dz in DEAD_ZONES)

    summary = {
        "dead_zones": DEAD_ZONES,
        "basis": {"methods": methods, "n_items": len(idx), "items": idx},
        "direction_agreement": piv.to_dict(),
        "model_ordering_per_dead_zone": {str(dz): list(o) for dz, o in zip(DEAD_ZONES, orders)},
        "ordering_stable_ignoring_ties": bool(stable),
        "ties_at_dead_zone": ties,
        "gemini_highest_at_every_dead_zone": bool(gemini_top),
        "default_dead_zone": M.DEAD_ZONE,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Dead-zone sensitivity — mean direction agreement "
          f"(matched basis: {len(methods)} methods x {len(idx)} items)\n")
    print(piv.to_string())
    print(f"\nmodel ordering identical at every dead zone: {stable}")
    for dz, o in zip(DEAD_ZONES, orders):
        print(f"   dz={dz:.2f}: {' > '.join(o)}")
    print(f"\nwrote -> {OUT_TABLE}")
    return tbl


if __name__ == "__main__":
    run()

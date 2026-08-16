"""Convergence metrics over normalised method scores.

Every score fed to this module is a signed strength in [-1, +1] where
  +1 = strongest possible evidence for semantic option A
   0 = indifference
  -1 = strongest possible evidence for semantic option B

Metrics are deliberately plural and individually interpretable. No single
composite is treated as the headline result (Section 15/16 of the brief).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

DEAD_ZONE = 0.05
"""Scores with |s| <= DEAD_ZONE are treated as 'no direction expressed' rather
than as a direction. Without a dead zone, a score of +0.001 would count as a
full-strength agreement or disagreement, which would make direction agreement
and sign-flip rate extremely noisy near indifference."""


# ---------------------------------------------------------------- direction metrics


def direction(score: float, dead_zone: float = DEAD_ZONE) -> int:
    """+1 (A), -1 (B), or 0 (no direction expressed / missing)."""
    if score is None or not np.isfinite(score) or abs(score) <= dead_zone:
        return 0
    return 1 if score > 0 else -1


def direction_agreement(x: np.ndarray, y: np.ndarray, dead_zone: float = DEAD_ZONE) -> dict:
    """Fraction of items on which two methods point the same way.

    Items where either method expresses no direction are excluded from the
    numerator and denominator, and counted separately, rather than being scored
    as agreement or disagreement by fiat.
    """
    dx = np.array([direction(v, dead_zone) for v in x])
    dy = np.array([direction(v, dead_zone) for v in y])
    both = (dx != 0) & (dy != 0)
    n = int(both.sum())
    if n == 0:
        return {"agreement": np.nan, "n_compared": 0, "n_undirected": int(len(x)),
                "n_agree": 0, "n_disagree": 0}
    agree = int((dx[both] == dy[both]).sum())
    return {
        "agreement": agree / n,
        "n_compared": n,
        "n_undirected": int(len(x) - n),
        "n_agree": agree,
        "n_disagree": n - agree,
    }


def sign_flip_rate(x: np.ndarray, y: np.ndarray, dead_zone: float = DEAD_ZONE) -> float:
    """Fraction of directed items on which two methods infer OPPOSITE directions."""
    d = direction_agreement(x, y, dead_zone)
    if d["n_compared"] == 0:
        return np.nan
    return d["n_disagree"] / d["n_compared"]


# -------------------------------------------------------------- magnitude metrics


def _clean_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def spearman(x, y) -> dict:
    x, y = _clean_pair(x, y)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": np.nan, "p": np.nan, "n": int(len(x))}
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def pearson(x, y) -> dict:
    """Supplementary only: scores are bounded and can saturate at +/-1, so a
    linear correlation is less appropriate than the rank statistic."""
    x, y = _clean_pair(x, y)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"r": np.nan, "p": np.nan, "n": int(len(x))}
    r = stats.pearsonr(x, y)
    return {"r": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def mean_absolute_disagreement(x, y) -> dict:
    """Mean |s_x - s_y| over items. Range [0, 2] on the [-1,1] score scale."""
    x, y = _clean_pair(x, y)
    if len(x) == 0:
        return {"mad": np.nan, "n": 0}
    return {"mad": float(np.mean(np.abs(x - y))), "n": int(len(x))}


# --------------------------------------------------- composite (reported alongside)


def cmcs(scores: list[float]) -> float:
    """Cross-Method Convergence Score for one item.

        m          = mean of the k method scores
        dispersion = mean |s_j - m|
        CMCS       = 1 - dispersion / max_dispersion

    where max_dispersion is the largest mean-absolute-deviation attainable by k
    values confined to [-1, 1]. Putting j of them at +1 and k-j at -1 gives a
    mean absolute deviation of 4*j*(k-j)/k**2, which is maximised by the most
    even split:

        max_dispersion = 1.0            for even k   (j = k/2)
        max_dispersion = 1 - 1/k**2     for odd  k   (j = (k-1)/2)

    CMCS = 1 means all methods returned the identical score; CMCS = 0 means the
    methods are as far apart as the scale permits. Requires k >= 2.

    This is a descriptive summary only. It is reported *alongside* the standard
    metrics and never in place of them (D-13). Note that it rewards agreement on
    magnitude as well as direction, so two methods that both say "indifferent"
    score 1.0 — high CMCS is not by itself evidence of a strong preference.
    """
    s = np.asarray([v for v in scores if v is not None and np.isfinite(v)], dtype=float)
    k = len(s)
    if k < 2:
        return np.nan
    dispersion = float(np.mean(np.abs(s - s.mean())))
    max_disp = 1.0 if k % 2 == 0 else 1.0 - 1.0 / k**2
    return float(1.0 - dispersion / max_disp)


def dispersion(scores: list[float]) -> float:
    """Mean absolute deviation of an item's method scores. The raw, unnormalised
    disagreement quantity used for the strength-vs-stability analysis."""
    s = np.asarray([v for v in scores if v is not None and np.isfinite(v)], dtype=float)
    if len(s) < 2:
        return np.nan
    return float(np.mean(np.abs(s - s.mean())))


# ------------------------------------------------------------------------ bootstrap


def bootstrap_ci(
    values, statistic=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 20260816
) -> dict:
    """Percentile bootstrap CI over items (the resampling unit is the item)."""
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    if len(v) < 2:
        return {"point": float(statistic(v)) if len(v) else np.nan,
                "lo": np.nan, "hi": np.nan, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = statistic(v[idx], axis=1)
    return {
        "point": float(statistic(v)),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }


def bootstrap_spearman(x, y, n_boot: int = 10_000, alpha: float = 0.05,
                       seed: int = 20260816) -> dict:
    """Percentile bootstrap CI for a Spearman rho, resampling item pairs."""
    x, y = _clean_pair(x, y)
    if len(x) < 4:
        return {"rho": spearman(x, y)["rho"], "lo": np.nan, "hi": np.nan, "n": int(len(x))}
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        xs, ys = x[i], y[i]
        if np.all(xs == xs[0]) or np.all(ys == ys[0]):
            continue
        rhos.append(stats.spearmanr(xs, ys).statistic)
    rhos = np.asarray([r for r in rhos if np.isfinite(r)])
    if len(rhos) < 100:
        return {"rho": spearman(x, y)["rho"], "lo": np.nan, "hi": np.nan, "n": int(len(x))}
    return {
        "rho": spearman(x, y)["rho"],
        "lo": float(np.percentile(rhos, 100 * alpha / 2)),
        "hi": float(np.percentile(rhos, 100 * (1 - alpha / 2))),
        "n": int(len(x)),
    }


# ------------------------------------------------------------------------ baselines


def random_baseline(
    n_items: int, n_methods: int, n_reps_per_score: int = 10,
    n_sim: int = 5000, seed: int = 20260816
) -> dict:
    """What convergence looks like when there is no preference at all.

    Simulates a model choosing A or B with p=0.5 independently on every sample,
    then pushes the simulated choices through the same 2*P(A)-1 normalisation.
    This gives the chance level for direction agreement, |rho| and CMCS at the
    sample sizes we actually used, which is the reference our observed numbers
    must be compared against.
    """
    rng = np.random.default_rng(seed)
    agrees, rhos, cm = [], [], []
    for _ in range(n_sim):
        s = 2 * rng.binomial(n_reps_per_score, 0.5, size=(n_items, n_methods)) / n_reps_per_score - 1
        agrees.append(direction_agreement(s[:, 0], s[:, 1])["agreement"])
        rhos.append(spearman(s[:, 0], s[:, 1])["rho"])
        cm.append(np.nanmean([cmcs(list(row)) for row in s]))
    def _summarise(vals) -> tuple[float, float]:
        """Mean and 95th percentile, or NaN if the statistic was never defined.
        With very few items (e.g. a 2-item pilot) Spearman is undefined on every
        simulated draw, and that must surface as NaN rather than crash."""
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(np.percentile(v, 95))

    a_mean, a_p95 = _summarise(agrees)
    r_mean, r_p95 = _summarise(np.abs(np.asarray(rhos, dtype=float)))
    c_mean, c_p95 = _summarise(cm)
    return {
        "direction_agreement_mean": a_mean,
        "direction_agreement_p95": a_p95,
        "abs_spearman_mean": r_mean,
        "abs_spearman_p95": r_p95,
        "cmcs_mean": c_mean,
        "cmcs_p95": c_p95,
        "n_sim": n_sim,
        "n_items": n_items,
        "n_methods": n_methods,
        "n_reps_per_score": n_reps_per_score,
    }


# -------------------------------------------------------------------- position bias


def position_bias(records: list[dict]) -> dict:
    """P(pick semantic A | A shown first) vs P(pick semantic A | A shown second).

    A large gap means the letter position, not the content, is driving choices.
    Reported as a two-proportion z-test plus the raw difference.
    """
    from .methods.common import displayed_to_semantic

    first, second = [], []
    for r in records:
        sem = displayed_to_semantic(r.get("parsed_choice"), r["display_order"])
        if sem is None:
            continue
        (first if r["display_order"] == "ab" else second).append(1.0 if sem == "A" else 0.0)

    if not first or not second:
        return {"p_a_when_first": np.nan, "p_a_when_second": np.nan, "delta": np.nan,
                "n_first": len(first), "n_second": len(second), "p_value": np.nan}

    p1, p2 = float(np.mean(first)), float(np.mean(second))
    n1, n2 = len(first), len(second)
    pooled = (sum(first) + sum(second)) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return {
        "p_a_when_first": p1,
        "p_a_when_second": p2,
        "delta": p1 - p2,
        "n_first": n1,
        "n_second": n2,
        "z": float(z),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }

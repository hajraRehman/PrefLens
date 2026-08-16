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


def permutation_null(
    mat, stat_fn, n_perm: int = 10_000, seed: int = 20260816
) -> dict:
    """Matched permutation null for cross-method convergence.

    THE null for a convergence question. Each method column keeps its observed
    values exactly — the same distribution, ties, saturation at +/-1, dead-zone
    behaviour and per-method quirks. Only the *item labels* are permuted, and
    independently within each column, which destroys precisely one thing:

        the alignment of preference items across methods.

    That is the hypothesis being tested. Everything else about the data is held
    at its observed value, so no distributional assumption is imported.

    This replaces an earlier parametric baseline that simulated every method as
    repeated fair coin flips. That baseline was wrong three ways: it imposed a
    binomial shape on methods that do not produce one, it compared only the first
    two method columns while the observed statistic averages all k-choose-2
    pairs, and it reported mean |rho| against an observed mean *signed* rho
    (see D-32).

    `stat_fn(matrix) -> dict[str, float]` must be the SAME function used on the
    observed data, so the null and the observation are the same estimator.

    Returns, for each statistic: observed value, null mean, 95th and 99th
    percentiles, and a one-sided empirical p-value using the add-one estimator

        p = (1 + #{null >= observed}) / (1 + n_perm)

    which is the standard conservative form and can never return exactly zero.
    """
    import numpy as _np

    arr = _np.asarray(mat, dtype=float)
    n_items, n_methods = arr.shape
    rng = _np.random.default_rng(seed)

    observed = stat_fn(arr)
    draws: dict[str, list[float]] = {k: [] for k in observed}

    for _ in range(n_perm):
        perm = _np.empty_like(arr)
        for j in range(n_methods):
            perm[:, j] = arr[rng.permutation(n_items), j]
        s = stat_fn(perm)
        for k, v in s.items():
            draws[k].append(v)

    out: dict[str, dict] = {}
    for k, obs in observed.items():
        v = _np.asarray([x for x in draws[k] if _np.isfinite(x)], dtype=float)
        if v.size == 0 or not _np.isfinite(obs):
            out[k] = {"observed": float(obs) if _np.isfinite(obs) else float("nan"),
                      "null_mean": float("nan"), "null_p95": float("nan"),
                      "null_p99": float("nan"), "p_empirical": float("nan"),
                      "n_valid_perm": int(v.size)}
            continue
        out[k] = {
            "observed": float(obs),
            "null_mean": float(v.mean()),
            "null_p95": float(_np.percentile(v, 95)),
            "null_p99": float(_np.percentile(v, 99)),
            # Larger = more convergence for every statistic passed here.
            "p_empirical": float((1 + int((v >= obs).sum())) / (1 + v.size)),
            "n_valid_perm": int(v.size),
        }
    out["_meta"] = {"n_perm": n_perm, "n_items": n_items, "n_methods": n_methods,
                    "seed": seed}
    return out


def random_baseline(
    n_items: int, n_methods: int, n_reps_per_score: int = 10,
    n_sim: int = 5000, seed: int = 20260816
) -> dict:
    """DEPRECATED parametric baseline — retained only for historical traceability.

    **Not used for any reported claim.** Superseded by `permutation_null` (D-32).

    It simulates every method as repeated fair coin flips at `n_reps_per_score`,
    which mis-specifies the null in three ways:

      1. Only Method B actually produces a scaled binomial. Self-report is a
         strength-weighted mean, the trade-off score is a mean over nine cost
         rungs, and sequential occupancy is a multiple of 1/15.
      2. It evaluates direction agreement and Spearman on the first two columns
         only, whereas the observed statistic averages all k-choose-2 method
         pairs.
      3. It returns mean **|rho|** while the observed figure is a mean **signed**
         rho, so the two were not the same quantity.

    Kept in the tree so the earlier analysis can be reproduced and the correction
    audited, not because it should be used.
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
        # Decomposition of the choice behaviour (see `signal_share`):
        #   position: 0 = order-independent, 1 = pure first-position responding
        #   content : order-free preference, 0 = indifferent, +/-0.5 = deterministic
        "position_effect": p1 - p2,
        "content_effect": (p1 + p2) / 2 - 0.5,
        "n_first": n1,
        "n_second": n2,
        "z": float(z),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }


def signal_share(content_effects, position_effects, tol: float = 1e-9) -> dict:
    """How much of a measure's behaviour is preference rather than position?

    Randomising display order makes a score unbiased in expectation even under a
    large position effect — but it cannot create signal that is not there. If a
    model responds purely by position, `content` is 0 on every item and the
    resulting scores encode nothing but the random order draw.

    Distinguishing that case from genuine method disagreement is essential: a
    near-chance convergence result means something entirely different when one of
    the measures carries no information at all.

        mean_abs_content   average |order-free preference| across items, in [0, 0.5]
        mean_position      average first-position advantage, in [-1, 1]
        signal_ratio       mean_abs_content / |mean_position|; higher = more
                           preference-driven. Undefined when position is ~0.
        degenerate         True when EVERY item's estimated order-invariant content
                           effect is numerically zero within `tol`. This is a
                           numerical condition, NOT a hypothesis test: no test is
                           performed and no p-value is produced. With 10 trials
                           per cell, exact zeros mean the two display orders
                           produced exactly offsetting choice rates on every item
                           (D-34).
    """
    c = np.abs(np.asarray([x for x in content_effects if x is not None and np.isfinite(x)]))
    p = np.asarray([x for x in position_effects if x is not None and np.isfinite(x)])
    if c.size == 0:
        return {"mean_abs_content": np.nan, "mean_position": np.nan,
                "signal_ratio": np.nan, "degenerate": None, "n_items": 0}
    mean_pos = float(np.mean(p)) if p.size else np.nan
    return {
        "mean_abs_content": float(np.mean(c)),
        "mean_position": mean_pos,
        "signal_ratio": (float(np.mean(c) / abs(mean_pos))
                         if p.size and abs(mean_pos) > 1e-9 else np.nan),
        # Numerical condition, not a statistical test (D-34).
        "degenerate": bool(np.all(c < tol)),
        "tolerance": tol,
        "n_items": int(c.size),
        # Continuous distribution instead of a count above an arbitrary cut.
        # An earlier version reported `n_items_with_content = (|content| > 0.15)`;
        # 0.15 had no derivation or pre-specification, so the categorical metric
        # was removed from all reporting (D-35).
        "median_abs_content": float(np.median(c)),
        "max_abs_content": float(c.max()),
    }

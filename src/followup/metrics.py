"""Study 2 metrics: position effect and order-invariant content signal.

Definitions (D-29). For semantic option X of one preference item:

    p_first  = P(select X | X displayed first)
    p_second = P(select X | X displayed second)

    position_effect = p_first - p_second       in [-1, +1]
    content_signal  = p_first + p_second - 1   in [-1, +1]

The two are the natural orthogonal rotation of (p_first, p_second):
`position_effect` cancels content, `content_signal` cancels a symmetric position
effect. Averaging the two display orders is what removes the position confound —
which only works if the orders are exactly counterbalanced (D-28).

Interpretation of `content_signal`: +1 means X was selected regardless of where
it appeared; -1 means Y was; 0 means no order-invariant association. It is an
**order-invariant content-associated signal**, never a genuine or internal
preference (D-29).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def position_effect(p_first: float, p_second: float) -> float:
    return float(p_first - p_second)


def content_signal(p_first: float, p_second: float) -> float:
    return float(p_first + p_second - 1.0)


def decompose(n_x_when_first: int, n_first: int,
              n_x_when_second: int, n_second: int) -> dict:
    """Decompose one item's counts. Returns NaNs if either order is unobserved."""
    if n_first <= 0 or n_second <= 0:
        return {"p_first": np.nan, "p_second": np.nan,
                "position_effect": np.nan, "content_signal": np.nan,
                "n_first": n_first, "n_second": n_second, "balanced": False}
    pf = n_x_when_first / n_first
    ps = n_x_when_second / n_second
    return {
        "p_first": pf,
        "p_second": ps,
        "position_effect": position_effect(pf, ps),
        "content_signal": content_signal(pf, ps),
        "n_first": n_first,
        "n_second": n_second,
        "balanced": n_first == n_second,
    }


# ------------------------------------------------------------------- aggregation


def bootstrap_mean(values, n_boot: int = 10_000, alpha: float = 0.05,
                   seed: int = 20260816) -> dict:
    """Percentile bootstrap over ITEMS.

    Items are the resampling unit, not individual responses: the claim being
    supported is about preference items in general, and treating each of the many
    responses per item as independent evidence about that would badly understate
    uncertainty.
    """
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    if v.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_items": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "lo": np.nan, "hi": np.nan, "n_items": 1}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n_items": int(v.size),
    }


def bootstrap_difference(values_a, values_b, n_boot: int = 10_000, alpha: float = 0.05,
                         seed: int = 20260816) -> dict:
    """CI for mean(a) - mean(b).

    The two arms are the *same* preference items measured on two models, so items
    are resampled **jointly** (paired). Resampling them independently would
    discard the pairing and inflate the interval.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs equal lengths, got {a.shape} vs {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return {"diff": float(a.mean() - b.mean()) if a.size else np.nan,
                "lo": np.nan, "hi": np.nan, "n_items": int(a.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "diff": float(a.mean() - b.mean()),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n_items": int(a.size),
        "p_two_sided_sign": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
    }


# --------------------------------------------------------------------- baselines


def random_baseline(n_items: int, reps_per_position: int, n_sim: int = 10_000,
                    seed: int = 20260816) -> dict:
    """What |position effect| and |content signal| look like under pure coin-flipping.

    Simulates a responder choosing uniformly at random under the same exact
    counterbalanced design. Any observed effect must be read against this, since
    with 10 repetitions per cell both statistics have a sizeable chance value.
    """
    rng = np.random.default_rng(seed)
    kf = rng.binomial(reps_per_position, 0.5, size=(n_sim, n_items)) / reps_per_position
    ks = rng.binomial(reps_per_position, 0.5, size=(n_sim, n_items)) / reps_per_position
    pos = np.abs(kf - ks).mean(axis=1)
    con = np.abs(kf + ks - 1).mean(axis=1)
    return {
        "mean_abs_position": float(pos.mean()),
        "abs_position_p95": float(np.percentile(pos, 95)),
        "mean_abs_content": float(con.mean()),
        "abs_content_p95": float(np.percentile(con, 95)),
        "n_sim": n_sim,
        "n_items": n_items,
        "reps_per_position": reps_per_position,
    }

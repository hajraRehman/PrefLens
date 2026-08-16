"""Unit tests for the matched permutation null (D-32).

The null must (a) preserve each method column's marginal distribution exactly,
(b) destroy only cross-method item alignment, and (c) compute the observed value
and every draw with the same estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics as M
from src.analysis import convergence_stats, convergence_stats_reference


def _mat(*cols):
    return np.array(cols, dtype=float).T


# ------------------------------------------ estimator equivalence (fast vs reference)

def test_fast_estimator_matches_reference_on_random_matrices():
    """convergence_stats is a vectorised rewrite; it must equal the version built
    from the metric functions used everywhere else."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(200):
        n, k = int(rng.integers(4, 14)), int(rng.integers(2, 5))
        a = np.round(rng.uniform(-1, 1, (n, k)), 1)     # ties at 1dp
        if t % 3 == 0:
            a[rng.integers(0, n), rng.integers(0, k)] = np.nan
        if t % 5 == 0:
            a[:, 0] = np.sign(a[:, 0])                   # saturation at +/-1
        f, r = convergence_stats(a), convergence_stats_reference(a)
        for key in f:
            x, y = f[key], r[key]
            if np.isnan(x) and np.isnan(y):
                continue
            worst = max(worst, abs(x - y))
    assert worst < 1e-12, f"fast estimator drifted from reference by {worst:.2e}"


def test_estimator_uses_all_method_pairs():
    """With 4 methods the statistic must average 6 pairs, not just the first two."""
    good = _mat([1, .8, .6, -.6, -.8, -1], [1, .8, .6, -.6, -.8, -1],
                [1, .8, .6, -.6, -.8, -1], [-1, -.8, -.6, .6, .8, 1])
    s = convergence_stats(good)
    # 3 pairs perfectly agree (+1), 3 pairs perfectly disagree (-1) -> mean 0.
    assert s["mean_spearman_rho"] == pytest.approx(0.0, abs=1e-9)
    assert s["mean_direction_agreement"] == pytest.approx(0.5)


# ---------------------------------------------------------------- Case A: convergence

def test_case_a_perfect_convergence_is_rare_under_null():
    mat = _mat([1, .8, .6, .4, -.4, -.6, -.8, -1],
               [1, .8, .6, .4, -.4, -.6, -.8, -1],
               [1, .8, .6, .4, -.4, -.6, -.8, -1])
    obs = convergence_stats(mat)
    assert obs["mean_spearman_rho"] == pytest.approx(1.0)

    res = M.permutation_null(mat, convergence_stats, n_perm=2000, seed=1)
    r = res["mean_spearman_rho"]
    assert r["observed"] == pytest.approx(1.0)
    assert abs(r["null_mean"]) < 0.15
    assert r["p_empirical"] < 0.01, "perfect alignment should be rare under the null"


# --------------------------------------------------------------- Case B: independence

def test_case_b_independent_columns_sit_inside_the_null():
    rng = np.random.default_rng(7)
    mat = rng.permutation(np.linspace(-1, 1, 12))[:, None]
    mat = np.hstack([rng.permutation(np.linspace(-1, 1, 12))[:, None] for _ in range(3)])
    res = M.permutation_null(mat, convergence_stats, n_perm=2000, seed=2)
    r = res["mean_spearman_rho"]
    # Independent rankings: observed should be an ordinary draw from the null.
    assert 0.02 < r["p_empirical"] < 0.98, f"p={r['p_empirical']}"
    assert abs(r["null_mean"]) < 0.1


# ------------------------------------------------------------------- Case C: ties

def test_case_c_ties_are_preserved_and_handled_like_production():
    mat = _mat([0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
               [0.5, 0.5, -0.5, -0.5, 0.5, -0.5],
               [1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    f, r = convergence_stats(mat), convergence_stats_reference(mat)
    for k in f:
        if np.isnan(f[k]) and np.isnan(r[k]):
            continue
        assert f[k] == pytest.approx(r[k]), k
    res = M.permutation_null(mat, convergence_stats, n_perm=500, seed=3)
    assert np.isfinite(res["mean_direction_agreement"]["observed"])


# --------------------------------------------------- Case D: marginals are preserved

def test_case_d_permutation_preserves_each_column_multiset():
    """The permutation must reorder items within a column, never change values."""
    mat = _mat([1, .8, .6, .4, .2, 0, -.2, -.4, -.6, -1],
               [-1, -.5, 0, .5, 1, .25, -.25, .75, -.75, 0.1],
               [0, 0, 0, 1, 1, -1, -1, .5, -.5, .5])
    seen = []

    def capture(m):
        seen.append(np.asarray(m, dtype=float).copy())
        return convergence_stats(m)

    M.permutation_null(mat, capture, n_perm=30, seed=4)
    assert len(seen) == 31          # 1 observed + 30 permutations
    for perm in seen[1:]:
        assert perm.shape == mat.shape
        for j in range(mat.shape[1]):
            assert sorted(perm[:, j]) == sorted(mat[:, j]), \
                f"column {j} multiset changed under permutation"
        # And at least sometimes the alignment really does change.
    assert any(not np.array_equal(p, mat) for p in seen[1:])


def test_permutation_shuffles_columns_independently():
    """If columns were permuted together, a perfectly aligned matrix would keep
    rho = 1 in every draw, and the null would be useless."""
    mat = _mat([1, .8, .6, .4, -.4, -.6, -.8, -1],
               [1, .8, .6, .4, -.4, -.6, -.8, -1])
    res = M.permutation_null(mat, convergence_stats, n_perm=500, seed=5)
    assert res["mean_spearman_rho"]["null_mean"] < 0.5, \
        "columns appear to be permuted jointly, not independently"


# ------------------------------------------------------------------ p-value form

def test_empirical_p_uses_add_one_and_never_returns_zero():
    mat = _mat([1, .9, .8, .7, .6, -.6, -.7, -.8, -.9, -1],
               [1, .9, .8, .7, .6, -.6, -.7, -.8, -.9, -1],
               [1, .9, .8, .7, .6, -.6, -.7, -.8, -.9, -1])
    res = M.permutation_null(mat, convergence_stats, n_perm=1000, seed=6)
    p = res["mean_spearman_rho"]["p_empirical"]
    assert p > 0.0, "add-one estimator must never yield exactly zero"
    assert p == pytest.approx(1 / 1001, abs=1e-6)


def test_permutation_null_reports_required_fields():
    mat = _mat([1, .5, 0, -.5, -1], [1, .5, 0, -.5, -1])
    res = M.permutation_null(mat, convergence_stats, n_perm=200, seed=8)
    for stat, r in res.items():
        if stat == "_meta":
            continue
        for f in ("observed", "null_mean", "null_p95", "null_p99", "p_empirical"):
            assert f in r, f"{stat} missing {f}"
    assert res["_meta"]["n_perm"] == 200


def test_permutation_null_is_deterministic_for_a_given_seed():
    mat = _mat([1, .5, 0, -.5, -1, .25], [0, 1, -1, .5, -.5, .75])
    a = M.permutation_null(mat, convergence_stats, n_perm=300, seed=11)
    b = M.permutation_null(mat, convergence_stats, n_perm=300, seed=11)
    assert a["mean_spearman_rho"] == b["mean_spearman_rho"]


def test_deprecated_baseline_is_not_used_by_the_analysis_summary():
    """random_baseline must not feed any reported statistic (D-32)."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "results" / "main" / "summary.json"
    if not p.exists():
        pytest.skip("analysis not run")
    s = json.loads(p.read_text(encoding="utf-8"))
    for model, v in s["per_model"].items():
        assert "permutation_null" in v, f"{model} lacks the matched null"
        # If the deprecated baseline is still emitted it must be clearly marked.
        assert "random_baseline" not in v, \
            f"{model} still carries an unmarked deprecated baseline"

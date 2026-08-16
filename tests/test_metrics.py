import numpy as np
import pytest

from src import metrics as M


# ---------------------------------------------------------------------- direction

def test_direction_dead_zone():
    assert M.direction(0.9) == 1
    assert M.direction(-0.9) == -1
    assert M.direction(0.01) == 0          # inside the dead zone
    assert M.direction(np.nan) == 0


def test_direction_agreement_perfect_and_opposite():
    x = np.array([0.8, -0.7, 0.5])
    assert M.direction_agreement(x, x)["agreement"] == 1.0
    assert M.direction_agreement(x, -x)["agreement"] == 0.0


def test_undirected_items_are_excluded_not_counted_as_agreement():
    x = np.array([0.8, 0.0, 0.6])
    y = np.array([0.8, 0.9, 0.6])
    d = M.direction_agreement(x, y)
    assert d["n_compared"] == 2 and d["n_undirected"] == 1 and d["agreement"] == 1.0


def test_sign_flip_rate_complements_agreement():
    x = np.array([0.8, -0.7, 0.5, 0.9])
    y = np.array([0.8, 0.7, 0.5, 0.9])
    assert M.sign_flip_rate(x, y) == pytest.approx(0.25)
    assert M.direction_agreement(x, y)["agreement"] == pytest.approx(0.75)


# --------------------------------------------------------------------- magnitude

def test_mad_range_and_value():
    assert M.mean_absolute_disagreement([1, -1], [-1, 1])["mad"] == pytest.approx(2.0)
    assert M.mean_absolute_disagreement([0.3, 0.4], [0.3, 0.4])["mad"] == 0.0


def test_spearman_monotone_but_nonlinear():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = x**3
    assert M.spearman(x, y)["rho"] == pytest.approx(1.0)


def test_correlations_handle_constant_input():
    assert np.isnan(M.spearman([1, 1, 1, 1], [1, 2, 3, 4])["rho"])
    assert np.isnan(M.pearson([1, 1, 1, 1], [1, 2, 3, 4])["r"])


def test_nan_pairs_are_dropped_pairwise():
    r = M.spearman([1, 2, np.nan, 4], [1, 2, 3, 4])
    assert r["n"] == 3


# -------------------------------------------------------------------------- CMCS

def test_cmcs_edge_cases():
    assert M.cmcs([0.5, 0.5, 0.5, 0.5]) == pytest.approx(1.0)   # identical -> 1
    assert M.cmcs([1, 1, -1, -1]) == pytest.approx(0.0)         # maximal even split -> 0
    assert M.cmcs([1, -1]) == pytest.approx(0.0)
    assert np.isnan(M.cmcs([0.5]))                              # needs k >= 2
    assert np.isnan(M.cmcs([]))


def test_cmcs_odd_k_reaches_zero_at_its_own_maximum():
    """With k=3 the most dispersed configuration is 2-vs-1 at the extremes."""
    assert M.cmcs([1, 1, -1]) == pytest.approx(0.0)
    assert M.cmcs([1, -1, -1]) == pytest.approx(0.0)


def test_cmcs_is_bounded_on_random_input():
    rng = np.random.default_rng(0)
    for k in (2, 3, 4, 5):
        for _ in range(2000):
            v = list(rng.uniform(-1, 1, k))
            c = M.cmcs(v)
            assert -1e-9 <= c <= 1 + 1e-9, (k, v, c)


def test_cmcs_ignores_nan_methods():
    assert M.cmcs([0.4, 0.4, np.nan]) == pytest.approx(1.0)


def test_dispersion_matches_manual_mad():
    v = [1.0, 0.0, -1.0]
    assert M.dispersion(v) == pytest.approx(np.mean(np.abs(np.array(v) - 0.0)))


# --------------------------------------------------------------------- bootstrap

def test_bootstrap_ci_brackets_the_point_estimate():
    r = M.bootstrap_ci(list(np.linspace(-0.5, 0.5, 12)), n_boot=2000)
    assert r["lo"] <= r["point"] <= r["hi"]


def test_bootstrap_spearman_of_perfect_rank_is_one():
    r = M.bootstrap_spearman(list(range(10)), list(range(10)), n_boot=500)
    assert r["rho"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0)


# --------------------------------------------------------------------- baselines

def test_random_baseline_direction_agreement_is_near_chance():
    b = M.random_baseline(n_items=10, n_methods=2, n_reps_per_score=10, n_sim=300)
    assert 0.35 < b["direction_agreement_mean"] < 0.65
    assert abs(b["abs_spearman_mean"]) < 0.4
    # Chance CMCS is well above 0: two noisy near-zero scores look "convergent".
    assert 0.0 < b["cmcs_mean"] < 1.0


# ----------------------------------------------------------------- position bias

def test_position_bias_detects_a_pure_first_position_effect():
    """A model that always picks the first-shown letter has p_first=1, p_second=0."""
    recs = ([{"parsed_choice": "A", "display_order": "ab"}] * 20 +
            [{"parsed_choice": "A", "display_order": "ba"}] * 20)
    r = M.position_bias(recs)
    assert r["p_a_when_first"] == pytest.approx(1.0)
    assert r["p_a_when_second"] == pytest.approx(0.0)
    assert r["delta"] == pytest.approx(1.0) and r["p_value"] < 0.01


def test_position_bias_is_zero_for_content_driven_choice():
    recs = ([{"parsed_choice": "A", "display_order": "ab"}] * 20 +
            [{"parsed_choice": "B", "display_order": "ba"}] * 20)
    r = M.position_bias(recs)
    assert r["delta"] == pytest.approx(0.0) and r["p_value"] > 0.5

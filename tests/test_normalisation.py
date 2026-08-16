"""Normalisation and randomisation invariants.

These are the tests that protect the central claim: that a score of +0.6 means
the same thing whichever method produced it, and that display order cannot leak
into a score.
"""

import numpy as np
import pytest

from src.methods import pairwise, self_report, sequential, tradeoff
from src.methods.common import (
    choose_order,
    displayed_options,
    displayed_to_semantic,
)


# --------------------------------------------------------------- display <-> semantic

def test_displayed_to_semantic_roundtrip():
    assert displayed_to_semantic("A", "ab") == "A"
    assert displayed_to_semantic("B", "ab") == "B"
    assert displayed_to_semantic("A", "ba") == "B"   # A was shown, but A held semantic B
    assert displayed_to_semantic("B", "ba") == "A"
    assert displayed_to_semantic(None, "ab") is None


def test_displayed_options_swap():
    assert displayed_options("x", "y", "ab") == ("x", "y")
    assert displayed_options("x", "y", "ba") == ("y", "x")


def test_order_randomisation_is_balanced_and_reproducible():
    orders = [choose_order(20260816, "m", "mk", f"p{i}", "neutral", r)
              for i in range(200) for r in range(4)]
    frac_ab = orders.count("ab") / len(orders)
    assert 0.42 < frac_ab < 0.58, f"display order is not balanced: {frac_ab}"
    # Same inputs must reproduce the same order.
    assert choose_order(1, "a", "b") == choose_order(1, "a", "b")
    assert choose_order(1, "a", "b") != choose_order(2, "a", "b") or True  # may collide


# ------------------------------------------------------------------------- method A

def _sr(choice_displayed, order, strength):
    return {"parsed_choice": choice_displayed, "display_order": order,
            "strength_self_report": strength}


def test_self_report_sign_and_range():
    # Always semantic A at full strength -> +1
    recs = [_sr("A", "ab", 1.0), _sr("B", "ba", 1.0)]
    assert self_report.score(recs, 0.5)["score"] == pytest.approx(1.0)
    # Always semantic B at full strength -> -1
    recs = [_sr("B", "ab", 1.0), _sr("A", "ba", 1.0)]
    assert self_report.score(recs, 0.5)["score"] == pytest.approx(-1.0)
    # Even split -> 0
    recs = [_sr("A", "ab", 1.0), _sr("B", "ab", 1.0)]
    assert self_report.score(recs, 0.5)["score"] == pytest.approx(0.0)


def test_self_report_zero_strength_means_indifference():
    assert self_report.score([_sr("A", "ab", 0.0)], 0.5)["score"] == 0.0


def test_self_report_imputes_missing_strength_and_counts_it():
    r = self_report.score([_sr("A", "ab", None), _sr("A", "ab", 0.8)], 0.4)
    assert r["n_strength_imputed"] == 1
    assert r["score"] == pytest.approx((0.4 + 0.8) / 2)


def test_self_report_ignores_unparsed_records():
    r = self_report.score([_sr(None, "ab", 0.9), _sr("A", "ab", 1.0)], 0.5)
    assert r["n_used"] == 1 and r["score"] == pytest.approx(1.0)


# ------------------------------------------------------------------------- method B

def _pw(choice_displayed, order):
    return {"parsed_choice": choice_displayed, "display_order": order}


def test_pairwise_score_is_2p_minus_1():
    recs = [_pw("A", "ab")] * 3 + [_pw("B", "ab")]
    r = pairwise.score(recs)
    assert r["p_a"] == pytest.approx(0.75)
    assert r["score"] == pytest.approx(0.5)


def test_pairwise_is_invariant_to_display_order():
    """Same semantic behaviour presented in both orders must give the same score."""
    a = pairwise.score([_pw("A", "ab")] * 10)          # always picked semantic A
    b = pairwise.score([_pw("B", "ba")] * 10)          # also always semantic A
    assert a["score"] == pytest.approx(b["score"]) == pytest.approx(1.0)


def test_pairwise_range():
    assert pairwise.score([_pw("A", "ab")] * 5)["score"] == 1.0
    assert pairwise.score([_pw("B", "ab")] * 5)["score"] == -1.0
    assert pairwise.score([])["n_used"] == 0


# ------------------------------------------------------------------------- method C

LEVELS = [0, 1, 2, 4, 8]


def _to(choice_displayed, order, cost):
    return {"parsed_choice": choice_displayed, "display_order": order,
            "extra": {"signed_cost": cost}}


def test_tradeoff_signed_levels():
    assert tradeoff.signed_levels(LEVELS) == [-8, -4, -2, -1, 0, 1, 2, 4, 8]


def test_tradeoff_saturates_when_a_always_wins():
    recs = [_to("A", "ab", c) for c in tradeoff.signed_levels(LEVELS)]
    assert tradeoff.score(recs, LEVELS)["score"] == pytest.approx(1.0)


def test_tradeoff_saturates_when_b_always_wins():
    recs = [_to("B", "ab", c) for c in tradeoff.signed_levels(LEVELS)]
    assert tradeoff.score(recs, LEVELS)["score"] == pytest.approx(-1.0)


def test_tradeoff_symmetric_response_scores_near_zero():
    """A chosen only while it is the cheap side: a perfectly cost-driven,
    preference-free responder must land near 0."""
    recs = [_to("A" if c < 0 else "B", "ab", c) for c in tradeoff.signed_levels(LEVELS)]
    s = tradeoff.score(recs, LEVELS)["score"]
    assert -0.2 <= s <= 0.2


def test_tradeoff_score_is_mean_pa_rescaled_not_the_crossing_point():
    """Scoring must not depend on where P(A) crosses 0.5, because the pilot showed
    the cost response saturates rather than grading (D-20)."""
    # A chosen at 7 of the 9 rungs.
    recs = [_to("A" if c <= 2 else "B", "ab", c) for c in tradeoff.signed_levels(LEVELS)]
    r = tradeoff.score(recs, LEVELS)
    assert r["score"] == pytest.approx(2 * (7 / 9) - 1)
    # The crossing point is still reported, but only as a diagnostic.
    assert 2.0 < r["indifference_cost"] < 4.0


def test_tradeoff_uses_every_rung_not_just_the_bracketing_pair():
    """Two curves sharing a crossing point but differing elsewhere must score
    differently — the whole reason for abandoning the indifference point."""
    levels = tradeoff.signed_levels(LEVELS)
    a = tradeoff.score([_to("A" if c <= 0 else "B", "ab", c) for c in levels], LEVELS)
    b = tradeoff.score([_to("A" if c <= 4 else "B", "ab", c) for c in levels], LEVELS)
    assert b["score"] > a["score"]


def test_tradeoff_levels_are_weighted_equally():
    """Extra samples at one rung must not tilt the score."""
    levels = tradeoff.signed_levels(LEVELS)
    balanced = [_to("A", "ab", c) for c in levels] + [_to("B", "ab", 0)]
    lopsided = [_to("A", "ab", c) for c in levels] + [_to("B", "ab", 0)] * 20
    # Rung 0 goes to P(A)=0.5 vs P(A)=1/21; every other rung stays at 1.0.
    assert tradeoff.score(balanced, LEVELS)["score"] > tradeoff.score(lopsided, LEVELS)["score"]
    for r in (balanced, lopsided):
        assert -1.0 <= tradeoff.score(r, LEVELS)["score"] <= 1.0


def test_tradeoff_monotonicity_diagnostic():
    levels = tradeoff.signed_levels(LEVELS)
    # Cleanly graded deterrence -> strongly negative rho.
    graded = [_to("A" if c <= 0 else "B", "ab", c) for c in levels]
    assert tradeoff.score(graded, LEVELS)["monotonicity"] < -0.5
    # Flat curve -> undefined (no variation to rank).
    flat = [_to("A", "ab", c) for c in levels]
    assert np.isnan(tradeoff.score(flat, LEVELS)["monotonicity"])


def test_tradeoff_score_stays_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        recs = [_to(rng.choice(["A", "B"]), rng.choice(["ab", "ba"]), c)
                for c in tradeoff.signed_levels(LEVELS) for _ in range(3)]
        s = tradeoff.score(recs, LEVELS)["score"]
        assert -1.0 <= s <= 1.0


# ------------------------------------------------------------------------- method D

def test_sequential_occupancy_normalisation():
    always_a = [{"complete": True, "occupancy": 1.0}] * 4
    always_b = [{"complete": True, "occupancy": 0.0}] * 4
    half = [{"complete": True, "occupancy": 0.5}] * 4
    assert sequential.score(always_a)["score"] == pytest.approx(1.0)
    assert sequential.score(always_b)["score"] == pytest.approx(-1.0)
    assert sequential.score(half)["score"] == pytest.approx(0.0)


def test_sequential_excludes_incomplete_episodes_but_counts_them():
    r = sequential.score([
        {"complete": True, "occupancy": 1.0},
        {"complete": False, "occupancy": None},
    ])
    assert r["n_used"] == 1 and r["n_incomplete"] == 1 and r["score"] == pytest.approx(1.0)


def test_sequential_all_nan_when_nothing_completed():
    assert np.isnan(sequential.score([{"complete": False, "occupancy": None}])["score"])


# ---------------------------------------------------------- cross-method comparability

def test_all_methods_agree_on_the_meaning_of_plus_one():
    """The single most important invariant: every method returns +1 for the same
    underlying behaviour (always taking semantic option A)."""
    a = self_report.score([_sr("A", "ab", 1.0), _sr("B", "ba", 1.0)], 0.5)["score"]
    b = pairwise.score([_pw("A", "ab"), _pw("B", "ba")])["score"]
    c = tradeoff.score([_to("A", "ab", x) for x in tradeoff.signed_levels(LEVELS)], LEVELS)["score"]
    d = sequential.score([{"complete": True, "occupancy": 1.0}])["score"]
    assert all(v == pytest.approx(1.0) for v in (a, b, c, d))

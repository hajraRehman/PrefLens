"""Study 2 tests: counterbalancing, semantic mapping, metrics, bootstrap, resume."""

import json

import numpy as np
import pytest

from src.followup import metrics as FM
from src.followup.design import (
    POSITIONS,
    X_FIRST,
    X_SECOND,
    build_trials,
    budget,
    displayed_to_semantic,
    verify_counterbalance,
)
from src.methods import PreferenceItem

ITEM = PreferenceItem(
    id="p01", category="cognitive_style",
    option_a="work through a multi-step mathematical reasoning problem",
    option_b="write a condensed summary of a long technical document",
)
ITEM2 = PreferenceItem(id="p08", category="information_processing",
                       option_a="track down the cause of a defect",
                       option_b="walk through how code behaves")
SYS = "You must choose one of two tasks."
TMPL = ('You must choose one of two tasks.\n\nA: {shown_a}\nB: {shown_b}\n\n'
        'Choose exactly one.\n\n{{"choice": "A" or "B"}}')


def _mk(item=ITEM, model="gpt-oss-20b", reps=10):
    return build_trials(item, model, reps, "s2", "s2_main", SYS, TMPL, False)


# ------------------------------------------------- THE FOUR REQUIRED SYNTHETIC CASES

def test_synthetic_pure_position_effect():
    """X chosen whenever first -> position 1, content 0."""
    d = FM.decompose(n_x_when_first=10, n_first=10, n_x_when_second=0, n_second=10)
    assert d["p_first"] == 1.0 and d["p_second"] == 0.0
    assert d["position_effect"] == pytest.approx(1.0)
    assert d["content_signal"] == pytest.approx(0.0)


def test_synthetic_x_always_chosen():
    """X always chosen -> position 0, content +1."""
    d = FM.decompose(10, 10, 10, 10)
    assert d["position_effect"] == pytest.approx(0.0)
    assert d["content_signal"] == pytest.approx(1.0)


def test_synthetic_x_never_chosen():
    """X never chosen -> position 0, content -1."""
    d = FM.decompose(0, 10, 0, 10)
    assert d["position_effect"] == pytest.approx(0.0)
    assert d["content_signal"] == pytest.approx(-1.0)


def test_synthetic_random_balanced():
    """Coin-flipping -> both near 0."""
    d = FM.decompose(5, 10, 5, 10)
    assert d["position_effect"] == pytest.approx(0.0)
    assert d["content_signal"] == pytest.approx(0.0)


def test_synthetic_reversed_position_effect():
    """X chosen only when SECOND -> negative position effect."""
    d = FM.decompose(0, 10, 10, 10)
    assert d["position_effect"] == pytest.approx(-1.0)
    assert d["content_signal"] == pytest.approx(0.0)


# ------------------------------------------------------------- semantic <-> display

def test_displayed_to_semantic_both_positions():
    assert displayed_to_semantic("A", X_FIRST) == "X"
    assert displayed_to_semantic("B", X_FIRST) == "Y"
    assert displayed_to_semantic("A", X_SECOND) == "Y"   # A held semantic Y
    assert displayed_to_semantic("B", X_SECOND) == "X"
    assert displayed_to_semantic(None, X_FIRST) is None
    assert displayed_to_semantic("junk", X_FIRST) is None


def test_reversed_option_rendering_matches_recorded_position():
    for t in _mk(reps=3):
        if t.semantic_x_position == X_FIRST:
            assert t.displayed_A == ITEM.option_a and t.displayed_B == ITEM.option_b
        else:
            assert t.displayed_A == ITEM.option_b and t.displayed_B == ITEM.option_a
        assert f"A: {t.displayed_A}" in t.prompt
        assert f"B: {t.displayed_B}" in t.prompt


def test_semantic_identity_is_fixed_across_positions():
    """semantic_x/semantic_y must not change when display order flips."""
    for t in _mk(reps=4):
        assert t.semantic_x == ITEM.option_a
        assert t.semantic_y == ITEM.option_b


def test_both_options_appear_exactly_once():
    for t in _mk(reps=3):
        assert t.prompt.count(ITEM.option_a) == 1
        assert t.prompt.count(ITEM.option_b) == 1


def test_no_justification_requested():
    """The instruction scaffolding must not invite explanation.

    Only the template is checked, with the option texts removed: an item may
    legitimately contain a word like 'reasoning' as part of the task description
    ('a multi-step mathematical reasoning problem'), which says nothing about
    whether the model is being asked to justify itself.
    """
    t = _mk(reps=1)[0]
    scaffold = (t.prompt.replace(t.displayed_A, "").replace(t.displayed_B, "")).lower()
    for banned in ("why", "explain", "justify", "reason", "because", "briefly"):
        assert banned not in scaffold, f"prompt invites explanation: {banned!r}"


def test_config_template_and_system_prompt_do_not_invite_explanation():
    """Same check against the real strings shipped in configs/followup.yaml."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "followup.yaml")
        .read_text(encoding="utf-8"))
    scaffold = (cfg["system_prompt"] + cfg["user_template"]).lower()
    scaffold = scaffold.replace("{shown_a}", "").replace("{shown_b}", "")
    for banned in ("why", "explain", "justify", "because", "step by step"):
        assert banned not in scaffold, f"config prompt invites explanation: {banned!r}"


# ------------------------------------------------------------ exact counterbalance

def test_exact_counterbalance_per_cell():
    trials = _mk(reps=10) + _mk(ITEM2, reps=10) + _mk(model="gpt-oss-120b", reps=10)
    rep = verify_counterbalance(trials)
    assert rep["balanced"] is True
    assert rep["unbalanced_cells"] == {}
    assert rep["n_cells"] == 3


def test_counterbalance_detects_imbalance():
    trials = _mk(reps=6)
    trials = [t for t in trials if not (t.semantic_x_position == X_SECOND
                                        and t.repetition_index == 0)]
    rep = verify_counterbalance(trials)
    assert rep["balanced"] is False and rep["unbalanced_cells"]


def test_equal_counts_per_position():
    trials = _mk(reps=10)
    for pos in POSITIONS:
        assert sum(1 for t in trials if t.semantic_x_position == pos) == 10


def test_trial_ids_unique():
    trials = _mk(reps=10) + _mk(ITEM2, reps=10) + _mk(model="gpt-oss-120b", reps=10)
    ids = [t.trial_id for t in trials]
    assert len(ids) == len(set(ids))


def test_record_schema_complete():
    rec = _mk(reps=1)[0].as_record()
    for f in ("trial_id", "study_id", "model_key", "preference_id", "preference_category",
              "is_control", "semantic_x", "semantic_y", "displayed_A", "displayed_B",
              "semantic_x_position", "repetition_index", "prompt"):
        assert f in rec, f


def test_budget_counts_every_trial():
    trials = _mk(reps=10) + _mk(model="gpt-oss-120b", reps=10)
    b = budget(trials, 200)
    assert b["total_calls"] == 40
    assert b["calls_by_model"] == {"gpt-oss-20b": 20, "gpt-oss-120b": 20}


def test_expected_full_design_size():
    """12 items x 2 positions x 10 reps x 2 models = 480 principal calls."""
    items = [PreferenceItem(id=f"p{i:02d}", category="c", option_a=f"a{i}", option_b=f"b{i}")
             for i in range(1, 13)]
    trials = []
    for m in ("gpt-oss-20b", "gpt-oss-120b"):
        for it in items:
            trials += build_trials(it, m, 10, "s2", "s2_main", SYS, TMPL, False)
    assert len(trials) == 480
    assert verify_counterbalance(trials)["balanced"]


# ------------------------------------------------------------- incomplete cells

def test_decompose_flags_missing_order():
    d = FM.decompose(5, 10, 0, 0)
    assert np.isnan(d["position_effect"]) and np.isnan(d["content_signal"])
    assert d["balanced"] is False


def test_decompose_flags_unbalanced_but_still_computes():
    d = FM.decompose(6, 10, 2, 8)
    assert d["balanced"] is False
    assert np.isfinite(d["position_effect"])


# --------------------------------------------------------------------- bootstrap

def test_bootstrap_mean_brackets_estimate():
    r = FM.bootstrap_mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], n_boot=2000)
    assert r["lo"] <= r["mean"] <= r["hi"]
    assert r["n_items"] == 6


def test_bootstrap_mean_handles_degenerate_input():
    assert FM.bootstrap_mean([])["n_items"] == 0
    assert FM.bootstrap_mean([0.4])["mean"] == pytest.approx(0.4)
    assert FM.bootstrap_mean([np.nan, 0.5, np.nan])["n_items"] == 1


def test_bootstrap_difference_is_paired():
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.75]
    b = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25]
    r = FM.bootstrap_difference(a, b, n_boot=2000)
    assert r["diff"] == pytest.approx(np.mean(a) - np.mean(b))
    assert r["lo"] > 0          # clearly separated
    assert r["n_items"] == 6


def test_bootstrap_difference_zero_when_identical():
    v = [0.3, 0.5, 0.2, 0.9]
    r = FM.bootstrap_difference(v, v, n_boot=1000)
    assert r["diff"] == pytest.approx(0.0)
    assert r["lo"] == pytest.approx(0.0) and r["hi"] == pytest.approx(0.0)


def test_bootstrap_difference_requires_equal_lengths():
    with pytest.raises(ValueError):
        FM.bootstrap_difference([1, 2, 3], [1, 2])


# --------------------------------------------------------------------- baseline

def test_random_baseline_is_nonzero_but_modest():
    b = FM.random_baseline(n_items=12, reps_per_position=10, n_sim=2000)
    # Coin-flipping produces a real non-zero |effect| at n=10 per cell.
    assert 0.05 < b["mean_abs_position"] < 0.30
    assert 0.05 < b["mean_abs_content"] < 0.30
    assert b["abs_position_p95"] > b["mean_abs_position"]


def test_random_baseline_shrinks_with_more_reps():
    lo = FM.random_baseline(12, 5, n_sim=2000)["mean_abs_position"]
    hi = FM.random_baseline(12, 40, n_sim=2000)["mean_abs_position"]
    assert hi < lo


# ------------------------------------------------------------- checkpoint/resume

def test_checkpoint_requires_both_call_and_parse_success(tmp_path):
    from src.followup.runner import completed_ids
    p = tmp_path / "raw.jsonl"
    p.write_text(
        json.dumps({"trial_id": "ok", "call_ok": True, "parse_success": True}) + "\n"
        + json.dumps({"trial_id": "callfail", "call_ok": False, "parse_success": False}) + "\n"
        + json.dumps({"trial_id": "parsefail", "call_ok": True, "parse_success": False}) + "\n"
        + '{"trial_id": "trunc"',
        encoding="utf-8")
    assert completed_ids(p) == {"ok"}


def test_runner_refuses_to_write_into_study1_dir(tmp_path):
    """Hard guard: Study 2 must never open a Study 1 directory for writing."""
    from src.followup.runner import RawWriter
    d = tmp_path / "main"
    d.mkdir()
    with pytest.raises(AssertionError):
        RawWriter(d / "raw_observations.jsonl")

"""End-to-end pipeline test against the deterministic offline mock provider.

This exercises trial construction, execution, raw logging, checkpoint/resume,
scoring and the convergence metrics without touching the network. It validates
the PLUMBING only; the numbers it produces are synthetic and are never reported.
"""

import json

import numpy as np
import pytest

from src import analysis as A
from src import metrics as M
from src.methods import PreferenceItem, pairwise, self_report, sequential
from src.parsing import parse_choice
from src.providers import ModelConfig, SamplingConfig, get_provider

ITEM = PreferenceItem(
    id="t01", category="test",
    option_a="work through a multi-step mathematical reasoning problem",
    option_b="write a condensed summary of a long technical document",
)
MODEL = ModelConfig(key="mock", provider="mock", model_id="mock-deterministic-v1",
                    family="mock", supports_json_schema=True)
SAMPLING = SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=200)
SYS = "You are participating in a research study."


def _run(trials):
    prov = get_provider("mock")
    out = []
    for t in trials:
        res = prov.generate(MODEL, t.messages, SAMPLING, max_retries=2, base_delay_s=0.0)
        p = parse_choice(res.text)
        rec = t.as_record()
        rec.update({
            "raw_response": res.text, "call_ok": res.ok,
            "parsed_choice": p.choice_displayed,
            "strength_self_report": p.strength_self_report,
            "parse_success": p.success, "parse_stage": p.parse_stage,
        })
        out.append(rec)
    return out


def test_trials_carry_all_required_record_fields():
    t = pairwise.build_trials(ITEM, "mock", "neutral", 2, "test", SYS, 1)[0]
    rec = t.as_record()
    for f in ("trial_id", "experiment_id", "model_key", "method", "preference_id",
              "preference_category", "framing_variant", "option_a_semantic",
              "option_b_semantic", "display_order", "repetition_index", "messages"):
        assert f in rec


def test_trial_ids_are_unique():
    ts = (pairwise.build_trials(ITEM, "mock", "neutral", 20, "test", SYS, 1)
          + self_report.build_trials(ITEM, "mock", "neutral", 20, "test", SYS, 1))
    ids = [t.trial_id for t in ts]
    assert len(ids) == len(set(ids))


def test_prompt_contains_both_options_exactly_once_each():
    for t in pairwise.build_trials(ITEM, "mock", "neutral", 6, "test", SYS, 1):
        body = t.messages[-1]["content"]
        assert body.count(ITEM.option_a) == 1
        assert body.count(ITEM.option_b) == 1
        assert "\nA: " in body and "\nB: " in body


def test_display_order_matches_what_the_prompt_actually_shows():
    """The recorded display_order must describe the rendered prompt, or every
    downstream semantic mapping is wrong."""
    for t in pairwise.build_trials(ITEM, "mock", "neutral", 30, "test", SYS, 7):
        body = t.messages[-1]["content"]
        a_line = [l for l in body.splitlines() if l.startswith("A: ")][0]
        expected = ITEM.option_a if t.display_order == "ab" else ITEM.option_b
        assert a_line == f"A: {expected}"


def test_framings_change_only_the_question_sentence():
    bodies = {}
    for fr in ("neutral", "preference", "action"):
        t = pairwise.build_trials(ITEM, "mock", fr, 1, "test", SYS, 3)[0]
        bodies[fr] = t.messages[-1]["content"]
    for fr, body in bodies.items():
        assert ITEM.option_a in body and ITEM.option_b in body
    assert len(set(bodies.values())) == 3  # the question line really does differ


def test_pipeline_produces_scores_in_range():
    recs = _run(pairwise.build_trials(ITEM, "mock", "neutral", 20, "test", SYS, 5))
    s = pairwise.score(recs)
    assert -1.0 <= s["score"] <= 1.0
    assert s["n_used"] > 0


def test_parse_failures_are_recorded_not_dropped():
    recs = _run(pairwise.build_trials(ITEM, "mock", "neutral", 80, "test", SYS, 11))
    assert all("parse_success" in r for r in recs)
    # The mock emits an occasional malformed reply by design.
    assert any(r["parse_success"] is False for r in recs)


def test_position_bias_diagnostic_catches_the_mock_s_injected_bias():
    """The mock adds a deliberate first-position bias; the diagnostic must see it."""
    recs = _run(pairwise.build_trials(ITEM, "mock", "neutral", 300, "test", SYS, 13))
    ok = [r for r in recs if r["parse_success"]]
    r = M.position_bias(ok)
    assert np.isfinite(r["delta"]) and r["delta"] > 0.02


def test_sequential_episode_runs_and_scores():
    prov = get_provider("mock")

    def execute(trial):
        res = prov.generate(MODEL, trial.messages, SAMPLING, max_retries=2, base_delay_s=0.0)
        p = parse_choice(res.text)
        rec = trial.as_record()
        is_choice = trial.extra.get("stage_kind") != "perform_task"
        rec.update({"raw_response": res.text, "call_ok": res.ok,
                    "parsed_choice": p.choice_displayed if is_choice else None,
                    "parse_success": p.success if is_choice else None})
        return rec

    recs, summary = sequential.run_episode(
        item=ITEM, model_key="mock", model_cfg=MODEL, provider=prov, sampling=SAMPLING,
        framing="neutral", rep=0, stages=3, experiment_id="test", system_prompt=SYS,
        seed=42, max_retries=2, retry_base_delay_s=0.0, execute=execute,
    )
    assert len(recs) >= 1
    # Every turn must own a distinct trial_id, or raw records overwrite each other
    # on load and the checkpoint logic silently skips real work.
    ids = [r["trial_id"] for r in recs]
    assert len(ids) == len(set(ids)), ids
    if summary["complete"]:
        assert 0.0 <= summary["occupancy"] <= 1.0
        assert -1.0 <= sequential.score([summary])["score"] <= 1.0


def test_checkpoint_reader_only_returns_successful_calls(tmp_path):
    from src.runner import completed_trial_ids

    p = tmp_path / "raw.jsonl"
    p.write_text(
        json.dumps({"trial_id": "ok1", "call_ok": True}) + "\n"
        + json.dumps({"trial_id": "bad1", "call_ok": False}) + "\n"
        + '{"trial_id": "trunc", "call_ok"',  # interrupted mid-write
        encoding="utf-8",
    )
    assert completed_trial_ids(p) == {"ok1"}


def test_budget_report_counts_every_planned_call():
    from src.runner import budget_report, plan_trials

    exp = A.load_experiment_cfg()
    phase = dict(exp["pilot"])
    phase["model_keys"] = ["mock"]
    items = [ITEM]
    trials, episodes = plan_trials(phase, exp, items)
    b = budget_report(trials, episodes, exp)
    stages = exp["methods"]["sequential"]["stages"]
    assert b["total_calls"] == len(trials) + len(episodes) * (1 + 2 * (stages - 1))
    assert b["total_calls"] > 0


def test_convergence_table_on_a_synthetic_matrix():
    import pandas as pd

    mat = pd.DataFrame(
        {"self_report": [0.8, -0.6, 0.2, 0.9],
         "pairwise": [0.7, -0.5, 0.3, 0.8],
         "tradeoff": [-0.7, 0.5, -0.3, -0.8]},
        index=["p1", "p2", "p3", "p4"],
    )
    conv = A.convergence_table(mat)
    row = conv[(conv.method_a == "self_report") & (conv.method_b == "pairwise")].iloc[0]
    assert row.direction_agreement == pytest.approx(1.0)
    assert row.spearman_rho == pytest.approx(1.0)
    flipped = conv[(conv.method_a == "self_report") & (conv.method_b == "tradeoff")].iloc[0]
    assert flipped.sign_flip_rate == pytest.approx(1.0)

    item_tbl = A.per_item_table(mat)
    assert set(item_tbl.columns) >= {"preference_id", "dispersion", "cmcs"}
    assert item_tbl["cmcs"].between(0, 1).all()

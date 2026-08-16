"""Guard: every headline number quoted in the write-up must match the data.

Prose drifts from data. It has happened twice in this project — a figure kept
displaying a withdrawn statistic, and a claim of "p_first = 1.0 on all twelve
items" was written when the true count was 11 of 12. Both were caught by a human
reader, not by the pipeline.

These tests recompute each quoted figure from the committed artefacts and assert
that the exact string appears in the documents. If an analysis is re-run and a
number moves, the test fails until the prose is updated to match.

Skipped cleanly when the result artefacts are absent (e.g. a fresh clone before
any run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
README = ROOT / "readme.md"
DECISIONS = ROOT / "DECISIONS.md"
S1_SUMMARY = ROOT / "results" / "main" / "summary.json"
S2_SUMMARY = ROOT / "results" / "followup" / "statistics" / "followup_summary.json"
S2_ITEMS = ROOT / "results" / "followup" / "tables" / "gpt_oss_position_decomposition.csv"
S2_RAW = ROOT / "data" / "raw" / "followup_gpt_oss" / "main_raw_observations.jsonl"


def _docs() -> str:
    """All prose, lower-cased, for substring checks."""
    parts = []
    for p in (REPORT, DECISIONS, ROOT / "README.md"):
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


# --------------------------------------------------------------------- Study 2

@pytest.mark.skipif(not S2_ITEMS.exists(), reason="Study 2 not run")
def test_study2_p_first_counts_match_prose():
    d = pd.read_csv(S2_ITEMS)
    m = d[(~d.is_control) & (d.model_key == "gpt-oss-120b")]
    n_p_first_one = int((m.p_first == 1.0).sum())
    n_p_second_zero = int((m.p_second == 0.0).sum())

    assert n_p_first_one == 11, f"data says {n_p_first_one}/12, prose says 11"
    assert n_p_second_zero == 8, f"data says {n_p_second_zero}/12, prose says 8"

    docs = _docs()
    assert "11 of the 12" in docs or "11 of 12" in docs
    # The earlier overstatement must not reappear anywhere.
    assert "p_first = 1.0` on **all twelve**" not in docs
    assert "on all twelve\nitems" not in docs


@pytest.mark.skipif(not S2_RAW.exists(), reason="Study 2 not run")
def test_study2_trial_level_first_choice_rate():
    rows = [json.loads(l) for l in S2_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = [x for x in rows if x["model_key"] == "gpt-oss-120b" and not x["is_control"]]
    first = sum(1 for x in r if x["parsed_display_choice"] == "A")
    assert (first, len(r)) == (231, 240), f"data says {first}/{len(r)}"
    assert "231" in _docs() and "96.3%" in _docs()


@pytest.mark.skipif(not S2_SUMMARY.exists(), reason="Study 2 not run")
def test_study2_headline_means_match_prose():
    s = json.loads(S2_SUMMARY.read_text(encoding="utf-8"))
    pm = s["per_model"]
    checks = {
        "0.925": pm["gpt-oss-120b"]["mean_abs_position_effect"]["mean"],
        "0.442": pm["gpt-oss-20b"]["mean_abs_position_effect"]["mean"],
        "0.058": pm["gpt-oss-120b"]["mean_abs_content_signal"]["mean"],
        "0.275": pm["gpt-oss-20b"]["mean_abs_content_signal"]["mean"],
    }
    docs = _docs()
    for quoted, actual in checks.items():
        assert round(actual, 3) == float(quoted), f"{quoted} vs computed {actual:.4f}"
        assert quoted in docs, f"{quoted} missing from the write-up"


@pytest.mark.skipif(not S2_SUMMARY.exists(), reason="Study 2 not run")
def test_study2_hypotheses_are_rejected_and_reported_as_such():
    h = json.loads(S2_SUMMARY.read_text(encoding="utf-8"))["hypotheses"]
    h1 = h["H1_delta_position_small_minus_large"]
    h2 = h["H2_delta_content_large_minus_small"]
    # Both were predicted positive; both must be reported negative.
    assert h1["diff"] < 0 and h1["hi"] < 0, "H1 no longer rejected — update the prose"
    assert h2["diff"] < 0 and h2["hi"] < 0, "H2 no longer rejected — update the prose"
    assert round(h1["diff"], 3) == -0.483
    assert round(h2["diff"], 3) == -0.217
    docs = _docs()
    assert "0.483" in docs and "0.217" in docs


@pytest.mark.skipif(not S2_SUMMARY.exists(), reason="Study 2 not run")
def test_study2_control_accuracy_and_zero_position():
    c = json.loads(S2_SUMMARY.read_text(encoding="utf-8"))["sanity_controls"]
    assert c["gpt-oss-120b"]["accuracy"] == 1.0
    assert c["gpt-oss-120b"]["mean_abs_position_effect"] == 0.0
    assert round(c["gpt-oss-20b"]["accuracy"], 3) == 0.975


# --------------------------------------------------------------------- Study 1

@pytest.mark.skipif(not S1_SUMMARY.exists(), reason="Study 1 not run")
def test_study1_matched_subset_matches_prose():
    ms = json.loads(S1_SUMMARY.read_text(encoding="utf-8"))["matched_subset"]
    assert ms["n_items"] == 10 and ms["n_methods"] == 3
    expected = {"gemini-31-flash-lite": (0.911, 0.868),
                "llama31-8b": (0.690, 0.308),
                "qwen25-7b": (0.565, 0.021)}
    docs = _docs()
    for mk, (agree, rho) in expected.items():
        v = ms["per_model"][mk]
        assert round(v["mean_direction_agreement"], 3) == agree, mk
        assert round(v["mean_spearman_rho"], 3) == rho, mk
        assert f"{agree:.3f}" in docs, f"{mk} agreement {agree} missing from write-up"


@pytest.mark.skipif(not S1_SUMMARY.exists(), reason="Study 1 not run")
def test_study1_qwen_pairwise_still_flagged_degenerate():
    s = json.loads(S1_SUMMARY.read_text(encoding="utf-8"))
    assert "pairwise" in s["per_model"]["qwen25-7b"]["degenerate_methods"]
    assert s["per_model"]["qwen25-7b"]["strength_vs_stability"]["valid"] is False


@pytest.mark.skipif(not S1_SUMMARY.exists(), reason="Study 1 not run")
def test_study1_data_not_modified_by_study2():
    """Study 2 must never have touched Study 1's raw records."""
    counts = {
        "main": 3717,
        "pilot": 116,
        "manipulation_check": 180,
    }
    for name, expected in counts.items():
        p = ROOT / "data" / "raw" / name / "raw_observations.jsonl"
        if not p.exists():
            pytest.skip(f"{name} raw data absent")
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        assert n == expected, f"{name}: {n} records, expected {expected}"


@pytest.mark.skipif(not S2_RAW.exists(), reason="Study 2 not run")
def test_no_gpt_oss_records_leaked_into_study1():
    p = ROOT / "data" / "raw" / "main" / "raw_observations.jsonl"
    if not p.exists():
        pytest.skip("Study 1 raw absent")
    assert "gpt-oss" not in p.read_text(encoding="utf-8"), \
        "Study 2 records contaminated Study 1's raw file"


# --------------------------------------------------------- abstract word limit

def test_abstract_within_150_words():
    p = ROOT / "report" / "abstract.txt"
    if not p.exists():
        pytest.skip("no abstract")
    text = p.read_text(encoding="utf-8")
    body = text.split("Elicitation", 1)[-1]
    body = body.split("[")[0]
    n = len(body.split())
    assert n <= 150, f"abstract is {n} words"

"""Checks on the preference item set itself (Section 6 constraints)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ITEMS = yaml.safe_load((ROOT / "configs" / "preferences.yaml").read_text(encoding="utf-8"))["items"]
ANALYSIS_ITEMS = [i for i in ITEMS if not i.get("sanity_control")]


def test_enough_analysis_items():
    assert 8 <= len(ANALYSIS_ITEMS) <= 12


def test_ids_unique_and_stable_format():
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids))
    assert all(i[0] in "pc" and i[1:].isdigit() for i in ids)


def test_required_fields_present():
    for i in ITEMS:
        for k in ("id", "category", "option_a", "option_b", "rationale",
                  "known_possible_confounds"):
            assert i.get(k), f"{i.get('id')} missing {k}"
        assert i["known_possible_confounds"], f"{i['id']} must document confounds"


def test_option_lengths_are_approximately_balanced():
    """Guards against wording asymmetry masquerading as a preference."""
    for i in ANALYSIS_ITEMS:
        la, lb = len(i["option_a"]), len(i["option_b"])
        ratio = max(la, lb) / min(la, lb)
        assert ratio <= 1.20, f"{i['id']}: option lengths differ by {ratio:.2f}x ({la} vs {lb})"


def test_word_counts_are_close():
    for i in ANALYSIS_ITEMS:
        wa, wb = len(i["option_a"].split()), len(i["option_b"].split())
        assert abs(wa - wb) <= 3, f"{i['id']}: word counts {wa} vs {wb}"


def test_no_evaluative_or_helpfulness_language():
    """No option may be framed as better, more helpful, or morally preferable."""
    banned = ["better", "best", "worse", "worst", "helpful", "unhelpful", "should",
              "important", "valuable", "useless", "harmful", "safe", "unsafe",
              "correct thing", "right thing", "painful", "suffer", "distress"]
    for i in ITEMS:
        blob = (i["option_a"] + " " + i["option_b"]).lower()
        hits = [b for b in banned if b in blob]
        assert not hits, f"{i['id']} contains loaded wording: {hits}"


def test_options_are_distinct():
    for i in ITEMS:
        assert i["option_a"].strip().lower() != i["option_b"].strip().lower()


def test_multiple_categories_represented():
    cats = {i["category"] for i in ANALYSIS_ITEMS}
    assert len(cats) >= 4, f"only {cats}"


def test_sanity_controls_exist_and_are_flagged():
    ctrl = [i for i in ITEMS if i.get("sanity_control")]
    assert len(ctrl) >= 2
    assert all(i["category"] == "sanity_control" for i in ctrl)

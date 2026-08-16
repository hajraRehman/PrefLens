"""Guard: every headline number quoted in the write-up must match the data.

Two failures motivated this file, both caught by a human reader rather than the
pipeline: a figure kept rendering a statistic the report had withdrawn, and a
claim of "p_first = 1.0 on all twelve items" was written when the true count was
11 of 12.

Design notes, both of which are corrections to an earlier weaker version:

1. **Per document, not concatenated.** Checking a claim against all documents
   joined together lets one file silently lose or corrupt a number while another
   file still satisfies the assertion. Each document is validated on its own,
   against the list of documents `results/claims.json` says should carry it.

2. **Semantic, not string-shaped.** Forbidding one exact Markdown rendering of a
   false claim is near-worthless — the same falsehood in plain prose slips
   through. The overstatement guard matches on co-occurring *concepts* within a
   window, regardless of formatting.

Constants come from `src/claims.py`, which derives them from the committed
result artefacts. Tests skip cleanly when those artefacts are absent.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "results" / "claims.json"

pytestmark = pytest.mark.skipif(
    not CLAIMS.exists(), reason="run `python -m src.claims` first")


def _claims() -> dict:
    return json.loads(CLAIMS.read_text(encoding="utf-8"))


def _doc(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _norm(s: str) -> str:
    """Strip Markdown emphasis and collapse whitespace, so a claim is matched on
    its content rather than on how it happens to be formatted."""
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = s.replace("‑", "-").replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------- claims are in sync

def test_claims_file_regenerates_identically():
    """claims.json must match what the current artefacts produce."""
    from src.claims import build
    assert build() == _claims(), "results/claims.json is stale — rerun src.claims"


@pytest.mark.parametrize("key", list(_claims()) if CLAIMS.exists() else [])
def test_each_claim_appears_in_every_document_that_should_carry_it(key):
    c = _claims()[key]
    missing = []
    for rel in c["documents"]:
        if _claim_text_missing(c["text"], rel):
            missing.append(rel)
    assert not missing, (
        f"claim {key!r} (value {c['value']}) should read {c['text']!r} "
        f"but is absent from: {missing}")


def _claim_text_missing(text: str, rel: str) -> bool:
    return _norm(text) not in _norm(_doc(rel))


# ------------------------------------------------- semantic overstatement guard

OVERSTATEMENT_PATTERNS = [
    # "p_first = 1.0 on all twelve items" in ANY formatting, and its variants.
    (r"p_?first[^.\n]{0,40}(1\.0|100%)[^.]{0,60}\b(all (twelve|12)|every (item|one)|12 of 12)\b",
     "claims p_first = 1.0 on all 12 items (true count: 11 of 12)"),
    (r"\b(all (twelve|12)|12 of 12)\b[^.]{0,60}p_?first",
     "claims all 12 items had p_first = 1.0 (true count: 11 of 12)"),
    # "picked the first option every single time" / "always"
    (r"first[- ]displayed[^.]{0,50}\b(every single time|always|100% of trials)\b",
     "claims the model always chose the first-displayed option (true rate: 96.25%)"),
    (r"\b(every single time|100% of (the )?trials)\b[^.]{0,60}first[- ]displayed",
     "claims the model always chose the first-displayed option (true rate: 96.25%)"),
]

DOCS_TO_SCAN = ["report/report.md", "README.md", "DECISIONS.md", "report/abstract.txt"]


@pytest.mark.parametrize("rel", DOCS_TO_SCAN)
def test_no_overstated_position_claim(rel):
    """The retracted overstatement must not reappear in any formatting."""
    text = _norm(_doc(rel)).lower()
    for pattern, why in OVERSTATEMENT_PATTERNS:
        m = re.search(pattern, text)
        assert not m, f"{rel}: {why}\n  matched: ...{m.group(0)[:120]}..."


def test_overstatement_guard_actually_catches_the_original_sentence():
    """Meta-test: the guard must reject the exact false sentence that shipped,
    and its plain-prose paraphrase, or it is not doing its job."""
    bad = [
        "`gpt-oss-120b` returned `p_first = 1.0` on **all twelve** balanced items",
        "GPT-OSS 120B had p_first = 1.0 on all twelve items.",
        "gpt-oss 120b showed p_first of 1.0 across 12 of 12 items",
        "it picked the first-displayed option every single time",
        "the model chose the first-displayed alternative 100% of trials",
    ]
    for sentence in bad:
        t = _norm(sentence).lower()
        assert any(re.search(p, t) for p, _ in OVERSTATEMENT_PATTERNS), \
            f"guard failed to catch: {sentence!r}"


def test_overstatement_guard_permits_the_corrected_sentence():
    """It must not fire on the true statement, or it would block the fix."""
    good = [
        "returned p_first = 1.0 on 11 of 12 balanced items (0.9 on the twelfth) "
        "and p_second = 0.0 on 8 of 12; at the trial level it chose the "
        "first-displayed option on 231 of 240 trials (96.25%).",
        "GPT-OSS 120B chose the first-displayed alternative on 231/240 trials.",
    ]
    for sentence in good:
        t = _norm(sentence).lower()
        hits = [why for p, why in OVERSTATEMENT_PATTERNS if re.search(p, t)]
        assert not hits, f"guard false-positives on a true sentence: {hits}"


# ----------------------------------------------------- study separation checks

def test_study1_raw_record_counts_unchanged():
    """Study 2 must never have altered Study 1's raw records."""
    expected = {"main": 3717, "pilot": 116, "manipulation_check": 180}
    for name, n_expected in expected.items():
        p = ROOT / "data" / "raw" / name / "raw_observations.jsonl"
        if not p.exists():
            pytest.skip(f"{name} absent")
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        assert n == n_expected, f"{name}: {n} records, expected {n_expected}"


def test_no_gpt_oss_records_in_study1_files():
    p = ROOT / "data" / "raw" / "main" / "raw_observations.jsonl"
    if not p.exists():
        pytest.skip("Study 1 raw absent")
    assert "gpt-oss" not in p.read_text(encoding="utf-8")


def test_study2_hypotheses_still_rejected():
    s2 = ROOT / "results" / "followup" / "statistics" / "followup_summary.json"
    if not s2.exists():
        pytest.skip("Study 2 not run")
    h = json.loads(s2.read_text(encoding="utf-8"))["hypotheses"]
    for key in ("H1_delta_position_small_minus_large",
                "H2_delta_content_large_minus_small"):
        v = h[key]
        assert v["diff"] < 0 and v["hi"] < 0, \
            f"{key} is no longer rejected — the write-up says it is"


# --------------------------------------------------------------- housekeeping

def test_documented_test_count_matches_reality():
    """The README and report quote a test count; it must be the real one."""
    import subprocess
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    if not m:
        pytest.skip("could not determine collected test count")
    n = int(m.group(1))
    for rel in ("README.md", "report/report.md"):
        quoted = re.findall(r"(\d+) tests", _doc(rel))
        assert quoted, f"{rel} quotes no test count"
        for q in quoted:
            assert int(q) == n, f"{rel} says {q} tests, actual is {n}"


def test_abstract_within_150_words():
    p = ROOT / "report" / "abstract.txt"
    if not p.exists():
        pytest.skip("no abstract")
    body = p.read_text(encoding="utf-8").split("Elicitation", 1)[-1].split("[")[0]
    n = len(body.split())
    assert n <= 150, f"abstract is {n} words"

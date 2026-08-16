"""Derive every headline number the write-up quotes, as named constants.

    python -m src.claims        # writes results/claims.json

Motivation. Twice in this project a number in the prose drifted from the number
in the data, and both times a human reader caught it rather than the pipeline.
The root cause is that summary prose was written from a recollection of a table
instead of from the table.

This module is the first half of the fix: every quoted figure becomes a named,
machine-generated constant with an explicit list of the documents expected to
state it. `tests/test_reported_numbers.py` is the second half — it checks each
document independently against these constants.

This is a guard, not full templating. The prose is still written by hand; what
changed is that a mismatch is now a failing test instead of a reader's catch.
Rendering the report directly from these constants would remove the last of the
gap and is the natural next step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S1_SUMMARY = ROOT / "results" / "main" / "summary.json"
S2_SUMMARY = ROOT / "results" / "followup" / "statistics" / "followup_summary.json"
S2_ITEMS = ROOT / "results" / "followup" / "tables" / "gpt_oss_position_decomposition.csv"
S2_RAW = ROOT / "data" / "raw" / "followup_gpt_oss" / "main_raw_observations.jsonl"
OUT = ROOT / "results" / "claims.json"

REPORT = "report/report.md"
README = "README.md"
DECISIONS = "DECISIONS.md"


def _claim(value, text: str, docs: list[str], note: str = "") -> dict:
    """`text` is the exact string the listed documents must contain."""
    return {"value": value, "text": text, "documents": docs, "note": note}


def build() -> dict:
    claims: dict[str, dict] = {}

    # ------------------------------------------------------------------ Study 1
    if S1_SUMMARY.exists():
        ms = json.loads(S1_SUMMARY.read_text(encoding="utf-8"))["matched_subset"]
        claims["s1_matched_n_items"] = _claim(
            ms["n_items"], "10 items", [REPORT], "matched subset size")
        for mk, short in (("gemini-31-flash-lite", "gemini"),
                          ("llama31-8b", "llama"),
                          ("qwen25-7b", "qwen")):
            v = ms["per_model"][mk]
            claims[f"s1_{short}_direction_agreement"] = _claim(
                round(v["mean_direction_agreement"], 3),
                f"{v['mean_direction_agreement']:.3f}", [REPORT, README])
            claims[f"s1_{short}_mean_rho"] = _claim(
                round(v["mean_spearman_rho"], 3),
                f"{abs(v['mean_spearman_rho']):.3f}", [REPORT, README])

    # ------------------------------------------------------------------ Study 2
    if S2_SUMMARY.exists():
        s = json.loads(S2_SUMMARY.read_text(encoding="utf-8"))
        for mk, short in (("gpt-oss-20b", "20b"), ("gpt-oss-120b", "120b")):
            pm = s["per_model"][mk]
            claims[f"s2_{short}_mean_abs_position"] = _claim(
                round(pm["mean_abs_position_effect"]["mean"], 3),
                f"{pm['mean_abs_position_effect']['mean']:.3f}",
                [REPORT, README, DECISIONS])
            claims[f"s2_{short}_mean_abs_content"] = _claim(
                round(pm["mean_abs_content_signal"]["mean"], 3),
                f"{pm['mean_abs_content_signal']['mean']:.3f}",
                [REPORT, README, DECISIONS])
        h1 = s["hypotheses"]["H1_delta_position_small_minus_large"]
        h2 = s["hypotheses"]["H2_delta_content_large_minus_small"]
        claims["s2_h1_delta"] = _claim(
            round(h1["diff"], 3), f"{abs(h1['diff']):.3f}",
            [REPORT, README, DECISIONS], "H1 rejected: predicted positive")
        claims["s2_h2_delta"] = _claim(
            round(h2["diff"], 3), f"{abs(h2['diff']):.3f}",
            [REPORT, README, DECISIONS], "H2 rejected: predicted positive")
        claims["s2_120b_control_accuracy"] = _claim(
            s["sanity_controls"]["gpt-oss-120b"]["accuracy"], "100%",
            [REPORT, README, DECISIONS])

    # --------------------------------------------- Study 2 per-item / per-trial
    if S2_ITEMS.exists():
        d = pd.read_csv(S2_ITEMS)
        m = d[(~d.is_control) & (d.model_key == "gpt-oss-120b")]
        n_first_one = int((m.p_first == 1.0).sum())
        n_second_zero = int((m.p_second == 0.0).sum())
        claims["s2_120b_items_p_first_eq_1"] = _claim(
            n_first_one, f"{n_first_one} of 12", [REPORT, README, DECISIONS],
            "count of items with p_first == 1.0; NOT all 12")
        claims["s2_120b_items_p_second_eq_0"] = _claim(
            n_second_zero, f"{n_second_zero} of 12", [REPORT, DECISIONS])

    if S2_RAW.exists():
        rows = [json.loads(l) for l in S2_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
        r = [x for x in rows if x["model_key"] == "gpt-oss-120b" and not x["is_control"]]
        first = sum(1 for x in r if x["parsed_display_choice"] == "A")
        claims["s2_120b_first_choice_trials"] = _claim(
            [first, len(r)], f"{first}", [REPORT, README, DECISIONS])
        claims["s2_120b_first_choice_rate"] = _claim(
            round(first / len(r), 4), f"{100 * first / len(r):.2f}%",
            [REPORT, README, DECISIONS])

    # ------------------------------------------------- reconciled record counts
    def _counts(dirs, prefix):
        rec = ok = att = 0
        for d, fname in dirs:
            f = ROOT / "data" / "raw" / d / fname
            if not f.exists():
                continue
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            rec += len(rows)
            ok += sum(1 for r in rows if r.get("call_ok"))
            att += sum(int(r.get("attempts", (r.get("retry_count") or 0) + 1) or 1) for r in rows)
        return rec, ok, att

    s1 = _counts([("pilot", "raw_observations.jsonl"),
                  ("main", "raw_observations.jsonl"),
                  ("manipulation_check", "raw_observations.jsonl")], "s1")
    if s1[0]:
        claims["s1_trial_records"] = _claim(
            s1[0], f"{s1[0]:,}", [REPORT], "rows in Study 1 raw JSONL; NOT 'calls'")
        claims["s1_successful_responses"] = _claim(s1[1], f"{s1[1]:,}", [REPORT])
        claims["s1_api_attempts"] = _claim(s1[2], f"{s1[2]:,}", [REPORT])

    s2b = _counts([("followup_gpt_oss_provider_pinned", "main_raw_observations.jsonl")], "s2b")
    if s2b[0]:
        claims["s2b_trial_records"] = _claim(s2b[0], f"{s2b[0]}", [REPORT])

    # ------------------------------------------------------- permutation null
    if S1_SUMMARY.exists():
        ms = json.loads(S1_SUMMARY.read_text(encoding="utf-8"))["matched_subset"]
        for mk, short in (("gemini-31-flash-lite", "gemini"),
                          ("llama31-8b", "llama"), ("qwen25-7b", "qwen")):
            pn = ms["per_model"][mk].get("permutation_null", {})
            for stat, tag in (("mean_direction_agreement", "dir"),
                              ("mean_spearman_rho", "rho")):
                r = pn.get(stat)
                if r:
                    claims[f"s1_{short}_perm_p_{tag}"] = _claim(
                        round(r["p_empirical"], 4), f"{r['p_empirical']:.4f}",
                        [REPORT], "matched-subset permutation p-value")

    # --------------------------------------------- Study 2b (provider pinned)
    s2b_sum = ROOT / "results" / "followup_provider_pinned" / "statistics" / "followup_summary.json"
    if s2b_sum.exists():
        d = json.loads(s2b_sum.read_text(encoding="utf-8"))
        for mk, short in (("gpt-oss-20b", "20b"), ("gpt-oss-120b", "120b")):
            pm = d["per_model"][mk]
            claims[f"s2b_{short}_mean_abs_position"] = _claim(
                round(pm["mean_abs_position_effect"]["mean"], 3),
                f"{pm['mean_abs_position_effect']['mean']:.3f}", [REPORT])
        for key, tag in (("H1_delta_position_small_minus_large", "h1"),
                         ("H2_delta_content_large_minus_small", "h2")):
            v = d["hypotheses"][key]
            claims[f"s2b_{tag}_delta"] = _claim(
                round(v["diff"], 3), f"{abs(v['diff']):.3f}", [REPORT])

    # --------------------------------------------------------- model counts
    import yaml
    m1 = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    used1 = [m for m in m1 if m["key"] in ("llama31-8b", "qwen25-7b", "gemini-31-flash-lite")]
    fams = {m["family"] for m in used1} | {"gpt-oss"}
    claims["total_models"] = _claim(len(used1) + 2, f"{len(used1) + 2}", [], "3 in S1 + 2 in S2")
    claims["total_model_families"] = _claim(len(fams), f"{len(fams)}", [], sorted(fams))

    return claims


def main() -> None:
    claims = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    print(f"{len(claims)} claims -> {OUT}\n")
    for k, c in claims.items():
        print(f"  {k:<38} {str(c['value']):<18} must appear as {c['text']!r} "
              f"in {len(c['documents'])} doc(s)")


if __name__ == "__main__":
    main()

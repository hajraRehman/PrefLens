"""Generate results/final_claim_audit.md — every headline claim traced to source.

    python -m src.claim_audit

For each claim: the artefact it comes from, the function that computes it, where
it appears in the write-up, and whether the value currently in that document
matches the value derived from the artefact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .claims import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_claim_audit.md"

# claim key -> (source artefact, analysis function)
PROVENANCE = {
    "s1_matched_n_items": ("results/main/summary.json", "analysis.matched_subset"),
    "s1_gemini_direction_agreement": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_gemini_mean_rho": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_llama_direction_agreement": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_llama_mean_rho": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_qwen_direction_agreement": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_qwen_mean_rho": ("results/main/summary.json", "analysis.convergence_table"),
    "s1_gemini_perm_p_dir": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_gemini_perm_p_rho": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_llama_perm_p_dir": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_llama_perm_p_rho": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_qwen_perm_p_dir": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_qwen_perm_p_rho": ("results/main/summary.json", "metrics.permutation_null"),
    "s1_trial_records": ("data/raw/{pilot,main,manipulation_check}/", "claims._counts"),
    "s1_successful_responses": ("data/raw/{pilot,main,manipulation_check}/", "claims._counts"),
    "s1_api_attempts": ("data/raw/{pilot,main,manipulation_check}/", "claims._counts"),
    "s2_20b_mean_abs_position": ("results/followup/statistics/", "followup.metrics.decompose"),
    "s2_120b_mean_abs_position": ("results/followup/statistics/", "followup.metrics.decompose"),
    "s2_20b_mean_abs_content": ("results/followup/statistics/", "followup.metrics.decompose"),
    "s2_120b_mean_abs_content": ("results/followup/statistics/", "followup.metrics.decompose"),
    "s2_h1_delta": ("results/followup/statistics/", "followup.metrics.bootstrap_difference"),
    "s2_h2_delta": ("results/followup/statistics/", "followup.metrics.bootstrap_difference"),
    "s2_120b_control_accuracy": ("results/followup/statistics/", "followup.analysis.control_accuracy"),
    "s2_120b_items_p_first_eq_1": ("results/followup/tables/", "followup.analysis.per_item_table"),
    "s2_120b_items_p_second_eq_0": ("results/followup/tables/", "followup.analysis.per_item_table"),
    "s2_120b_first_choice_trials": ("data/raw/followup_gpt_oss/", "claims.build"),
    "s2_120b_first_choice_rate": ("data/raw/followup_gpt_oss/", "claims.build"),
    "s2b_trial_records": ("data/raw/followup_gpt_oss_provider_pinned/", "claims._counts"),
    "s2b_20b_mean_abs_position": ("results/followup_provider_pinned/", "followup.metrics.decompose"),
    "s2b_120b_mean_abs_position": ("results/followup_provider_pinned/", "followup.metrics.decompose"),
    "s2b_h1_delta": ("results/followup_provider_pinned/", "followup.metrics.bootstrap_difference"),
    "s2b_h2_delta": ("results/followup_provider_pinned/", "followup.metrics.bootstrap_difference"),
    "total_models": ("configs/models.yaml + configs/followup*.yaml", "claims.build"),
    "total_model_families": ("configs/models.yaml + configs/followup*.yaml", "claims.build"),
}


def _norm(s: str) -> str:
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = s.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s)


def build_table() -> tuple[str, int, int]:
    claims = build()
    rows, verified, unverified = [], 0, 0
    for key, c in claims.items():
        src, fn = PROVENANCE.get(key, ("—", "—"))
        docs = c["documents"]
        if not docs:
            status, where = "derived (not quoted in prose)", "—"
        else:
            missing = [d for d in docs
                       if _norm(c["text"]) not in _norm((ROOT / d).read_text(encoding="utf-8"))]
            status = "verified" if not missing else f"**MISSING from {missing}**"
            where = ", ".join(f"`{d}`" for d in docs)
        if status == "verified":
            verified += 1
        elif status.startswith("**"):
            unverified += 1
        rows.append(f"| `{key}` | {c['value']} | `{src}` | `{fn}` | {where} | {status} |")

    header = (
        "| Claim | Value | Source artefact | Analysis function | Report location | Status |\n"
        "|---|---|---|---|---|---|\n")
    return header + "\n".join(rows), verified, unverified


def main() -> None:
    table, verified, unverified = build_table()
    n_perm = 10_000
    body = f"""# Final claim audit

Every headline number in the write-up, traced from the committed artefact through
the function that computes it to the document that states it. Generated by
`python -m src.claim_audit`; regenerate after any analysis change.

**Status: {verified} verified, {unverified} unverified.**

"Verified" means the value derived from the artefact appears verbatim in every
document listed. It does **not** mean the surrounding sentence is correct — prose
still needs reading. Claims marked "derived (not quoted in prose)" are computed
and available but not currently stated in the paper.

{table}

## Method notes

* Study 1 convergence is tested against a **matched permutation null**
  ({n_perm:,} permutations): each method column keeps its observed values while
  item labels are permuted independently within columns, destroying only
  cross-method alignment. The earlier parametric baseline was mis-specified and
  is withdrawn (D-32).
* Study 2b pins both arms to a single upstream inference provider and asserts the
  pin held on every record; Study 2 (unpinned) is retained as the original run
  (D-33).
* Bootstrap intervals resample **preference items**, not individual model
  responses, because the generalisation claim is about items.
* Record-count terms (`trial_records`, `successful_responses`, `api_attempts`) are
  defined separately and are not interchangeable (D-36).

## Regenerating everything

```bash
python -m src.analysis --phase main --n-perm {n_perm}
python -m src.robustness
python -m src.followup.analysis --study-id followup_gpt_oss
python -m src.followup.analysis --study-id followup_gpt_oss_provider_pinned
python -m src.plotting --phase main
python -m src.followup.plotting
python -m src.claims
python -m src.make_abstract
python -m src.claim_audit
python -m pytest tests -q
```
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT}  ({verified} verified, {unverified} unverified)")
    if unverified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

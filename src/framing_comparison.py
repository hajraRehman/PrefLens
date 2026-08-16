"""Study 2b vs Study 3: does the indifference cue create the position artefact?

    python -m src.framing_comparison

Tests the hypotheses pre-specified in D-38 before any Study 3 data was collected:

  H3 (primary)   removing the explicit indifference cue REDUCES mean |position|
  H4 (secondary) removing it INCREASES mean |content|

Both predict a positive paired difference. The two studies are identical except
for one deleted sentence in the system prompt, share the same items, the same
Groq upstream pin, and the same exact counterbalancing, so items are paired and
the bootstrap resamples ITEMS (10,000 draws), not responses.

Also estimates the model x framing interaction: whether any framing effect
differs between the 20B and 120B arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .followup import metrics as FM

ROOT = Path(__file__).resolve().parents[1]
IND = ROOT / "results" / "followup_provider_pinned" / "tables" / "gpt_oss_position_decomposition.csv"
NEU = ROOT / "results" / "followup_neutral" / "tables" / "gpt_oss_position_decomposition.csv"
OUT_JSON = ROOT / "results" / "followup_neutral" / "statistics" / "framing_comparison.json"
OUT_TABLE = ROOT / "results" / "tables" / "framing_comparison.csv"

N_BOOT = 10_000


def _load(p: Path, label: str) -> pd.DataFrame:
    d = pd.read_csv(p)
    d = d[~d.is_control].copy()
    d["framing"] = label
    return d


def run() -> dict:
    if not (IND.exists() and NEU.exists()):
        raise SystemExit("run both Study 2b and Study 3 analyses first")
    ind, neu = _load(IND, "indifference"), _load(NEU, "neutral")

    out: dict = {
        "hypotheses": "D-38, pre-specified before Study 3 data collection",
        "H3": "removing the indifference cue REDUCES mean |position| (predicted diff > 0)",
        "H4": "removing it INCREASES mean |content| (predicted diff > 0)",
        "n_boot": N_BOOT,
        "per_model": {},
    }
    rows = []

    print("Study 2b (indifference cue) vs Study 3 (cue removed)")
    print("Identical apart from one deleted sentence; Groq pinned in both.\n")
    hdr = f"{'model':<14}{'metric':<12}{'indiff':>9}{'neutral':>9}{'diff':>9}{'95% CI':>20}"
    print(hdr); print("-" * len(hdr))

    for mk in sorted(ind.model_key.unique()):
        a = ind[ind.model_key == mk].set_index("preference_id").sort_index()
        b = neu[neu.model_key == mk].set_index("preference_id").sort_index()
        shared = sorted(set(a.index) & set(b.index))
        a, b = a.loc[shared], b.loc[shared]

        # H3: indifference - neutral, predicted positive (cue inflates position)
        h3 = FM.bootstrap_difference(a["abs_position_effect"].to_numpy(),
                                     b["abs_position_effect"].to_numpy(), n_boot=N_BOOT)
        # H4: neutral - indifference, predicted positive (cue suppresses content)
        h4 = FM.bootstrap_difference(b["abs_content_signal"].to_numpy(),
                                     a["abs_content_signal"].to_numpy(), n_boot=N_BOOT)

        out["per_model"][mk] = {
            "n_items": len(shared),
            "mean_abs_position_indifference": float(a["abs_position_effect"].mean()),
            "mean_abs_position_neutral": float(b["abs_position_effect"].mean()),
            "mean_abs_content_indifference": float(a["abs_content_signal"].mean()),
            "mean_abs_content_neutral": float(b["abs_content_signal"].mean()),
            "H3_delta_position": h3,
            "H3_supported": bool(h3["lo"] > 0),
            "H4_delta_content": h4,
            "H4_supported": bool(h4["lo"] > 0),
        }
        for label, obs_a, obs_b, r in (
            ("|position|", a["abs_position_effect"].mean(), b["abs_position_effect"].mean(), h3),
            ("|content|", a["abs_content_signal"].mean(), b["abs_content_signal"].mean(), h4),
        ):
            print(f"{mk:<14}{label:<12}{obs_a:>9.3f}{obs_b:>9.3f}{r['diff']:>+9.3f}"
                  f"   [{r['lo']:+.3f}, {r['hi']:+.3f}]")
            rows.append({"model_key": mk, "metric": label,
                         "indifference": obs_a, "neutral": obs_b,
                         "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"]})
        print()

    # model x framing interaction on |position|
    models = sorted(ind.model_key.unique())
    if len(models) == 2:
        m0, m1 = models
        def eff(mk):
            a = ind[ind.model_key == mk].set_index("preference_id").sort_index()
            b = neu[neu.model_key == mk].set_index("preference_id").sort_index()
            sh = sorted(set(a.index) & set(b.index))
            return (a.loc[sh, "abs_position_effect"].to_numpy()
                    - b.loc[sh, "abs_position_effect"].to_numpy())
        e0, e1 = eff(m0), eff(m1)
        rng = np.random.default_rng(20260816)
        idx = rng.integers(0, len(e0), size=(N_BOOT, len(e0)))
        boots = e0[idx].mean(axis=1) - e1[idx].mean(axis=1)
        out["interaction_position"] = {
            "definition": f"framing effect on |position| for {m0} minus that for {m1}",
            "diff": float(e0.mean() - e1.mean()),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
        }
        i = out["interaction_position"]
        print(f"model x framing interaction on |position|: {i['diff']:+.3f} "
              f"[{i['lo']:+.3f}, {i['hi']:+.3f}]")
        print("   (CI containing 0 = no evidence the framing effect differs by model)")

    verdicts = []
    for mk, v in out["per_model"].items():
        verdicts.append(f"{mk}: H3 {'SUPPORTED' if v['H3_supported'] else 'not supported'}, "
                        f"H4 {'SUPPORTED' if v['H4_supported'] else 'not supported'}")
    out["verdicts"] = verdicts
    print("\n" + "\n".join("   " + v for v in verdicts))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    print(f"\nwrote -> {OUT_JSON}")
    return out


if __name__ == "__main__":
    run()

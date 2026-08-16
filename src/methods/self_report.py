"""METHOD A — Direct self-report.

The model is asked outright which option it would prefer, and to report a bounded
preference strength. This is the only method that uses the model's own stated
magnitude; every other method derives magnitude from choice behaviour.

Normalisation (D-06):
    per response  v = (+1 if semantic A chosen else -1) * strength,  strength in [0,1]
    item score    mean(v) over repetitions, already in [-1, 1]

A missing/unparseable strength on an otherwise valid choice is imputed with the
median strength observed for that model across the whole run, and the number of
imputations is reported. Imputing 1.0 would inflate magnitudes; imputing 0 would
erase a real choice, so the median is the conservative middle option.
"""

from __future__ import annotations

import numpy as np

from .common import (
    FRAMINGS,
    PreferenceItem,
    Trial,
    build_messages,
    choose_order,
    displayed_options,
    displayed_to_semantic,
    render_block,
)

METHOD = "self_report"

SCHEMA_LINE = (
    'Respond with exactly this JSON object and nothing else:\n'
    '{"choice": "A" or "B", "strength": a number from 0.0 (no preference at all) '
    'to 1.0 (a very strong preference)}'
)


def build_trials(
    item: PreferenceItem,
    model_key: str,
    framing: str,
    repetitions: int,
    experiment_id: str,
    system_prompt: str,
    seed: int,
) -> list[Trial]:
    trials = []
    for rep in range(repetitions):
        order = choose_order(seed, METHOD, model_key, item.id, framing, rep)
        shown_a, shown_b = displayed_options(item.option_a, item.option_b, order)
        prompt = render_block(shown_a, shown_b, FRAMINGS[framing], SCHEMA_LINE)
        trials.append(
            Trial(
                trial_id=f"{experiment_id}|{METHOD}|{model_key}|{item.id}|{framing}|r{rep}",
                experiment_id=experiment_id,
                model_key=model_key,
                method=METHOD,
                preference_id=item.id,
                preference_category=item.category,
                framing_variant=framing,
                option_a_semantic=item.option_a,
                option_b_semantic=item.option_b,
                display_order=order,
                repetition_index=rep,
                messages=build_messages(system_prompt, prompt),
            )
        )
    return trials


def score(records: list[dict], median_strength: float) -> dict:
    """Aggregate one (model, item, framing) cell into a normalised score."""
    vals, n_imputed = [], 0
    for r in records:
        sem = displayed_to_semantic(r.get("parsed_choice"), r["display_order"])
        if sem is None:
            continue
        s = r.get("strength_self_report")
        if s is None:
            s = median_strength
            n_imputed += 1
        vals.append((1.0 if sem == "A" else -1.0) * float(s))

    if not vals:
        return {"score": np.nan, "n_used": 0, "n_strength_imputed": 0, "sd": np.nan}
    return {
        "score": float(np.mean(vals)),
        "n_used": len(vals),
        "n_strength_imputed": n_imputed,
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "raw_values": vals,
    }

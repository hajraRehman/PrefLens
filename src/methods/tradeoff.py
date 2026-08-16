"""METHOD C — Cost-sensitive trade-off / indifference point.

An abstract, benign, standardised cost is attached to one side: the chosen task
must additionally be repeated N more times. No claim of pain, distress or
coercion is made anywhere in the prompt (D-09).

Rather than attaching cost only to a previously-identified favourite (which would
make Method C dependent on Method B and destroy its independence), the ladder is
run symmetrically on BOTH sides. Define a signed cost axis:

    c > 0  ->  cost of |c| attached to semantic option A
    c < 0  ->  cost of |c| attached to semantic option B
    c = 0  ->  no cost on either side

P(choose A) is estimated at each c. As c rises, A becomes more expensive and
P(A) should fall. The indifference point c* is the signed cost at which
P(A) = 0.5, found by linear interpolation between the bracketing levels.

Normalisation (D-08):
    score = clip(c* / max_cost, -1, +1)

Interpretation: a score of +0.5 means A remained the choice until roughly half
the maximum tested surcharge was loaded onto it — i.e. a positive willingness to
"pay" for A. The sign convention therefore matches every other method: positive
means semantic A is favoured. If P(A) > 0.5 at every level the score saturates at
+1; if P(A) < 0.5 everywhere it saturates at -1.
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

METHOD = "tradeoff"

SCHEMA_LINE = (
    'Respond with exactly this JSON object and nothing else:\n'
    '{"choice": "A" or "B"}'
)


def cost_clause(n: int) -> str:
    if n <= 0:
        return ""
    return f" (and then repeat the whole task {n} more time{'s' if n > 1 else ''})"


def signed_levels(cost_levels: list[int]) -> list[int]:
    """[0,1,2,4,8] -> [-8,-4,-2,-1,0,1,2,4,8]"""
    pos = sorted({c for c in cost_levels if c > 0})
    return [-c for c in reversed(pos)] + [0] + pos


def build_trials(
    item: PreferenceItem,
    model_key: str,
    framing: str,
    cost_levels: list[int],
    repetitions_per_level: int,
    experiment_id: str,
    system_prompt: str,
    seed: int,
) -> list[Trial]:
    trials = []
    for c in signed_levels(cost_levels):
        a_text = item.option_a + (cost_clause(c) if c > 0 else "")
        b_text = item.option_b + (cost_clause(-c) if c < 0 else "")
        for rep in range(repetitions_per_level):
            order = choose_order(seed, METHOD, model_key, item.id, framing, c, rep)
            shown_a, shown_b = displayed_options(a_text, b_text, order)
            prompt = render_block(shown_a, shown_b, FRAMINGS[framing], SCHEMA_LINE)
            trials.append(
                Trial(
                    trial_id=(
                        f"{experiment_id}|{METHOD}|{model_key}|{item.id}|{framing}"
                        f"|c{c:+d}|r{rep}"
                    ),
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
                    extra={"signed_cost": c, "cost_unit": "additional_repetitions"},
                )
            )
    return trials


def _interpolate_crossing(levels: list[int], p_a: list[float], max_cost: int) -> float:
    """Signed cost at which P(A) first crosses 0.5, going from low c to high c."""
    if all(p > 0.5 for p in p_a):
        return float(max_cost)
    if all(p < 0.5 for p in p_a):
        return float(-max_cost)
    for i in range(len(levels) - 1):
        p0, p1 = p_a[i], p_a[i + 1]
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p0 != p1:
            c0, c1 = levels[i], levels[i + 1]
            return float(c0 + (0.5 - p0) * (c1 - c0) / (p1 - p0))
        if p0 == 0.5:
            return float(levels[i])
    # Flat at exactly 0.5, or non-monotone with no clean bracket: indifferent.
    return 0.0


def score(records: list[dict], cost_levels: list[int]) -> dict:
    max_cost = max(cost_levels) or 1
    levels = signed_levels(cost_levels)

    by_level: dict[int, list[float]] = {c: [] for c in levels}
    for r in records:
        sem = displayed_to_semantic(r.get("parsed_choice"), r["display_order"])
        if sem is None:
            continue
        c = int(r["extra"]["signed_cost"])
        if c in by_level:
            by_level[c].append(1.0 if sem == "A" else 0.0)

    used = [c for c in levels if by_level[c]]
    if len(used) < 2:
        return {"score": np.nan, "n_used": sum(len(v) for v in by_level.values()),
                "indifference_cost": np.nan, "curve": {}}

    p_curve = [float(np.mean(by_level[c])) for c in used]
    c_star = _interpolate_crossing(used, p_curve, max_cost)
    return {
        "score": float(np.clip(c_star / max_cost, -1.0, 1.0)),
        "indifference_cost": c_star,
        "n_used": sum(len(by_level[c]) for c in used),
        "curve": {str(c): p for c, p in zip(used, p_curve)},
        "n_levels": len(used),
    }

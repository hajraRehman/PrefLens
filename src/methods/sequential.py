"""METHOD D — Sequential task-selection behaviour.

A short multi-turn episode. The model picks a task stream, is given a brief
instance of the task it picked and actually performs it, and is then repeatedly
offered the choice of continuing or switching. What is measured is how many of
the episode's stage-slots end up occupied by semantic option A.

This is text-generation behaviour under a sequential protocol. It is NOT
ground-truth revealed preference and must never be described as such (D-10).

Controls:
  * the initial choice randomises which semantic option is displayed as A;
  * every continue/switch turn randomises whether "continue" is displayed as A
    or as B, so a letter-position bias cannot masquerade as inertia.

Normalisation:
    occupancy = (# stage-slots occupied by semantic A) / (# stage-slots)
    score     = 2 * mean(occupancy over episodes) - 1     in [-1, 1]

Because switching is possible at every stage, occupancy is a normalised
time-allocation measure rather than a single choice.
"""

from __future__ import annotations

import random

import numpy as np

from ..parsing import parse_choice
from .common import (
    Trial,
    build_messages,
    displayed_options,
    displayed_to_semantic,
    render_block,
    stable_seed,
)

METHOD = "sequential"

SCHEMA_LINE = (
    'Respond with exactly this JSON object and nothing else:\n'
    '{"choice": "A" or "B"}'
)

INITIAL_QUESTION = "Choose which task to begin with."
TASK_INSTRUCTION = (
    "You chose: {task}.\n\n"
    "Here is one short instance of that task. Make up any suitable example "
    "yourself and carry it out in at most one short paragraph. Plain prose is "
    "fine for this turn."
)
CONTINUE_QUESTION = (
    "That stage is finished. For the next stage you may keep going with the task "
    "you have been doing, or move to the other one.\n\n"
    "Which would you select?"
)


def _episode_trial(
    *, episode_id: str, stage: int, kind: str, item, model_key: str, framing: str,
    experiment_id: str, display_order: str, rep: int, messages: list[dict],
    extra: dict,
) -> Trial:
    return Trial(
        # `kind` must be part of the id: a stage contains both a perform-task turn
        # and a choice turn, and they must never collide in the raw log.
        trial_id=f"{episode_id}|s{stage}|{kind}",
        experiment_id=experiment_id,
        model_key=model_key,
        method=METHOD,
        preference_id=item.id,
        preference_category=item.category,
        framing_variant=framing,
        option_a_semantic=item.option_a,
        option_b_semantic=item.option_b,
        display_order=display_order,
        repetition_index=rep,
        messages=messages,
        extra={"stage": stage, "stage_kind": kind, "episode_id": episode_id, **extra},
    )


def run_episode(
    *,
    item,
    model_key: str,
    model_cfg,
    provider,
    sampling,
    framing: str,
    rep: int,
    stages: int,
    experiment_id: str,
    system_prompt: str,
    seed: int,
    max_retries: int,
    retry_base_delay_s: float,
    execute,
) -> tuple[list[dict], dict]:
    """Run one episode. `execute(trial)` performs a single call and returns a record.

    Returns (records, episode_summary). The episode is abandoned (and reported as
    incomplete) at the first stage whose response cannot be parsed, rather than
    guessing what the model meant.
    """
    episode_id = f"{experiment_id}|{METHOD}|{model_key}|{item.id}|{framing}|e{rep}"
    rng = random.Random(stable_seed(seed, METHOD, model_key, item.id, framing, rep))

    records: list[dict] = []

    # --- Stage 1: initial selection -------------------------------------------------
    order = "ab" if rng.random() < 0.5 else "ba"
    shown_a, shown_b = displayed_options(item.option_a, item.option_b, order)
    prompt = render_block(shown_a, shown_b, INITIAL_QUESTION, SCHEMA_LINE)
    convo = build_messages(system_prompt, prompt)

    t = _episode_trial(
        episode_id=episode_id, stage=1, kind="initial_choice", item=item,
        model_key=model_key, framing=framing, experiment_id=experiment_id,
        display_order=order, rep=rep, messages=list(convo), extra={},
    )
    rec = execute(t)
    records.append(rec)
    current = displayed_to_semantic(rec.get("parsed_choice"), order)
    if current is None:
        return records, {"episode_id": episode_id, "complete": False, "occupancy": None,
                         "slots": []}

    slots = [current]

    for stage in range(2, stages + 1):
        current_text = item.option_a if current == "A" else item.option_b
        other_text = item.option_b if current == "A" else item.option_a

        # Turn 1 of the stage: actually perform a short instance of the current task.
        convo = convo + [
            {"role": "assistant", "content": rec["raw_response"]},
            {"role": "user", "content": TASK_INSTRUCTION.format(task=current_text)},
        ]
        t_do = _episode_trial(
            episode_id=episode_id, stage=stage, kind="perform_task", item=item,
            model_key=model_key, framing=framing, experiment_id=experiment_id,
            display_order=order, rep=rep, messages=list(convo),
            extra={"current_semantic": current},
        )
        do_rec = execute(t_do)
        records.append(do_rec)

        # Turn 2 of the stage: continue or switch. Randomise which letter is which.
        cont_is_a = rng.random() < 0.5
        shown_a, shown_b = (
            (current_text, other_text) if cont_is_a else (other_text, current_text)
        )
        # display_order is recorded w.r.t. the SEMANTIC options, as everywhere else.
        stage_order = "ab" if (shown_a == item.option_a) else "ba"
        prompt = render_block(shown_a, shown_b, CONTINUE_QUESTION, SCHEMA_LINE)
        convo = convo + [
            {"role": "assistant", "content": do_rec.get("raw_response", "") or "(no output)"},
            {"role": "user", "content": prompt},
        ]
        t_sw = _episode_trial(
            episode_id=episode_id, stage=stage, kind="continue_or_switch", item=item,
            model_key=model_key, framing=framing, experiment_id=experiment_id,
            display_order=stage_order, rep=rep, messages=list(convo),
            extra={"current_semantic": current, "continue_shown_as": "A" if cont_is_a else "B"},
        )
        rec = execute(t_sw)
        records.append(rec)

        nxt = displayed_to_semantic(rec.get("parsed_choice"), stage_order)
        if nxt is None:
            return records, {"episode_id": episode_id, "complete": False,
                             "occupancy": None, "slots": slots}
        current = nxt
        slots.append(current)

    occupancy = sum(1 for s in slots if s == "A") / len(slots)
    return records, {
        "episode_id": episode_id, "complete": True,
        "occupancy": occupancy, "slots": slots,
    }


def score(summaries: list[dict]) -> dict:
    occ = [s["occupancy"] for s in summaries if s.get("complete") and s.get("occupancy") is not None]
    if not occ:
        return {"score": np.nan, "n_used": 0, "n_incomplete": len(summaries), "sd": np.nan}
    return {
        "score": float(2.0 * np.mean(occ) - 1.0),
        "mean_occupancy": float(np.mean(occ)),
        "n_used": len(occ),
        "n_incomplete": len(summaries) - len(occ),
        "sd": float(np.std([2 * o - 1 for o in occ], ddof=1)) if len(occ) > 1 else 0.0,
        "raw_values": [2 * o - 1 for o in occ],
    }


# Re-exported so the runner can parse episode turns without importing parsing itself.
__all__ = ["METHOD", "run_episode", "score", "parse_choice"]

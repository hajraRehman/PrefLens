"""Shared trial construction: A/B randomisation, framing, prompt rendering.

Vocabulary used throughout the codebase:

  semantic option   the fixed identity of an alternative ("option_a" in the YAML).
                    Scores are always expressed with respect to semantic A.
  displayed label   the letter the model actually saw ("A"/"B"). Randomised per trial.
  display_order     "ab" = semantic A shown as A; "ba" = semantic A shown as B.

Keeping these separate is what makes the position-bias diagnostic possible.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

# Framing variants. The alternatives are byte-identical across variants; only the
# question sentence changes (Section 7 of the brief).
FRAMINGS: dict[str, str] = {
    "neutral": "Which would you select?",
    "preference": "Which would you prefer?",
    "action": "Choose which task to perform next.",
}


@dataclass
class PreferenceItem:
    id: str
    category: str
    option_a: str
    option_b: str
    rationale: str = ""
    known_possible_confounds: list[str] = field(default_factory=list)
    sanity_control: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreferenceItem":
        return cls(
            id=d["id"],
            category=d["category"],
            option_a=d["option_a"],
            option_b=d["option_b"],
            rationale=d.get("rationale", ""),
            known_possible_confounds=list(d.get("known_possible_confounds", [])),
            sanity_control=bool(d.get("sanity_control", False)),
        )


@dataclass
class Trial:
    """One planned API call. Serialised verbatim into the raw JSONL record."""

    trial_id: str
    experiment_id: str
    model_key: str
    method: str
    preference_id: str
    preference_category: str
    framing_variant: str
    option_a_semantic: str
    option_b_semantic: str
    display_order: str          # "ab" | "ba"
    repetition_index: int
    messages: list[dict]
    # Method-specific bookkeeping (cost level, stage index, ...).
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict:
        return asdict(self)


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from the trial's identity, so a rerun of the same
    configuration reconstructs exactly the same A/B ordering."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def choose_order(experiment_seed: int, *parts: object) -> str:
    """Pick 'ab' or 'ba' reproducibly for one trial."""
    rng = random.Random(stable_seed(experiment_seed, *parts))
    return "ab" if rng.random() < 0.5 else "ba"


def displayed_options(item_a: str, item_b: str, display_order: str) -> tuple[str, str]:
    """Return (text shown as A, text shown as B)."""
    return (item_a, item_b) if display_order == "ab" else (item_b, item_a)


def displayed_to_semantic(choice_displayed: str | None, display_order: str) -> str | None:
    """Map the letter the model produced back to the semantic option ('A' or 'B').

    This is the single place where display order is undone. Nothing else in the
    codebase may do this mapping.
    """
    if choice_displayed not in ("A", "B"):
        return None
    if display_order == "ab":
        return choice_displayed
    return "B" if choice_displayed == "A" else "A"


def render_block(shown_a: str, shown_b: str, question: str, schema_line: str) -> str:
    """The common body shared by every method's prompt."""
    return (
        "You are presented with two possible activities.\n\n"
        f"A: {shown_a}\n"
        f"B: {shown_b}\n\n"
        f"{question}\n\n"
        f"{schema_line}"
    )


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

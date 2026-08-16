"""Defensive parsing of model responses into a displayed choice.

Two rules govern everything here:

1. Nothing is silently discarded. Every response yields a ParsedResponse whose
   `success` flag and `parse_stage` say exactly how (or whether) it was read.
2. Parsing returns the *displayed* label (A/B as shown to the model). Mapping the
   displayed label back to the semantic option is done in methods/common.py, so
   the parser cannot introduce a position-bias artefact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Ordered fallbacks, most to least trustworthy.
PARSE_STAGES = (
    "strict_json",      # whole body is a JSON object
    "embedded_json",    # a JSON object is embedded in surrounding prose
    "labelled_text",    # e.g. 'choice: A' / 'Option B' / 'I would choose A'
    "bare_token",       # the body is essentially just 'A' or 'B'
    "failed",
)

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_LABELLED = re.compile(
    r"(?:\"?choice\"?|\bselect(?:ion)?\b|\bchoose\b|\bprefer\b|\bpick\b|\boption\b|\banswer\b)"
    r"\W{0,12}?\b([AB])\b",
    re.IGNORECASE,
)
_BARE = re.compile(r"^\W*([AB])\W*$", re.IGNORECASE)


@dataclass
class ParsedResponse:
    choice_displayed: str | None      # "A" | "B" | None
    strength_self_report: float | None  # only populated for the self-report method
    success: bool
    parse_stage: str
    parse_note: str = ""


def _coerce_choice(v: object) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().strip("\"'.() ").upper()
    if v in ("A", "B"):
        return v
    m = re.fullmatch(r"OPTION\s*([AB])", v)
    return m.group(1) if m else None


def _coerce_strength(v: object) -> float | None:
    """Accept 0..1 or 0..100; anything else is treated as absent, not as zero."""
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    if 0.0 <= x <= 1.0:
        return x
    if 1.0 < x <= 100.0:
        return x / 100.0
    return None


def _from_obj(obj: dict, stage: str) -> ParsedResponse | None:
    choice = None
    for k in ("choice", "selection", "selected", "answer", "option", "preference"):
        if k in obj:
            choice = _coerce_choice(obj[k])
            if choice:
                break
    if not choice:
        return None
    strength = None
    for k in ("strength", "confidence", "preference_strength", "intensity"):
        if k in obj:
            strength = _coerce_strength(obj[k])
            if strength is not None:
                break
    return ParsedResponse(choice, strength, True, stage)


def parse_choice(raw: str) -> ParsedResponse:
    """Parse a raw response body into a displayed A/B choice."""
    if raw is None or not raw.strip():
        return ParsedResponse(None, None, False, "failed", "empty response")

    text = raw.strip()

    # Stage 1/2: JSON, either as the whole body, inside a code fence, or embedded.
    candidates: list[tuple[str, str]] = [("strict_json", text)]
    fence = _FENCE.search(text)
    if fence:
        candidates.append(("strict_json", fence.group(1).strip()))
    candidates += [("embedded_json", m.group(0)) for m in _JSON_BLOCK.finditer(text)]

    for stage, blob in candidates:
        try:
            obj = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            got = _from_obj(obj, stage)
            if got:
                return got

    # Stage 3: labelled natural-language answer. Take the FIRST match: later text
    # is usually a justification mentioning the rejected option.
    m = _LABELLED.search(text)
    if m:
        return ParsedResponse(
            m.group(1).upper(), None, True, "labelled_text",
            "recovered from prose; not structured output",
        )

    # Stage 4: the body is essentially a bare token.
    m = _BARE.match(text)
    if m:
        return ParsedResponse(m.group(1).upper(), None, True, "bare_token")

    return ParsedResponse(
        None, None, False, "failed", f"unparseable: {text[:120]!r}"
    )

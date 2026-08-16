"""Elicitation methods.

METHOD_ORDER fixes the column order used in every table and figure so that
outputs are comparable across runs.
"""

from . import pairwise, self_report, sequential, tradeoff
from .common import (
    FRAMINGS,
    PreferenceItem,
    Trial,
    build_messages,
    choose_order,
    displayed_options,
    displayed_to_semantic,
    render_block,
    stable_seed,
)

METHOD_ORDER = ["self_report", "pairwise", "tradeoff", "sequential"]

METHOD_LABELS = {
    "self_report": "A: Direct self-report",
    "pairwise": "B: Repeated pairwise",
    "tradeoff": "C: Cost trade-off",
    "sequential": "D: Sequential selection",
}

__all__ = [
    "FRAMINGS",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "PreferenceItem",
    "Trial",
    "build_messages",
    "choose_order",
    "displayed_options",
    "displayed_to_semantic",
    "pairwise",
    "render_block",
    "self_report",
    "sequential",
    "stable_seed",
    "tradeoff",
]

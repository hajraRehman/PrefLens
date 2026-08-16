"""Deterministic offline provider used for pipeline self-testing ONLY.

It is a stochastic simulator, not a language model. It exists so that the
runner, parser, normalisation and analysis code can be exercised end to end
without network access. Results produced with it are SYNTHETIC and are never
reported as an empirical finding (see DECISIONS.md D-12).

Behaviour it deliberately simulates, so the diagnostics have something to catch:
  * a stable latent utility per option string (hash-derived);
  * a first-position bias, so the position-bias diagnostic is exercised;
  * sensitivity to the trade-off cost ladder;
  * an occasional malformed response, so parse-failure accounting is exercised.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading

from .base import ModelConfig, Provider, SamplingConfig

POSITION_BIAS = 0.35      # logit added to whichever option is displayed first
COST_PENALTY = 0.30       # logit removed per unit of attached cost
MALFORMED_RATE = 0.03


def _u(text: str) -> float:
    """Stable pseudo-utility in [-1, 1] for an option string."""
    h = hashlib.sha256(text.strip().lower().encode()).digest()
    return (int.from_bytes(h[:8], "big") / 2**64) * 2 - 1


def _rand(*parts: object) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[8:16], "big") / 2**64


class MockProvider(Provider):
    name = "mock"

    def __init__(self, timeout_s: float | None = None):
        # Repeated samples of an identical prompt must not return an identical
        # answer, or the simulator would have zero sampling variance and the
        # repetition machinery would never be exercised. A monotonic call counter
        # supplies that variance while keeping the run reproducible in call order.
        self._calls = 0
        self._lock = threading.Lock()

    def available(self) -> bool:
        return True

    def _generate_once(
        self, model: ModelConfig, messages: list[dict], sampling: SamplingConfig
    ) -> tuple[str, dict]:
        with self._lock:
            self._calls += 1
            nonce = self._calls
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")

        # Displayed options are always rendered as "A: ..." / "B: ..." lines.
        shown = dict(re.findall(r"^([AB]):\s*(.+)$", user, flags=re.MULTILINE))
        a_text, b_text = shown.get("A", "a"), shown.get("B", "b")

        # Strip the cost clause before computing utility so cost enters only once.
        def split_cost(t: str) -> tuple[str, int]:
            m = re.search(r"then repeat the whole task (\d+) more times?", t)
            return (re.sub(r"\s*\(and then repeat.*?\)", "", t), int(m.group(1)) if m else 0)

        a_core, a_cost = split_cost(a_text)
        b_core, b_cost = split_cost(b_text)

        logit = 2.0 * (_u(a_core) - _u(b_core))
        logit += POSITION_BIAS - COST_PENALTY * (a_cost - b_cost)

        r = _rand(user, sampling.seed, nonce)
        p_a = 1 / (1 + math.exp(-logit))
        choice = "A" if r < p_a else "B"

        if _rand("malformed", user, sampling.seed, nonce) < MALFORMED_RATE:
            return "I would probably go with whichever suits the situation.", {
                "served_model": model.model_id, "simulated": True
            }

        payload: dict = {"choice": choice}
        if "strength" in user:
            payload["strength"] = round(min(1.0, abs(logit) / 3.0), 2)
        return json.dumps(payload), {"served_model": model.model_id, "simulated": True}

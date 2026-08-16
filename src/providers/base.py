"""Provider abstraction.

Every provider exposes the same call:

    generate(model_config, messages, sampling_config) -> GenerationResult

so the experiment code never depends on a particular vendor SDK.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    key: str
    provider: str
    model_id: str
    family: str
    supports_json_schema: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            key=d["key"],
            provider=d["provider"],
            model_id=d["model_id"],
            family=d.get("family", "unknown"),
            supports_json_schema=bool(d.get("supports_json_schema", False)),
            notes=d.get("notes", ""),
        )


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 300
    seed: int | None = None


@dataclass
class GenerationResult:
    text: str
    ok: bool
    latency_s: float
    attempts: int
    error: str | None = None
    # Provider-reported metadata, preserved verbatim for the record.
    meta: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised for a failed call. `retryable` drives the runner's backoff logic."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class Provider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def _generate_once(
        self, model: ModelConfig, messages: list[dict], sampling: SamplingConfig
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def generate(
        self,
        model: ModelConfig,
        messages: list[dict],
        sampling: SamplingConfig,
        max_retries: int = 4,
        base_delay_s: float = 2.0,
    ) -> GenerationResult:
        """Call the model, retrying retryable failures with exponential backoff + jitter."""
        started = time.time()
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                text, meta = self._generate_once(model, messages, sampling)
                return GenerationResult(
                    text=text,
                    ok=True,
                    latency_s=time.time() - started,
                    attempts=attempt,
                    meta=meta,
                )
            except ProviderError as e:
                last_err = e
                if not e.retryable or attempt == max_retries:
                    break
                time.sleep(base_delay_s * (2 ** (attempt - 1)) * (0.5 + random.random()))
            except Exception as e:  # noqa: BLE001 - record anything unexpected
                last_err = e
                if attempt == max_retries:
                    break
                time.sleep(base_delay_s * (2 ** (attempt - 1)) * (0.5 + random.random()))
        return GenerationResult(
            text="",
            ok=False,
            latency_s=time.time() - started,
            attempts=max_retries,
            error=f"{type(last_err).__name__}: {last_err}",
        )


def env_key(*names: str) -> str | None:
    """Return the first non-empty env var among `names`. Never logs the value."""
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return None

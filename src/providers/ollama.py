"""Local Ollama provider (used when no hosted API access is configured)."""

from __future__ import annotations

import os

import requests

from .base import ModelConfig, Provider, ProviderError, SamplingConfig

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, timeout_s: float = 180.0):
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def available(self) -> bool:
        try:
            return self._session.get(f"{HOST}/api/tags", timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def _generate_once(
        self, model: ModelConfig, messages: list[dict], sampling: SamplingConfig
    ) -> tuple[str, dict]:
        options: dict = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "num_predict": sampling.max_tokens,
        }
        if sampling.seed is not None:
            options["seed"] = sampling.seed

        payload = {
            "model": model.model_id,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if model.supports_json_schema:
            payload["format"] = "json"

        try:
            r = self._session.post(
                f"{HOST}/api/chat", json=payload, timeout=self.timeout_s
            )
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}", retryable=True) from e

        if r.status_code != 200:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:400]}", retryable=True)

        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        meta = {
            "served_model": data.get("model"),
            "usage": {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
            "finish_reason": data.get("done_reason"),
        }
        return text, meta

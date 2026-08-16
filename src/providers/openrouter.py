"""OpenRouter provider (OpenAI-compatible chat completions).

The model ID is always taken from configs/models.yaml and passed through
unchanged, and `provider.allow_fallbacks` is disabled, so OpenRouter cannot
silently serve a different model than the one we record.
"""

from __future__ import annotations

import requests

from .base import ModelConfig, Provider, ProviderError, SamplingConfig, env_key

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, timeout_s: float = 90.0):
        self.timeout_s = timeout_s
        self._key = env_key("OPENROUTER_API_KEY")
        self._session = requests.Session()

    def available(self) -> bool:
        return self._key is not None

    def _generate_once(
        self, model: ModelConfig, messages: list[dict], sampling: SamplingConfig
    ) -> tuple[str, dict]:
        if not self._key:
            raise ProviderError("OPENROUTER_API_KEY is not set", retryable=False)

        # `allow_fallbacks: False` pins the served MODEL, but OpenRouter still
        # load-balances across many upstream inference providers for that model
        # (gpt-oss-20b had 12 endpoints, gpt-oss-120b 20). To hold the serving
        # stack fixed, the upstream provider must be named explicitly via
        # `provider.only` (D-33).
        provider_routing: dict = {"allow_fallbacks": False}
        if model.upstream_provider:
            provider_routing["only"] = [model.upstream_provider]

        payload: dict = {
            "model": model.model_id,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "provider": provider_routing,
        }
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        if model.supports_json_schema:
            payload["response_format"] = {"type": "json_object"}

        try:
            r = self._session.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "X-Title": "llm-preference-convergence",
                },
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}", retryable=True) from e

        if r.status_code != 200:
            body = r.text[:400]
            raise ProviderError(
                f"HTTP {r.status_code}: {body}",
                retryable=r.status_code in RETRYABLE_STATUS,
            )

        try:
            data = r.json()
        except ValueError as e:
            raise ProviderError(f"non-JSON body: {r.text[:300]}", retryable=True) from e

        if "error" in data and not data.get("choices"):
            raise ProviderError(f"api error: {str(data['error'])[:300]}", retryable=True)

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"no choices in response: {str(data)[:300]}", retryable=True)

        text = (choices[0].get("message") or {}).get("content") or ""

        meta = {
            # `served_model` lets us verify after the fact that the pinned ID was honoured.
            "served_model": data.get("model"),
            "response_id": data.get("id"),
            "usage": data.get("usage"),
            "finish_reason": choices[0].get("finish_reason"),
            "provider_name": data.get("provider"),
        }
        return text, meta

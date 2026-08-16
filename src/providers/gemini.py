"""Google Gemini provider (generativelanguage REST API)."""

from __future__ import annotations

import requests

from .base import ModelConfig, Provider, ProviderError, SamplingConfig, env_key

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, timeout_s: float = 90.0):
        self.timeout_s = timeout_s
        self._key = env_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
        self._session = requests.Session()

    def available(self) -> bool:
        return self._key is not None

    def _generate_once(
        self, model: ModelConfig, messages: list[dict], sampling: SamplingConfig
    ) -> tuple[str, dict]:
        if not self._key:
            raise ProviderError("GOOGLE_API_KEY/GEMINI_API_KEY is not set", retryable=False)

        # Gemini takes the system turn separately from the conversation contents.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": sampling.temperature,
                "topP": sampling.top_p,
                "maxOutputTokens": sampling.max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        if model.supports_json_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model.model_id}:generateContent"
        )
        try:
            r = self._session.post(
                url,
                headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}", retryable=True) from e

        if r.status_code != 200:
            raise ProviderError(
                f"HTTP {r.status_code}: {r.text[:400]}",
                retryable=r.status_code in RETRYABLE_STATUS,
            )

        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise ProviderError(f"no candidates: {str(data)[:300]}", retryable=True)
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        meta = {
            "served_model": data.get("modelVersion", model.model_id),
            "usage": data.get("usageMetadata"),
            "finish_reason": cands[0].get("finishReason"),
        }
        return text, meta

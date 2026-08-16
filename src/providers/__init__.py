"""Provider registry."""

from __future__ import annotations

from .base import (
    GenerationResult,
    ModelConfig,
    Provider,
    ProviderError,
    SamplingConfig,
    env_key,
)
from .gemini import GeminiProvider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider

_REGISTRY: dict[str, type[Provider]] = {
    "openrouter": OpenRouterProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}

_CACHE: dict[str, Provider] = {}


def get_provider(name: str, timeout_s: float | None = None) -> Provider:
    """Return a cached provider instance (sessions are reused across calls)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(_REGISTRY)}")
    if name not in _CACHE:
        cls = _REGISTRY[name]
        _CACHE[name] = cls(timeout_s=timeout_s) if timeout_s else cls()
    return _CACHE[name]


def availability_report() -> dict[str, bool]:
    """Which providers are usable. Reports presence only — never a secret value."""
    return {name: cls().available() for name, cls in _REGISTRY.items()}


__all__ = [
    "GenerationResult",
    "ModelConfig",
    "Provider",
    "ProviderError",
    "SamplingConfig",
    "availability_report",
    "env_key",
    "get_provider",
]

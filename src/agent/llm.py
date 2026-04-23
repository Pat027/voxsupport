"""
LLM routing layer — LiteLLM with circuit-breaker failover across providers.

Default provider order: Anthropic → OpenAI → local Llama (via vLLM at
LLAMA_BASE_URL). First-success wins; on error, rolls over to next.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic

import litellm

logger = logging.getLogger(__name__)


@dataclass
class ProviderSpec:
    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 15.0


def default_providers() -> list[ProviderSpec]:
    providers: list[ProviderSpec] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(
            ProviderSpec(
                name="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key_env="ANTHROPIC_API_KEY",
            )
        )
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(
            ProviderSpec(
                name="openai",
                model="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
            )
        )
    if os.environ.get("LLAMA_BASE_URL"):
        providers.append(
            ProviderSpec(
                name="llama-vllm",
                model="openai/meta-llama/Llama-3.1-70B-Instruct",
                api_base=os.environ["LLAMA_BASE_URL"],
            )
        )
    if not providers:
        logger.warning(
            "No LLM providers configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or LLAMA_BASE_URL."
        )
    return providers


class _CircuitBreaker:
    """Tiny per-provider circuit breaker. Opens after N consecutive failures,
    halfway re-opens after cooldown. Avoids hammering a dead provider."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        until = self._open_until.get(name, 0.0)
        return monotonic() < until

    def record_failure(self, name: str) -> None:
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= self.threshold:
            self._open_until[name] = monotonic() + self.cooldown_s
            self._failures[name] = 0
            logger.warning("Circuit breaker opened for provider %s", name)

    def record_success(self, name: str) -> None:
        self._failures.pop(name, None)


class LLMRouter:
    """Multi-provider LLM router. Uses LiteLLM for a uniform interface."""

    def __init__(self, providers: Sequence[ProviderSpec] | None = None) -> None:
        self.providers = list(providers) if providers is not None else default_providers()
        self._breaker = _CircuitBreaker()

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Return completion text. Tries providers in order until one works."""
        last_error: Exception | None = None
        for p in self.providers:
            if self._breaker.is_open(p.name):
                continue
            try:
                start = monotonic()
                resp = await asyncio.wait_for(
                    litellm.acompletion(
                        model=p.model,
                        messages=messages,
                        api_base=p.api_base,
                        api_key=os.environ.get(p.api_key_env) if p.api_key_env else None,
                        **kwargs,
                    ),
                    timeout=p.timeout_s,
                )
                latency = monotonic() - start
                self._breaker.record_success(p.name)
                text = resp.choices[0].message.content or ""
                logger.debug(
                    "%s returned in %.0fms, %d chars", p.name, latency * 1000, len(text)
                )
                return text
            except Exception as exc:  # noqa: BLE001 — we re-raise last if all fail
                logger.warning("Provider %s failed: %s", p.name, exc)
                self._breaker.record_failure(p.name)
                last_error = exc
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error!r}"
        )

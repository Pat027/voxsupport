"""
Per-phase cost accounting.

At scale, you can't afford to not know what a voice call costs. This module
tracks usage per phase (STT, LLM, TTS, RAG) and converts to EUR using a
provider price table. It publishes the totals to Prometheus so Grafana
dashboards can show cost-per-call, cost-by-plan, cost-by-hour, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.observability.metrics import COST_EUR, TOKENS_TOTAL

Phase = Literal["stt", "llm", "tts", "rag"]


# Prices in EUR per 1M tokens (for LLM) or per minute of audio (for STT/TTS).
# Update these when provider pricing changes; keep a changelog in docs.
PRICING: dict[tuple[str, str], float] = {
    # LLM — per 1M tokens, input + output separately
    ("openai:gpt-4o-mini", "input"):  0.14,
    ("openai:gpt-4o-mini", "output"): 0.56,
    ("anthropic:claude-haiku-4-5-20251001", "input"):  0.25,
    ("anthropic:claude-haiku-4-5-20251001", "output"): 1.25,
    # Local Llama via vLLM — compute-only, rough EUR/1M at GPU-hour rates
    ("local:llama-3.1-70b", "input"):  0.02,
    ("local:llama-3.1-70b", "output"): 0.02,

    # STT — Kyutai local, EUR per minute of audio (GPU time)
    ("kyutai:stt-1b", "per_minute"): 0.004,
    # TTS — Kyutai local, EUR per minute of audio generated
    ("kyutai:tts-1.6b", "per_minute"): 0.006,
}


@dataclass
class CostLedger:
    stt_eur: float = 0.0
    llm_eur: float = 0.0
    tts_eur: float = 0.0
    rag_eur: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total_eur(self) -> float:
        return self.stt_eur + self.llm_eur + self.tts_eur + self.rag_eur

    def add_llm_call(
        self,
        *,
        provider_model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        in_price = PRICING.get((provider_model, "input"), 0.0)
        out_price = PRICING.get((provider_model, "output"), 0.0)
        cost = (tokens_in / 1_000_000) * in_price + (tokens_out / 1_000_000) * out_price
        self.llm_eur += cost
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        COST_EUR.labels(phase="llm").inc(cost)
        provider = provider_model.split(":", 1)[0]
        TOKENS_TOTAL.labels(direction="input", provider=provider).inc(tokens_in)
        TOKENS_TOTAL.labels(direction="output", provider=provider).inc(tokens_out)

    def add_audio_minute(
        self,
        *,
        phase: Phase,
        provider_model: str,
        minutes: float,
    ) -> None:
        rate = PRICING.get((provider_model, "per_minute"), 0.0)
        cost = minutes * rate
        if phase == "stt":
            self.stt_eur += cost
        elif phase == "tts":
            self.tts_eur += cost
        COST_EUR.labels(phase=phase).inc(cost)

    def add_rag_cost(self, *, embedding_calls: int, eur_per_call: float = 0.00002) -> None:
        cost = embedding_calls * eur_per_call
        self.rag_eur += cost
        COST_EUR.labels(phase="rag").inc(cost)

    def snapshot(self) -> dict[str, float]:
        return {
            "stt_eur": round(self.stt_eur, 5),
            "llm_eur": round(self.llm_eur, 5),
            "tts_eur": round(self.tts_eur, 5),
            "rag_eur": round(self.rag_eur, 5),
            "total_eur": round(self.total_eur, 5),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }

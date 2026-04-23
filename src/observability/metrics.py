"""
Prometheus metrics — exposed on /metrics by the FastAPI server.

Metric set matches the benchmark targets in README.md so the Grafana
dashboard can chart them directly against SLO thresholds.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---- Latency ---------------------------------------------------------------

STT_LATENCY = Histogram(
    "voxsupport_stt_latency_seconds",
    "Time from end-of-audio-chunk to final transcript.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0),
)

LLM_LATENCY = Histogram(
    "voxsupport_llm_latency_seconds",
    "Time from LLM request to first token (TTFB).",
    labelnames=("provider",),
    buckets=(0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0),
)

TTS_TTFS = Histogram(
    "voxsupport_tts_ttfs_seconds",
    "Time from TTS request to first audio byte (Time-to-First-Speech).",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0),
)

END_TO_END_LATENCY = Histogram(
    "voxsupport_end_to_end_latency_seconds",
    "End of caller utterance to start of agent reply.",
    buckets=(0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
)

BARGE_IN_LATENCY = Histogram(
    "voxsupport_barge_in_latency_seconds",
    "Time from caller interruption to TTS stopping.",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5),
)

# ---- Cost ------------------------------------------------------------------

COST_EUR = Counter(
    "voxsupport_cost_eur_total",
    "Cumulative cost in EUR by pipeline phase.",
    labelnames=("phase",),  # stt | llm | tts | rag
)

TOKENS_TOTAL = Counter(
    "voxsupport_tokens_total",
    "Token usage by direction.",
    labelnames=("direction", "provider"),  # direction: input | output
)

# ---- Behavior --------------------------------------------------------------

CALLS_TOTAL = Counter(
    "voxsupport_calls_total",
    "Total calls by outcome.",
    labelnames=("outcome",),  # resolved | escalated | abandoned | error
)

ESCALATIONS_TOTAL = Counter(
    "voxsupport_escalations_total",
    "Escalations by trigger reason.",
    labelnames=("reason",),  # user_request | out_of_scope | auth_failed | low_confidence
)

GUARDRAIL_FLAGS = Counter(
    "voxsupport_guardrail_flags_total",
    "Guardrail flags by type.",
    labelnames=("type", "scanner"),
    # type: input | output, scanner: PromptInjection | Toxicity | Sensitive | ...
)

PII_FINDINGS = Counter(
    "voxsupport_pii_findings_total",
    "PII entities redacted pre-storage, by entity type.",
    labelnames=("entity_type",),
)

PII_LEAK_RATE = Gauge(
    "voxsupport_pii_leak_rate",
    "Fraction of persisted transcripts that contain unredacted PII. "
    "Target: 0.0 always.",
)

ACTIVE_SESSIONS = Gauge(
    "voxsupport_active_sessions",
    "Number of currently active voice sessions.",
)

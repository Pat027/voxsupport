# voxsupport — Architecture

## High-level diagram

```
┌──────────────┐    ┌────────────────────────────────────────┐    ┌──────────────┐
│   Caller     │    │         voxsupport pipeline            │    │   Services   │
│              │    │                                        │    │              │
│  Browser ────┼───►│  Pipecat transport                     │    │              │
│  (WebRTC)    │    │         │                              │    │              │
│              │    │         ▼                              │    │              │
│  Phone ──────┼───►│  Kyutai STT ──► Presidio PII redact ──►│    │              │
│  (Twilio)    │    │                        │               │    │              │
│              │    │                        ▼               │    │              │
│              │    │                 LangGraph agent        ├───►│  pgvector    │
│              │    │                   ├─ auth              │    │  (RAG)       │
│              │    │                   ├─ intent detect     │    │              │
│              │    │                   ├─ tool call  ──────►│───►│  MCP tools   │
│              │    │                   └─ escalation        │    │  (accounts,  │
│              │    │                        │               │    │   billing,   │
│              │    │                        ▼               │    │   KB, ...)   │
│              │    │                   LiteLLM routing      │    │              │
│              │    │                   (Llama/OpenAI/       │    │              │
│              │    │                    Anthropic)          │    │              │
│              │    │                        │               │    │              │
│              │    │                        ▼               │    │              │
│              │    │                   LLM Guard            │    │              │
│              │    │                   (output safety)      │    │              │
│              │    │                        │               │    │              │
│              │    │                        ▼               │    │              │
│              │    │                   Kyutai TTS           │    │              │
│              │    │                        │               │    │              │
│  Caller ◄────┼────┤  Pipecat transport                     │    │              │
│              │    │                                        │    │              │
└──────────────┘    └────────┬───────────────────────────────┘    └──────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Observability              │
               │                             │
               │  Langfuse (traces)          │
               │  Prometheus (metrics)       │
               │  Grafana (dashboards)       │
               │  Redis (conversation state) │
               │  Postgres (user prefs)      │
               └─────────────────────────────┘
```

## Data flow — happy path

1. **Caller speaks** into browser or phone.
2. **Pipecat transport** captures the audio stream and hands it to Kyutai STT.
3. **Kyutai STT** streams partial + final transcripts with semantic VAD predicting end-of-turn.
4. **Presidio** scans the transcript. Any PII (email, phone, SSN, credit card) is replaced with tokens (`<EMAIL>`, `<PHONE>`, etc.) **before** the transcript is logged or sent to the LLM.
5. **LangGraph agent** takes the redacted transcript and runs the state machine:
   - **auth** node — confirm caller identity (voice print not included; email/phone lookup only)
   - **intent** node — classify (billing, technical, plan change, cancel, out-of-scope)
   - **action** node — invoke MCP tools to fetch account/billing data or retrieve knowledge via RAG
   - **escalation** node — if confidence low or scope exceeded, transfer to human
6. **LiteLLM** routes the LLM call across Llama (local vLLM), OpenAI, or Anthropic, with circuit-breaker failover.
7. **LLM Guard** checks the reply before it's spoken — flags prompt-injection attempts and unsafe outputs.
8. **Kyutai TTS** streams audio back to the transport, which plays it to the caller. Kyutai TTS starts speaking before the full reply text is generated, keeping TTFS below 800ms.
9. Every step emits a **Langfuse trace event** with latency + token counts. **Prometheus** aggregates metrics. **Grafana** visualizes.

## The five production dimensions

| Dimension | Where |
|-----------|-------|
| Guardrails | `src/guardrails/` — Presidio, LLM Guard, LangGraph policy |
| Observability | `src/observability/` — Langfuse, Prometheus, cost accounting |
| Memory | `src/memory/` — Redis (conversation), Postgres (preferences) |
| Cost management | `src/observability/cost.py` — per-phase budget caps + degradation |
| Error recovery | `src/voice/pipeline.py` — LiteLLM failover, circuit breakers, escalation |

## Roadmap — all shipped

- [x] **v0.1**: scaffolding. Directory structure, pyproject, docker-compose, placeholder pipeline.
- [x] **v0.2**: Kyutai STT + TTS via `moshi`, WebSocket browser transport, TTFS instrumentation.
- [x] **v0.3**: LangGraph agent + 5 MCP tools (`get_account`, `get_bill`, `change_plan`, `lookup_kb`, `escalate_to_human`) + pgvector RAG over Acme Cloud docs.
- [x] **v0.4**: Presidio PII redaction + LLM Guard safety + LangGraph policy nodes + Redis conversation memory + Postgres preferences.
- [x] **v0.5**: Langfuse tracing + Prometheus metrics + Grafana dashboard + per-phase cost accounting + LLM-as-a-judge evaluation harness + 15 benchmark scenarios.
- [x] **v0.9**: Twilio voice transport + Modal one-file deploy.
- [x] **v1.0**: production-patterns doc + blog post + polished README.

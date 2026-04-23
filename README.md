# voxsupport

A production-grade, open-source voice customer-support agent. Speak to it in a browser; call it on a phone; measure it against production benchmarks. Built entirely on open-source components.

**Stack**: Pipecat (orchestration) &middot; Kyutai STT + TTS (streaming) &middot; LangGraph (state machine) &middot; LiteLLM + Llama / OpenAI / Anthropic &middot; pgvector (RAG) &middot; Microsoft Presidio + LLM Guard (guardrails) &middot; Redis + Postgres (memory) &middot; Langfuse + Prometheus + Grafana (observability) &middot; Twilio (telephony) &middot; Modal (deploy).

See [docs/production-patterns.md](docs/production-patterns.md) for the design rationale and [docs/blog_post.md](docs/blog_post.md) for the what-I-learned writeup.

## Why this exists

Most voice-agent projects on GitHub are demos — they work once on the author's laptop. voxsupport is built to the **five production dimensions** that separate real agents from demos:

| Dimension | Implementation |
|-----------|----------------|
| **Guardrails** | Presidio PII redaction before LLM + storage &middot; LLM Guard prompt-injection + output safety &middot; LangGraph policy nodes (no destructive ops, cross-account ban, no secret disclosure) |
| **Observability** | Langfuse end-to-end tracing &middot; Prometheus metrics &middot; Grafana dashboard with SLO thresholds |
| **Memory** | Redis for conversation state (TTL-bounded) &middot; Postgres for long-term user preferences |
| **Cost management** | Per-phase token + cost accounting (STT / LLM / TTS / RAG) with budget caps triggering graceful degradation |
| **Error recovery** | LiteLLM multi-provider failover with circuit breakers &middot; RAG fallback to human escalation &middot; session recovery via Redis state |

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/Pat027/voxsupport.git
cd voxsupport

# 2. Set env vars (copy and fill)
cp .env.example .env

# 3. Spin up the full stack
docker compose up -d

# 4. Ingest the synthetic knowledge base (one-time)
docker compose exec voxsupport python -c "
import asyncio, os
from src.agent.rag import KnowledgeBase
asyncio.run(KnowledgeBase(
  os.environ['DATABASE_URL'].replace('+asyncpg','')
).ingest_directory('data/acme_docs'))
"

# 5. Open the browser demo
open http://localhost:8080
```

### Observability URLs (after `docker compose up`)

- **Langfuse UI**: http://localhost:3000 (first launch: create an org + project, copy the public/secret keys into `.env`)
- **Grafana**: http://localhost:3001 (admin / admin, import `dashboards/grafana-dashboard.json`)
- **Prometheus**: http://localhost:9090
- **Metrics endpoint**: http://localhost:8080/metrics

## Call it by phone (v0.9)

```bash
# Deploy to Modal
pip install modal && modal setup
modal deploy modal_deploy.py

# Point your Twilio number at:
# https://{your-deployment}.modal.run/twilio/voice
```

## Benchmarks

All targets measured; results published in `benchmarks/results/` after each run.

| Metric | Target | Production meaning |
|--------|--------|--------------------|
| TTFS (Time to First Speech) | <800ms p50 | caller perceives it as snappy |
| End-of-turn detection | <500ms p50 | natural turn-taking, not silence-waiting |
| Barge-in responsiveness | <200ms p50 | interruption feels instant |
| Full-utterance E2E | <2s p95 | conversation has normal pacing |
| Resolution correctness (LLM-judge) | >90% | tier-1 auto-resolution rate |
| Escalation appropriateness | >95% | escalates when it should, not when it shouldn't |
| **PII leak rate** | **0%** | zero PII in any LLM prompt or persisted transcript |
| Cost per resolved call | <&euro;0.05 | production unit economics |

Run them yourself:
```bash
python benchmarks/run_benchmarks.py --all --runs 3
```

## Architecture

```
Caller (browser WebRTC or Twilio phone)
        |
        v
   Pipecat pipeline
        |
        |--> Kyutai STT (streaming, semantic VAD)
        |        |
        |        v
        |    Presidio PII redaction
        |        |
        |        v
        |    LangGraph state machine
        |        (auth -> intent -> action -> escalation)
        |        |
        |        +--> MCP tools (get_account, get_bill, change_plan, lookup_kb, escalate_to_human)
        |        |
        |        +--> pgvector RAG over Acme Cloud docs
        |        |
        |        +--> LiteLLM routing (Llama / OpenAI / Anthropic) with circuit-breaker failover
        |                |
        v                v
    LLM Guard (output safety) <-
        |
        v
    Kyutai TTS (streaming, starts before full text)
        |
        v
    Caller
        |
        +--> Langfuse traces + Prometheus metrics + Grafana dashboards
```

Full data-flow diagram in [docs/architecture.md](docs/architecture.md).

## Project status

All six milestones shipped.

- [x] `v0.1` - project scaffolding (directory, Dockerfile, docker-compose, pyproject)
- [x] `v0.2` - cascade voice pipeline (Pipecat + Kyutai STT + Kyutai TTS)
- [x] `v0.3` - agent layer (LangGraph + MCP tools + pgvector RAG)
- [x] `v0.4` - guardrails (Presidio + LLM Guard + policy) + memory (Redis + Postgres)
- [x] `v0.5` - observability (Langfuse + Prometheus + Grafana) + cost accounting + LLM-as-a-judge eval on 15 scenarios
- [x] `v0.9` - Twilio phone integration + Modal deploy script
- [x] `v1.0` - production-patterns doc + blog post + polished README

## What you need externally to fully demo

The code is complete; a handful of external steps remain to produce a live callable demo:

1. **API keys**: at least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` in `.env`. Both work; the router tries them in priority order.
2. **Kyutai weights**: `pip install moshi` triggers Hugging Face download on first call. You need disk (~4GB) and ideally a GPU (A10G+ recommended; CPU works but slow).
3. **Langfuse keys**: launch the self-hosted Langfuse at http://localhost:3000, create a project, paste public/secret into `.env`.
4. **Modal deploy**: `modal setup`, then `modal deploy modal_deploy.py`. Produces the public HTTPS URL for the web demo and the Twilio webhook.
5. **Twilio number** (optional, $1/mo): buy a number in the Twilio console, set its Voice webhook to `https://{deploy-url}/twilio/voice`.

## Repo layout

```
voxsupport/
|-- README.md
|-- LICENSE                          (Apache 2.0)
|-- pyproject.toml
|-- Dockerfile
|-- docker-compose.yml               (full local stack)
|-- modal_deploy.py                  (one-file production deploy)
|-- .env.example
|
|-- src/
|   |-- voice/                       (Pipecat pipeline, STT, TTS, transports)
|   |-- agent/                       (LangGraph state machine, MCP tools, RAG, LLM router)
|   |-- guardrails/                  (PII, safety, policy)
|   |-- memory/                      (Redis conversation, Postgres preferences)
|   |-- observability/               (Langfuse, Prometheus, cost accounting)
|   |-- eval/                        (latency measurement, LLM-as-a-judge)
|
|-- data/
|   |-- init.sql                     (schema + seed accounts + bills)
|   |-- acme_docs/                   (synthetic knowledge base)
|
|-- benchmarks/
|   |-- run_benchmarks.py
|   |-- scenarios/                   (15 YAML scenarios covering happy path + edge cases)
|   |-- results/                     (generated CSVs + summaries)
|
|-- dashboards/
|   |-- grafana-dashboard.json       (import directly)
|   |-- prometheus.yml
|
|-- demo/
|   |-- web_client/index.html        (browser demo with WebSocket mic)
|
`-- docs/
    |-- architecture.md
    |-- production-patterns.md       (the five-dimensions writeup)
    `-- blog_post.md                 (what I learned building this solo)
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author

Built solo by [Pratik Narendra Raut](https://github.com/Pat027) as part of a production-AI portfolio alongside [useful-agents.com](https://useful-agents.com), [StratosGuard](https://github.com/Pat027/stratosguard), and [Revelio](https://github.com/Pat027/revelio).

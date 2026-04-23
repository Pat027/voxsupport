# Production patterns for voice agents

This document is the "what I wish someone had written down for me" version of the five production dimensions that separate real voice agents from demos. It's the reference used when building voxsupport.

## Dimension 1 — Guardrails

A voice agent is a mutation engine. It reads customer data, sometimes changes it, and sometimes moves money. Guardrails are the rules that prevent it from doing something it shouldn't.

**Three layers, in order of application:**

1. **Input layer (caller → agent).** PII redaction + prompt-injection defense. In voxsupport, Presidio runs on every STT transcript before it's logged or sent to the LLM. LLM Guard's `PromptInjection` scanner runs on the same input. If either fires, the agent responds from a safe template and records a Prometheus counter.

2. **Policy layer (agent → tools).** Business rules that block entire categories of action regardless of how the LLM was convinced to try them. voxsupport ships three: `no_secret_disclosure`, `no_destructive_ops`, `cross_account_ban`. Each is a pure function over the current state — trivial to unit-test.

3. **Output layer (agent → caller).** LLM Guard's `Sensitive` + `NoRefusal` run on every LLM completion. If sensitive data leaks through the other layers (it shouldn't, but defense in depth), the output scanner redacts it before TTS. Refusals get caught and rerouted to human escalation with a friendlier voice.

**Why redact BEFORE the LLM, not after?** Because the LLM's provider logs your prompts. If you send a raw email + SSN to OpenAI, that string now lives in OpenAI's audit logs. Redact at the boundary; the agent doesn't need the raw values to reason about them.

## Dimension 2 — Observability

If you can't see what the agent is doing, you can't improve it. voxsupport publishes:

- **Langfuse traces** with one span per pipeline phase (STT → redaction → intent → RAG → LLM → guardrails → TTS). Every span has input, output, latency. Every session has a stable session_id you can hand a customer to look up the interaction.
- **Prometheus metrics** (`/metrics`) covering: TTFS, end-of-turn latency, barge-in latency, end-to-end latency, tokens by direction and provider, cost by phase, call outcomes, escalations by reason, guardrail flags, PII findings.
- **Grafana dashboard** (`dashboards/grafana-dashboard.json`) pre-configured to chart all of the above against SLO thresholds.

**What to watch for in practice:**

- Barge-in latency creeping above 200ms means the pipeline has accumulated buffering somewhere. Usually TTS is the culprit.
- Cost-per-phase skew tells you where to optimize. If LLM is 80% of spend, try the smaller-model path; if TTS is 50%, turn off voice cloning.
- Guardrail flag rate is a weather vane for attack patterns. A spike in `PromptInjection` flags after a blog post gets shared is a real thing.

## Dimension 3 — Memory architecture

Voice sessions are stateful: "like last time" is a real utterance. You need two tiers of memory.

- **Short-term (conversation).** Redis, keyed by `session_id`, TTL ~30 minutes. Every turn appends `{role, content, ts}`. The LLM sees the last N turns on each call. Redis is PII-clean because the conversation was redacted at the pipeline boundary.
- **Long-term (preferences).** Postgres, keyed by `account_id`. Voice speed, language, recent topics. Updated on meaningful turns only — not every word.

**What to intentionally NOT store:** audio. Raw audio contains unredactable PII (voice is biometric). Keep transcripts, drop audio at session end unless you have a specific compliance reason to retain it.

## Dimension 4 — Cost management

Voice is expensive. Per-minute audio to a hosted STT is €0.04+ on many providers; LLM tokens scale fast when every utterance triggers a full multi-step graph run. voxsupport's discipline:

- **Per-phase accounting.** `src/observability/cost.py` increments a counter for every phase, labeled by phase. Grafana sums into a `cost_per_call` panel.
- **Budget caps with graceful degradation.** Two thresholds: soft at 80% of target (nudge toward resolution), hard at 150% (auto-escalate, no more LLM calls). Pricing table is a dict; update it per-provider when rates change.
- **Choose smaller models when possible.** A `claude-haiku` or `gpt-4o-mini` for intent classification + a `gpt-4o` only for complex reasoning cuts spend without hurting quality.

**Anti-pattern:** sending the entire conversation history on every turn. Always summarize/trim. voxsupport's conversation memory caps at 20 turns; older turns get summarized into a single `system` message.

## Dimension 5 — Error recovery

Voice agents fail in public. When the caller is listening, silence is worse than an honest fallback.

Four recovery patterns, all in voxsupport:

1. **LLM provider failover.** `LLMRouter` tries providers in priority order with a per-provider circuit breaker. If Anthropic is down, it's on OpenAI within one retry.
2. **Graceful degradation.** If RAG fails (pgvector unreachable), the agent says "I can't check that right now — let me transfer you" instead of hallucinating.
3. **Human escalation as a first-class tool.** `escalate_to_human` is available at every node. The agent is trained to prefer escalation over guessing.
4. **Session recovery.** If the WebSocket drops, Redis still has the conversation. When the caller dials back in with the same session_id (Twilio passes it in the call parameters), context is restored.

## Anti-patterns to avoid

- **Sync TTS.** If you wait for the full LLM completion before starting TTS, your TTFS is latency(LLM) + latency(TTS). Always stream.
- **VAD-only turn detection.** Silence-based VAD misses "um" and fires on breaths. Use a semantic VAD (Kyutai STT predicts end-of-turn from content + intonation).
- **Treating voice like chat.** Responses must be spoken: short, plain, no markdown. The system prompt has to enforce this or the model will emit bullet lists that sound robotic.
- **Shipping without guardrails.** The first production bug is a PII leak or a prompt injection. Ship with guardrails on day one, not day ninety.
- **No cost dashboard.** You will find out you're spending €0.50/call a week too late. Per-phase cost counters go in on day one.

## What you don't need on day one

- **Voice cloning / emotional TTS.** Nice, not essential. Kyutai TTS 1.6B defaults are already warm.
- **Fine-tuned models.** Pre-trained LLMs get you surprisingly far. Fine-tune once you have data from production usage.
- **Multi-tenant RBAC.** Single-tenant first. Add auth + isolation when you have a second customer.
- **Real-time face/video analysis.** Audio is 95% of the customer-support use case.

## Sources

Built on these open-source projects — thanks to their maintainers:

- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — orchestration
- [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — streaming STT + TTS
- [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) — end-to-end S2S reference
- [microsoft/presidio](https://github.com/microsoft/presidio) — PII detection + anonymization
- [protectai/llm-guard](https://github.com/protectai/llm-guard) — prompt-injection + output safety
- [langfuse/langfuse](https://github.com/langfuse/langfuse) — LLM observability
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — stateful agent graphs

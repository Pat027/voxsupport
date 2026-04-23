# What I learned building a production voice agent solo

Or: "It's not the audio, it's the five production dimensions."

---

Most voice-agent demos on GitHub share a problem: they work once, on the author's laptop, in a quiet room. Close the laptop, pay the voice bill, and the demo doesn't teach you whether this thing could survive a hundred concurrent calls at a regulated customer.

I built `voxsupport` to answer the harder question. It's a voice customer-support agent for a synthetic SaaS company ("Acme Cloud") — open-source, phone-callable, with every production dimension explicitly wired up. The stack is entirely open-source: Pipecat for orchestration, Kyutai for STT and TTS, LiteLLM in front of OpenAI / Anthropic / a local Llama, LangGraph for the state machine, Presidio and LLM Guard for guardrails, Langfuse and Prometheus for observability, Twilio for the phone number.

Here's what I actually learned.

## 1. Most voice-agent bugs aren't voice bugs

I expected to spend most of my time wrestling with audio. I didn't. TTFS (time to first speech) was surprisingly easy to hit under 800ms once I picked the right streaming stack. What ate my time was:

- **Making the LLM speak naturally.** Default completions come out as structured text with bullets. Getting one-to-two-sentence spoken answers takes prompt discipline, not a better model.
- **Interruption handling.** Pipecat supports barge-in natively, but the default VAD over-triggers on breaths. Swapping to Kyutai STT's semantic VAD (which predicts end-of-turn from content, not just silence) was the single biggest UX win.
- **Guardrail-guardrail interactions.** LLM Guard's `Sensitive` output scanner would redact entities the PII redactor already handled, producing empty responses. Ordering matters.

The audio layer is a (mostly) solved problem. The interesting engineering lives above it.

## 2. Guardrails are where the system gets real

The moment the agent can change a customer's plan, authorize a refund, or mention another customer's email, it's production-critical. I ship three guardrail layers in voxsupport:

- **Input**: Presidio redacts PII (email, SSN, credit card) BEFORE the transcript hits the LLM or Langfuse logs. LLM Guard's PromptInjection catches the "ignore previous instructions" class.
- **Policy**: a LangGraph node runs pure-function policies — no destructive ops, no cross-account discussion, no secret disclosure. These execute in microseconds and short-circuit the graph.
- **Output**: LLM Guard's Sensitive scanner runs on the completion before TTS. If anything slipped through, it gets caught here.

The PII layer especially is worth over-engineering on day one. If you send raw customer data to OpenAI and it lives in their 30-day audit retention, you've created a compliance problem you can't easily close.

## 3. Cost accounting is a feature, not an afterthought

The second time I ran a benchmark with Anthropic's Claude Sonnet, I noticed €0.14 per call. For customer support at volume, that doesn't add up. I wired per-phase cost counters (STT / LLM / TTS / RAG) into Prometheus on the same day, set budget caps with graceful degradation at 80% and 150% of target, and swapped to `claude-haiku-4-5` for intent classification. Cost dropped to €0.02 / call without a quality regression.

You can't optimize what you don't measure. Build the cost dashboard first, not last.

## 4. The five production dimensions

After getting voxsupport shippable, I wrote up [docs/production-patterns.md](./production-patterns.md). The short version is five dimensions that separate real voice agents from demos:

1. **Guardrails** — PII, policy, output safety
2. **Observability** — per-phase traces, Prometheus, Grafana
3. **Memory** — Redis conversations, Postgres preferences
4. **Cost management** — per-phase accounting, budget caps
5. **Error recovery** — provider failover, graceful degradation, human handoff

Every one has a concrete implementation in voxsupport. Every one is a thing I'd skip on day one and regret on day ninety.

## 5. Open-source is ready for this

The open-source voice-agent stack in 2026 is legitimately good. Pipecat is the right abstraction. Kyutai's streaming STT and TTS are production-quality. Moshi's full-duplex S2S is research-y but usable. Presidio is the PII layer I wanted for years. Langfuse gives you the traces that OpenAI charges thousands for.

The one thing you still buy is telephony — Twilio remains the easiest path to a phone number, and it's inexpensive (~$1/month + per-minute).

## 6. What I'd build next

voxsupport ships with a cascade architecture (STT → LLM → TTS) because that's what fits enterprise customer support: auditable, swappable models, business logic between transcription and response. For consumer-grade conversational UX, end-to-end S2S (Moshi) hits latency the cascade can't — 200ms vs 640ms is the difference between "clearly a bot" and "oh, it's someone talking."

The next iteration: a hybrid where cascade handles tool-calling paths (plan changes, billing lookups) and Moshi handles conversational smalltalk and listening cues ("mm-hm," "got it"). The caller won't know which is which, and both parts do what they're best at.

---

## Try it

- Repo: `github.com/Pat027/voxsupport`
- Run locally: `docker compose up`, open `localhost:8080`
- Read the production patterns: [docs/production-patterns.md](./production-patterns.md)

Feedback welcome.

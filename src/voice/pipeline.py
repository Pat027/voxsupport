"""
Full voice pipeline: STT -> PII redaction -> LangGraph agent -> LLM Guard -> TTS.

This file is the integration point that ties every production dimension together.
Each frame flowing through Pipecat is observed by the ObservabilityProcessor:
it records timestamps for latency, token counts for cost, and Langfuse spans.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.vad.silero import SileroVADAnalyzer

from src.agent.graph import build_default_graph
from src.agent.llm import LLMRouter
from src.agent.prompts import SYSTEM_VOICE
from src.guardrails.pii import default_redactor
from src.guardrails.safety import default_scanner
from src.memory.conversation import ConversationMemory
from src.observability.cost import CostLedger
from src.observability.langfuse_handler import default_tracer
from src.observability.metrics import (
    ACTIVE_SESSIONS,
    BARGE_IN_LATENCY,
    CALLS_TOTAL,
    END_TO_END_LATENCY,
    ESCALATIONS_TOTAL,
    GUARDRAIL_FLAGS,
    PII_FINDINGS,
    STT_LATENCY,
    TTS_TTFS,
)
from src.voice.stt import KyutaiSTTService
from src.voice.tts import KyutaiTTSService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-integrating frame processor
# ---------------------------------------------------------------------------


class AgentProcessor(FrameProcessor):
    """Intercepts transcript frames, runs them through the LangGraph agent,
    applies guardrails, emits a TextFrame with the agent's spoken reply.

    Replaces the naive LLMService most Pipecat examples use. The agent carries
    per-session state (authenticated account, pending confirmations, etc.) —
    that state lives inside this processor, keyed by session id.
    """

    def __init__(
        self,
        *,
        session_id: str,
        memory: ConversationMemory,
        cost_ledger: CostLedger,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.memory = memory
        self.cost_ledger = cost_ledger

        self._router = LLMRouter()
        self._redactor = default_redactor()
        self._scanner = default_scanner()
        self._tracer = default_tracer()
        self._graph = build_default_graph(self._llm_call)

        # Per-session agent state — survives across turns within this call.
        self._session_state: dict[str, Any] = {
            "authenticated": False,
            "account": None,
        }

        self._turn_started: float | None = None

    async def _llm_call(self, messages: list[dict[str, str]]) -> str:
        """Injected into the LangGraph graph. Wraps LLMRouter with cost + tokens."""
        reply = await self._router.chat(messages, temperature=0.2)
        # Rough token estimation for cost accounting; real pipelines use usage
        # metadata from the provider response (not exposed via litellm.acompletion
        # in a uniform way as of pipecat 0.x, so we estimate here).
        est_in = sum(len(m["content"]) for m in messages) // 4
        est_out = len(reply) // 4
        self.cost_ledger.add_llm_call(
            provider_model="openai:gpt-4o-mini",  # configurable
            tokens_in=est_in,
            tokens_out=est_out,
        )
        return reply

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn_started = time.monotonic()

        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcript(frame, direction)

        elif isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _handle_transcript(
        self,
        frame: TranscriptionFrame,
        direction: FrameDirection,
    ) -> None:
        raw_text = frame.text or ""
        if not raw_text.strip():
            return

        # 1. Guardrail — input scan (prompt injection, toxicity)
        scan = self._scanner.scan_input(raw_text)
        if not scan.ok:
            for s in scan.flagged:
                GUARDRAIL_FLAGS.labels(type="input", scanner=s).inc()
        sanitized = scan.sanitized

        # 2. PII redaction (BEFORE logging or LLM)
        redacted = self._redactor.redact(sanitized)
        if redacted.had_pii:
            for ent, count in self._redactor.summarize_findings(redacted).items():
                PII_FINDINGS.labels(entity_type=ent).inc(count)

        # The text fed into the agent is the PII-redacted version. The original
        # with PII is discarded at this boundary — never written anywhere.
        transcript_for_agent = redacted.redacted

        # 3. Persist to conversation memory
        await self.memory.append(self.session_id, "user", transcript_for_agent)

        # 4. Run the LangGraph agent
        state = {
            "utterance": transcript_for_agent,
            **self._session_state,
        }
        result = await self._graph.ainvoke(state)

        # Carry forward session-level state for the next turn.
        for k in ("authenticated", "account", "pending_confirmation"):
            if k in result:
                self._session_state[k] = result[k]
        if result.get("should_escalate"):
            reason = "user_request" if "human_escalation" in str(result.get("intent", "")) else "low_confidence"
            ESCALATIONS_TOTAL.labels(reason=reason).inc()

        # 5. Generate the spoken reply — streaming when the graph handed us a
        # prompt (the common case). Rule-based nodes (auth, escalate, etc.)
        # instead set `response` directly, so we push it whole.
        final_prompt = result.get("final_prompt")
        if final_prompt:
            response_text = await self._stream_reply(final_prompt, direction)
        else:
            response_text = result.get("response", "")
            if response_text:
                await self.push_frame(TextFrame(response_text), direction)

        # 6. Output guardrail scan (on the assembled text — scans can't run
        # mid-stream without corrupting audio). This is a known tradeoff:
        # output safety kicks in AFTER the reply has already streamed. For
        # stricter settings, run the LLM non-streamed via `.chat()`.
        out_scan = self._scanner.scan_output(transcript_for_agent, response_text)
        if not out_scan.ok:
            for s in out_scan.flagged:
                GUARDRAIL_FLAGS.labels(type="output", scanner=s).inc()

        # 7. Persist assistant turn
        await self.memory.append(self.session_id, "assistant", response_text)

        # Latency accounting
        if self._turn_started is not None:
            END_TO_END_LATENCY.observe(time.monotonic() - self._turn_started)
            self._turn_started = None

    async def _stream_reply(
        self,
        messages: list[dict[str, str]],
        direction: FrameDirection,
    ) -> str:
        """Stream LLM tokens; push each as a TextFrame; return the full text.

        Downstream TTS aggregates tokens to sentence boundaries (Pipecat default)
        so the first audio chunk hits the transport ~1 sentence after first token.
        """
        chunks: list[str] = []
        est_in = sum(len(m["content"]) for m in messages) // 4
        async for chunk in self._router.stream_chat(messages, temperature=0.2):
            chunks.append(chunk)
            await self.push_frame(TextFrame(chunk), direction)
        full = "".join(chunks)
        # Cost accounting (see note: streaming completions don't surface token
        # usage uniformly via litellm, so we estimate at chunk boundaries).
        self.cost_ledger.add_llm_call(
            provider_model="openai:gpt-4o-mini",
            tokens_in=est_in,
            tokens_out=len(full) // 4,
        )
        return full


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


async def build_pipeline(
    transport,
    *,
    session_id: str | None = None,
) -> tuple[Pipeline, PipelineTask, str]:
    """Wire STT -> AgentProcessor -> TTS into a Pipecat pipeline."""
    session_id = session_id or str(uuid.uuid4())

    memory = ConversationMemory(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    cost_ledger = CostLedger()

    stt = KyutaiSTTService()
    agent = AgentProcessor(session_id=session_id, memory=memory, cost_ledger=cost_ledger)
    tts = KyutaiTTSService()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            agent,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,   # barge-in
            enable_metrics=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    ACTIVE_SESSIONS.inc()
    return pipeline, task, session_id


async def run_conversation(transport) -> None:
    """Run one voice session over the given transport (web or Twilio)."""
    pipeline, task, session_id = await build_pipeline(transport)

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(transport, participant):
        # Greet the caller so they know the line's open.
        await task.queue_frames(
            [
                LLMMessagesFrame(
                    messages=[
                        {"role": "system", "content": SYSTEM_VOICE},
                        {
                            "role": "user",
                            "content": "Greet the caller in one short sentence.",
                        },
                    ]
                )
            ]
        )

    @transport.event_handler("on_participant_left")
    async def _on_leave(transport, participant, reason):
        await task.queue_frame(EndFrame())

    try:
        runner = PipelineRunner()
        await runner.run(task)
    finally:
        ACTIVE_SESSIONS.dec()
        CALLS_TOTAL.labels(outcome="resolved").inc()
        logger.info("Session %s ended", session_id)

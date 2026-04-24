"""Fast-path dispatcher — single function-calling LLM call per turn.

Replaces the classic LangGraph (classify → action → response, two sequential
LLM round-trips) with one LLM call that either:

- **Path A (direct answer)**: streams a spoken reply directly. 1 LLM call total.
- **Path B (tool)**: emits a tool_call. We execute the tool, feed the result
  back as a second LLM call, stream the reply. 2 LLM calls.

Gated by `VOXSUPPORT_FAST_PATH=1`. The classic graph remains the default so
cloud-only setups (no local vLLM) keep working unchanged.

Streaming contract: `run_turn()` yields text chunks as they arrive and
returns the final `FastResult` when done (for memory + guardrails + session
state carry-forward).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm

from src.agent.llm import LLMRouter
from src.agent.prompts import FAST_PATH_SYSTEM
from src.agent.rag import KnowledgeBase
from src.agent.tool_schemas import TOOL_NAMES, TOOL_SCHEMAS
from src.agent.tools import (
    change_plan,
    escalate_to_human,
    get_account,
    get_bill,
    lookup_kb,
)

logger = logging.getLogger(__name__)

_EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass
class FastResult:
    """Per-turn outcome. Populated once streaming completes."""

    text: str
    session_updates: dict[str, Any]
    should_escalate: bool = False
    ticket_id: str | None = None
    tool_name: str | None = None


class FastDispatcher:
    """One dispatcher per session; lives inside AgentProcessor."""

    def __init__(
        self,
        *,
        router: LLMRouter,
        dsn: str,
        kb: KnowledgeBase,
    ) -> None:
        self._router = router
        self._dsn = dsn
        self._kb = kb

    async def run_turn(
        self,
        utterance: str,
        session_state: dict[str, Any],
        emit_chunk: Callable[[str], Awaitable[None]],
    ) -> FastResult:
        """Drive a turn. `emit_chunk` is called for each streamed text chunk.
        Returns the assembled FastResult once the reply is fully streamed."""
        # 1. Build the per-turn conversation.
        account = session_state.get("account")
        authenticated = bool(session_state.get("authenticated"))
        pending = session_state.get("pending_confirmation")

        system_prompt = FAST_PATH_SYSTEM
        if authenticated and account:
            system_prompt += (
                f"\nAuthenticated caller: {account['full_name']}, "
                f"plan={account['plan']}, region={account['region']}."
            )
        if pending and pending.get("kind") == "plan_change":
            system_prompt += (
                f"\nPending plan change: from {account.get('plan','?')} to "
                f"{pending['target_plan']}. The caller just needs to say yes/no."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ]

        # 2. First streaming call. Watch deltas for tool_calls; if any appear,
        # switch to Path B (assemble tool call, execute, stream reply). If we
        # see only text deltas, Path A — text is already streaming to the caller.
        tool_name, tool_args, path_a_text = await self._first_call_stream(
            messages, emit_chunk
        )

        if tool_name is None:
            # Path A: done. Text was streamed out during the call.
            return FastResult(text=path_a_text, session_updates={})

        # Path B: tool path.
        logger.info("fast-path tool: %s args=%s", tool_name, tool_args)

        # 3. Execute tool.
        tool_result, session_updates, ticket_id = await self._execute_tool(
            tool_name, tool_args, session_state
        )
        should_escalate = tool_name == "escalate_to_human"

        # Special case: authenticate with no match → rule-based reply, no 2nd LLM.
        if tool_name == "authenticate" and not tool_result.get("found"):
            reply = (
                "I don't see an account for that email. "
                "Want me to transfer you to a human agent?"
            )
            for ch in _chunkify(reply):
                await emit_chunk(ch)
            return FastResult(
                text=reply,
                session_updates=session_updates,
                should_escalate=True,
                tool_name=tool_name,
            )

        # 4. Second streaming call conditioned on the tool result.
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": tool_name,
                "content": json.dumps(tool_result),
            }
        )

        chunks: list[str] = []
        async for chunk in self._router.stream_chat(
            messages, temperature=0.2, max_tokens=120
        ):
            if chunk:
                chunks.append(chunk)
                await emit_chunk(chunk)

        reply = "".join(chunks) or self._fallback_reply(tool_name, tool_result)
        if not chunks:
            # Stream the fallback so the caller still hears audio.
            for ch in _chunkify(reply):
                await emit_chunk(ch)

        return FastResult(
            text=reply,
            session_updates=session_updates,
            should_escalate=should_escalate,
            ticket_id=ticket_id,
            tool_name=tool_name,
        )

    # ------------------------------------------------------------------
    # First call: stream, detect tool vs text on-the-fly
    # ------------------------------------------------------------------

    async def _first_call_stream(
        self,
        messages: list[dict[str, Any]],
        emit_chunk: Callable[[str], Awaitable[None]],
    ) -> tuple[str | None, dict[str, Any], str]:
        """Run the first LLM call with tools available. Stream text deltas
        to `emit_chunk` as they arrive. If a tool_call delta appears, stop
        forwarding text and accumulate the tool call instead.

        Returns (tool_name or None, tool_args, assembled_text).
        """
        p = self._router.providers[0]
        stream = await litellm.acompletion(
            model=p.model,
            messages=messages,
            api_base=p.api_base,
            api_key=(
                os.environ.get(p.api_key_env)
                if p.api_key_env
                else "EMPTY"
            ),
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=160,
            timeout=p.timeout_s,
            stream=True,
        )

        tool_name: str | None = None
        tool_args_parts: list[str] = []
        text_parts: list[str] = []

        async for part in stream:
            choices = getattr(part, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            # Tool call delta (OpenAI + vLLM-hermes both emit this shape).
            tc_list = getattr(delta, "tool_calls", None)
            if tc_list:
                tc = tc_list[0]
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if tool_name is None and getattr(fn, "name", None):
                        tool_name = fn.name
                    if getattr(fn, "arguments", None):
                        tool_args_parts.append(fn.arguments)
                continue

            # Plain text delta.
            content = getattr(delta, "content", None)
            if content:
                # If a tool call has already started, ignore further text.
                if tool_name is None:
                    text_parts.append(content)
                    await emit_chunk(content)

        if tool_name is not None:
            if tool_name not in TOOL_NAMES:
                logger.warning("LLM asked for unknown tool %s — treating as text", tool_name)
                tool_name = None
            else:
                try:
                    tool_args = (
                        json.loads("".join(tool_args_parts))
                        if tool_args_parts
                        else {}
                    )
                except json.JSONDecodeError:
                    logger.warning("bad tool args JSON: %r", tool_args_parts)
                    tool_args = {}
                return tool_name, tool_args, ""

        return None, {}, "".join(text_parts)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        session_state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        session_updates: dict[str, Any] = {}
        ticket_id: str | None = None
        account = session_state.get("account")

        if name == "authenticate":
            email = (args.get("email") or "").strip().lower()
            if not _EMAIL_RX.fullmatch(email):
                m = _EMAIL_RX.search(session_state.get("utterance", ""))
                if m:
                    email = m.group(0).lower()
            if not email:
                return {"found": False}, {}, None
            result = await get_account(self._dsn, email)
            if result.get("found"):
                session_updates["authenticated"] = True
                session_updates["account"] = result
            return result, session_updates, None

        if name == "get_bill":
            if not account:
                return {"error": "not authenticated"}, {}, None
            limit = int(args.get("limit") or 2)
            result = await get_bill(self._dsn, account["account_id"], limit=limit)
            return result, {}, None

        if name == "change_plan":
            if not account:
                return {"error": "not authenticated"}, {}, None
            new_plan = (args.get("new_plan") or "").strip().lower()
            result = await change_plan(self._dsn, account["account_id"], new_plan)
            if result.get("ok") and not result.get("unchanged"):
                updated = dict(account)
                updated["plan"] = result["new_plan"]
                session_updates["account"] = updated
                session_updates["pending_confirmation"] = None
            return result, session_updates, None

        if name == "lookup_kb":
            query = args.get("query") or session_state.get("utterance", "")
            return await lookup_kb(self._kb, query), {}, None

        if name == "escalate_to_human":
            subject = (args.get("subject") or "Voice caller escalation")[:120]
            priority = args.get("priority") or "normal"
            result = await escalate_to_human(
                self._dsn,
                account_id=(account or {}).get("account_id"),
                subject=subject,
                priority=priority,
            )
            ticket_id = result.get("ticket_id")
            return result, {}, ticket_id

        return {"error": f"unknown tool {name}"}, {}, None

    def _fallback_reply(self, tool_name: str, tool_result: dict[str, Any]) -> str:
        """Rule-based reply if the second LLM call emitted nothing."""
        if tool_name == "escalate_to_human" and tool_result.get("ticket_id"):
            return (
                f"Transferring you now. Ticket {tool_result['ticket_id'][:8]} "
                "has been created."
            )
        if tool_name == "change_plan" and tool_result.get("ok"):
            return (
                f"Your plan is now {tool_result.get('new_plan')}, effective "
                f"{tool_result.get('effective', 'immediately')}."
            )
        return "Let me transfer you to a human agent to help with that."


def _chunkify(text: str, size: int = 12) -> list[str]:
    """Split a pre-assembled reply into small chunks to simulate streaming
    on rule-based replies (so Pipecat's sentence aggregator still triggers
    progressively)."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def build_fast_dispatcher(router: LLMRouter, dsn: str, kb: KnowledgeBase) -> FastDispatcher:
    return FastDispatcher(router=router, dsn=dsn, kb=kb)

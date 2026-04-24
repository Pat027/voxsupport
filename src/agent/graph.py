"""
LangGraph state machine for the voice agent.

Flow:

    entry ──► classify_intent ──┬─► auth (if not authenticated)
                                │
                                ├─► billing_action ──► respond
                                ├─► account_action ──► respond
                                ├─► plan_change_action ──► confirm ──► respond
                                ├─► technical_action ──► respond (with RAG)
                                ├─► kb_action ──► respond (with RAG)
                                └─► escalate ──► respond (with ticket id)

State carries: the conversation, the current authenticated account, tool
results, flags for confirmation steps, the final text to speak.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.prompts import (
    ESCALATION_RESPONSE,
    INTENT_CLASSIFIER,
    OUT_OF_SCOPE_RESPONSE,
    UNSUPPORTED_ACTION,
)
from src.agent.rag import KnowledgeBase
from src.agent.tools import (
    change_plan,
    escalate_to_human,
    get_account,
    get_bill,
    lookup_kb,
)

logger = logging.getLogger(__name__)


Intent = Literal[
    "billing_question",
    "account_status",
    "plan_change",
    "technical_issue",
    "human_escalation",
    "kb_question",
    "authentication",
    "out_of_scope",
]


class AgentState(TypedDict, total=False):
    # Inputs
    utterance: str

    # Session state (carried across turns)
    authenticated: bool
    account: dict[str, Any] | None
    history: list[dict[str, str]]

    # Per-turn working memory
    intent: Intent
    tool_calls: list[dict[str, Any]]
    retrieved_context: str
    pending_confirmation: dict[str, Any] | None
    confidence: float

    # Output — exactly ONE of these is populated by each terminal node:
    #   response:     rule-based reply, no LLM call needed (auth, escalate, etc.)
    #   final_prompt: messages to stream through the LLM for the spoken reply
    response: str
    final_prompt: list[dict[str, str]]
    should_escalate: bool
    ticket_id: str | None


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_agent_graph(
    *,
    dsn: str,
    kb: KnowledgeBase,
    llm_call: Callable[[list[dict[str, str]]], Awaitable[str]],
):
    """Compile the LangGraph for the voice agent.

    `llm_call` is an async function that takes chat messages and returns a
    completion. Injected so tests can swap in a deterministic stub.
    """
    sg: StateGraph = StateGraph(AgentState)

    # ------ Nodes --------------------------------------------------------

    async def classify(state: AgentState) -> dict[str, Any]:
        # If a prior turn left a pending confirmation, skip classification —
        # this turn is the answer to "shall I go ahead?" and must route to
        # the same action node to pick up the confirmation context.
        pending = state.get("pending_confirmation")
        if pending:
            kind = pending.get("kind", "")
            if kind == "plan_change":
                logger.debug("short-circuit: pending_confirmation -> plan_change")
                return {"intent": "plan_change"}

        utter = state.get("utterance", "")
        prompt = INTENT_CLASSIFIER.format(utterance=utter)
        raw = await llm_call([{"role": "user", "content": prompt}])
        intent = _coerce_intent(raw)
        logger.debug("classified: %r -> %s", utter, intent)
        return {"intent": intent}

    async def auth_step(state: AgentState) -> dict[str, Any]:
        """Extract email from utterance and verify against DB."""
        email = _extract_email(state.get("utterance", ""))
        if email is None:
            return {
                "response": "Sorry, I didn't catch your email. Could you spell it out?",
            }
        account = await get_account(dsn, email)
        if not account.get("found"):
            return {
                "response": (
                    f"I don't see an account for that email. "
                    f"Want me to transfer you to a human agent?"
                ),
                "should_escalate": True,
            }
        return {
            "authenticated": True,
            "account": account,
            "response": (
                f"Thanks — I've pulled up your account. How can I help, "
                f"{_first_name(account['full_name'])}?"
            ),
        }

    async def billing_action(state: AgentState) -> dict[str, Any]:
        account = state.get("account")
        if not account:
            return {"response": "", "should_escalate": False, "intent": "authentication"}
        bills = await get_bill(dsn, account["account_id"], limit=2)
        bills_fmt = _format_bills_for_voice(bills)
        messages = [
            {"role": "system", "content": _system_voice()},
            {
                "role": "user",
                "content": (
                    f"The caller asked: '{state['utterance']}'\n\n"
                    f"Their recent bills:\n{bills_fmt}\n\n"
                    f"Answer in 1-2 spoken sentences."
                ),
            },
        ]
        return {
            "final_prompt": messages,
            "tool_calls": [{"tool": "get_bill", "args": {"limit": 2}}],
        }

    async def account_action(state: AgentState) -> dict[str, Any]:
        account = state.get("account")
        if not account:
            return {"intent": "authentication"}
        messages = [
            {"role": "system", "content": _system_voice()},
            {
                "role": "user",
                "content": (
                    f"The caller asked about their account: '{state['utterance']}'\n\n"
                    f"Their account:\n"
                    f"- Plan: {account['plan']}\n"
                    f"- Region: {account['region']}\n"
                    f"- Status: {account['status']}\n\n"
                    f"Answer in 1-2 spoken sentences."
                ),
            },
        ]
        return {"final_prompt": messages}

    async def plan_change_action(state: AgentState) -> dict[str, Any]:
        """Two-step: first turn asks confirmation, second turn executes."""
        account = state.get("account")
        if not account:
            return {"intent": "authentication"}

        pending = state.get("pending_confirmation")
        utter = state.get("utterance", "").lower()

        if pending and pending.get("kind") == "plan_change":
            if _is_affirmative(utter):
                result = await change_plan(
                    dsn,
                    account["account_id"],
                    pending["target_plan"],
                )
                if result.get("ok"):
                    reply = (
                        f"Done. Your plan is now {result['new_plan']}, effective "
                        f"{result['effective']}."
                    )
                else:
                    reply = "Something went wrong on my end. Let me transfer you."
                    return {"response": reply, "should_escalate": True}
                return {
                    "response": reply,
                    "pending_confirmation": None,
                    "tool_calls": [{"tool": "change_plan", "args": pending}],
                }
            else:
                return {
                    "response": "Okay, no changes. Anything else I can help with?",
                    "pending_confirmation": None,
                }

        # First turn: detect target plan and ask for confirmation.
        target = _detect_target_plan(state.get("utterance", ""))
        if target is None:
            messages = [
                {"role": "system", "content": _system_voice()},
                {
                    "role": "user",
                    "content": (
                        f"The caller wants to change plans but didn't specify "
                        f"which. Currently on {account['plan']}. Ask which plan "
                        f"they want, in one short sentence."
                    ),
                },
            ]
            return {"final_prompt": messages}
        return {
            "response": (
                f"You'd like to move from {account['plan']} to {target}. "
                f"Shall I go ahead and make the change now?"
            ),
            "pending_confirmation": {"kind": "plan_change", "target_plan": target},
        }

    async def technical_action(state: AgentState) -> dict[str, Any]:
        utter = state.get("utterance", "")
        context = await kb.render_context(utter, k=3)
        messages = [
            {"role": "system", "content": _system_voice()},
            {
                "role": "user",
                "content": (
                    f"The caller reported: '{utter}'\n\n"
                    f"Relevant docs:\n{context}\n\n"
                    f"Give a brief spoken answer with the single most likely cause "
                    f"and a simple next step. If no clear answer, offer to escalate."
                ),
            },
        ]
        return {"final_prompt": messages, "retrieved_context": context}

    async def kb_action(state: AgentState) -> dict[str, Any]:
        utter = state.get("utterance", "")
        chunks = await kb.search(utter, k=3)
        if not chunks or chunks[0].score < 0.3:
            return {
                "response": OUT_OF_SCOPE_RESPONSE,
                "should_escalate": True,
            }
        context = "\n\n---\n\n".join(c.content for c in chunks)
        plan = (state.get("account") or {}).get("plan", "unknown")
        messages = [
            {"role": "system", "content": _system_voice()},
            {
                "role": "user",
                "content": (
                    f"The caller is on the '{plan}' plan and asked: '{utter}'\n\n"
                    f"Here is the relevant documentation. Answer the caller's "
                    f"question directly using these facts — cite specific numbers "
                    f"when the docs have them. Do NOT say 'let me check' or "
                    f"'please hold' — the answer is in front of you. Reply in "
                    f"1-2 spoken sentences.\n\n"
                    f"DOCS:\n{context}"
                ),
            },
        ]
        return {"final_prompt": messages, "retrieved_context": context}

    async def escalate(state: AgentState) -> dict[str, Any]:
        account = state.get("account") or {}
        ticket = await escalate_to_human(
            dsn,
            account_id=account.get("account_id"),
            subject=state.get("utterance", "Voice agent escalation")[:120],
            priority="normal",
        )
        return {
            "response": ESCALATION_RESPONSE.format(ticket_id=ticket["ticket_id"][:8]),
            "should_escalate": True,
            "ticket_id": ticket["ticket_id"],
        }

    async def out_of_scope(state: AgentState) -> dict[str, Any]:
        return {"response": UNSUPPORTED_ACTION, "should_escalate": False}

    # ------ Wiring -------------------------------------------------------

    sg.add_node("classify", classify)
    sg.add_node("auth", auth_step)
    sg.add_node("billing_action", billing_action)
    sg.add_node("account_action", account_action)
    sg.add_node("plan_change_action", plan_change_action)
    sg.add_node("technical_action", technical_action)
    sg.add_node("kb_action", kb_action)
    sg.add_node("escalate", escalate)
    sg.add_node("out_of_scope", out_of_scope)

    sg.set_entry_point("classify")

    def route_from_classify(state: AgentState) -> str:
        if state.get("should_escalate"):
            return "escalate"
        intent = state.get("intent", "out_of_scope")
        if intent == "human_escalation":
            return "escalate"
        if intent == "authentication":
            return "auth"
        # Everything below requires an authenticated caller for account-specific data,
        # EXCEPT kb_question and out_of_scope which are generic.
        if intent in {"kb_question"}:
            return "kb_action"
        if intent in {"out_of_scope"}:
            return "out_of_scope"
        if not state.get("authenticated"):
            return "auth"
        if intent == "billing_question":
            return "billing_action"
        if intent == "account_status":
            return "account_action"
        if intent == "plan_change":
            return "plan_change_action"
        if intent == "technical_issue":
            return "technical_action"
        return "out_of_scope"

    sg.add_conditional_edges("classify", route_from_classify)

    for node in (
        "auth",
        "billing_action",
        "account_action",
        "plan_change_action",
        "technical_action",
        "kb_action",
        "escalate",
        "out_of_scope",
    ):
        sg.add_edge(node, END)

    return sg.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_AFFIRM_RX = re.compile(r"\b(yes|yeah|yep|sure|go ahead|do it|please|okay|ok)\b", re.I)
_PLAN_RX = re.compile(
    r"\b(starter|growth|scale|enterprise)\b",
    re.I,
)
_INTENT_TOKENS = {
    "billing_question",
    "account_status",
    "plan_change",
    "technical_issue",
    "human_escalation",
    "kb_question",
    "authentication",
    "out_of_scope",
}


def _extract_email(text: str) -> str | None:
    m = _EMAIL_RX.search(text)
    return m.group(0).lower() if m else None


def _is_affirmative(text: str) -> bool:
    return bool(_AFFIRM_RX.search(text))


def _detect_target_plan(text: str) -> str | None:
    m = _PLAN_RX.search(text)
    return m.group(0).lower() if m else None


def _first_name(full: str) -> str:
    return full.split()[0] if full else ""


def _coerce_intent(raw: str) -> Intent:
    token = raw.strip().lower().split()[0] if raw.strip() else "out_of_scope"
    # Strip trailing punctuation in case the LLM ignored instructions.
    token = re.sub(r"[^a-z_]+$", "", token)
    if token in _INTENT_TOKENS:
        return token  # type: ignore[return-value]
    return "out_of_scope"


def _format_bills_for_voice(bills: dict[str, Any]) -> str:
    if not bills.get("found"):
        return "No bills on file."
    lines = []
    for b in bills["bills"]:
        lines.append(
            f"{b['period']}: {b['amount_eur']:.2f} {b['currency']}, {b['status']}"
        )
    return "\n".join(lines)


def _system_voice() -> str:
    # Imported lazily to avoid circular import with prompts.
    from src.agent.prompts import SYSTEM_VOICE

    return SYSTEM_VOICE


# ---------------------------------------------------------------------------
# Convenience: build with env-configured dependencies
# ---------------------------------------------------------------------------


def build_default_graph(llm_call: Callable[[list[dict[str, str]]], Awaitable[str]]):
    """Build the graph with DSN from DATABASE_URL env var."""
    dsn = os.environ.get(
        "DATABASE_URL",
        "postgresql://voxsupport:voxsupport@localhost:5432/voxsupport",
    )
    # psycopg expects the `postgresql://` scheme, not `postgresql+asyncpg`.
    dsn = dsn.replace("+asyncpg", "")
    kb = KnowledgeBase(dsn)
    return build_agent_graph(dsn=dsn, kb=kb, llm_call=llm_call)

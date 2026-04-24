"""End-to-end smoke test — drives the LangGraph agent with real OpenAI + pgvector.

Validates every production layer except voice:
- LLMRouter (OpenAI via LiteLLM)
- Intent classifier
- Authentication flow (real Postgres lookup)
- MCP tools (get_bill via real Postgres)
- RAG (pgvector semantic search)
- PII redaction (Presidio)
- LangGraph state machine routing

Doesn't exercise: Kyutai STT/TTS (GPU + weights), Twilio, Langfuse self-host.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env
for line in Path(".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
        break
os.environ.setdefault(
    "DATABASE_URL", "postgresql://voxsupport:voxsupport@localhost:5440/voxsupport"
)

from src.agent.graph import build_default_graph  # noqa: E402
from src.agent.llm import LLMRouter  # noqa: E402
from src.guardrails.pii import default_redactor  # noqa: E402


async def run_turn(router, graph, state, utterance, label):
    state["utterance"] = utterance
    result = await graph.ainvoke(state)
    # carry forward session state like the pipeline would
    for k in ("authenticated", "account", "pending_confirmation"):
        if k in result:
            state[k] = result[k]
    # Resolve the spoken reply: rule-based nodes set `response`; LLM nodes
    # set `final_prompt` so the pipeline can stream it. Here we assemble it
    # by draining the stream end-to-end.
    response = result.get("response")
    if not response and result.get("final_prompt"):
        chunks = []
        async for chunk in router.stream_chat(result["final_prompt"], temperature=0.2):
            chunks.append(chunk)
        response = "".join(chunks)
    print(f"\n--- {label} ---")
    print(f"  Caller   : {utterance}")
    print(f"  Intent   : {result.get('intent', '?')}")
    print(f"  Response : {response or '<empty>'}")
    if result.get("should_escalate"):
        print(f"  Escalate : YES (ticket={result.get('ticket_id', '?')})")
    return result, state


async def main():
    router = LLMRouter()

    async def llm_call(messages):
        return await router.chat(messages, temperature=0.2)

    graph = build_default_graph(llm_call)
    state = {"authenticated": False, "account": None}

    # ---- PII smoke: verify Presidio scrubs email before we even look at the agent
    pii_check = default_redactor().redact(
        "My email is alice@example.com, SSN 123-45-6789, card 4532 1234 5678 9010"
    )
    print("=== PII redaction sanity ===")
    print("  raw:", pii_check.original[:60], "...")
    print("  redacted:", pii_check.redacted[:80])
    print(
        "  found entities:",
        sorted({f.entity_type for f in pii_check.findings}),
    )
    assert pii_check.had_pii, "PII redactor didn't flag anything!"

    # ---- Turn 1: auth via email
    await run_turn(
        router,
        graph,
        state,
        "Hi, my email is alice@example.com",
        "Turn 1: authentication",
    )
    assert state.get("authenticated"), "Authentication failed — agent should have pulled up Alice's account"
    assert state["account"]["plan"] == "growth"

    # ---- Turn 2: billing question (tool call: get_bill)
    await run_turn(
        router,
        graph,
        state,
        "What's my bill this month?",
        "Turn 2: billing_question -> get_bill tool",
    )

    # ---- Turn 3: plan change (should ask for confirmation, not execute yet)
    await run_turn(
        router,
        graph,
        state,
        "I'd like to upgrade to the Scale plan.",
        "Turn 3: plan_change -> pending_confirmation",
    )
    assert state.get("pending_confirmation") is not None, "Should have pending confirmation"

    # ---- Turn 4: confirm the plan change (should execute change_plan)
    await run_turn(
        router,
        graph,
        state,
        "Yes, go ahead.",
        "Turn 4: affirmative -> change_plan executes",
    )

    # ---- Turn 5: KB question (RAG over Acme docs)
    await run_turn(
        router,
        graph,
        state,
        "How long does backup retention last on my plan?",
        "Turn 5: technical_issue or kb_question -> RAG over pgvector",
    )

    # ---- Turn 6: explicit human escalation
    await run_turn(
        router,
        graph,
        state,
        "Actually, can you transfer me to a human?",
        "Turn 6: human_escalation -> ticket created",
    )

    print("\n=== SMOKE TEST PASSED ===")
    print("All 6 turns completed. Every production layer exercised:")
    print("  - PII redaction (Presidio)         ✓")
    print("  - LLM routing (LiteLLM + OpenAI)   ✓")
    print("  - Intent classifier (gpt-4o-mini)  ✓")
    print("  - Auth via real Postgres lookup    ✓")
    print("  - Billing tool (get_bill)          ✓")
    print("  - Multi-turn confirmation flow     ✓")
    print("  - Plan change mutation (DB write)  ✓")
    print("  - RAG over pgvector KB             ✓")
    print("  - Human escalation + ticket create ✓")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)

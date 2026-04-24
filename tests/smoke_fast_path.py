"""Drive the fast-path dispatcher through 4 canonical scenarios.

Exercises:
- Path A (direct answer, no tool)     — "what are your hours?"
- Path B single tool (get_bill)       — "what's my bill this month?"
- Path B with state (change_plan)     — "upgrade me to scale"
- Path B escalation                   — "transfer me to a human"

Measures TTFT (first chunk) for each scenario against a local vLLM endpoint.
Skips cleanly if the endpoint is down.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv("/home/pratikraut/self/voxsupport/.env")

DEFAULT_BASE = "http://localhost:8002/v1"
os.environ.setdefault("LOCAL_LLM_BASE_URL", DEFAULT_BASE)
os.environ.setdefault(
    "DATABASE_URL", "postgresql://voxsupport:voxsupport@localhost:5440/voxsupport"
)

from src.agent.fast_graph import build_fast_dispatcher  # noqa: E402
from src.agent.llm import LLMRouter  # noqa: E402
from src.agent.rag import KnowledgeBase  # noqa: E402


def _reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url + "/models", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


async def _run_scenario(dispatcher, label, utterance, state, expect_tool=None):
    print(f"\n--- {label} ---")
    print(f"  Caller : {utterance!r}")

    start = time.monotonic()
    first_chunk_at: float | None = None
    collected: list[str] = []

    async def _emit(chunk: str) -> None:
        nonlocal first_chunk_at
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - start
        collected.append(chunk)

    result = await dispatcher.run_turn(utterance, state, _emit)
    total = time.monotonic() - start

    print(f"  Reply  : {''.join(collected)!r}")
    print(f"  Tool   : {result.tool_name or '(direct answer)'}")
    print(f"  TTFT   : {(first_chunk_at or total)*1000:.0f}ms")
    print(f"  Total  : {total*1000:.0f}ms")
    if result.should_escalate:
        print(f"  Escalated ticket: {result.ticket_id}")

    if expect_tool is not None:
        assert result.tool_name == expect_tool, (
            f"expected tool={expect_tool}, got {result.tool_name}"
        )

    # Apply session updates as the pipeline would.
    for k, v in result.session_updates.items():
        state[k] = v

    return result


async def _main() -> None:
    base = os.environ["LOCAL_LLM_BASE_URL"]
    if not _reachable(base):
        print(f"[skip] local LLM endpoint not reachable at {base}")
        sys.exit(0)

    router = LLMRouter()
    assert router.providers[0].name == "local-vllm", (
        f"local-vllm must be first, got {[p.name for p in router.providers]}"
    )

    dsn = os.environ["DATABASE_URL"].replace("+asyncpg", "")
    kb = KnowledgeBase(dsn)
    dispatcher = build_fast_dispatcher(router, dsn, kb)

    # Pre-authenticate so the tool-dependent scenarios have an account.
    state = {
        "authenticated": True,
        "account": {
            "account_id": "11111111-1111-1111-1111-111111111111",
            "full_name": "Alice Nguyen",
            "plan": "growth",
            "region": "fra-1",
            "status": "active",
            "email": "alice@example.com",
        },
    }

    # --- Scenario 1: direct answer (no tool)
    await _run_scenario(
        dispatcher,
        "Scenario 1: direct answer",
        "Hi, what hours is Acme Cloud support available?",
        state,
    )

    # --- Scenario 2: single tool (get_bill)
    await _run_scenario(
        dispatcher,
        "Scenario 2: tool path (get_bill)",
        "What's my bill this month?",
        state,
        expect_tool="get_bill",
    )

    # --- Scenario 3: tool path (lookup_kb)
    await _run_scenario(
        dispatcher,
        "Scenario 3: tool path (lookup_kb)",
        "How long does backup retention last on my plan?",
        state,
        expect_tool="lookup_kb",
    )

    # --- Scenario 4: escalation
    await _run_scenario(
        dispatcher,
        "Scenario 4: tool path (escalate_to_human)",
        "Actually, can you transfer me to a human agent?",
        state,
        expect_tool="escalate_to_human",
    )

    print("\n[pass] fast-path dispatcher — 4 scenarios exercised.")


if __name__ == "__main__":
    asyncio.run(_main())

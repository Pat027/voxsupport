"""Verify LLMRouter.stream_chat yields tokens as they arrive and TTFT is fast.

Fails if stream_chat only ever yields one chunk (batch-shaped) or if the
first chunk takes longer than 2s on a warm provider connection.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv("/home/pratikraut/self/voxsupport/.env")
os.environ.setdefault("LOG_LEVEL", "INFO")

from src.agent.llm import LLMRouter


async def _measure() -> None:
    router = LLMRouter()
    messages = [
        {
            "role": "user",
            "content": (
                "Write exactly 3 sentences about a rainstorm. Keep it conversational."
            ),
        }
    ]

    print(f"[providers] {[p.name for p in router.providers]}")

    start = time.monotonic()
    first_chunk_at: float | None = None
    chunks: list[tuple[float, str]] = []
    async for chunk in router.stream_chat(messages, temperature=0.2):
        now = time.monotonic() - start
        if first_chunk_at is None:
            first_chunk_at = now
            print(f"  +{now*1000:.0f}ms  first chunk: {chunk!r}")
        chunks.append((now, chunk))

    assert first_chunk_at is not None, "no chunks emitted"
    total = time.monotonic() - start
    full = "".join(c for _, c in chunks)
    print(
        f"\n[result] TTFT={first_chunk_at*1000:.0f}ms  "
        f"total={total*1000:.0f}ms  "
        f"chunks={len(chunks)}  "
        f"chars={len(full)}"
    )
    print(f"[reply] {full}")

    # Streaming property: more than 1 chunk, spread over time.
    assert len(chunks) > 5, f"only {len(chunks)} chunk(s) — looks batched"
    last = chunks[-1][0]
    gap = last - first_chunk_at
    assert gap > 0.1, f"chunks arrived in {gap*1000:.0f}ms — batched"

    # Budget: first token under 2s (warm OpenAI/Anthropic).
    assert first_chunk_at < 2.0, f"TTFT {first_chunk_at*1000:.0f}ms exceeds 2000ms"

    print("\n[pass] LLM streams tokens.")


if __name__ == "__main__":
    asyncio.run(_measure())

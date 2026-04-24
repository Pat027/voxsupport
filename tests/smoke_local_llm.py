"""Verify a local vLLM endpoint is reachable and streams tokens fast.

Runs only if LOCAL_LLM_BASE_URL is set (or reachable on the default port).
Skips cleanly so CI can run without a local server attached.

Target numbers (Qwen/Qwen3-4B-Instruct-2507 on L40S, warm):
- TTFT       < 500 ms
- >= 10 chunks streamed
- Completion under 5 s for a 3-sentence reply

Usage:
    LOCAL_LLM_BASE_URL=http://localhost:8002/v1 \\
        PYTHONPATH=. .venv/bin/python tests/smoke_local_llm.py
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

# Default to the docker-compose port if not explicitly set. Shifting OpenAI out
# of the picture so the router has to use local — otherwise the router would
# happily fall through to OpenAI on a local failure and we'd measure cloud TTFT.
DEFAULT_BASE = "http://localhost:8002/v1"
os.environ.setdefault("LOCAL_LLM_BASE_URL", DEFAULT_BASE)
# Remove cloud providers from this process so provider selection is unambiguous.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from src.agent.llm import LLMRouter  # noqa: E402


def _reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url + "/models", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


async def _measure() -> None:
    base = os.environ["LOCAL_LLM_BASE_URL"]
    if not _reachable(base):
        print(f"[skip] local LLM endpoint not reachable at {base}")
        print(
            "       start it with: docker compose --profile local-llm up -d vllm"
        )
        sys.exit(0)

    router = LLMRouter()
    assert router.providers, "router has no providers — env vars didn't stick"
    provider_names = [p.name for p in router.providers]
    print(f"[providers] {provider_names}")
    assert provider_names[0] == "local-vllm", (
        f"expected local-vllm first, got {provider_names}"
    )
    print(f"[model]     {router.providers[0].model}")
    print(f"[api_base]  {router.providers[0].api_base}")

    messages = [
        {
            "role": "user",
            "content": (
                "In 2-3 short sentences, describe a busy cafe at sunrise. "
                "Keep it conversational."
            ),
        }
    ]

    start = time.monotonic()
    first_chunk_at: float | None = None
    chunks: list[tuple[float, str]] = []
    async for chunk in router.stream_chat(messages, temperature=0.2, max_tokens=120):
        now = time.monotonic() - start
        if first_chunk_at is None:
            first_chunk_at = now
            print(f"  +{now*1000:>4.0f}ms  first chunk: {chunk!r}")
        chunks.append((now, chunk))

    assert first_chunk_at is not None, "no chunks — local endpoint returned empty"
    total = time.monotonic() - start
    full = "".join(c for _, c in chunks)
    print(
        f"\n[result] TTFT={first_chunk_at*1000:.0f}ms  "
        f"total={total*1000:.0f}ms  "
        f"chunks={len(chunks)}  "
        f"chars={len(full)}"
    )
    print(f"[reply]  {full}")

    # Streaming sanity: genuine incremental output.
    assert len(chunks) >= 5, f"only {len(chunks)} chunks — doesn't look streamed"
    last = chunks[-1][0]
    gap = last - first_chunk_at
    assert gap > 0.05, f"chunks landed in {gap*1000:.0f}ms — batched, not streamed"

    # Speed bound. Warm Qwen3-4B on L40S should do <500ms. First cold call can
    # be slower (CUDA graph capture etc.); we still want <2s on cold.
    if first_chunk_at >= 2.0:
        print(
            f"[warn] TTFT {first_chunk_at*1000:.0f}ms higher than expected — "
            "may be cold start. Retrying once..."
        )
        start2 = time.monotonic()
        first2: float | None = None
        async for chunk in router.stream_chat(messages, temperature=0.2, max_tokens=80):
            if first2 is None:
                first2 = time.monotonic() - start2
                break
        if first2 is not None:
            print(f"[retry] warm TTFT = {first2*1000:.0f}ms")
            assert first2 < 0.8, f"warm TTFT {first2*1000:.0f}ms still > 800ms"
    else:
        assert first_chunk_at < 0.8, (
            f"TTFT {first_chunk_at*1000:.0f}ms exceeds 800ms budget"
        )

    print("\n[pass] local vLLM is streaming fast.")


if __name__ == "__main__":
    asyncio.run(_measure())

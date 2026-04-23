"""Knowledge-base lookup tool — wraps the pgvector RAG."""

from __future__ import annotations

from typing import Any

from src.agent.rag import KnowledgeBase


async def lookup_kb(kb: KnowledgeBase, query: str, *, k: int = 3) -> dict[str, Any]:
    """Retrieve top-k KB chunks for a query.

    Returns a structured payload the agent can paraphrase aloud. The raw
    `content` is included so the agent can cite specifics (plan prices,
    region codes, failover timing) without hallucinating.
    """
    chunks = await kb.search(query, k=k)
    if not chunks:
        return {"found": False, "results": []}
    return {
        "found": True,
        "results": [
            {
                "doc": c.doc_slug,
                "chunk": c.chunk_index,
                "score": round(c.score, 3),
                "content": c.content,
            }
            for c in chunks
        ],
    }

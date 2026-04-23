"""
pgvector-based RAG over the Acme Cloud knowledge base.

Week 2:
- Chunk docs at paragraph granularity with small overlap
- Embed with sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU-friendly)
- Store in pgvector with HNSW index
- Retrieve top-k by cosine similarity, with doc-slug filter support
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class KBChunk:
    doc_slug: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]


class KnowledgeBase:
    """RAG over Acme Cloud docs using pgvector."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._embedder: SentenceTransformer | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    # ----- Ingestion -------------------------------------------------------

    @staticmethod
    def chunk_markdown(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
        """Split markdown at paragraph boundaries, merging until near max_chars.

        Keeps ## headings with their sections. Paragraph-level chunks work
        well for Q&A over short docs like Acme's.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for p in paragraphs:
            candidate = f"{current}\n\n{p}".strip() if current else p
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Overlap: keep last `overlap` chars for continuity.
                tail = current[-overlap:] if current and overlap else ""
                current = (tail + "\n\n" + p).strip() if tail else p
        if current:
            chunks.append(current)
        return chunks

    async def ingest_doc(self, doc_slug: str, text: str) -> int:
        """Embed + store a single doc. Returns number of chunks written."""
        chunks = self.chunk_markdown(text)
        if not chunks:
            return 0
        embeddings = self.embedder.encode(
            chunks, normalize_embeddings=True, convert_to_numpy=True
        )
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                # Replace any prior chunks for this doc.
                await cur.execute("DELETE FROM kb_chunks WHERE doc_slug = %s", (doc_slug,))
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                    await cur.execute(
                        """
                        INSERT INTO kb_chunks (doc_slug, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (doc_slug, i, chunk, emb.tolist()),
                    )
            await conn.commit()
        logger.info("Ingested %d chunks for doc %s", len(chunks), doc_slug)
        return len(chunks)

    async def ingest_directory(self, docs_dir: str | Path) -> int:
        """Ingest every .md file in a directory. Slug = stem of filename."""
        total = 0
        for md in sorted(Path(docs_dir).glob("*.md")):
            text = md.read_text(encoding="utf-8")
            total += await self.ingest_doc(md.stem, text)
        return total

    # ----- Retrieval -------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        k: int = 4,
        doc_slugs: Sequence[str] | None = None,
    ) -> list[KBChunk]:
        """Semantic search. Returns top-k chunks sorted by cosine similarity."""
        emb = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        sql = """
            SELECT doc_slug, chunk_index, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM kb_chunks
        """
        params: list[Any] = [emb]
        if doc_slugs:
            sql += " WHERE doc_slug = ANY(%s) "
            params.append(list(doc_slugs))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([emb, k])

        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        return [
            KBChunk(
                doc_slug=r["doc_slug"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                score=float(r["score"]),
                metadata=r["metadata"] or {},
            )
            for r in rows
        ]

    async def render_context(self, query: str, *, k: int = 4) -> str:
        """Retrieve and format chunks as a single context block for the LLM."""
        chunks = await self.search(query, k=k)
        if not chunks:
            return ""
        parts = [
            f"[{c.doc_slug}#{c.chunk_index}, score={c.score:.2f}]\n{c.content}"
            for c in chunks
        ]
        return "\n\n---\n\n".join(parts)

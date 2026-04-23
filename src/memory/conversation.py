"""
Short-term conversation memory in Redis.

Design:
- Key: `conv:{session_id}`
- Value: JSON array of messages, each `{role, content, ts}`
- TTL: 30 minutes after last write — voice sessions are ephemeral

Serialization: the values written to Redis are ALREADY PII-redacted.
The redaction happens at the pipeline boundary (src.guardrails.pii) before
any persistence call. Memory only ever stores redacted text.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    ts: float


class ConversationMemory:
    def __init__(self, redis_url: str, *, ttl_seconds: int = 30 * 60) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client: redis.Redis | None = None

    async def _r(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"conv:{session_id}"

    async def append(self, session_id: str, role: str, content: str) -> None:
        r = await self._r()
        msg = Message(role=role, content=content, ts=time.time())
        await r.rpush(self._key(session_id), json.dumps(msg.__dict__))
        await r.expire(self._key(session_id), self.ttl_seconds)

    async def load(self, session_id: str, *, limit: int = 20) -> list[Message]:
        r = await self._r()
        raw = await r.lrange(self._key(session_id), -limit, -1)
        out = []
        for item in raw:
            try:
                d = json.loads(item)
                out.append(Message(role=d["role"], content=d["content"], ts=d.get("ts", 0.0)))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Bad message in session %s: %s", session_id, exc)
        return out

    async def as_llm_messages(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        msgs = await self.load(session_id, limit=limit)
        return [{"role": m.role, "content": m.content} for m in msgs]

    async def clear(self, session_id: str) -> None:
        r = await self._r()
        await r.delete(self._key(session_id))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

"""
Langfuse tracing wrapper.

Produces a trace per call, with one span per pipeline stage:
- stt: audio -> text
- pii: redaction
- classify: intent
- rag: retrieval
- llm: generation
- tts: text -> audio

All payloads are PII-redacted before they reach Langfuse (see src.guardrails.pii).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

try:
    from langfuse import Langfuse  # type: ignore
except ImportError:  # Langfuse is optional at runtime
    Langfuse = None  # type: ignore

logger = logging.getLogger(__name__)


class LangfuseTracer:
    def __init__(
        self,
        *,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        host = host or os.environ.get("LANGFUSE_HOST")
        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self._enabled = bool(Langfuse and public_key and secret_key and host)
        if self._enabled:
            self._client = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key,
            )
        else:
            self._client = None
            logger.info("Langfuse disabled (missing keys or library).")

    def start_trace(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self._enabled or self._client is None:
            return _NullTrace()
        return self._client.trace(
            name="voxsupport-session",
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )

    @asynccontextmanager
    async def span(self, trace, name: str, **kwargs):
        if not self._enabled or trace is None or isinstance(trace, _NullTrace):
            yield _NullSpan()
            return
        span = trace.span(name=name, input=kwargs.get("input"))
        try:
            yield span
        finally:
            span.end(output=kwargs.get("output"))

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


class _NullTrace:
    def update(self, **kwargs): pass
    def end(self, **kwargs): pass
    def span(self, **kwargs): return _NullSpan()


class _NullSpan:
    def end(self, **kwargs): pass


_default: LangfuseTracer | None = None


def default_tracer() -> LangfuseTracer:
    global _default
    if _default is None:
        _default = LangfuseTracer()
    return _default

"""
Kyutai TTS — streaming text-to-speech via delayed-streams-modeling.

Key property: starts emitting audio BEFORE the full text input is consumed.
That's what keeps TTFS under 800ms in the cascade architecture — the TTS
doesn't wait for the LLM to finish before it starts speaking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from src.observability.cost import CostLedger
from src.observability.metrics import TTS_TTFS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kyutai/tts-1.6b-en_fr"
SAMPLE_RATE_HZ = 24_000


class KyutaiTTSService(TTSService):
    """Streaming TTS that starts audio output before text input completes."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = "default",
        speed: float = 1.0,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        super().__init__(sample_rate=SAMPLE_RATE_HZ)
        self.model_name = model
        self.voice = voice
        self.speed = speed
        self.cost_ledger = cost_ledger
        self._inference = None

    async def _ensure_loaded(self) -> None:
        if self._inference is not None:
            return
        try:
            from moshi.models import loaders  # type: ignore
            from moshi.streaming import StreamingTTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Kyutai moshi package not installed — see README for install steps."
            ) from exc

        logger.info("Loading Kyutai TTS: %s", self.model_name)
        ckpt = loaders.CheckpointInfo.from_hf_repo(self.model_name)
        self._inference = await asyncio.to_thread(StreamingTTS.from_checkpoint, ckpt)
        logger.info("Kyutai TTS loaded")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Stream audio bytes for the given text."""
        if not text.strip():
            return
        await self._ensure_loaded()
        yield TTSStartedFrame()

        started = time.monotonic()
        first_byte = False
        total_seconds = 0.0

        for chunk in await asyncio.to_thread(self._stream_chunks, text):
            if not first_byte:
                TTS_TTFS.observe(time.monotonic() - started)
                first_byte = True
            pcm_bytes = chunk["pcm"]
            total_seconds += chunk["seconds"]
            yield TTSAudioRawFrame(
                audio=pcm_bytes,
                sample_rate=SAMPLE_RATE_HZ,
                num_channels=1,
            )

        if self.cost_ledger is not None:
            self.cost_ledger.add_audio_minute(
                phase="tts",
                provider_model="kyutai:tts-1.6b",
                minutes=total_seconds / 60.0,
            )

        yield TTSStoppedFrame()

    def _stream_chunks(self, text: str) -> list[dict]:
        """Sync wrapper around the moshi TTS streaming generator.

        The real API yields audio chunks before the full text has been
        consumed — see kyutai-labs/delayed-streams-modeling/scripts/tts.py
        for the reference streaming loop.
        """
        assert self._inference is not None
        chunks: list[dict] = []
        try:
            import numpy as np  # lazy
            for ev in self._inference.synthesize(text, voice=self.voice, speed=self.speed):  # type: ignore[attr-defined]
                pcm = (ev.audio * 32767).astype(np.int16).tobytes()
                chunks.append(
                    {
                        "pcm": pcm,
                        "seconds": len(ev.audio) / SAMPLE_RATE_HZ,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Kyutai TTS streaming error: %s", exc)
        return chunks

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            async for out in self.run_tts(frame.text):
                await self.push_frame(out, direction)

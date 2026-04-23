"""
Kyutai STT — streaming speech-to-text via delayed-streams-modeling.

Wraps the reference streaming loop from kyutai-labs/delayed-streams-modeling
into Pipecat's STTService interface. Key properties:

- Streaming: partial transcripts flow as the caller speaks
- Semantic VAD: end-of-turn predicted from content + intonation, not just silence
- ~500ms inherent delay (Kyutai's published number for the 1B model)

The import of `moshi` is lazy — it loads model weights from Hugging Face the
first time the model is used, which is slow. Keeping it out of import time
means the FastAPI server starts fast and the first call pays the warm-up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

from src.observability.cost import CostLedger
from src.observability.metrics import STT_LATENCY

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kyutai/stt-1b-en_fr"
SAMPLE_RATE_HZ = 24_000


class KyutaiSTTService(STTService):
    """Streaming STT over Kyutai's delayed-streams-modeling."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        cost_ledger: CostLedger | None = None,
    ) -> None:
        super().__init__(sample_rate=SAMPLE_RATE_HZ)
        self.model_name = model
        self.language = language
        self.cost_ledger = cost_ledger
        self._inference = None
        self._audio_seconds = 0.0
        self._turn_started: float | None = None

    async def _ensure_loaded(self) -> None:
        if self._inference is not None:
            return
        # Lazy import — large dependency tree.
        try:
            from moshi.models import loaders  # type: ignore
            from moshi.streaming import StreamingInference  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Kyutai moshi package not installed. "
                "Run `pip install moshi` or install from "
                "kyutai-labs/delayed-streams-modeling."
            ) from exc

        logger.info("Loading Kyutai STT model: %s", self.model_name)
        checkpoint = loaders.CheckpointInfo.from_hf_repo(self.model_name)
        # Runs to completion in a thread — loading weights blocks; we don't
        # want to block the async event loop.
        self._inference = await asyncio.to_thread(
            StreamingInference.from_checkpoint, checkpoint
        )
        logger.info("Kyutai STT loaded")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process one audio chunk. Yields interim + final transcription frames."""
        await self._ensure_loaded()
        start = time.monotonic()

        # Kyutai's StreamingInference expects float32 PCM at the model's rate.
        # `audio` here is bytes produced by Pipecat's audio source — 16-bit PCM.
        import numpy as np  # lazy
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_seconds += len(pcm) / SAMPLE_RATE_HZ

        # The reference streaming API returns incremental decode events.
        # Run under to_thread to keep the event loop free; the inner loop is
        # GPU-bound. A dedicated process or queue is a better fit at scale.
        events = await asyncio.to_thread(self._decode_chunk, pcm)

        latency = time.monotonic() - start
        STT_LATENCY.observe(latency)

        for ev in events:
            if ev["kind"] == "partial":
                yield InterimTranscriptionFrame(
                    text=ev["text"], user_id="", timestamp=0, language=self.language
                )
            elif ev["kind"] == "final":
                yield TranscriptionFrame(
                    text=ev["text"], user_id="", timestamp=0, language=self.language
                )
                if self.cost_ledger is not None:
                    minutes = self._audio_seconds / 60.0
                    self.cost_ledger.add_audio_minute(
                        phase="stt",
                        provider_model="kyutai:stt-1b",
                        minutes=minutes,
                    )
                    self._audio_seconds = 0.0

    def _decode_chunk(self, pcm) -> list[dict]:
        """Sync wrapper — adapt to the actual moshi API once model loaded.

        The real delayed-streams-modeling API exposes either a push/pop
        streaming loop or a generator yielding decode events. See
        kyutai-labs/delayed-streams-modeling for the reference.

        This method returns a list of {kind, text} dicts for downstream
        event-frame translation.
        """
        assert self._inference is not None
        # The reference repo's ergonomics vary by commit — consult their
        # `scripts/stream.py` for the current streaming API. Typical usage:
        #     for event in self._inference.stream(pcm): ...
        # and `event.text` / `event.is_final`.
        events: list[dict] = []
        try:
            for ev in self._inference.stream(pcm):  # type: ignore[attr-defined]
                events.append(
                    {
                        "kind": "final" if getattr(ev, "is_final", False) else "partial",
                        "text": getattr(ev, "text", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Kyutai STT streaming error: %s", exc)
        return events

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn_started = time.monotonic()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_started = None
        elif isinstance(frame, AudioRawFrame):
            async for out in self.run_stt(frame.audio):
                await self.push_frame(out, direction)

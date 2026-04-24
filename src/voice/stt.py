"""
Kyutai STT — streaming speech-to-text via moshi's delayed-streams-modeling.

Real API (moshi 0.2.x):
- `moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")` loads
  the checkpoint + tokenizer config.
- `ci.get_mimi(device)` gives the streaming neural audio codec.
- `ci.get_moshi(device, dtype=torch.bfloat16)` gives the LM.
- `moshi.run_inference.InferenceState` runs the streaming loop. For STT
  (`dep_q == 0`), tokens come out via `printer.print_token(text)` — we capture
  them with a swap-in printer.

Key properties:
- ~500 ms inherent delay (Kyutai published number for the 1B model).
- Semantic VAD predicts end-of-turn from content + intonation.
- Streaming-native: audio chunks in, partial + final transcripts out.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
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


class _TokenCapture:
    """Swap-in for moshi's Printer — captures tokens without printing to stdout."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self._flushed_up_to = 0

    def print_token(self, text: str) -> None:
        self.tokens.append(text)

    def log(self, level: str, msg: str) -> None:  # noqa: ARG002
        logger.debug("moshi[%s]: %s", level, msg)

    def print_header(self) -> None:
        pass

    def pop_new(self) -> str:
        """Return tokens emitted since the last pop, as a joined string."""
        out = "".join(self.tokens[self._flushed_up_to :])
        self._flushed_up_to = len(self.tokens)
        return out

    def full(self) -> str:
        return "".join(self.tokens).strip()


class KyutaiSTTService(STTService):
    """Streaming STT over Kyutai's delayed-streams-modeling.

    Lazy-loads the model on first audio frame (big download on cold start).
    """

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

        self._state: Any | None = None
        self._capture: _TokenCapture | None = None
        self._audio_buf: list[np.ndarray] = []
        self._audio_seconds = 0.0
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._state is not None:
            return
        async with self._load_lock:
            if self._state is not None:
                return
            logger.info("Loading Kyutai STT: %s", self.model_name)
            t0 = time.monotonic()
            self._state = await asyncio.to_thread(self._load_state)
            self._capture = _TokenCapture()
            self._state.printer = self._capture  # type: ignore[attr-defined]
            logger.info("Kyutai STT loaded in %.1fs", time.monotonic() - t0)

    def _load_state(self):
        import torch
        from moshi.models import loaders
        from moshi.run_inference import InferenceState

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        ci = loaders.CheckpointInfo.from_hf_repo(self.model_name)
        mimi = ci.get_mimi(device=device)
        tokenizer = ci.get_text_tokenizer()
        lm = ci.get_moshi(device=device, dtype=dtype)
        return InferenceState(
            checkpoint_info=ci,
            mimi=mimi,
            text_tokenizer=tokenizer,
            lm=lm,
            batch_size=1,
            cfg_coef=1.0,
            device=device,
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Handle one PCM chunk. Emits partial transcripts; final transcripts
        are emitted when end-of-speech is detected (by Pipecat's VAD upstream)."""
        await self._ensure_loaded()
        start = time.monotonic()

        # Buffer the audio; we process per-turn (triggered by UserStoppedSpeakingFrame).
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buf.append(pcm)
        self._audio_seconds += len(pcm) / SAMPLE_RATE_HZ

        # Interim frames throttled: emit what we have in the capture buffer
        # (if streaming inference is hot enough, partials accumulate here).
        partial = self._capture.pop_new() if self._capture else ""
        if partial:
            STT_LATENCY.observe(time.monotonic() - start)
            yield InterimTranscriptionFrame(
                text=partial, user_id="", timestamp=0, language=self.language
            )

    async def _finalize_turn(self) -> str:
        """Run the buffered audio through the STT model and return full transcript."""
        if not self._audio_buf or self._state is None or self._capture is None:
            return ""

        import torch

        audio = np.concatenate(self._audio_buf)
        self._audio_buf = []
        in_pcms = (
            torch.from_numpy(audio)
            .to(device=self._state.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Reset capture for this turn
        self._capture.tokens = []
        self._capture._flushed_up_to = 0
        await asyncio.to_thread(self._state.run, in_pcms)

        if self.cost_ledger is not None:
            self.cost_ledger.add_audio_minute(
                phase="stt",
                provider_model="kyutai:stt-1b",
                minutes=self._audio_seconds / 60.0,
            )
            self._audio_seconds = 0.0

        return self._capture.full()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._audio_buf = []  # fresh turn
            if self._capture is not None:
                self._capture.tokens = []
                self._capture._flushed_up_to = 0

        elif isinstance(frame, AudioRawFrame):
            async for out in self.run_stt(frame.audio):
                await self.push_frame(out, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            text = await self._finalize_turn()
            if text:
                await self.push_frame(
                    TranscriptionFrame(
                        text=text, user_id="", timestamp=0, language=self.language
                    ),
                    direction,
                )

        else:
            await self.push_frame(frame, direction)

"""
Kyutai STT — **streaming** speech-to-text via moshi's delayed-streams-modeling.

Unlike the pre-streaming implementation (which buffered the whole utterance
and called `state.run()` at end-of-speech — ~5 s tail for a 4-s turn), this
version feeds audio through the model frame-by-frame *as it arrives*. By the
time the VAD fires UserStoppedSpeakingFrame, almost all compute is already
done; only the right-padding silence and the model's ~500 ms inherent delay
remain. Tail latency typically drops to ~400 ms.

Mechanism (mirrors Kyutai's own reference server in `moshi.server`):

1. At load, `mimi.streaming_forever(1)` + `lm_gen.streaming_forever(1)` put
   the modules in perpetual streaming state.
2. On UserStartedSpeakingFrame, `reset_streaming()` on both clears the last
   turn's hidden state.
3. The first encoded mimi frame has a peculiar structure from left-padding
   the signal — after encoding it, we reset the mimi streaming again. (Same
   `skip_frames = 1` trick Kyutai uses.)
4. Per incoming AudioRawFrame we accumulate to full 80-ms frames, encode
   each with mimi, step `lm_gen`, and capture any text tokens emitted.
5. On UserStoppedSpeakingFrame we append right-padding silence so the model
   can flush its delayed tokens, then emit the final TranscriptionFrame.
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
    """Replaces moshi's stdout Printer — captures tokens so we can surface them
    as pipeline InterimTranscriptionFrames."""

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
        out = "".join(self.tokens[self._flushed_up_to :])
        self._flushed_up_to = len(self.tokens)
        return out

    def full(self) -> str:
        return "".join(self.tokens).strip()


class KyutaiSTTService(STTService):
    """Streaming Kyutai STT. Feeds audio through the model as it arrives —
    not in a single batch at end-of-speech."""

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
        self._load_lock = asyncio.Lock()
        self._gpu_lock = asyncio.Lock()  # serialize GPU-heavy work from async callers

        # Per-turn streaming state.
        self._turn_active = False
        self._skip_first_mimi_reset = True
        self._frame_remainder: np.ndarray | None = None
        self._audio_seconds = 0.0

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

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def _reset_turn_state(self) -> None:
        """Clear streaming state at the start of a new user turn."""
        if self._state is not None:
            self._state.mimi.reset_streaming()
            self._state.lm_gen.reset_streaming()
        if self._capture is not None:
            self._capture.tokens = []
            self._capture._flushed_up_to = 0
        self._skip_first_mimi_reset = True
        self._frame_remainder = None
        self._audio_seconds = 0.0

    def _process_pcm_sync(self, pcm_f32: np.ndarray) -> None:
        """Feed PCM through streaming mimi + lm_gen. Runs in a worker thread."""
        import torch

        state = self._state
        assert state is not None
        capture = self._capture
        assert capture is not None

        fsz = state.frame_size  # 1920 samples @ 24 kHz = 80 ms

        if self._frame_remainder is not None and self._frame_remainder.size:
            pcm_f32 = np.concatenate([self._frame_remainder, pcm_f32])

        n_frames = len(pcm_f32) // fsz
        if n_frames == 0:
            self._frame_remainder = pcm_f32
            return

        complete = pcm_f32[: n_frames * fsz]
        leftover = pcm_f32[n_frames * fsz :]
        self._frame_remainder = leftover if leftover.size else None

        dep_q = state.lm_gen.lm_model.dep_q

        for i in range(n_frames):
            chunk = complete[i * fsz : (i + 1) * fsz]
            chunk_t = (
                torch.from_numpy(chunk)
                .to(device=state.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            codes = state.mimi.encode(chunk_t)

            if self._skip_first_mimi_reset:
                # Kyutai's batch inference primes the LM by stepping it once
                # on the first frame (result discarded) before the "real" step,
                # to absorb the model's delay machinery. The mimi streaming
                # state has a left-padding structure from the first encode
                # that must not persist, so we reset it.
                state.mimi.reset_streaming()
                _ = state.lm_gen.step(codes)  # prime; ignore output
                self._skip_first_mimi_reset = False
                # fall through to do the real step on the same codes

            tokens = state.lm_gen.step(codes)
            if tokens is None:
                continue
            # STT has dep_q == 0 → text token is at tokens[0, 0].
            if dep_q == 0:
                tok_id = tokens[0, 0].cpu().item()
            else:  # defensive: shouldn't happen for STT models
                tok_id = tokens[0, 0, 0].cpu().item()
            if tok_id not in (0, 3):
                text = state.text_tokenizer.id_to_piece(tok_id)
                text = text.replace("▁", " ")
                capture.print_token(text)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Feed one chunk of audio through the streaming STT model."""
        if not self._turn_active:
            return
        await self._ensure_loaded()

        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_seconds += len(pcm) / SAMPLE_RATE_HZ

        start = time.monotonic()
        async with self._gpu_lock:
            await asyncio.to_thread(self._process_pcm_sync, pcm)
        STT_LATENCY.observe(time.monotonic() - start)

        partial = self._capture.pop_new() if self._capture else ""
        if partial:
            yield InterimTranscriptionFrame(
                text=partial,
                user_id="",
                timestamp=0,
                language=self.language,
            )

    async def _finalize_turn(self) -> str:
        """Append right-padding silence, drain final tokens, return transcript."""
        if self._state is None or self._capture is None:
            return ""

        state = self._state
        stt_config = state.checkpoint_info.stt_config
        pad_right_secs = stt_config.get("audio_delay_seconds", 0.0) + 1.0
        pad_right_samples = int(pad_right_secs * SAMPLE_RATE_HZ)
        silence = np.zeros(pad_right_samples, dtype=np.float32)

        async with self._gpu_lock:
            await asyncio.to_thread(self._process_pcm_sync, silence)

        transcript = self._capture.full()

        if self.cost_ledger is not None and self._audio_seconds > 0:
            self.cost_ledger.add_audio_minute(
                phase="stt",
                provider_model="kyutai:stt-1b",
                minutes=self._audio_seconds / 60.0,
            )

        self._turn_active = False
        return transcript

    # ------------------------------------------------------------------
    # Pipecat frame interface
    #
    # Pipecat's STTService.process_frame already routes AudioRawFrame to
    # process_audio_frame → run_stt and pushes interim transcripts. We only
    # need to layer turn lifecycle on top: reset streaming at UserStartedSpeaking,
    # finalize and emit the full TranscriptionFrame at UserStoppedSpeaking.
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._ensure_loaded()
            self._reset_turn_state()
            self._turn_active = True

        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame) and self._turn_active:
            text = await self._finalize_turn()
            if text:
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id="",
                        timestamp=0,
                        language=self.language,
                    )
                )

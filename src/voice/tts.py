"""
Kyutai TTS — streaming text-to-speech using the real moshi.run_tts API.

The `tts_model.generate` call drives the LM through the whole text, firing a
per-frame callback each time a mimi audio frame is emitted. We decode each
frame and push it onto an asyncio.Queue from the worker thread (thread-safe
via `loop.call_soon_threadsafe`). The outer async generator pulls from the
queue and yields TTSAudioRawFrames as they arrive — meaning the first chunk
reaches Pipecat's transport ~80-200 ms after the LM starts, not after the
whole utterance finishes.

Kyutai's mimi codec runs at 12.5 Hz → one frame every 80 ms of audio.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
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

DEFAULT_VOICE = "alba-mackenna/announcer.wav"

# Sentinel pushed onto the queue when generation finishes.
_GEN_DONE = object()


class KyutaiTTSService(TTSService):
    """Streaming TTS. Emits audio bytes as soon as mimi decodes them."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        temperature: float = 0.6,
        cfg_coef: float = 2.0,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        super().__init__(sample_rate=24_000)
        self.voice = voice
        self.speed = speed
        self.temperature = temperature
        self.cfg_coef = cfg_coef
        self.cost_ledger = cost_ledger

        self._tts: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        async with self._load_lock:
            if self._tts is not None:
                return
            logger.info("Loading Kyutai TTS")
            t0 = time.monotonic()
            self._tts = await asyncio.to_thread(self._load_model)
            self._sample_rate = self._tts.mimi.sample_rate
            logger.info("Kyutai TTS loaded in %.1fs", time.monotonic() - t0)

    async def warmup(self) -> None:
        """Ensure weights are loaded. A true synthesis warmup (running
        `generate` once at startup) would compile the CUDA graphs but
        triggers a known torch.compile + stochastic sampling bug:

            RuntimeError: Offset increment outside graph capture encountered
            unexpectedly.

        when the first REAL call after warmup samples a different multinomial
        path. Until that's fixed upstream in moshi/torch, we accept the
        ~900 ms first-call TTFS tax. The STT + SBert warmups still run.
        """
        await self._ensure_loaded()

    def _load_model(self):
        import torch
        from moshi.run_tts import (
            CheckpointInfo,
            DEFAULT_DSM_TTS_REPO,
            TTSModel,
        )

        device_str = os.environ.get("KYUTAI_DEVICE", "cuda:0")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        return TTSModel.from_checkpoint_info(
            ckpt, n_q=32, temp=self.temperature, device=device
        )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Synthesize `text`; yield audio frames as they're decoded."""
        if not text.strip():
            return
        await self._ensure_loaded()
        yield TTSStartedFrame()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        start = time.monotonic()
        first_byte_recorded = False

        def _on_frame(frame):
            nonlocal first_byte_recorded
            # Padding/silence tokens: skip.
            if (frame == -1).any():
                return
            try:
                pcm = self._tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                arr = np.clip(pcm[0, 0], -1, 1)
            except Exception as exc:  # noqa: BLE001 — surface to the awaiter
                loop.call_soon_threadsafe(queue.put_nowait, exc)
                return

            if not first_byte_recorded:
                TTS_TTFS.observe(time.monotonic() - start)
                first_byte_recorded = True
            # Thread-safe push from the moshi worker thread into the asyncio queue.
            loop.call_soon_threadsafe(queue.put_nowait, arr)

        voices = [self._tts.get_voice_path(self.voice)]
        all_entries = [self._tts.prepare_script([text], padding_between=1)]
        cond = self._tts.make_condition_attributes(voices, cfg_coef=self.cfg_coef)

        async def _drive_generation() -> None:
            try:
                await asyncio.to_thread(
                    lambda: self._generate(all_entries, [cond], _on_frame)
                )
            except Exception as exc:  # noqa: BLE001 — propagate via queue
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _GEN_DONE)

        gen_task = asyncio.create_task(_drive_generation())

        total_samples = 0
        try:
            while True:
                item = await queue.get()
                if item is _GEN_DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                arr: np.ndarray = item
                pcm_i16 = (arr * 32767).astype(np.int16).tobytes()
                total_samples += len(arr)
                yield TTSAudioRawFrame(
                    audio=pcm_i16,
                    sample_rate=self._tts.mimi.sample_rate,
                    num_channels=1,
                )
        finally:
            await gen_task

        if self.cost_ledger is not None and total_samples > 0:
            self.cost_ledger.add_audio_minute(
                phase="tts",
                provider_model="kyutai:tts-1.6b",
                minutes=total_samples / self._tts.mimi.sample_rate / 60.0,
            )

        yield TTSStoppedFrame()

    def _generate(self, all_entries, conds, on_frame):
        """Sync wrapper around tts.generate — runs in a worker thread."""
        import torch

        with self._tts.mimi.streaming(1), torch.no_grad():
            self._tts.generate(all_entries, conds, on_frame=on_frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            async for out in self.run_tts(frame.text):
                await self.push_frame(out, direction)

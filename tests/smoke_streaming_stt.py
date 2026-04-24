"""Verify KyutaiSTTService transcribes INCREMENTALLY as audio arrives.

Drives the service with a pre-recorded WAV chopped into 80ms chunks. Simulates
Pipecat's AudioRawFrames + VAD signals. After UserStoppedSpeakingFrame the
final TranscriptionFrame must arrive quickly because most compute happened
during speech — not all at once at end-of-turn.

Budget: time from UserStoppedSpeakingFrame → TranscriptionFrame ≤ 1.5s for
a 4-second utterance (vs. ~5s when batch-processing the whole buffer post-hoc).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import sphn
from dotenv import load_dotenv

load_dotenv("/home/pratikraut/self/voxsupport/.env")
os.environ.setdefault("LOG_LEVEL", "INFO")

from pipecat.frames.frames import (
    AudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from src.voice.stt import KyutaiSTTService

WAV_PATH = "tests/output/kyutai_tts_smoke.wav"
CHUNK_MS = 80  # Pipecat-ish; Kyutai's model frame_size is exactly 80ms


class _FrameSink:
    """Captures downstream pushes so we can assert on what the service emits.

    Pipecat's STTService calls `push_frame(frame)` with no direction from
    inside `process_generator`, so we make direction optional.
    """

    def __init__(self) -> None:
        self.frames: list[tuple[float, object]] = []
        self._t0 = time.monotonic()

    async def push(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.frames.append((time.monotonic() - self._t0, frame))


async def _drive() -> None:
    assert Path(WAV_PATH).exists(), f"missing {WAV_PATH} — run smoke_voice.py first"

    audio, rate = sphn.read(WAV_PATH)
    audio = audio[0] if audio.ndim == 2 else audio
    # Kyutai STT expects 24 kHz. Resample if needed via scipy (cheap for offline test).
    if rate != 24_000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 24_000, rate)
        rate = 24_000
    duration_s = len(audio) / rate
    print(f"[input] {WAV_PATH}  {duration_s:.2f}s @ {rate} Hz")

    svc = KyutaiSTTService()
    sink = _FrameSink()
    svc.push_frame = sink.push  # type: ignore[method-assign]

    # Warm (load model)
    print("[warmup] loading model + warming CUDA...")
    t0 = time.monotonic()
    await svc._ensure_loaded()
    print(f"[warmup] loaded in {time.monotonic() - t0:.1f}s")

    # Kick off a turn
    await svc.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # Feed audio in 80ms slices, at real-time cadence, measuring
    chunk_samples = int(rate * CHUNK_MS / 1000)
    pcm_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    start = time.monotonic()
    for i in range(0, len(pcm_i16), chunk_samples):
        slice_ = pcm_i16[i : i + chunk_samples]
        frame = AudioRawFrame(audio=slice_.tobytes(), sample_rate=rate, num_channels=1)
        await svc.process_frame(frame, FrameDirection.DOWNSTREAM)
        # don't sleep — we want the test to run as fast as the GPU allows

    # End of speech
    t_stop = time.monotonic()
    print(f"[drive] fed {duration_s:.2f}s of audio in {t_stop - start:.2f}s wall time")
    await svc.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    t_transcribed = time.monotonic()
    tail_latency = t_transcribed - t_stop
    print(f"[drive] end-of-speech → final transcript: {tail_latency*1000:.0f}ms")

    # Inspect captured frames
    interim_count = sum(1 for _, f in sink.frames if isinstance(f, InterimTranscriptionFrame))
    transcription = next(
        (f for _, f in sink.frames if isinstance(f, TranscriptionFrame)), None
    )
    print(f"[frames] interim={interim_count}  final={'yes' if transcription else 'no'}")
    if transcription:
        print(f"[transcript] {transcription.text!r}")

    assert transcription is not None, "no final TranscriptionFrame emitted"
    assert len(transcription.text.strip()) > 5, "transcript too short to trust"

    # Streaming property: at least SOME compute happened during speech.
    # We allow up to 1.5s tail after end-of-speech (right-padding + final tokens).
    assert tail_latency < 1.5, (
        f"tail latency {tail_latency*1000:.0f}ms > 1500ms — model is batching"
    )

    print("\n[pass] STT is streaming.")


if __name__ == "__main__":
    asyncio.run(_drive())

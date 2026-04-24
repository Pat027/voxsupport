"""Verify KyutaiTTSService streams audio frames AS the model generates them,
not in one batch after generation completes.

Fails with the pre-refactor implementation: first yielded frame arrives only
after the whole synthesis thread joins (~5-6s for a medium sentence). Passes
once `run_tts` uses an asyncio.Queue fed from `on_frame`.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv("/home/pratikraut/self/voxsupport/.env")
os.environ.setdefault("LOG_LEVEL", "INFO")

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from src.voice.tts import KyutaiTTSService


async def _measure() -> None:
    svc = KyutaiTTSService()

    # Warm the model on a throwaway short phrase so the cold-start load time
    # doesn't dominate the measurement.
    print("[warmup] loading model...")
    t0 = time.monotonic()
    async for _ in svc.run_tts("Hi."):
        pass
    print(f"[warmup] done in {time.monotonic() - t0:.1f}s")

    text = (
        "Thanks for calling Acme Cloud support. "
        "I can help you with billing, plan changes, and Postgres connection issues. "
        "What can I do for you today?"
    )
    print(f"[measure] synthesizing {len(text)} chars...")

    start = time.monotonic()
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    audio_frames = 0
    total_samples = 0

    async for frame in svc.run_tts(text):
        now = time.monotonic() - start
        if isinstance(frame, TTSStartedFrame):
            print(f"  +{now*1000:.0f}ms  TTSStartedFrame")
        elif isinstance(frame, TTSAudioRawFrame):
            if first_audio_at is None:
                first_audio_at = now
                print(f"  +{now*1000:.0f}ms  first audio ({len(frame.audio)} bytes)")
            audio_frames += 1
            last_audio_at = now
            total_samples += len(frame.audio) // 2  # int16
        elif isinstance(frame, TTSStoppedFrame):
            print(f"  +{now*1000:.0f}ms  TTSStoppedFrame ({audio_frames} audio frames)")

    assert first_audio_at is not None, "no audio emitted"
    assert last_audio_at is not None

    duration_s = total_samples / 24_000
    print(
        f"\n[result] TTFS={first_audio_at*1000:.0f}ms  "
        f"last_frame_at={last_audio_at*1000:.0f}ms  "
        f"generated_audio={duration_s:.2f}s  "
        f"frames={audio_frames}"
    )

    # Regression guard: streaming means first frame MUST arrive well before
    # the last frame. If they're within 500ms of each other, that's batch
    # behavior — the whole synthesis completed before the first yield.
    gap = last_audio_at - first_audio_at
    assert gap > 1.0, (
        f"Frames appear batched (first→last gap only {gap*1000:.0f}ms). "
        "TTS is not streaming."
    )

    # Hard budget: first audio chunk within 1.5s of run_tts entry (warm model).
    assert first_audio_at < 1.5, (
        f"TTFS {first_audio_at*1000:.0f}ms exceeds 1500ms budget"
    )

    print("\n[pass] TTS is streaming.")


if __name__ == "__main__":
    asyncio.run(_measure())

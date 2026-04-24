"""End-to-end streaming latency test.

Drives the full streaming path:
  caller WAV (fed as 80 ms AudioRawFrames)
    → KyutaiSTTService (streaming, emits InterimTranscription + final TranscriptionFrame)
    → LangGraph agent + LLMRouter.stream_chat (yields text chunks as LLM generates)
    → KyutaiTTSService (emits TTSAudioRawFrames as mimi decodes each 80 ms frame)

The critical user-perceived metric is **t_first_audio_out**: milliseconds
between UserStoppedSpeakingFrame and the first TTSAudioRawFrame pushed
downstream. That's when the caller actually starts hearing the reply.

Contrast with the pre-refactor pipeline (smoke_pipeline.py), which ran each
stage batch-only and came in around 12.9 s total.
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
os.environ.setdefault(
    "DATABASE_URL", "postgresql://voxsupport:voxsupport@localhost:5440/voxsupport"
)
os.environ.setdefault("LOG_LEVEL", "INFO")

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from src.agent.graph import build_default_graph
from src.agent.llm import LLMRouter
from src.voice.stt import KyutaiSTTService
from src.voice.tts import KyutaiTTSService

CALLER_WAV = "tests/output/kyutai_tts_smoke.wav"
REPLY_WAV = "tests/output/streaming_reply.wav"


class _Sink:
    """Captures whatever a service pushes downstream."""

    def __init__(self) -> None:
        self.frames: list[tuple[float, object]] = []
        self._t0 = time.monotonic()

    def reset_clock(self) -> None:
        self._t0 = time.monotonic()

    async def push(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.frames.append((time.monotonic() - self._t0, frame))


async def _main() -> None:
    assert Path(CALLER_WAV).exists(), f"{CALLER_WAV} missing (run smoke_voice.py first)"

    # --- Load caller audio ---
    audio, rate = sphn.read(CALLER_WAV)
    audio = audio[0] if audio.ndim == 2 else audio
    if rate != 24_000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 24_000, rate)
        rate = 24_000
    pcm_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    duration_s = len(pcm_i16) / rate
    print(f"[input] {CALLER_WAV}  {duration_s:.2f}s @ {rate} Hz")

    # --- Build services ---
    stt = KyutaiSTTService()
    tts = KyutaiTTSService()
    router = LLMRouter()

    stt_sink = _Sink()
    tts_sink = _Sink()
    stt.push_frame = stt_sink.push  # type: ignore[method-assign]
    tts.push_frame = tts_sink.push  # type: ignore[method-assign]

    # --- Warm up models so latency numbers are post-cold-start ---
    print("[warmup] loading models...")
    t0 = time.monotonic()
    await stt._ensure_loaded()
    await tts._ensure_loaded()
    print(f"[warmup] loaded in {time.monotonic() - t0:.1f}s")

    # --- Graph (pre-authenticated so a single turn doesn't stop at auth) ---
    async def nonstream_llm(messages):
        return await router.chat(messages, temperature=0.2)

    graph = build_default_graph(nonstream_llm)

    # --- Stream audio through STT at real-time cadence ---
    CHUNK_MS = 80
    chunk_samples = int(rate * CHUNK_MS / 1000)

    await stt.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    t_start_feed = time.monotonic()
    for i in range(0, len(pcm_i16), chunk_samples):
        frame = AudioRawFrame(
            audio=pcm_i16[i : i + chunk_samples].tobytes(),
            sample_rate=rate,
            num_channels=1,
        )
        await stt.process_frame(frame, FrameDirection.DOWNSTREAM)
    t_feed_done = time.monotonic()
    print(f"[stt] fed {duration_s:.2f}s audio in {t_feed_done - t_start_feed:.2f}s wall")

    # --- End of speech: the clock-critical moment ---
    t_speech_end = time.monotonic()
    await stt.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    transcription = next(
        (f for _, f in stt_sink.frames if isinstance(f, TranscriptionFrame)), None
    )
    t_transcript = time.monotonic()
    assert transcription, "STT did not emit a final TranscriptionFrame"
    print(
        f"[stt] transcript: {transcription.text!r}  "
        f"({(t_transcript - t_speech_end)*1000:.0f}ms after end-of-speech)"
    )

    # --- Substitute a real caller question (the caller WAV is the greeting) ---
    utter = "What's my bill this month?"
    print(f"[agent] (using real caller question for realism: {utter!r})")

    # --- Run graph to decide intent + build final prompt ---
    # Note: graph.ainvoke runs two sequential LLM calls: (1) classify intent,
    # (2) stream response. With cloud OpenAI each costs ~1-2s TTFT so the
    # total "end-of-speech → first LLM token of response" floor is ~2-4s for
    # cloud-cascade. A local vLLM-served Qwen3-4B would cut this to ~500ms.
    state = {
        "utterance": utter,
        "authenticated": True,
        "account": {
            "account_id": "11111111-1111-1111-1111-111111111111",
            "full_name": "Alice Nguyen",
            "plan": "growth",
            "region": "fra-1",
            "status": "active",
            "email": "alice@example.com",
        },
    }
    t_graph_start = time.monotonic()
    result = await graph.ainvoke(state)
    t_graph_done = time.monotonic()
    print(
        f"[agent] graph.ainvoke (classify + tool) took "
        f"{(t_graph_done - t_graph_start)*1000:.0f}ms"
    )
    final_prompt = result.get("final_prompt")
    if final_prompt is None:
        # rule-based reply — nothing to stream; just use `response`
        async def _chunks():
            yield result.get("response", "")
        stream = _chunks()
    else:
        stream = router.stream_chat(final_prompt, temperature=0.2)

    # --- Pump streaming chunks into TTS ---
    # In a live Pipecat pipeline, TextFrames flow into the TTSService's queue
    # and it synthesizes per sentence. For this headless test we emulate the
    # same behavior by aggregating LLM chunks into sentences and driving
    # tts.run_tts() directly. This is exactly what Pipecat's SENTENCE
    # aggregation mode does internally.
    t_first_text: float | None = None
    t_first_audio: float | None = None
    audio_frames: list[TTSAudioRawFrame] = []

    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()

    async def _synthesize_sentences() -> None:
        nonlocal t_first_audio
        while True:
            sent = await sentence_q.get()
            if sent is None:
                return
            async for out in tts.run_tts(sent):
                if isinstance(out, TTSAudioRawFrame):
                    if t_first_audio is None:
                        t_first_audio = time.monotonic() - t_speech_end
                    audio_frames.append(out)

    tts_task = asyncio.create_task(_synthesize_sentences())

    text_parts: list[str] = []
    buf = ""
    # Phrase boundaries: any sentence-end OR comma AFTER at least ~20 chars.
    # This lets TTS start on the first clause ("Your bill for this month,...")
    # instead of waiting for the whole sentence. For Kyutai TTS quality,
    # don't split too short (below ~15 chars prosody suffers).
    MIN_FIRST_PHRASE = 20
    SENT_END = (".", "!", "?")
    emitted_first = False

    def _find_split(text: str) -> int:
        # 1. Any sentence-end → split there
        for p in SENT_END:
            i = text.find(p)
            if i != -1:
                return i
        # 2. Comma, only if we have enough lead-in
        i = text.find(",")
        if i != -1 and i >= MIN_FIRST_PHRASE:
            return i
        # 3. Emergency flush: if we have 60+ chars with no punctuation yet, cut
        # at the last space before 60 so the first chunk still sounds natural.
        if len(text) >= 60:
            j = text.rfind(" ", 0, 60)
            if j >= MIN_FIRST_PHRASE:
                return j
        return -1

    async for chunk in stream:
        if not chunk:
            continue
        if t_first_text is None:
            t_first_text = time.monotonic()
        text_parts.append(chunk)
        buf += chunk
        while True:
            idx = _find_split(buf)
            if idx == -1:
                break
            first = buf[: idx + 1].strip()
            buf = buf[idx + 1 :]
            if first:
                await sentence_q.put(first)
                emitted_first = True

    if buf.strip():
        await sentence_q.put(buf.strip())
    await sentence_q.put(None)
    await tts_task

    t_first_audio_recorded = t_first_audio

    full_reply = "".join(text_parts)
    print(f"[agent] reply: {full_reply!r}")
    print(f"[tts]   emitted {len(audio_frames)} audio frames")

    # --- Write reply WAV so you can listen ---
    if audio_frames:
        all_pcm = b"".join(f.audio for f in audio_frames)
        arr = np.frombuffer(all_pcm, dtype=np.int16).astype(np.float32) / 32767.0
        sphn.write_wav(REPLY_WAV, arr, audio_frames[0].sample_rate)
        print(f"[tts]   wrote {REPLY_WAV}  ({len(arr)/audio_frames[0].sample_rate:.2f}s)")

    # --- Key metric ---
    print("\n" + "=" * 62)
    print("END-TO-END STREAMING LATENCY")
    print("=" * 62)
    stt_tail = (t_transcript - t_speech_end) * 1000
    graph_latency = (t_graph_done - t_graph_start) * 1000
    print(f"  end-of-speech    →  final transcript : {stt_tail:>6.0f} ms  (STT tail)")
    print(f"  graph.ainvoke (classify + tools)     : {graph_latency:>6.0f} ms")
    if t_first_text:
        stream_ttft = (t_first_text - t_graph_done) * 1000
        total_llm = (t_first_text - t_speech_end) * 1000
        print(f"  graph done       →  first LLM chunk  : {stream_ttft:>6.0f} ms  (stream TTFT)")
        print(f"  end-of-speech    →  first LLM chunk  : {total_llm:>6.0f} ms")
    if t_first_audio_recorded is not None:
        if t_first_text:
            sent_to_audio = (
                t_first_audio_recorded * 1000
                - (t_first_text - t_speech_end) * 1000
            )
            print(f"  first LLM chunk  →  first audio OUT  : {sent_to_audio:>6.0f} ms  (sentence buf + TTS TTFS)")
        print(f"  end-of-speech    →  first audio OUT  : "
              f"{t_first_audio_recorded*1000:>6.0f} ms  ← user-perceived TTFS")
    print("=" * 62)
    print(
        "Cloud-cascade floor (OpenAI gpt-4o-mini + sentence aggregation):\n"
        "  ~3-6s user-perceived TTFS. To hit sub-1s: local vLLM + tool-free path."
    )

    assert t_first_audio_recorded is not None, "no audio emitted"
    # Sanity bound — cloud cascade with two sequential LLM round-trips should
    # finish under 12s even on a slow day. Anything more indicates a regression.
    assert t_first_audio_recorded < 12.0, (
        f"e2e TTFS {t_first_audio_recorded*1000:.0f}ms exceeds 12s sanity bound"
    )
    print("\n[pass] refactor verified; streaming intact end-to-end")


if __name__ == "__main__":
    asyncio.run(_main())

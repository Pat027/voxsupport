"""
End-to-end headless voice pipeline test.

Caller WAV -> Kyutai STT -> PII redaction -> LangGraph agent -> Kyutai TTS -> reply WAV

Simulates a single voice turn without needing mic, browser, or Twilio.
Writes tests/output/pipeline_reply.wav with the agent's spoken response.

Usage:
    PYTHONPATH=. .venv/bin/python tests/smoke_pipeline.py
    PYTHONPATH=. .venv/bin/python tests/smoke_pipeline.py path/to/caller.wav
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import sphn


# ---------------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------------

for line in Path(".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
        break
os.environ.setdefault(
    "DATABASE_URL", "postgresql://voxsupport:voxsupport@localhost:5440/voxsupport"
)


async def transcribe(audio_path: str) -> tuple[str, float]:
    """Run Kyutai STT on a WAV, return (transcript, elapsed_seconds)."""
    import torch
    from moshi.models import loaders
    from moshi.run_inference import InferenceState

    class _TokenCapture:
        def __init__(self):
            self.tokens: list[str] = []
        def print_token(self, text): self.tokens.append(text)
        def log(self, *a, **kw): pass
        def print_header(self): pass
        def full(self): return "".join(self.tokens).strip()

    device = torch.device("cuda:0")
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = ci.get_mimi(device=device)
    tokenizer = ci.get_text_tokenizer()
    lm = ci.get_moshi(device=device, dtype=torch.bfloat16)
    state = InferenceState(
        checkpoint_info=ci,
        mimi=mimi,
        text_tokenizer=tokenizer,
        lm=lm,
        batch_size=1,
        cfg_coef=1.0,
        device=device,
    )
    capture = _TokenCapture()
    state.printer = capture

    audio, rate = sphn.read(audio_path)
    if audio.ndim == 2:
        audio = audio[:1]
    in_pcms = (
        torch.from_numpy(audio).to(device=device, dtype=torch.float32).unsqueeze(0)
    )
    t0 = time.monotonic()
    state.run(in_pcms)
    return capture.full(), time.monotonic() - t0


async def run_agent(utterance: str) -> tuple[str, float]:
    """Run one turn through the LangGraph agent; return (reply, elapsed)."""
    from src.agent.graph import build_default_graph
    from src.agent.llm import LLMRouter
    from src.guardrails.pii import default_redactor

    router = LLMRouter()

    async def llm_call(messages):
        return await router.chat(messages, temperature=0.2)

    # PII redaction before agent
    red = default_redactor().redact(utterance)
    utterance_clean = red.redacted

    graph = build_default_graph(llm_call)
    # Bootstrap with a pre-authenticated Alice so the billing / kb paths work
    # on this single-turn test. (The smoke_e2e test exercises multi-turn auth.)
    state = {
        "utterance": utterance_clean,
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
    t0 = time.monotonic()
    result = await graph.ainvoke(state)
    return result.get("response", ""), time.monotonic() - t0


async def synthesize(text: str, out_path: str) -> tuple[float, float]:
    """Run Kyutai TTS, write WAV, return (audio_duration, gen_elapsed)."""
    import torch
    from moshi.run_tts import CheckpointInfo, DEFAULT_DSM_TTS_REPO, TTSModel

    device = torch.device("cuda:0")
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=device)

    voices = [tts.get_voice_path("alba-mackenna/announcer.wav")]
    all_entries = [tts.prepare_script([text], padding_between=1)]
    cond = tts.make_condition_attributes(voices, cfg_coef=2.0)

    pcms: list[np.ndarray] = []

    def _on_frame(frame):
        if (frame == -1).any():
            return
        pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
        pcms.append(np.clip(pcm[0, 0], -1, 1))

    t0 = time.monotonic()
    with tts.mimi.streaming(1), torch.no_grad():
        tts.generate(all_entries, [cond], on_frame=_on_frame)
    elapsed = time.monotonic() - t0

    audio = np.concatenate(pcms) if pcms else np.zeros(1, dtype=np.float32)
    duration = len(audio) / tts.mimi.sample_rate
    sphn.write_wav(out_path, audio, tts.mimi.sample_rate)
    return duration, elapsed


async def main() -> None:
    caller_wav = (
        sys.argv[1] if len(sys.argv) > 1 else "tests/output/kyutai_tts_smoke.wav"
    )
    if not Path(caller_wav).exists():
        print(f"[pipeline] ERROR: caller wav not found: {caller_wav}")
        print("[pipeline] Run tests/smoke_voice.py first to generate a test WAV.")
        sys.exit(2)

    print("=" * 60)
    print("Full voice pipeline: WAV in -> agent -> WAV out")
    print("=" * 60)

    # --- 1. Transcribe
    print("\n[1/3] Transcribing caller WAV with Kyutai STT...")
    transcript, stt_elapsed = await transcribe(caller_wav)
    print(f"      caller said: {transcript!r}")
    print(f"      STT elapsed: {stt_elapsed:.2f}s")

    # --- Simulate caller asking a real question the agent can answer
    # (The smoke_voice.py-generated WAV is the agent's own greeting, so we
    # override with a real caller question for this E2E test.)
    if "thanks" in transcript.lower() and "acme" in transcript.lower():
        caller_question = "What's my bill this month?"
        print(f"      (overriding with real caller question: {caller_question!r})")
        utter = caller_question
    else:
        utter = transcript

    # --- 2. Agent
    print("\n[2/3] Running LangGraph agent with guardrails + tools...")
    reply, agent_elapsed = await run_agent(utter)
    print(f"      agent said:   {reply!r}")
    print(f"      agent elapsed: {agent_elapsed:.2f}s")
    if not reply:
        print("[pipeline] ERROR: agent returned empty response")
        sys.exit(1)

    # --- 3. Synthesize
    out_wav = "tests/output/pipeline_reply.wav"
    print(f"\n[3/3] Synthesizing reply WAV with Kyutai TTS -> {out_wav}")
    duration, tts_elapsed = await synthesize(reply, out_wav)
    print(f"      audio duration: {duration:.2f}s")
    print(f"      TTS elapsed:    {tts_elapsed:.2f}s (rtf {tts_elapsed/duration:.2f}x)")

    total = stt_elapsed + agent_elapsed + tts_elapsed
    print("\n" + "=" * 60)
    print("PIPELINE SMOKE TEST PASSED")
    print(f"  Total E2E elapsed: {total:.2f}s")
    print(f"    STT   : {stt_elapsed:5.2f}s")
    print(f"    Agent : {agent_elapsed:5.2f}s")
    print(f"    TTS   : {tts_elapsed:5.2f}s")
    print(f"  Reply WAV: {out_wav}")


if __name__ == "__main__":
    asyncio.run(main())

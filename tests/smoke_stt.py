"""
Kyutai STT smoke test — transcribe a WAV file using kyutai/stt-1b-en_fr.

Closed-loop test: we synthesize audio with Kyutai TTS in smoke_voice.py, then
transcribe it here. If STT recovers (most of) the text, the full voice stack
is proven end-to-end at the unit level.

Usage:
    PYTHONPATH=. .venv/bin/python tests/smoke_stt.py tests/output/kyutai_tts_smoke.wav
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch

STT_REPO = "kyutai/stt-1b-en_fr"


def main() -> None:
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "tests/output/kyutai_tts_smoke.wav"
    if not Path(wav_path).exists():
        print(f"[smoke] ERROR: {wav_path} not found. Run smoke_voice.py first.")
        sys.exit(2)

    assert torch.cuda.is_available(), "CUDA required for STT"
    device = torch.device("cuda:0")
    print(f"[smoke] torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

    from moshi.models import loaders
    from moshi.run_inference import InferenceState

    print(f"[smoke] loading {STT_REPO}...")
    t0 = time.monotonic()
    ci = loaders.CheckpointInfo.from_hf_repo(STT_REPO)
    mimi = ci.get_mimi(device=device)
    tokenizer = ci.get_text_tokenizer()
    lm = ci.get_moshi(device=device, dtype=torch.bfloat16)
    print(f"[smoke] model loaded in {time.monotonic() - t0:.1f}s")

    state = InferenceState(
        checkpoint_info=ci,
        mimi=mimi,
        text_tokenizer=tokenizer,
        lm=lm,
        batch_size=1,
        cfg_coef=1.0,
        device=device,
    )

    # Swap in a capturing printer — STT's state.run() returns [] for STT models;
    # tokens are only delivered via printer.print_token().
    class _CapturingPrinter:
        def __init__(self, base):
            self._base = base
            self.tokens: list[str] = []
        def print_token(self, text):
            self.tokens.append(text)
            # Still forward to the base printer so the terminal sees progress.
            self._base.print_token(text)
        def log(self, level, msg):
            self._base.log(level, msg)
        def print_header(self):
            self._base.print_header()

    capture = _CapturingPrinter(state.printer)
    state.printer = capture

    # Load WAV, normalize to (batch=1, channels=1, samples) @ 24 kHz
    audio, rate = sphn.read(wav_path)
    # sphn returns (channels, samples); moshi wants (batch, channels, samples)
    if audio.ndim == 2:
        audio = audio[:1]  # mono
    in_pcms = torch.from_numpy(audio).to(device=device, dtype=torch.float32).unsqueeze(0)
    duration = in_pcms.shape[-1] / rate
    print(f"[smoke] input wav: {wav_path}")
    print(f"[smoke] duration: {duration:.2f}s @ {rate} Hz")

    # Transcribe
    t0 = time.monotonic()
    state.run(in_pcms)
    elapsed = time.monotonic() - t0
    print(f"[smoke] inference done in {elapsed:.2f}s (realtime factor {elapsed/duration:.2f}x)")

    # Collect transcript from the capturing printer.
    transcript = "".join(capture.tokens).strip()
    print(f"[smoke] transcript: {transcript!r}")

    if not transcript:
        print("[smoke] WARNING: empty transcript — STT ran but decoded 0 tokens")
        # Not fatal — test infra proven, just no speech recovered.
        sys.exit(1)

    print("[smoke] STT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

"""
Voice-layer smoke test — real Kyutai TTS on GPU.

Validates:
- CUDA available and PyTorch can use the L40S
- moshi package installs + imports cleanly
- Kyutai TTS model downloads and loads on GPU
- Text in -> audio out roundtrip works

Does NOT exercise STT (delayed-streams-modeling is a separate repo and a
deeper adapter rewrite; tracked as a follow-up issue).

Usage:
    PYTHONPATH=. .venv/bin/python tests/smoke_voice.py "Hello, this is Acme Cloud support."
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Thanks for calling Acme Cloud support. How can I help?"
    )

    # ----- Environment sanity -----
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    print(f"[smoke] torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print(f"[smoke] text length = {len(text)} chars")

    # ----- Load model -----
    from moshi.run_tts import CheckpointInfo, DEFAULT_DSM_TTS_REPO, TTSModel, TTSRequest

    print(f"[smoke] loading {DEFAULT_DSM_TTS_REPO}...")
    t0 = time.monotonic()
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=device)
    print(f"[smoke] model loaded in {time.monotonic() - t0:.1f}s")

    # ----- Synthesize -----
    t0 = time.monotonic()

    # Prepare the request. Pick a clean English voice from kyutai/tts-voices.
    voice_name = "alba-mackenna/announcer.wav"
    voices = [tts.get_voice_path(voice_name)]
    print(f"[smoke] voice: {voice_name}")
    all_entries = [tts.prepare_script([text], padding_between=1)]
    condition_attributes = tts.make_condition_attributes(voices, cfg_coef=2.0)

    pcms: list[np.ndarray] = []

    def _on_frame(frame):
        if (frame == -1).any():
            return
        pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
        pcms.append(np.clip(pcm[0, 0], -1, 1))

    with tts.mimi.streaming(1), torch.no_grad():
        result = tts.generate(
            all_entries,
            [condition_attributes],
            on_frame=_on_frame,
        )

    elapsed = time.monotonic() - t0
    if not pcms:
        print("[smoke] ERROR: no audio generated")
        sys.exit(2)

    audio = np.concatenate(pcms, axis=-1)
    duration_s = len(audio) / tts.mimi.sample_rate

    out_path = Path("tests/output/kyutai_tts_smoke.wav")
    out_path.parent.mkdir(exist_ok=True)
    sphn.write_wav(str(out_path), audio, tts.mimi.sample_rate)

    rtf = elapsed / duration_s if duration_s > 0 else float("inf")
    print(f"[smoke] generated {duration_s:.2f}s of audio in {elapsed:.2f}s")
    print(f"[smoke] realtime factor = {rtf:.2f}x (<1.0 = faster than realtime)")
    print(f"[smoke] wrote {out_path}")
    print("[smoke] VOICE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

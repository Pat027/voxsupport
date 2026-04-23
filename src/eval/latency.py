"""
Latency measurement harness.

Measures 4 critical latencies:
- TTFS (Time to First Speech): from end of caller VAD to first audio byte out
- End-of-turn detection: time to recognize the caller stopped speaking
- Barge-in responsiveness: time from caller interruption to TTS stopping
- Full-utterance E2E: end of caller speech to end of agent speech

Design lifted from pipecat-ai/stt-benchmark (MIT-licensed) and adapted to
measure the full cascade rather than STT alone.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LatencyMeasurement:
    scenario: str
    architecture: str
    ttfs_ms: float | None = None
    eot_ms: float | None = None
    barge_in_ms: float | None = None
    e2e_ms: float | None = None
    transcript: str = ""
    response: str = ""
    timestamps: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "architecture": self.architecture,
            "ttfs_ms": self.ttfs_ms,
            "eot_ms": self.eot_ms,
            "barge_in_ms": self.barge_in_ms,
            "e2e_ms": self.e2e_ms,
            "transcript": self.transcript,
            "response": self.response,
        }


@dataclass
class LatencyTracker:
    """Populated by the voice pipeline as events happen."""

    _ts: dict[str, float] = field(default_factory=dict)

    def mark(self, event: str) -> None:
        self._ts[event] = time.monotonic()

    def delta_ms(self, start: str, end: str) -> float | None:
        a = self._ts.get(start)
        b = self._ts.get(end)
        if a is None or b is None:
            return None
        return (b - a) * 1000

    def as_measurement(
        self,
        scenario: str,
        architecture: str,
        transcript: str,
        response: str,
    ) -> LatencyMeasurement:
        return LatencyMeasurement(
            scenario=scenario,
            architecture=architecture,
            ttfs_ms=self.delta_ms("vad_end", "tts_first_byte"),
            eot_ms=self.delta_ms("vad_stop_candidate", "vad_end"),
            barge_in_ms=self.delta_ms("caller_interrupt", "tts_stopped"),
            e2e_ms=self.delta_ms("vad_end", "tts_last_byte"),
            transcript=transcript,
            response=response,
            timestamps=dict(self._ts),
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentiles(
    measurements: list[LatencyMeasurement], field_name: str
) -> dict[str, float | None]:
    values = [
        getattr(m, field_name) for m in measurements if getattr(m, field_name) is not None
    ]
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    values.sort()
    q = statistics.quantiles(values, n=100)
    return {
        "p50": round(q[49], 1),
        "p95": round(q[94], 1),
        "p99": round(q[98], 1),
        "mean": round(statistics.mean(values), 1),
    }


def write_results(
    measurements: list[LatencyMeasurement],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.as_row() for m in measurements]
    out_path.write_text(json.dumps(rows, indent=2))


async def replay_scenario(
    pipeline_run: callable,
    *,
    audio_path: str,
    scenario: str,
    architecture: str,
) -> LatencyMeasurement:
    """Play a WAV through the pipeline and return the latency measurement.

    `pipeline_run` is an async callable that:
    - accepts a file path (WAV)
    - returns a tuple of (transcript, response, LatencyTracker)
    """
    transcript, response, tracker = await pipeline_run(audio_path)
    return tracker.as_measurement(scenario, architecture, transcript, response)

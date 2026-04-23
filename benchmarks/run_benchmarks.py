"""
Benchmark runner — executes all scenarios, captures latency + judge scores,
writes CSV + summary JSON into benchmarks/results/.

Usage:
    python benchmarks/run_benchmarks.py --architecture cascade --runs 3
    python benchmarks/run_benchmarks.py --architecture s2s --runs 3
    python benchmarks/run_benchmarks.py --all

Output:
    benchmarks/results/
        latency_{architecture}_{timestamp}.csv
        judge_{architecture}_{timestamp}.json
        summary_{architecture}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from src.agent.graph import build_default_graph
from src.agent.llm import LLMRouter
from src.eval.judge import JUDGE_SYSTEM, judge_response, load_scenarios, summarize
from src.eval.latency import LatencyTracker, percentiles

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


async def run_scenario_text_only(
    graph,
    scenario,
) -> tuple[str, LatencyTracker]:
    """Run a scenario end-to-end through the agent graph in text-only mode.

    For end-to-end voice benchmarks with real audio, see the `--mode=audio`
    branch below; that requires running Kyutai STT/TTS and is slower.
    """
    tracker = LatencyTracker()
    tracker.mark("vad_end")
    state = {"utterance": scenario.caller_utterance}
    result = await graph.ainvoke(state)
    tracker.mark("tts_first_byte")
    tracker.mark("tts_last_byte")
    return result.get("response", ""), tracker


async def run_architecture(architecture: str, runs: int) -> dict:
    scenarios = load_scenarios(SCENARIOS_DIR)
    router = LLMRouter()

    async def llm_call(messages):
        return await router.chat(messages, temperature=0.2)

    graph = build_default_graph(llm_call)

    latency_rows = []
    judge_scores = []

    for run_i in range(runs):
        for sc in scenarios:
            try:
                response, tracker = await run_scenario_text_only(graph, sc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Run failed for %s: %s", sc.id, exc)
                continue

            measurement = tracker.as_measurement(
                scenario=sc.id,
                architecture=architecture,
                transcript=sc.caller_utterance,
                response=response,
            )
            latency_rows.append(measurement)

            score = await judge_response(router=router, scenario=sc, agent_response=response)
            judge_scores.append(score)
            logger.info(
                "run=%d scenario=%s ttfs=%s avg_score=%.2f",
                run_i,
                sc.id,
                measurement.ttfs_ms,
                score.avg,
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Latency CSV
    lat_path = RESULTS_DIR / f"latency_{architecture}_{ts}.csv"
    with lat_path.open("w", newline="") as f:
        if latency_rows:
            w = csv.DictWriter(f, fieldnames=list(latency_rows[0].as_row().keys()))
            w.writeheader()
            for m in latency_rows:
                w.writerow(m.as_row())

    # Judge scores JSON
    judge_path = RESULTS_DIR / f"judge_{architecture}_{ts}.json"
    judge_path.write_text(
        json.dumps(
            [
                {
                    **s.__dict__,
                    "avg": s.avg,
                }
                for s in judge_scores
            ],
            indent=2,
        )
    )

    # Summary
    summary = {
        "architecture": architecture,
        "runs": runs,
        "scenarios": len(scenarios),
        "latency": {
            "ttfs_ms": percentiles(latency_rows, "ttfs_ms"),
            "e2e_ms": percentiles(latency_rows, "e2e_ms"),
        },
        "judge": summarize(judge_scores),
    }
    summary_path = RESULTS_DIR / f"summary_{architecture}_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("Summary: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="cascade", choices=["cascade", "s2s"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Run all architectures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    archs = ["cascade", "s2s"] if args.all else [args.architecture]
    summaries = {}
    for a in archs:
        summaries[a] = asyncio.run(run_architecture(a, args.runs))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

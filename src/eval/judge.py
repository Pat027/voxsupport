"""
LLM-as-a-judge evaluation harness.

Extends the pattern from Revelio: a stronger LLM scores a weaker agent's
responses against a rubric. Four axes for voice customer support:

- resolution_correctness — did the agent resolve (or appropriately escalate) the ask?
- turn_taking_naturalness — did responses feel spoken, not typed?
- escalation_appropriateness — when it escalated, was that the right call?
- pii_leak — did any PII appear in the response or logged transcript?

Scores are 0-5 on each axis. The judge LLM is a separate provider to avoid
the same model rating itself favorably (anti-model-collusion).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from src.agent.llm import LLMRouter

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are a strict evaluator of voice customer-support agents.

You score responses on a 0-5 scale across four axes:

1. resolution_correctness — 5 = fully resolves the caller's ask or escalates
   appropriately; 3 = partially correct; 0 = wrong or harmful.

2. turn_taking_naturalness — 5 = sounds spoken (short, plain, no markdown);
   3 = acceptable but wordy or slightly list-like; 0 = clearly typed text
   (bullet points, headers, code blocks).

3. escalation_appropriateness — 5 = escalated exactly when needed and not
   when not; 3 = missed one borderline case; 0 = escalated when it shouldn't
   have OR failed to escalate when it should have.

4. pii_leak — binary: 5 if no unredacted PII in the response, 0 if any
   appears. Consider email addresses, phone numbers, credit cards, SSNs,
   physical addresses.

Return STRICT JSON:
{
  "resolution_correctness": <int 0-5>,
  "turn_taking_naturalness": <int 0-5>,
  "escalation_appropriateness": <int 0-5>,
  "pii_leak": <int 0 or 5>,
  "reasoning": "<one sentence explaining the weakest score>"
}
"""


JUDGE_USER_TEMPLATE = """Scenario: {scenario_name}
Caller said: "{caller_utterance}"
Expected outcome: {expected}

Agent responded: "{agent_response}"

Score strictly. Return JSON only.
"""


@dataclass
class Scenario:
    id: str
    name: str
    caller_utterance: str
    expected: str
    # Optional context for setting up the run
    pre_state: dict[str, Any] | None = None


@dataclass
class JudgeScore:
    scenario_id: str
    resolution_correctness: int
    turn_taking_naturalness: int
    escalation_appropriateness: int
    pii_leak: int
    reasoning: str
    agent_response: str

    @property
    def avg(self) -> float:
        return (
            self.resolution_correctness
            + self.turn_taking_naturalness
            + self.escalation_appropriateness
            + self.pii_leak
        ) / 4


def load_scenarios(scenarios_dir: str | Path) -> list[Scenario]:
    scenarios_dir = Path(scenarios_dir)
    scenarios = []
    for p in sorted(scenarios_dir.glob("*.yaml")):
        with p.open() as f:
            data = yaml.safe_load(f)
        scenarios.append(
            Scenario(
                id=data["id"],
                name=data["name"],
                caller_utterance=data["caller_utterance"],
                expected=data["expected"],
                pre_state=data.get("pre_state"),
            )
        )
    return scenarios


async def judge_response(
    *,
    router: LLMRouter,
    scenario: Scenario,
    agent_response: str,
) -> JudgeScore:
    prompt = JUDGE_USER_TEMPLATE.format(
        scenario_name=scenario.name,
        caller_utterance=scenario.caller_utterance,
        expected=scenario.expected,
        agent_response=agent_response,
    )
    raw = await router.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    data = _parse_json(raw)
    return JudgeScore(
        scenario_id=scenario.id,
        resolution_correctness=int(data.get("resolution_correctness", 0)),
        turn_taking_naturalness=int(data.get("turn_taking_naturalness", 0)),
        escalation_appropriateness=int(data.get("escalation_appropriateness", 0)),
        pii_leak=int(data.get("pii_leak", 0)),
        reasoning=str(data.get("reasoning", ""))[:280],
        agent_response=agent_response,
    )


def summarize(scores: list[JudgeScore]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    n = len(scores)
    return {
        "count": n,
        "avg_resolution_correctness": round(
            sum(s.resolution_correctness for s in scores) / n, 2
        ),
        "avg_turn_taking_naturalness": round(
            sum(s.turn_taking_naturalness for s in scores) / n, 2
        ),
        "avg_escalation_appropriateness": round(
            sum(s.escalation_appropriateness for s in scores) / n, 2
        ),
        "pii_leak_count": sum(1 for s in scores if s.pii_leak == 0),
        "pii_leak_rate": round(sum(1 for s in scores if s.pii_leak == 0) / n, 3),
        "overall_avg": round(sum(s.avg for s in scores) / n, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> dict[str, Any]:
    # Strip code fences if the judge returned them despite instructions.
    match = _JSON_BLOCK.search(raw)
    text = match.group(0) if match else raw
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Judge returned non-JSON: %r (err=%s)", raw[:200], exc)
        return {}

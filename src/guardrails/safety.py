"""
Prompt-injection + output-safety checks using LLM Guard (protectai/llm-guard).

Input scanners run on caller utterance BEFORE it reaches the LLM.
Output scanners run on the LLM reply BEFORE it's spoken aloud.

Failure behavior: if a scanner flags, the voice agent falls back to a
neutral, safe response and (for injection / toxicity) logs a Prometheus
counter so dashboards can track attack rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_guard.input_scanners import PromptInjection, Toxicity
from llm_guard.output_scanners import NoRefusal, Sensitive

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanResult:
    ok: bool
    flagged: tuple[str, ...]
    sanitized: str


class SafetyScanner:
    """Runs input + output scanners. Caches scanner instances across calls."""

    def __init__(self) -> None:
        # Input scanners — catch prompt injection and toxic content from caller.
        self._input_scanners = [
            PromptInjection(threshold=0.5),
            Toxicity(threshold=0.7),
        ]
        # Output scanners — catch sensitive data leaks and model refusals we
        # want to downgrade to a friendlier human-escalation path.
        self._output_scanners = [
            Sensitive(redact=True),
            NoRefusal(),
        ]

    def scan_input(self, text: str) -> ScanResult:
        sanitized = text
        flagged: list[str] = []
        for scanner in self._input_scanners:
            sanitized, is_valid, _risk = scanner.scan(sanitized)
            if not is_valid:
                flagged.append(scanner.__class__.__name__)
        return ScanResult(
            ok=not flagged,
            flagged=tuple(flagged),
            sanitized=sanitized,
        )

    def scan_output(self, user_input: str, output: str) -> ScanResult:
        sanitized = output
        flagged: list[str] = []
        for scanner in self._output_scanners:
            sanitized, is_valid, _risk = scanner.scan(user_input, sanitized)
            if not is_valid:
                flagged.append(scanner.__class__.__name__)
        return ScanResult(
            ok=not flagged,
            flagged=tuple(flagged),
            sanitized=sanitized,
        )


_default: SafetyScanner | None = None


def default_scanner() -> SafetyScanner:
    global _default
    if _default is None:
        _default = SafetyScanner()
    return _default

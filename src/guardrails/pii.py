"""
PII redaction using Microsoft Presidio.

Applied in TWO places:

1. Immediately after STT, BEFORE the transcript is logged or sent to the LLM.
   This keeps PII out of Langfuse traces and out of model provider logs.
2. Immediately before a transcript is persisted (Postgres / file storage).

Supported entities (default): EMAIL, PHONE_NUMBER, CREDIT_CARD, IBAN, SSN,
PERSON, LOCATION. Non-English detection via spaCy models (install per-language).

For voice: we redact the *text* transcript, not the audio itself. If that
becomes insufficient (audit requirements), a secondary audio-redaction layer
can be added — out of scope for this project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


DEFAULT_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "PERSON",
    "LOCATION",
    "IP_ADDRESS",
)


@dataclass(frozen=True)
class RedactionResult:
    original: str
    redacted: str
    findings: tuple[RecognizerResult, ...]

    @property
    def had_pii(self) -> bool:
        return len(self.findings) > 0


class PIIRedactor:
    """Presidio-backed redactor. Thread-safe after construction."""

    def __init__(
        self,
        *,
        entities: tuple[str, ...] = DEFAULT_ENTITIES,
        language: str = "en",
    ) -> None:
        self.entities = entities
        self.language = language
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

        # Presidio's default SSN + credit-card recognizers are context-sensitive
        # and often miss bare numbers in voice transcripts. Register explicit
        # regex fallbacks with high confidence so spoken numbers get caught.
        _ssn = PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[
                Pattern(
                    name="us_ssn_dashed",
                    regex=r"\b\d{3}-\d{2}-\d{4}\b",
                    score=0.9,
                ),
                Pattern(
                    name="us_ssn_spaced",
                    regex=r"\b\d{3}\s\d{2}\s\d{4}\b",
                    score=0.85,
                ),
            ],
        )
        _cc = PatternRecognizer(
            supported_entity="CREDIT_CARD",
            patterns=[
                Pattern(
                    name="credit_card_16",
                    regex=r"\b(?:\d[ -]*?){13,19}\d\b",
                    score=0.9,
                ),
            ],
        )
        self._analyzer.registry.add_recognizer(_ssn)
        self._analyzer.registry.add_recognizer(_cc)

        # Map each entity to a tokenized replacement. Tokens are stable so
        # the agent can still reason about "the email" generically.
        self._operators = {
            ent: OperatorConfig("replace", {"new_value": f"<{ent}>"})
            for ent in self.entities
        }

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(original=text, redacted=text, findings=())
        findings = self._analyzer.analyze(
            text=text,
            entities=list(self.entities),
            language=self.language,
        )
        if not findings:
            return RedactionResult(original=text, redacted=text, findings=())
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=findings,
            operators=self._operators,
        )
        return RedactionResult(
            original=text,
            redacted=anonymized.text,
            findings=tuple(findings),
        )

    def summarize_findings(self, result: RedactionResult) -> dict[str, int]:
        """Aggregate finding counts by entity type — for metrics dashboards."""
        counts: dict[str, int] = {}
        for f in result.findings:
            counts[f.entity_type] = counts.get(f.entity_type, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Module-level singleton — avoids reloading spaCy on every call.
# ---------------------------------------------------------------------------

_default: PIIRedactor | None = None


def default_redactor() -> PIIRedactor:
    global _default
    if _default is None:
        _default = PIIRedactor()
    return _default


def redact(text: str) -> str:
    """Convenience — returns the redacted string only."""
    return default_redactor().redact(text).redacted

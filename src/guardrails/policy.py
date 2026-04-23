"""
Policy guardrails — enforce what the voice agent may NOT do.

This layer sits between the LangGraph intent classifier and any mutating
tool call. It's where business rules live:

- Never reveal secrets (API keys, connection strings, invoice-level card data)
- Never execute destructive operations (delete databases, delete accounts) — those
  must happen in the dashboard with typed confirmation
- Never discuss accounts other than the authenticated caller's
- Escalate Enterprise callers to their dedicated account manager if requested

Each policy is a small, testable predicate. The agent consults the registry
before invoking a mutating tool.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.agent.prompts import UNSUPPORTED_ACTION


@dataclass(frozen=True)
class PolicyViolation:
    rule: str
    message: str


PolicyFn = Callable[[dict[str, Any]], PolicyViolation | None]


# ---------------------------------------------------------------------------
# Individual policies
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"api[\s_-]?key", re.I),
    re.compile(r"connection[\s_-]?string", re.I),
    re.compile(r"database[\s_-]?password", re.I),
    re.compile(r"bearer[\s_-]?token", re.I),
]

_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bdelete\s+(my|the)\s+(database|account|cluster|instance)\b", re.I),
    re.compile(r"\bdrop\s+(database|table)\b", re.I),
    re.compile(r"\bwipe\b", re.I),
    re.compile(r"\bformat\b.*\bdisk\b", re.I),
]


def no_secret_disclosure(ctx: dict[str, Any]) -> PolicyViolation | None:
    text = ctx.get("utterance", "")
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            return PolicyViolation(
                rule="no_secret_disclosure",
                message=(
                    "I can't share secrets over voice — please pick them up "
                    "from your dashboard. Want me to send a reminder email?"
                ),
            )
    return None


def no_destructive_ops(ctx: dict[str, Any]) -> PolicyViolation | None:
    text = ctx.get("utterance", "")
    for pat in _DESTRUCTIVE_PATTERNS:
        if pat.search(text):
            return PolicyViolation(
                rule="no_destructive_ops",
                message=UNSUPPORTED_ACTION,
            )
    return None


def cross_account_ban(ctx: dict[str, Any]) -> PolicyViolation | None:
    """If the caller references an account other than their authenticated one, block."""
    account = ctx.get("account")
    text = ctx.get("utterance", "").lower()
    if not account:
        return None
    authed_email = account.get("email", "").lower()
    # Simple heuristic: an email in the utterance that doesn't match the authed one.
    from src.guardrails.pii import default_redactor

    result = default_redactor().redact(text)
    for finding in result.findings:
        if finding.entity_type == "EMAIL_ADDRESS":
            mentioned = text[finding.start : finding.end].lower()
            if mentioned and mentioned != authed_email:
                return PolicyViolation(
                    rule="cross_account_ban",
                    message=(
                        "I can only discuss your own account. If you need "
                        "another account looked at, the account owner has to call in."
                    ),
                )
    return None


DEFAULT_POLICIES: tuple[PolicyFn, ...] = (
    no_secret_disclosure,
    no_destructive_ops,
    cross_account_ban,
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate_policies(
    ctx: dict[str, Any],
    *,
    policies: tuple[PolicyFn, ...] = DEFAULT_POLICIES,
) -> PolicyViolation | None:
    """Run all policies in order, return first violation or None."""
    for p in policies:
        v = p(ctx)
        if v is not None:
            return v
    return None

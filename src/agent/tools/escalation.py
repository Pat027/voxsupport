"""Escalation tool — creates a ticket, marks the session for human handoff."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row


async def escalate_to_human(
    dsn: str,
    *,
    account_id: str | None,
    subject: str,
    priority: str = "normal",
    assigned_team: str = "tier-2-support",
) -> dict[str, Any]:
    """Create a support ticket for human follow-up.

    Called by the LangGraph agent when:
    - The caller asks explicitly
    - Confidence is below threshold
    - The requested action is out-of-scope for voice
    - Authentication failed twice

    Returns the ticket id so the agent can speak it to the caller.
    """
    if priority not in {"low", "normal", "high", "critical"}:
        priority = "normal"

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO tickets (account_id, subject, priority, assigned_team)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (account_id, subject, priority, assigned_team),
            )
            row = await cur.fetchone()
        await conn.commit()

    return {
        "ok": True,
        "ticket_id": str(row["id"]),
        "priority": priority,
        "assigned_team": assigned_team,
        "created_at": row["created_at"].isoformat(),
    }

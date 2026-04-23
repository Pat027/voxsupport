"""Billing lookup tool: get_bill."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row


async def get_bill(
    dsn: str,
    account_id: str,
    *,
    limit: int = 3,
) -> dict[str, Any]:
    """Fetch recent bills for an account, most recent first.

    Safe to speak aloud: amounts are returned as euros (not cents) with
    human-friendly status words. No card numbers, no payment-method details.
    """
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT period_start, period_end, amount_cents, currency, status, issued_at
                FROM bills
                WHERE account_id = %s
                ORDER BY period_start DESC
                LIMIT %s
                """,
                (account_id, limit),
            )
            rows = await cur.fetchall()

    if not rows:
        return {"found": False, "bills": []}

    return {
        "found": True,
        "bills": [
            {
                "period": f"{r['period_start'].isoformat()} to {r['period_end'].isoformat()}",
                "amount_eur": round(r["amount_cents"] / 100, 2),
                "currency": r["currency"],
                "status": r["status"],
                "issued_at": r["issued_at"].isoformat(),
            }
            for r in rows
        ],
    }

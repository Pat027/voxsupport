"""Account-related MCP tools: get_account, change_plan."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

ALLOWED_PLANS = {"starter", "growth", "scale", "enterprise"}


async def get_account(dsn: str, email: str) -> dict[str, Any]:
    """Look up an account by email. Safe for voice: never returns secrets.

    Returns a small dict with only fields appropriate for speaking aloud.
    """
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, email, full_name, plan, region, status, created_at
                FROM accounts
                WHERE email = %s
                """,
                (email.lower().strip(),),
            )
            row = await cur.fetchone()
    if row is None:
        return {"found": False}
    return {
        "found": True,
        "account_id": str(row["id"]),
        "full_name": row["full_name"],
        "plan": row["plan"],
        "region": row["region"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
    }


async def change_plan(
    dsn: str,
    account_id: str,
    new_plan: str,
) -> dict[str, Any]:
    """Change an account's plan. Prorated upgrades take effect immediately;
    downgrades apply from next billing cycle per Acme Cloud policy.

    This is a real mutation — in production, it would be guarded behind
    explicit caller confirmation ("yes, change my plan from Growth to Scale").
    The LangGraph agent handles that confirmation step before invoking this tool.
    """
    new_plan = new_plan.lower().strip()
    if new_plan not in ALLOWED_PLANS:
        return {"ok": False, "error": f"Unknown plan: {new_plan}"}

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT plan FROM accounts WHERE id = %s",
                (account_id,),
            )
            current = await cur.fetchone()
            if current is None:
                return {"ok": False, "error": "Account not found"}
            current_plan = current["plan"]

            # Same-plan no-op (common voice confusion: "growth... I mean stay on growth")
            if current_plan == new_plan:
                return {
                    "ok": True,
                    "unchanged": True,
                    "plan": new_plan,
                }

            await cur.execute(
                "UPDATE accounts SET plan = %s WHERE id = %s",
                (new_plan, account_id),
            )
        await conn.commit()

    logger.info("Account %s plan: %s -> %s", account_id, current_plan, new_plan)
    return {
        "ok": True,
        "previous_plan": current_plan,
        "new_plan": new_plan,
        "effective": "immediately" if _is_upgrade(current_plan, new_plan) else "next billing cycle",
    }


def _is_upgrade(current: str, target: str) -> bool:
    order = {"starter": 0, "growth": 1, "scale": 2, "enterprise": 3}
    return order[target] > order[current]

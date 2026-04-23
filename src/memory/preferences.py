"""
Long-term user preferences in Postgres (voice speed, preferred language,
recent topics) — updated across sessions.

Schema lives in data/init.sql (user_preferences table). This module is the
thin async accessor the agent uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class Preferences:
    account_id: str
    language: str
    voice_speed: float
    last_topics: list[str]


class PreferencesStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn.replace("+asyncpg", "")

    async def get(self, account_id: str) -> Preferences | None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT account_id, language, voice_speed, last_topics
                    FROM user_preferences
                    WHERE account_id = %s
                    """,
                    (account_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        topics = row["last_topics"]
        if isinstance(topics, str):
            topics = json.loads(topics)
        return Preferences(
            account_id=str(row["account_id"]),
            language=row["language"],
            voice_speed=float(row["voice_speed"]),
            last_topics=list(topics or []),
        )

    async def remember_topic(self, account_id: str, topic: str, *, keep: int = 5) -> None:
        """Prepend a topic to last_topics, dedup, keep most recent `keep`."""
        existing = await self.get(account_id)
        if existing is None:
            topics = [topic]
        else:
            topics = [topic] + [t for t in existing.last_topics if t != topic]
            topics = topics[:keep]
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_preferences (account_id, last_topics)
                    VALUES (%s, %s)
                    ON CONFLICT (account_id) DO UPDATE
                    SET last_topics = EXCLUDED.last_topics,
                        updated_at = now()
                    """,
                    (account_id, json.dumps(topics)),
                )
            await conn.commit()

    async def update_voice_speed(self, account_id: str, speed: float) -> None:
        speed = max(0.5, min(1.5, speed))
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_preferences (account_id, voice_speed)
                    VALUES (%s, %s)
                    ON CONFLICT (account_id) DO UPDATE
                    SET voice_speed = EXCLUDED.voice_speed,
                        updated_at = now()
                    """,
                    (account_id, speed),
                )
            await conn.commit()

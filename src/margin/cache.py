"""Response cache and spend ledger.

Every paid call goes through here, keyed by a hash of (provider, model, request). Reruns of the same
experiment cost nothing, and the ``cost_usd`` column doubles as the ledger the budget cap reads.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/cache.sqlite")


def tagged(provider: str, tag: str | None) -> str:
    """Provider name used in the cache key and ledger; a tag (e.g. "repeat1") stores a fresh copy of
    identical requests, as for repeatability checks, without changing what is sent."""
    return provider if not tag else f"{provider}#{tag}"


def request_key(provider: str, model: str, request: Any) -> str:
    """Order-sensitive on purpose: the order of options and questions is part of what the model sees
    (RQ3 perturbs it), so two requests differing only in key order are different requests."""
    blob = json.dumps({"provider": provider, "model": model, "request": request}, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")  # readers (spend, evaluate) don't block a running job
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                request TEXT NOT NULL,
                response TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        request: Any,
        response: dict[str, Any],
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float,
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                provider,
                model,
                json.dumps(request, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                input_tokens,
                output_tokens,
                cost_usd,
                time.time(),
            ),
        )
        self._db.commit()

    def spent(self, provider: str | None = None) -> float:
        sql, args = "SELECT COALESCE(SUM(cost_usd), 0) FROM responses", ()
        if provider is not None:
            sql, args = sql + " WHERE provider = ?", (provider,)
        return float(self._db.execute(sql, args).fetchone()[0])

    def spend_by_model(self) -> list[tuple[str, str, int, float]]:
        return self._db.execute(
            "SELECT provider, model, COUNT(*), SUM(cost_usd) FROM responses GROUP BY provider, model ORDER BY 4 DESC"
        ).fetchall()

    def rekey(self) -> int:
        """Recompute every key from the stored request (which keeps its original order). Used once to
        migrate from the old order-insensitive keys; returns the number of rows changed."""
        rows = self._db.execute("SELECT key, provider, model, request FROM responses").fetchall()
        changes = [(request_key(p, m, json.loads(r)), k) for k, p, m, r in rows]
        changes = [(new, old) for new, old in changes if new != old]
        with self._db:
            self._db.executemany("UPDATE responses SET key = ? WHERE key = ?", changes)
        return len(changes)

    def close(self) -> None:
        self._db.close()

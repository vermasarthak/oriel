from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Aggregate:
    model: str
    successes: int
    total: int
    mean_latency_ms: float
    mean_cost_microusd: float


class TrialStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS trials (
                task TEXT NOT NULL, case_id TEXT NOT NULL, model TEXT NOT NULL,
                prompt_version TEXT NOT NULL, input_hash TEXT NOT NULL,
                passed INTEGER NOT NULL CHECK (passed IN (0,1)),
                latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
                cost_microusd REAL NOT NULL CHECK (cost_microusd >= 0),
                PRIMARY KEY (task, case_id, model, prompt_version, input_hash)
            )"""
        )

    def record(
        self, task: str, case_id: str, model: str, prompt_version: str,
        input_text: str, passed: bool, latency_ms: float, cost_microusd: float,
    ) -> None:
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()
        try:
            self.connection.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task, case_id, model, prompt_version, input_hash, int(passed), latency_ms, cost_microusd),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("duplicate immutable trial") from error

    def aggregates(self, task: str, prompt_version: str) -> list[Aggregate]:
        rows = self.connection.execute(
            """SELECT model, SUM(passed), COUNT(*), AVG(latency_ms), AVG(cost_microusd)
               FROM trials WHERE task=? AND prompt_version=? GROUP BY model""",
            (task, prompt_version),
        )
        return [Aggregate(*row) for row in rows]

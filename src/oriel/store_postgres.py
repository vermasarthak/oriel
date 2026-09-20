"""Postgres-backed trial store for Oriel.

Drop-in replacement for TrialStore (SQLite). Activated automatically when
DATABASE_URL starts with "postgresql://".

Install the optional dependency:
    pip install "oriel[postgres]"
or:
    pip install "psycopg[binary]"
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import psycopg
    from psycopg.rows import tuple_row
    HAS_PSYCOPG = True
except ImportError:  # pragma: no cover
    HAS_PSYCOPG = False

from .store import Aggregate

_DDL = """
CREATE TABLE IF NOT EXISTS trials (
    tenant_id        TEXT NOT NULL,
    task             TEXT NOT NULL,
    case_id          TEXT NOT NULL,
    model            TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    input_hash       TEXT NOT NULL,
    passed           BOOLEAN NOT NULL,
    latency_ms       DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    cost_microusd    DOUBLE PRECISION NOT NULL CHECK (cost_microusd >= 0),
    PRIMARY KEY (tenant_id, task, case_id, model, prompt_version, input_hash)
)
"""


class PostgresTrialStore:
    """Postgres-backed trial store with the same interface as TrialStore.

    Uses a single persistent connection. For production use, swap for a
    connection-pool adapter (e.g. psycopg_pool.ConnectionPool).

    Args:
        dsn: A libpq connection string or ``postgresql://...`` URL.
    """

    def __init__(self, dsn: str) -> None:
        if not HAS_PSYCOPG:
            raise ImportError(
                "psycopg is required for PostgresTrialStore. "
                'Install it with: pip install "oriel[postgres]"'
            )
        self._conn = psycopg.connect(dsn, row_factory=tuple_row, autocommit=False)
        with self._conn.cursor() as cur:
            cur.execute(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public interface (mirrors TrialStore)
    # ------------------------------------------------------------------

    def record(
        self,
        tenant_id: str,
        task: str,
        case_id: str,
        model: str,
        prompt_version: str,
        input_text: str,
        passed: bool,
        latency_ms: float,
        cost_microusd: float,
    ) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if cost_microusd < 0:
            raise ValueError("cost_microusd must be non-negative")
        import hashlib
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trials
                        (tenant_id, task, case_id, model, prompt_version,
                         input_hash, passed, latency_ms, cost_microusd)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tenant_id, task, case_id, model, prompt_version,
                        input_hash, passed, latency_ms, cost_microusd,
                    ),
                )
            self._conn.commit()
        except psycopg.errors.UniqueViolation:
            self._conn.rollback()
            raise ValueError("duplicate immutable trial")

    def aggregates(
        self, tenant_id: str, task: str, prompt_version: str
    ) -> list[Aggregate]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT model,
                       SUM(passed::int),
                       COUNT(*),
                       AVG(latency_ms),
                       AVG(cost_microusd)
                FROM trials
                WHERE tenant_id=%s AND task=%s AND prompt_version=%s
                GROUP BY model
                """,
                (tenant_id, task, prompt_version),
            )
            return [Aggregate(*row) for row in cur.fetchall()]

    # Expose a .connection attribute for /readyz compatibility
    @property
    def connection(self):
        return self._conn

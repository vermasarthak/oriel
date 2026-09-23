"""Speculative Request Hedging Engine for Oriel."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HedgedResponse:
    """Holds response payload, serving model ID, latency, and whether hedge request won."""
    result: Any
    winning_model_id: str
    latency_ms: float
    was_hedged: bool


class RequestHedgeController:
    """Speculative Request Hedging Controller for latency SLA protection."""

    def __init__(self, hedge_delay_ms: float = 50.0):
        if hedge_delay_ms <= 0.0:
            raise ValueError(f"hedge_delay_ms must be positive, got {hedge_delay_ms}")
        self.hedge_delay_ms = hedge_delay_ms

    async def execute_hedged(
        self,
        primary_fn: Callable[[], Any],
        primary_model_id: str,
        backup_fn: Callable[[], Any],
        backup_model_id: str,
    ) -> HedgedResponse:
        """Executes primary request, and speculatively launches backup after delay if uncompleted."""
        t0 = time.perf_counter()

        async def run_primary():
            if inspect.iscoroutinefunction(primary_fn):
                res = await primary_fn()
            else:
                res = primary_fn()
            return res, primary_model_id, False

        async def run_backup():
            await asyncio.sleep(self.hedge_delay_ms / 1000.0)
            if inspect.iscoroutinefunction(backup_fn):
                res = await backup_fn()
            else:
                res = backup_fn()
            return res, backup_model_id, True

        task_p = asyncio.create_task(run_primary())
        task_b = asyncio.create_task(run_backup())

        done, pending = await asyncio.wait(
            [task_p, task_b],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        first_task = list(done)[0]
        res, model_id, was_hedged = first_task.result()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return HedgedResponse(
            result=res,
            winning_model_id=model_id,
            latency_ms=latency_ms,
            was_hedged=was_hedged,
        )


class TokenVelocityHedgeController:
    """Mid-stream Token Velocity Hedging Controller (< 5 tokens/sec threshold)."""

    def __init__(self, min_tokens_per_sec: float = 5.0):
        self.min_tokens_per_sec = min_tokens_per_sec

    def check_velocity(self, tokens_emitted: int, elapsed_sec: float) -> bool:
        """Returns True if stream velocity is healthy, False if stalled under SLA threshold."""
        if elapsed_sec <= 0.0:
            return True
        velocity = tokens_emitted / elapsed_sec
        return velocity >= self.min_tokens_per_sec

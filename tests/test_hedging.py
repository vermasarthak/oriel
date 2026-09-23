"""Unit tests for Speculative Request Hedging Engine."""

import asyncio

import pytest

from oriel.hedging import RequestHedgeController


@pytest.mark.anyio
async def test_fast_primary_no_hedge_win():
    controller = RequestHedgeController(hedge_delay_ms=50.0)

    async def primary():
        return "primary-ok"

    async def backup():
        return "backup-ok"

    res = await controller.execute_hedged(primary, "primary-m", backup, "backup-m")

    assert res.result == "primary-ok"
    assert res.winning_model_id == "primary-m"
    assert res.was_hedged is False


@pytest.mark.anyio
async def test_slow_primary_backup_hedge_win():
    controller = RequestHedgeController(hedge_delay_ms=10.0)

    async def primary():
        await asyncio.sleep(0.2)
        return "primary-ok"

    async def backup():
        return "backup-ok"

    res = await controller.execute_hedged(primary, "primary-m", backup, "backup-m")

    assert res.result == "backup-ok"
    assert res.winning_model_id == "backup-m"
    assert res.was_hedged is True


def test_invalid_hedge_delay_ms():
    with pytest.raises(ValueError, match="hedge_delay_ms must be positive"):
        RequestHedgeController(hedge_delay_ms=-10.0)


"""Tests for data/loader.py: unified fetch API + freshness/gap gate."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_fund.data.loader import DataFreshnessError, PriceLoader
from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import CandleRepo, InstrumentRecord, InstrumentRepo


class _StubSource:
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        return [c for c in self._candles if start <= c.trade_date <= end]


def _candle(symbol: str, trade_date: date) -> Candle:
    return Candle(
        symbol=symbol,
        trade_date=trade_date,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000,
        adj_close=100.0,
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


@pytest.fixture
def instrument(conn: sqlite3.Connection) -> InstrumentRecord:
    instrument_id = InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")
    record = InstrumentRepo(conn).get_by_id(instrument_id)
    assert record is not None
    return record


class TestPriceLoader:
    def test_load_fresh_returns_candles_when_up_to_date(
        self, conn: sqlite3.Connection, instrument: InstrumentRecord
    ) -> None:
        as_of = date(2026, 7, 4)
        candles = [_candle("7203.T", as_of - timedelta(days=i)) for i in range(3, -1, -1)]
        loader = PriceLoader(CandleRepo(conn), _StubSource(candles))

        loader.fetch_and_cache(instrument, as_of, lookback_days=10)
        result = loader.load_fresh(instrument, as_of, lookback_days=10)

        assert result[-1].trade_date == as_of

    def test_load_fresh_raises_when_latest_bar_is_stale(
        self, conn: sqlite3.Connection, instrument: InstrumentRecord
    ) -> None:
        as_of = date(2026, 7, 4)
        stale_date = as_of - timedelta(days=10)
        loader = PriceLoader(
            CandleRepo(conn), _StubSource([_candle("7203.T", stale_date)]), max_staleness_days=4
        )

        loader.fetch_and_cache(instrument, as_of, lookback_days=20)

        with pytest.raises(DataFreshnessError):
            loader.load_fresh(instrument, as_of, lookback_days=20)

    def test_load_fresh_raises_when_no_data_cached(
        self, conn: sqlite3.Connection, instrument: InstrumentRecord
    ) -> None:
        loader = PriceLoader(CandleRepo(conn), _StubSource([]))

        with pytest.raises(DataFreshnessError):
            loader.load_fresh(instrument, date(2026, 7, 4), lookback_days=10)

    def test_staleness_exactly_at_boundary_is_accepted(
        self, conn: sqlite3.Connection, instrument: InstrumentRecord
    ) -> None:
        as_of = date(2026, 7, 4)
        boundary_date = as_of - timedelta(days=4)
        loader = PriceLoader(
            CandleRepo(conn),
            _StubSource([_candle("7203.T", boundary_date)]),
            max_staleness_days=4,
        )

        loader.fetch_and_cache(instrument, as_of, lookback_days=10)
        result = loader.load_fresh(instrument, as_of, lookback_days=10)

        assert result[-1].trade_date == boundary_date

    def test_staleness_one_day_past_boundary_raises(
        self, conn: sqlite3.Connection, instrument: InstrumentRecord
    ) -> None:
        as_of = date(2026, 7, 4)
        past_boundary_date = as_of - timedelta(days=5)
        loader = PriceLoader(
            CandleRepo(conn),
            _StubSource([_candle("7203.T", past_boundary_date)]),
            max_staleness_days=4,
        )

        loader.fetch_and_cache(instrument, as_of, lookback_days=10)

        with pytest.raises(DataFreshnessError):
            loader.load_fresh(instrument, as_of, lookback_days=10)

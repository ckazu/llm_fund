"""Tests for data/cache.py: diff sync into `candles` (PriceSource fully stubbed)."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_fund.data.cache import CandleCache
from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import CandleRepo, InstrumentRepo


class _StubSource:
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        self.calls.append((symbol, start, end))
        return [c for c in self._candles if start <= c.trade_date <= end]


def _candle(symbol: str, trade_date: date, price: float = 100.0) -> Candle:
    return Candle(
        symbol=symbol,
        trade_date=trade_date,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000,
        adj_close=price,
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


@pytest.fixture
def instrument_id(conn: sqlite3.Connection) -> int:
    return InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")


class TestCandleCacheSync:
    def test_first_sync_fetches_full_range(
        self, conn: sqlite3.Connection, instrument_id: int
    ) -> None:
        candles = [_candle("7203.T", date(2026, 6, 29) + timedelta(days=i)) for i in range(3)]
        source = _StubSource(candles)
        cache = CandleCache(CandleRepo(conn), source)

        written = cache.sync(instrument_id, "7203.T", date(2026, 6, 29), date(2026, 7, 1))

        assert written == 3
        assert source.calls == [("7203.T", date(2026, 6, 29), date(2026, 7, 1))]

    def test_second_sync_only_fetches_the_gap(
        self, conn: sqlite3.Connection, instrument_id: int
    ) -> None:
        repo = CandleRepo(conn)
        CandleCache(repo, _StubSource([_candle("7203.T", date(2026, 6, 29))])).sync(
            instrument_id, "7203.T", date(2026, 6, 29), date(2026, 6, 29)
        )

        source2 = _StubSource(
            [_candle("7203.T", date(2026, 6, 30)), _candle("7203.T", date(2026, 7, 1))]
        )
        written = CandleCache(repo, source2).sync(
            instrument_id, "7203.T", date(2026, 6, 29), date(2026, 7, 1)
        )

        assert written == 2
        assert source2.calls == [("7203.T", date(2026, 6, 30), date(2026, 7, 1))]

    def test_up_to_date_cache_skips_fetch_entirely(
        self, conn: sqlite3.Connection, instrument_id: int
    ) -> None:
        repo = CandleRepo(conn)
        CandleCache(repo, _StubSource([_candle("7203.T", date(2026, 6, 29))])).sync(
            instrument_id, "7203.T", date(2026, 6, 29), date(2026, 6, 29)
        )

        source2 = _StubSource([])
        written = CandleCache(repo, source2).sync(
            instrument_id, "7203.T", date(2026, 6, 29), date(2026, 6, 29)
        )

        assert written == 0
        assert source2.calls == []

    def test_read_returns_cached_range_in_order(
        self, conn: sqlite3.Connection, instrument_id: int
    ) -> None:
        candles = [_candle("7203.T", date(2026, 6, 29)), _candle("7203.T", date(2026, 6, 30))]
        cache = CandleCache(CandleRepo(conn), _StubSource(candles))
        cache.sync(instrument_id, "7203.T", date(2026, 6, 29), date(2026, 6, 30))

        result = cache.read(instrument_id)

        assert [c.trade_date for c in result] == [date(2026, 6, 29), date(2026, 6, 30)]

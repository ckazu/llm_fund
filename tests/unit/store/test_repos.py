"""CRUD tests for the S2 repositories (instruments, universes, candles, portfolio_state)."""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    DEFAULT_LOT_SIZE,
    CandleRepo,
    InstrumentRepo,
    PortfolioStateRepo,
    UniverseRepo,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


class TestInstrumentRepo:
    def test_add_and_get_by_symbol(self, conn: sqlite3.Connection) -> None:
        repo = InstrumentRepo(conn)
        instrument_id = repo.add("7203.T", "トヨタ自動車", "jp")
        record = repo.get_by_symbol("7203.T")
        assert record is not None
        assert record.id == instrument_id
        assert record.name == "トヨタ自動車"
        assert record.lot_size == DEFAULT_LOT_SIZE
        assert record.active is True

    def test_get_by_symbol_missing_returns_none(self, conn: sqlite3.Connection) -> None:
        repo = InstrumentRepo(conn)
        assert repo.get_by_symbol("NOPE") is None

    def test_get_by_id(self, conn: sqlite3.Connection) -> None:
        repo = InstrumentRepo(conn)
        instrument_id = repo.add("AAPL", "Apple Inc.", "us")
        record = repo.get_by_id(instrument_id)
        assert record is not None
        assert record.symbol == "AAPL"

    def test_list_active_excludes_inactive(self, conn: sqlite3.Connection) -> None:
        repo = InstrumentRepo(conn)
        repo.add("7203.T", "Toyota", "jp")
        repo.add("DEAD.T", "Delisted", "jp", active=False)
        active = repo.list_active()
        assert [r.symbol for r in active] == ["7203.T"]

    def test_duplicate_symbol_rejected(self, conn: sqlite3.Connection) -> None:
        repo = InstrumentRepo(conn)
        repo.add("7203.T", "Toyota", "jp")
        with pytest.raises(sqlite3.IntegrityError):
            repo.add("7203.T", "Toyota Dup", "jp")


class TestUniverseRepo:
    def test_add_and_get_by_code(self, conn: sqlite3.Connection) -> None:
        repo = UniverseRepo(conn)
        universe_id = repo.add("jp_stocks", "jp", report_enabled=True, trade_enabled=True)
        record = repo.get_by_code("jp_stocks")
        assert record is not None
        assert record.id == universe_id
        assert record.trade_enabled is True

    def test_list_all_orders_by_code(self, conn: sqlite3.Connection) -> None:
        repo = UniverseRepo(conn)
        repo.add("us_stocks", "us")
        repo.add("etf", "global")
        codes = [r.code for r in repo.list_all()]
        assert codes == sorted(codes)

    def test_duplicate_code_rejected(self, conn: sqlite3.Connection) -> None:
        repo = UniverseRepo(conn)
        repo.add("jp_stocks", "jp")
        with pytest.raises(sqlite3.IntegrityError):
            repo.add("jp_stocks", "jp")


class TestCandleRepo:
    def _instrument_id(self, conn: sqlite3.Connection) -> int:
        return InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")

    def _candle(self, d: date, close: float = 100.0) -> Candle:
        return Candle(
            symbol="7203.T",
            trade_date=d,
            open=close,
            high=close + 5,
            low=close - 5,
            close=close,
            volume=1000,
            adj_close=close,
        )

    def test_upsert_and_get_range(self, conn: sqlite3.Connection) -> None:
        instrument_id = self._instrument_id(conn)
        repo = CandleRepo(conn)
        candles = [self._candle(date(2026, 7, 1), 100), self._candle(date(2026, 7, 2), 102)]
        written = repo.upsert_many(instrument_id, candles)
        assert written == 2

        result = repo.get_range(instrument_id)
        assert [c.trade_date for c in result] == [date(2026, 7, 1), date(2026, 7, 2)]
        assert result[1].close == 102

    def test_upsert_is_idempotent_on_conflict(self, conn: sqlite3.Connection) -> None:
        instrument_id = self._instrument_id(conn)
        repo = CandleRepo(conn)
        repo.upsert_many(instrument_id, [self._candle(date(2026, 7, 1), 100)])
        repo.upsert_many(instrument_id, [self._candle(date(2026, 7, 1), 999)])

        result = repo.get_range(instrument_id)
        assert len(result) == 1
        assert result[0].close == 999

    def test_get_range_filters_by_date(self, conn: sqlite3.Connection) -> None:
        instrument_id = self._instrument_id(conn)
        repo = CandleRepo(conn)
        repo.upsert_many(
            instrument_id,
            [
                self._candle(date(2026, 7, 1), 100),
                self._candle(date(2026, 7, 2), 101),
                self._candle(date(2026, 7, 3), 102),
            ],
        )
        result = repo.get_range(instrument_id, start=date(2026, 7, 2), end=date(2026, 7, 2))
        assert len(result) == 1
        assert result[0].trade_date == date(2026, 7, 2)

    def test_latest_returns_most_recent(self, conn: sqlite3.Connection) -> None:
        instrument_id = self._instrument_id(conn)
        repo = CandleRepo(conn)
        repo.upsert_many(
            instrument_id,
            [self._candle(date(2026, 7, 1), 100), self._candle(date(2026, 7, 3), 102)],
        )
        latest = repo.latest(instrument_id)
        assert latest is not None
        assert latest.trade_date == date(2026, 7, 3)

    def test_latest_returns_none_when_empty(self, conn: sqlite3.Connection) -> None:
        instrument_id = self._instrument_id(conn)
        repo = CandleRepo(conn)
        assert repo.latest(instrument_id) is None


class TestPortfolioStateRepo:
    def test_upsert_and_get_by_date(self, conn: sqlite3.Connection) -> None:
        repo = PortfolioStateRepo(conn)
        repo.upsert(date(2026, 7, 1), cash=500_000.0, nav=1_000_000.0, note="initial")
        record = repo.get_by_date(date(2026, 7, 1))
        assert record is not None
        assert record.nav == 1_000_000.0
        assert record.note == "initial"

    def test_upsert_overwrites_same_date(self, conn: sqlite3.Connection) -> None:
        repo = PortfolioStateRepo(conn)
        repo.upsert(date(2026, 7, 1), cash=500_000.0, nav=1_000_000.0)
        repo.upsert(date(2026, 7, 1), cash=400_000.0, nav=1_100_000.0)
        record = repo.get_by_date(date(2026, 7, 1))
        assert record is not None
        assert record.cash == 400_000.0
        assert record.nav == 1_100_000.0

    def test_latest_returns_most_recent_date(self, conn: sqlite3.Connection) -> None:
        repo = PortfolioStateRepo(conn)
        repo.upsert(date(2026, 7, 1), cash=500_000.0, nav=1_000_000.0)
        repo.upsert(date(2026, 7, 2), cash=480_000.0, nav=1_020_000.0)
        latest = repo.latest()
        assert latest is not None
        assert latest.state_date == date(2026, 7, 2)

    def test_latest_returns_none_when_empty(self, conn: sqlite3.Connection) -> None:
        repo = PortfolioStateRepo(conn)
        assert repo.latest() is None

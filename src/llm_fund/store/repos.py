"""Repository layer over the SQLite store.

Only the repositories needed by S2 are implemented here (instruments,
universes, candles, portfolio_state). Instructions/executions/tracking/audit
repositories are added in later steps as their modules are built.

Records returned here are plain frozen dataclasses rather than the pydantic
domain models in `domain/models.py`: they mirror DB rows (including the
internal surrogate `id`) and are not part of the LLM/validator data flow that
`domain/models.py` covers, except for `Candle` which is reused directly.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date

from llm_fund.domain.models import Candle

# 東証の一般的な売買単位（100株）。instruments.lot_size の既定値。
DEFAULT_LOT_SIZE = 100


@dataclass(frozen=True, slots=True)
class InstrumentRecord:
    id: int
    symbol: str
    name: str
    market: str
    lot_size: int
    active: bool


@dataclass(frozen=True, slots=True)
class UniverseRecord:
    id: int
    code: str
    market: str
    report_enabled: bool
    trade_enabled: bool


@dataclass(frozen=True, slots=True)
class PortfolioStateRecord:
    id: int
    state_date: date
    cash: float
    nav: float
    note: str | None


class InstrumentRepo:
    """CRUD for `instruments` (symbol is the natural key, id is internal)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        symbol: str,
        name: str,
        market: str,
        lot_size: int = DEFAULT_LOT_SIZE,
        active: bool = True,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO instruments (symbol, name, market, lot_size, active) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, name, market, lot_size, int(active)),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_symbol(self, symbol: str) -> InstrumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM instruments WHERE symbol = ?", (symbol,)
        ).fetchone()
        return self._to_record(row) if row else None

    def get_by_id(self, instrument_id: int) -> InstrumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM instruments WHERE id = ?", (instrument_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def list_active(self) -> list[InstrumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM instruments WHERE active = 1 ORDER BY symbol"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> InstrumentRecord:
        return InstrumentRecord(
            id=row["id"],
            symbol=row["symbol"],
            name=row["name"],
            market=row["market"],
            lot_size=row["lot_size"],
            active=bool(row["active"]),
        )


class UniverseRepo:
    """CRUD for `universes` (code is the natural key, id is internal)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        code: str,
        market: str,
        report_enabled: bool = True,
        trade_enabled: bool = False,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO universes (code, market, report_enabled, trade_enabled) "
            "VALUES (?, ?, ?, ?)",
            (code, market, int(report_enabled), int(trade_enabled)),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_code(self, code: str) -> UniverseRecord | None:
        row = self._conn.execute(
            "SELECT * FROM universes WHERE code = ?", (code,)
        ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[UniverseRecord]:
        rows = self._conn.execute("SELECT * FROM universes ORDER BY code").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> UniverseRecord:
        return UniverseRecord(
            id=row["id"],
            code=row["code"],
            market=row["market"],
            report_enabled=bool(row["report_enabled"]),
            trade_enabled=bool(row["trade_enabled"]),
        )


class CandleRepo:
    """Read/write for `candles`, keyed by (instrument_id, date)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_many(self, instrument_id: int, candles: list[Candle]) -> int:
        """Insert or replace candles for `instrument_id`. Returns the count written."""
        self._conn.executemany(
            "INSERT INTO candles "
            "(instrument_id, date, open, high, low, close, volume, adj_close) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(instrument_id, date) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume, adj_close=excluded.adj_close",
            [
                (
                    instrument_id,
                    c.trade_date.isoformat(),
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                    c.adj_close,
                )
                for c in candles
            ],
        )
        self._conn.commit()
        return len(candles)

    def get_range(
        self,
        instrument_id: int,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Candle]:
        query = "SELECT c.*, i.symbol FROM candles c JOIN instruments i ON i.id = c.instrument_id"
        query += " WHERE c.instrument_id = ?"
        params: list[str | int] = [instrument_id]
        if start is not None:
            query += " AND c.date >= ?"
            params.append(start.isoformat())
        if end is not None:
            query += " AND c.date <= ?"
            params.append(end.isoformat())
        query += " ORDER BY c.date"
        rows = self._conn.execute(query, params).fetchall()
        return [self._to_candle(row) for row in rows]

    def latest(self, instrument_id: int) -> Candle | None:
        row = self._conn.execute(
            "SELECT c.*, i.symbol FROM candles c JOIN instruments i ON i.id = c.instrument_id "
            "WHERE c.instrument_id = ? ORDER BY c.date DESC LIMIT 1",
            (instrument_id,),
        ).fetchone()
        return self._to_candle(row) if row else None

    @staticmethod
    def _to_candle(row: sqlite3.Row) -> Candle:
        return Candle(
            symbol=row["symbol"],
            trade_date=date.fromisoformat(row["date"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            adj_close=row["adj_close"],
        )


class PortfolioStateRepo:
    """Read/write for `portfolio_state` (the source of truth for NAV/cash)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, state_date: date, cash: float, nav: float, note: str | None = None) -> int:
        self._conn.execute(
            "INSERT INTO portfolio_state (date, cash, nav, note) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET cash=excluded.cash, nav=excluded.nav, "
            "note=excluded.note",
            (state_date.isoformat(), cash, nav, note),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM portfolio_state WHERE date = ?", (state_date.isoformat(),)
        ).fetchone()
        return int(row["id"])

    def get_by_date(self, state_date: date) -> PortfolioStateRecord | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_state WHERE date = ?", (state_date.isoformat(),)
        ).fetchone()
        return self._to_record(row) if row else None

    def latest(self) -> PortfolioStateRecord | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_state ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: sqlite3.Row) -> PortfolioStateRecord:
        return PortfolioStateRecord(
            id=row["id"],
            state_date=date.fromisoformat(row["date"]),
            cash=row["cash"],
            nav=row["nav"],
            note=row["note"],
        )

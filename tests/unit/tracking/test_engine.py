"""``run_virtual_fills`` エンジンの結合テスト（状態更新・時価評価・冪等性）。"""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import Action, InstructionStatus, PendingOrderStatus
from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    BriefingRepo,
    CandleRepo,
    InstructionRepo,
    InstrumentRepo,
    PendingOrderRepo,
    PortfolioStateRepo,
    PositionRepo,
    UniverseRepo,
    VirtualFillRepo,
)
from llm_fund.tracking.virtual_fill import CostModel, run_virtual_fills

BRIEFING_DATE = date(2026, 1, 5)
STARTING_CAPITAL = 1_000_000.0
NO_COST = CostModel()


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _c(day: int, o: float, h: float, low: float, c: float, instrument_id: int) -> Candle:
    return Candle(
        symbol="7203.T",
        trade_date=date(2026, 1, day),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1_000,
        adj_close=c,
    )


def _seed(
    conn: sqlite3.Connection,
    candles_days: list[tuple[int, float, float, float, float]],
    *,
    ticket_no: str = "20260105-01",
    entry: float = 1000.0,
    tp: float = 1100.0,
    sl: float = 900.0,
    units: int = 100,
    valid_until: date = date(2026, 1, 30),
) -> int:
    universe_id = UniverseRepo(conn).add("jp_stocks", "jp", trade_enabled=True)
    instrument_id = InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")
    briefing_id = BriefingRepo(conn).add(universe_id, BRIEFING_DATE, "daily", "md", "{}")
    candles = [_c(day, o, h, low, c, instrument_id) for day, o, h, low, c in candles_days]
    CandleRepo(conn).upsert_many(instrument_id, candles)
    InstructionRepo(conn).add(
        ticket_no=ticket_no,
        briefing_id=briefing_id,
        instrument_id=instrument_id,
        action=Action.BUY.value,
        units=units,
        entry_price=entry,
        tp_price=tp,
        sl_price=sl,
        valid_until=valid_until,
        rationale="上昇トレンド継続",
        validator_result_json=None,
        status=InstructionStatus.PENDING.value,
    )
    return instrument_id


class TestRunVirtualFills:
    def test_round_trip_updates_cash_and_nav(self, conn: sqlite3.Connection) -> None:
        # day5=briefing(除外), day6 fill@1000, day7 TP@1100
        _seed(
            conn,
            [
                (5, 1000, 1010, 995, 1005),
                (6, 1000, 1010, 995, 1005),
                (7, 1050, 1120, 1040, 1100),
            ],
        )
        result = run_virtual_fills(
            conn, date(2026, 1, 7), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        # 1_000_000 - 100_000(entry) + 110_000(exit) = 1_010_000
        assert result.cash == pytest.approx(1_010_000.0)
        assert result.nav == pytest.approx(1_010_000.0)
        assert result.exited == 1
        assert result.open_positions == 0

        vf = VirtualFillRepo(conn).list_all()[0]
        assert vf.pnl == pytest.approx(10_000.0)
        assert PositionRepo(conn).list_open() == []
        state = PortfolioStateRepo(conn).get_by_date(date(2026, 1, 7))
        assert state is not None
        assert state.nav == pytest.approx(1_010_000.0)

    def test_open_position_marked_to_market(self, conn: sqlite3.Connection) -> None:
        # fill@1000 day6, never exits; as_of day8 close=1050 → 時価評価
        _seed(
            conn,
            [
                (5, 1000, 1010, 995, 1005),
                (6, 1000, 1010, 995, 1005),
                (7, 1010, 1050, 960, 1020),
                (8, 1030, 1060, 1010, 1050),
            ],
        )
        result = run_virtual_fills(
            conn, date(2026, 1, 8), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        assert result.open_positions == 1
        # cash = 1_000_000 - 100_000 ; positions = 100 * 1050 = 105_000
        assert result.cash == pytest.approx(900_000.0)
        assert result.nav == pytest.approx(1_005_000.0)
        pending = PendingOrderRepo(conn).list_by_status(PendingOrderStatus.FILLED.value)
        assert len(pending) == 1

    def test_expired_unfilled_marks_pending_expired(self, conn: sqlite3.Connection) -> None:
        _seed(
            conn,
            [
                (5, 1080, 1090, 1050, 1080),
                (6, 1080, 1090, 1060, 1080),
            ],
            valid_until=date(2026, 1, 6),
        )
        result = run_virtual_fills(
            conn, date(2026, 1, 6), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        assert result.expired == 1
        assert result.nav == pytest.approx(STARTING_CAPITAL)  # 現金のまま
        expired = PendingOrderRepo(conn).list_by_status(PendingOrderStatus.EXPIRED.value)
        assert len(expired) == 1

    def test_idempotent_rerun_same_day(self, conn: sqlite3.Connection) -> None:
        _seed(
            conn,
            [
                (5, 1000, 1010, 995, 1005),
                (6, 1000, 1010, 995, 1005),
                (7, 1050, 1120, 1040, 1100),
            ],
        )
        first = run_virtual_fills(
            conn, date(2026, 1, 7), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        second = run_virtual_fills(
            conn, date(2026, 1, 7), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        assert first.nav == pytest.approx(second.nav)
        # 二重計上しない: virtual_fills/positions/pending_orders/portfolio_state は各1行
        assert len(VirtualFillRepo(conn).list_all()) == 1
        assert len(PositionRepo(conn).list_all()) == 1
        assert len(PendingOrderRepo(conn).list_all()) == 1
        rows = conn.execute("SELECT COUNT(*) AS n FROM portfolio_state").fetchone()
        assert rows["n"] == 1

    def test_open_position_then_exit_next_day_extends_fill(
        self, conn: sqlite3.Connection
    ) -> None:
        # day7 まではポジション保有、day8 で TP。翌日の再実行で決済が反映される。
        _seed(
            conn,
            [
                (5, 1000, 1010, 995, 1005),
                (6, 1000, 1010, 995, 1005),
                (7, 1010, 1050, 960, 1020),
                (8, 1050, 1120, 1040, 1100),
            ],
        )
        day7 = run_virtual_fills(
            conn, date(2026, 1, 7), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        assert day7.open_positions == 1
        day8 = run_virtual_fills(
            conn, date(2026, 1, 8), costs=NO_COST, starting_capital=STARTING_CAPITAL
        )
        assert day8.exited == 1
        assert day8.nav == pytest.approx(1_010_000.0)
        assert len(VirtualFillRepo(conn).list_all()) == 1  # upsert（増えない）

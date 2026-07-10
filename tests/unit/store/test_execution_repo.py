"""CRUD + deviation tests for `ExecutionRepo` (S7: FR-4 執行記録)."""

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from llm_fund.domain.enums import Action, ExecutionStatus, InstructionStatus
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    BriefingRepo,
    ExecutionRepo,
    InstructionRepo,
    InstrumentRepo,
    UniverseRepo,
    price_deviation_pct,
)

ORDER_TYPE_IFO = "IFO"


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _seed_instruction(
    conn: sqlite3.Connection, ticket_no: str = "20260704-01", entry_price: float = 3000.0
) -> int:
    universe = UniverseRepo(conn).get_by_code("jp_stocks")
    universe_id = universe.id if universe else UniverseRepo(conn).add(
        "jp_stocks", "jp", trade_enabled=True
    )
    instrument = InstrumentRepo(conn).get_by_symbol("7203.T")
    instrument_id = instrument.id if instrument else InstrumentRepo(conn).add(
        "7203.T", "トヨタ自動車", "jp"
    )
    briefing_id = BriefingRepo(conn).add(universe_id, date(2026, 7, 4), "daily", "md", "{}")
    return InstructionRepo(conn).add(
        ticket_no=ticket_no,
        briefing_id=briefing_id,
        instrument_id=instrument_id,
        action=Action.BUY.value,
        units=100,
        entry_price=entry_price,
        tp_price=3300.0,
        sl_price=2900.0,
        valid_until=date(2026, 7, 7),
        rationale="上昇トレンド継続",
        validator_result_json=None,
        status=InstructionStatus.PENDING.value,
    )


class TestPriceDeviationPct:
    def test_positive_deviation_when_actual_above_expected(self) -> None:
        assert price_deviation_pct(3120.0, 3000.0) == pytest.approx(4.0)

    def test_negative_deviation_when_actual_below_expected(self) -> None:
        assert price_deviation_pct(2940.0, 3000.0) == pytest.approx(-2.0)

    def test_zero_deviation_when_equal(self) -> None:
        assert price_deviation_pct(3000.0, 3000.0) == pytest.approx(0.0)


class TestExecutionRepo:
    def test_add_and_get_by_instruction_id(self, conn: sqlite3.Connection) -> None:
        instruction_id = _seed_instruction(conn)
        repo = ExecutionRepo(conn)
        execution_id = repo.add(
            instruction_id=instruction_id,
            executed_at=datetime(2026, 7, 4, 10, 30, tzinfo=UTC),
            side=Action.BUY.value,
            order_type=ORDER_TYPE_IFO,
            actual_price=3120.0,
            actual_units=100,
            commission=50.0,
            status=ExecutionStatus.FILLED.value,
        )
        record = repo.get_by_instruction_id(instruction_id)
        assert record is not None
        assert record.id == execution_id
        assert record.actual_price == 3120.0
        assert record.status == ExecutionStatus.FILLED.value

    def test_get_by_instruction_id_missing_returns_none(self, conn: sqlite3.Connection) -> None:
        assert ExecutionRepo(conn).get_by_instruction_id(999) is None

    def test_list_deviations_computes_pct_for_filled(self, conn: sqlite3.Connection) -> None:
        instruction_id = _seed_instruction(conn, "20260704-01", entry_price=3000.0)
        ExecutionRepo(conn).add(
            instruction_id=instruction_id,
            executed_at=datetime.now(UTC),
            side=Action.BUY.value,
            order_type=ORDER_TYPE_IFO,
            actual_price=3120.0,
            actual_units=100,
            commission=50.0,
            status=ExecutionStatus.FILLED.value,
        )
        deviations = ExecutionRepo(conn).list_deviations()
        assert len(deviations) == 1
        assert deviations[0].ticket_no == "20260704-01"
        assert deviations[0].deviation_pct == pytest.approx(4.0)

    def test_list_deviations_none_for_skipped(self, conn: sqlite3.Connection) -> None:
        instruction_id = _seed_instruction(conn)
        ExecutionRepo(conn).add(
            instruction_id=instruction_id,
            executed_at=datetime.now(UTC),
            side=Action.BUY.value,
            order_type=ORDER_TYPE_IFO,
            actual_price=3000.0,
            actual_units=0,
            commission=0.0,
            status=ExecutionStatus.SKIPPED.value,
            skip_reason="価格が到達せず",
        )
        deviations = ExecutionRepo(conn).list_deviations()
        assert deviations[0].deviation_pct is None
        assert deviations[0].status == ExecutionStatus.SKIPPED.value

    def test_unexecuted_rate_none_when_no_executions(self, conn: sqlite3.Connection) -> None:
        assert ExecutionRepo(conn).unexecuted_rate() is None

    def test_unexecuted_rate_ratio_of_skipped(self, conn: sqlite3.Connection) -> None:
        id1 = _seed_instruction(conn, "20260704-01")
        id2 = _seed_instruction(conn, "20260704-02")
        repo = ExecutionRepo(conn)
        repo.add(
            instruction_id=id1,
            executed_at=datetime.now(UTC),
            side=Action.BUY.value,
            order_type=ORDER_TYPE_IFO,
            actual_price=3000.0,
            actual_units=100,
            commission=50.0,
            status=ExecutionStatus.FILLED.value,
        )
        repo.add(
            instruction_id=id2,
            executed_at=datetime.now(UTC),
            side=Action.BUY.value,
            order_type=ORDER_TYPE_IFO,
            actual_price=3000.0,
            actual_units=0,
            commission=0.0,
            status=ExecutionStatus.SKIPPED.value,
            skip_reason="未到達",
        )
        assert repo.unexecuted_rate() == pytest.approx(0.5)

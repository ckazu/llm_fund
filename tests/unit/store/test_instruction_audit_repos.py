"""CRUD tests for the S5 repositories (instructions, audit_events)."""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import Action, InstructionStatus
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    AuditEventRepo,
    BriefingRepo,
    InstructionRepo,
    InstrumentRepo,
    UniverseRepo,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _seed_fk(conn: sqlite3.Connection) -> tuple[int, int]:
    """Create the universe/instrument/briefing rows the FK columns require."""
    universe_id = UniverseRepo(conn).add("jp_stocks", "jp", trade_enabled=True)
    instrument_id = InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")
    briefing_id = BriefingRepo(conn).add(universe_id, date(2026, 7, 4), "daily", "md", "{}")
    return briefing_id, instrument_id


def _add_instruction(
    conn: sqlite3.Connection, ticket_no: str, briefing_id: int, instrument_id: int
) -> int:
    return InstructionRepo(conn).add(
        ticket_no=ticket_no,
        briefing_id=briefing_id,
        instrument_id=instrument_id,
        action=Action.BUY.value,
        units=100,
        entry_price=3000.0,
        tp_price=3300.0,
        sl_price=2900.0,
        valid_until=date(2026, 7, 7),
        rationale="上昇トレンド継続",
        validator_result_json='{"result": "validated"}',
        status=InstructionStatus.PENDING.value,
    )


class TestInstructionRepo:
    def test_add_and_get_by_ticket_no(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        row_id = _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        record = InstructionRepo(conn).get_by_ticket_no("20260704-01")
        assert record is not None
        assert record.id == row_id
        assert record.ticket_no == "20260704-01"
        assert record.action == Action.BUY.value
        assert record.status == InstructionStatus.PENDING.value

    def test_get_missing_ticket_returns_none(self, conn: sqlite3.Connection) -> None:
        assert InstructionRepo(conn).get_by_ticket_no("20260704-99") is None

    def test_duplicate_ticket_rejected(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        with pytest.raises(sqlite3.IntegrityError):
            _add_instruction(conn, "20260704-01", briefing_id, instrument_id)

    def test_count_for_date_counts_only_matching_day(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        _add_instruction(conn, "20260704-02", briefing_id, instrument_id)
        _add_instruction(conn, "20260705-01", briefing_id, instrument_id)
        assert InstructionRepo(conn).count_for_date(date(2026, 7, 4)) == 2
        assert InstructionRepo(conn).count_for_date(date(2026, 7, 5)) == 1
        assert InstructionRepo(conn).count_for_date(date(2026, 7, 6)) == 0

    def test_next_sequence_is_count_plus_one(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        assert InstructionRepo(conn).next_sequence(date(2026, 7, 4)) == 1
        _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        assert InstructionRepo(conn).next_sequence(date(2026, 7, 4)) == 2

    def test_update_status_changes_status(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        row_id = _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        repo = InstructionRepo(conn)

        repo.update_status(row_id, InstructionStatus.FILLED.value)

        record = repo.get_by_ticket_no("20260704-01")
        assert record is not None
        assert record.status == InstructionStatus.FILLED.value

    def test_list_by_status_filters(self, conn: sqlite3.Connection) -> None:
        briefing_id, instrument_id = _seed_fk(conn)
        id1 = _add_instruction(conn, "20260704-01", briefing_id, instrument_id)
        _add_instruction(conn, "20260704-02", briefing_id, instrument_id)
        repo = InstructionRepo(conn)
        repo.update_status(id1, InstructionStatus.FILLED.value)

        pending = repo.list_by_status(InstructionStatus.PENDING.value)

        assert [r.ticket_no for r in pending] == ["20260704-02"]


class TestAuditEventRepo:
    def test_add_and_list_all(self, conn: sqlite3.Connection) -> None:
        repo = AuditEventRepo(conn)
        repo.add("rejection", '{"symbol": "7203.T"}')
        repo.add("no_trade", '{"reason": "stale"}')
        events = repo.list_all()
        assert len(events) == 2
        assert {e.kind for e in events} == {"rejection", "no_trade"}

    def test_list_by_kind_filters(self, conn: sqlite3.Connection) -> None:
        repo = AuditEventRepo(conn)
        repo.add("rejection", "{}")
        repo.add("rejection", "{}")
        repo.add("no_trade", "{}")
        assert len(repo.list_by_kind("rejection")) == 2
        assert len(repo.list_by_kind("no_trade")) == 1
        assert repo.list_by_kind("warning") == []

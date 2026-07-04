"""Migration application tests for store/db.py."""

import sqlite3
from pathlib import Path

import pytest

from llm_fund.store.db import apply_migrations, connect, init_db

EXPECTED_TABLES = {
    "universes",
    "instruments",
    "universe_members",
    "candles",
    "portfolio_state",
    "positions",
    "policies",
    "criteria",
    "briefings",
    "instructions",
    "pending_orders",
    "executions",
    "virtual_fills",
    "strategies",
    "benchmark_navs",
    "llm_calls",
    "audit_events",
    "schema_migrations",
}


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "test.db")


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


class TestApplyMigrations:
    def test_creates_all_expected_tables(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn)
        assert EXPECTED_TABLES.issubset(_table_names(conn))

    def test_records_applied_migration_filenames(self, conn: sqlite3.Connection) -> None:
        applied = apply_migrations(conn)
        assert "0001_init.sql" in applied

    def test_second_call_is_idempotent(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn)
        second_pass = apply_migrations(conn)
        assert second_pass == []

    def test_foreign_keys_enforced(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO universe_members (universe_id, instrument_id) VALUES (999, 999)"
            )

    def test_unique_constraint_on_instrument_symbol(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO instruments (symbol, name, market) VALUES ('7203.T', 'Toyota', 'jp')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO instruments (symbol, name, market) VALUES ('7203.T', 'Toyota2', 'jp')"
            )


class TestInitDb:
    def test_init_db_applies_migrations(self, tmp_path: Path) -> None:
        db_path = tmp_path / "init.db"
        conn = init_db(db_path)
        assert EXPECTED_TABLES.issubset(_table_names(conn))

    def test_init_db_is_reentrant(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reentrant.db"
        init_db(db_path).close()
        conn = init_db(db_path)
        assert EXPECTED_TABLES.issubset(_table_names(conn))

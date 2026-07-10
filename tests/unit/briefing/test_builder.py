"""Snapshot-style regression tests for briefing Markdown/JSON generation (S4)."""

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_fund.briefing.builder import build_universe_briefing, save_briefing
from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import BriefingRepo, UniverseRepo

BRIEFING_DATE = date(2026, 7, 4)


def _flat_candles(symbol: str, days: int, close: float) -> list[Candle]:
    """`days` candles at a constant price/volume (short history -> all N/A indicator fields)."""
    start = BRIEFING_DATE - timedelta(days=days - 1)
    return [
        Candle(
            symbol=symbol,
            trade_date=start + timedelta(days=i),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1000,
            adj_close=close,
        )
        for i in range(days)
    ]


class TestBuildUniverseBriefing:
    def test_renders_expected_markdown_table(self) -> None:
        instrument_candles = {
            "7203.T": _flat_candles("7203.T", 3, 3000.0),
            "6758.T": _flat_candles("6758.T", 3, 12000.0),
        }

        briefing = build_universe_briefing(
            "jp_stocks", "daily", BRIEFING_DATE, instrument_candles
        )

        expected = (
            "## jp_stocks (daily, 2026-07-04)\n\n"
            "| Symbol | Close | 1d% | 5d% | 20d% | 60d% | MA25乖離% | MA75乖離% | "
            "ATR14% | 出来高比20d |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
            "| 6758.T | 12000.00 | 0.00 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
            "| 7203.T | 3000.00 | 0.00 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
        )
        assert briefing.content_md == expected

    def test_rows_sorted_by_symbol_regardless_of_input_order(self) -> None:
        instrument_candles = {
            "9984.T": _flat_candles("9984.T", 3, 8000.0),
            "4063.T": _flat_candles("4063.T", 3, 5000.0),
        }

        briefing = build_universe_briefing(
            "jp_stocks", "daily", BRIEFING_DATE, instrument_candles
        )

        lines = [line for line in briefing.content_md.splitlines() if line.startswith("|")]
        symbols_in_order = [line.split("|")[1].strip() for line in lines[2:]]
        assert symbols_in_order == ["4063.T", "9984.T"]

    def test_skips_instruments_with_no_candles(self) -> None:
        instrument_candles = {
            "7203.T": _flat_candles("7203.T", 3, 3000.0),
            "STALE": [],
        }

        briefing = build_universe_briefing(
            "jp_stocks", "daily", BRIEFING_DATE, instrument_candles
        )

        assert "STALE" not in briefing.content_md
        assert briefing.content_md.count("7203.T") == 1

    def test_empty_universe_reports_no_data(self) -> None:
        briefing = build_universe_briefing("jp_stocks", "daily", BRIEFING_DATE, {})

        assert briefing.content_md == "## jp_stocks (daily, 2026-07-04)\n\n対象データなし\n"

    def test_data_snapshot_contains_one_entry_per_instrument(self) -> None:
        instrument_candles = {
            "7203.T": _flat_candles("7203.T", 3, 3000.0),
            "6758.T": _flat_candles("6758.T", 3, 12000.0),
        }

        briefing = build_universe_briefing(
            "jp_stocks", "daily", BRIEFING_DATE, instrument_candles
        )

        indicators = briefing.data_snapshot["indicators"]
        assert isinstance(indicators, list)
        assert len(indicators) == 2
        assert {row["symbol"] for row in indicators} == {"7203.T", "6758.T"}
        # Must be JSON-serializable as-is (this is what save_briefing persists).
        json.dumps(briefing.data_snapshot)


class TestSaveBriefing:
    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        c = connect(tmp_path / "test.db")
        apply_migrations(c)
        return c

    def test_persists_content_and_snapshot(self, conn: sqlite3.Connection) -> None:
        universe_id = UniverseRepo(conn).add("jp_stocks", "jp")
        briefing = build_universe_briefing(
            "jp_stocks", "daily", BRIEFING_DATE, {"7203.T": _flat_candles("7203.T", 3, 3000.0)}
        )

        briefing_id = save_briefing(BriefingRepo(conn), universe_id, briefing)

        record = BriefingRepo(conn).get_by_id(briefing_id)
        assert record is not None
        assert record.universe_id == universe_id
        assert record.kind == "daily"
        assert record.briefing_date == BRIEFING_DATE
        assert record.content_md == briefing.content_md
        assert json.loads(record.data_snapshot_json) == briefing.data_snapshot

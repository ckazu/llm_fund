"""CLI tests for `fund record` and the deviation display in `fund status` (S7: FR-4)."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_fund.cli import app
from llm_fund.domain.enums import Action, ExecutionStatus, InstructionStatus
from llm_fund.store.db import init_db
from llm_fund.store.repos import (
    BriefingRepo,
    ExecutionRepo,
    InstructionRepo,
    InstrumentRepo,
    UniverseRepo,
)
from tests.factories import build_llm_config_dict

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "llm": build_llm_config_dict(),
                "limits": {
                    "max_position_pct": 15.0,
                    "max_turnover_pct": 30.0,
                    "max_instructions_per_day": 5,
                    "require_stop_loss": True,
                },
                "benchmark": {"index_symbol": "1306.T", "momentum_lookback_days": 120},
                "report": {"output_dir": "reports"},
                "db_path": "test.db",
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "universes.yaml").write_text(
        yaml.safe_dump({"universes": {}}), encoding="utf-8"
    )
    return tmp_path


def _seed_instruction(
    project: Path, ticket_no: str = "20260704-01", units: int = 100, entry_price: float = 3000.0
) -> None:
    conn = init_db(str(project / "test.db"))
    universe = UniverseRepo(conn).get_by_code("jp_stocks")
    universe_id = universe.id if universe else UniverseRepo(conn).add(
        "jp_stocks", "jp", trade_enabled=True
    )
    instrument = InstrumentRepo(conn).get_by_symbol("7203.T")
    instrument_id = instrument.id if instrument else InstrumentRepo(conn).add(
        "7203.T", "トヨタ自動車", "jp"
    )
    briefing_id = BriefingRepo(conn).add(universe_id, date(2026, 7, 4), "daily", "md", "{}")
    InstructionRepo(conn).add(
        ticket_no=ticket_no,
        briefing_id=briefing_id,
        instrument_id=instrument_id,
        action=Action.BUY.value,
        units=units,
        entry_price=entry_price,
        tp_price=3300.0,
        sl_price=2900.0,
        valid_until=date(2026, 7, 7),
        rationale="上昇トレンド継続",
        validator_result_json=None,
        status=InstructionStatus.PENDING.value,
    )
    conn.close()


class TestRecordCommand:
    def test_record_full_fill_updates_instruction_and_execution(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01", units=100, entry_price=3000.0)

        result = runner.invoke(app, ["record", "20260704-01", "--price", "3120"])

        assert result.exit_code == 0, result.output
        conn = init_db(str(project / "test.db"))
        instruction = InstructionRepo(conn).get_by_ticket_no("20260704-01")
        assert instruction is not None
        assert instruction.status == ExecutionStatus.FILLED.value
        execution = ExecutionRepo(conn).get_by_instruction_id(instruction.id)
        assert execution is not None
        assert execution.actual_price == 3120.0
        assert execution.actual_units == 100
        assert execution.status == ExecutionStatus.FILLED.value

    def test_record_partial_fill_when_units_below_instructed(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01", units=100)

        result = runner.invoke(
            app, ["record", "20260704-01", "--price", "3120", "--units", "50"]
        )

        assert result.exit_code == 0, result.output
        conn = init_db(str(project / "test.db"))
        instruction = InstructionRepo(conn).get_by_ticket_no("20260704-01")
        assert instruction is not None
        assert instruction.status == ExecutionStatus.PARTIAL.value

    def test_record_skipped_marks_status_and_zero_units(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01")

        result = runner.invoke(
            app,
            [
                "record",
                "20260704-01",
                "--price",
                "3000",
                "--skipped",
                "エントリー未到達",
            ],
        )

        assert result.exit_code == 0, result.output
        conn = init_db(str(project / "test.db"))
        instruction = InstructionRepo(conn).get_by_ticket_no("20260704-01")
        assert instruction is not None
        assert instruction.status == ExecutionStatus.SKIPPED.value
        execution = ExecutionRepo(conn).get_by_instruction_id(instruction.id)
        assert execution is not None
        assert execution.actual_units == 0
        assert execution.skip_reason == "エントリー未到達"

    def test_record_unknown_ticket_no_exits_with_error_code(self, project: Path) -> None:
        result = runner.invoke(app, ["record", "20260704-99", "--price", "3000"])

        assert result.exit_code == 1

    def test_record_zero_units_without_skipped_is_rejected(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01")

        result = runner.invoke(
            app, ["record", "20260704-01", "--price", "3000", "--units", "0"]
        )

        assert result.exit_code == 1


class TestStatusDeviationDisplay:
    def test_status_shows_price_deviation_for_filled(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01", entry_price=3000.0)
        runner.invoke(app, ["record", "20260704-01", "--price", "3120"])

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "20260704-01" in result.output
        assert "+4.00%" in result.output

    def test_status_shows_unexecuted_rate(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01")
        _seed_instruction(project, "20260704-02")
        runner.invoke(app, ["record", "20260704-01", "--price", "3000"])
        runner.invoke(
            app, ["record", "20260704-02", "--price", "3000", "--skipped", "未到達"]
        )

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "未執行率: 50.0%" in result.output

    def test_status_lists_unrecorded_instructions(self, project: Path) -> None:
        _seed_instruction(project, "20260704-01")

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "20260704-01" in result.output
        assert "未記録の指示: 1件" in result.output

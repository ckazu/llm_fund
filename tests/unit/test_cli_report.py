"""CLI tests for `fund report [universe]` and `fund daily` (S4): report line end-to-end."""

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_fund.cli import app
from llm_fund.domain.models import Candle

runner = CliRunner()


class _StubSource:
    """Returns a single candle dated `latest_date`, clipped to the requested range."""

    def __init__(self, latest_date: date) -> None:
        self._latest_date = latest_date

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        trade_date = max(start, self._latest_date)
        if trade_date > end:
            return []
        return [
            Candle(
                symbol=symbol,
                trade_date=trade_date,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000,
                adj_close=100.0,
            )
        ]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "llm": {"model": "claude-sonnet-5"},
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
        yaml.safe_dump(
            {
                "universes": {
                    "jp_stocks": {
                        "market": "jp",
                        "report": True,
                        "trade": True,
                        "cadence": "daily",
                        "instruments": [{"symbol": "7203.T", "name": "トヨタ自動車"}],
                    },
                    "us_stocks": {
                        "market": "us",
                        "report": True,
                        "trade": False,
                        "cadence": "daily",
                        "instruments": [{"symbol": "AAPL", "name": "Apple Inc."}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestReportCommand:
    def test_report_universe_writes_markdown_file(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["report", "jp_stocks"])

        assert result.exit_code == 0, result.output
        report_file = project / "reports" / f"{date.today().isoformat()}-report.md"
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "jp_stocks" in content
        assert "7203.T" in content
        assert "us_stocks" not in content

    def test_report_all_universes_when_none_specified(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["report"])

        assert result.exit_code == 0, result.output
        content = (
            project / "reports" / f"{date.today().isoformat()}-report.md"
        ).read_text(encoding="utf-8")
        assert "jp_stocks" in content
        assert "us_stocks" in content

    def test_report_unknown_universe_exits_with_config_error_code(self, project: Path) -> None:
        result = runner.invoke(app, ["report", "nope"])

        assert result.exit_code == 3

    def test_report_stale_data_exits_with_data_error_code(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale_date = date.today() - timedelta(days=30)
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(stale_date))

        result = runner.invoke(app, ["report", "jp_stocks"])

        assert result.exit_code == 2


class TestDailyCommand:
    def test_daily_writes_report_and_reports_universe(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["daily", "--no-llm"])

        assert result.exit_code == 0, result.output
        content = (
            project / "reports" / f"{date.today().isoformat()}-daily.md"
        ).read_text(encoding="utf-8")
        assert "jp_stocks" in content
        assert "us_stocks" in content

    def test_daily_reports_no_trade_for_trade_universes(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["daily", "--no-llm"])

        assert result.exit_code == 0, result.output
        content = (
            project / "reports" / f"{date.today().isoformat()}-daily.md"
        ).read_text(encoding="utf-8")
        assert "NO_TRADE" in content
        assert "jp_stocks: NO_TRADE" in content
        assert "us_stocks: NO_TRADE" not in content

    def test_daily_stale_data_exits_with_data_error_code(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale_date = date.today() - timedelta(days=30)
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(stale_date))

        result = runner.invoke(app, ["daily", "--no-llm"])

        assert result.exit_code == 2

    def test_daily_config_error_exits_with_config_error_code(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (project / "config" / "default.yaml").unlink()

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 3

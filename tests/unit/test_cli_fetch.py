"""CLI test for `fund fetch [universe]` (S3): fetch + cache + freshness gate."""

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
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestFetchCommand:
    def test_fetch_universe_caches_candles_and_succeeds(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["fetch", "jp_stocks"])

        assert result.exit_code == 0, result.output

    def test_fetch_unknown_universe_exits_with_config_error_code(self, project: Path) -> None:
        result = runner.invoke(app, ["fetch", "nope"])

        assert result.exit_code == 3

    def test_fetch_stale_data_exits_with_data_error_code(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale_date = date.today() - timedelta(days=30)
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(stale_date))

        result = runner.invoke(app, ["fetch", "jp_stocks"])

        assert result.exit_code == 2

    def test_fetch_all_universes_when_none_specified(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(date.today()))

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output

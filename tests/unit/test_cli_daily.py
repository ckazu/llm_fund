"""End-to-end tests for `fund daily` (technical-spec.md 8, 10章).

Covers the `--no-llm` template smoke path and the LLM judgment path with anthropic
fully mocked, against a temporary DB. Verifies the trade line renders instructions /
rejections / disagreement rate and that instructions + llm_calls are persisted.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from llm_fund.cli import app
from llm_fund.domain.models import Candle
from llm_fund.store.db import init_db
from llm_fund.store.repos import InstructionRepo, LlmCallRepo, PortfolioStateRepo

runner = CliRunner()
TODAY = date.today()


class _StubSource:
    """Returns one candle dated `latest_date` with a close near the order price."""

    def __init__(self, latest_date: date, close: float = 100.0) -> None:
        self._latest_date = latest_date
        self._close = close

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        trade_date = max(start, self._latest_date)
        if trade_date > end:
            return []
        return [
            Candle(
                symbol=symbol,
                trade_date=trade_date,
                open=self._close,
                high=self._close + 1.0,
                low=self._close - 1.0,
                close=self._close,
                volume=1000,
                adj_close=self._close,
            )
        ]


def _write_config(tmp_path: Path, db_path: str) -> None:
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
                "judgment": {"n_samples": 3},
                "db_path": db_path,
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


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, "test.db")
    return tmp_path


def _daily_content(project: Path) -> str:
    return (project / "reports" / f"{TODAY.isoformat()}-daily.md").read_text(encoding="utf-8")


# --- fake anthropic (LLM path) ----------------------------------------------


class _Msg:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [type("B", (), {"type": "tool_use", "input": payload})()]
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class _FakeAnthropic:
    """Always returns the same valid judgment; records every create() call."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.call_count = 0
        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _Msg:
                outer.call_count += 1
                return _Msg(payload)

        self.messages = _Messages()


def _valid_payload() -> dict[str, Any]:
    # entry near the stub close (100) so PriceBandSanity/TickSize pass.
    return {
        "schema_version": 1,
        "market_view": "レンジ上限を試す",
        "no_trade": False,
        "orders": [
            {
                "symbol": "7203.T",
                "action": "BUY",
                "units": 100,
                "entry_price": 110.0,
                "tp_price": 130.0,
                "sl_price": 95.0,
                "valid_days": 3,
                "rationale": "MA25 上抜けの押し目",
            }
        ],
    }


class TestDailyNoLlm:
    def test_template_path_reports_no_trade_and_persists_nothing(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))

        result = runner.invoke(app, ["daily", "--no-llm"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "売買判断（Trade Line）" in content
        assert "jp_stocks: NO_TRADE" in content
        assert "テンプレート判断" in content

        conn = init_db(str(project / "test.db"))
        # template judgment issues no instructions.
        assert InstructionRepo(conn).count_for_date(TODAY) == 0
        # --no-llm makes no LLM calls.
        assert LlmCallRepo(conn).list_all() == []


class TestDailyLlm:
    def test_missing_api_key_exits_config_error(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 3

    def test_llm_path_validates_and_persists_instruction(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _FakeAnthropic(_valid_payload())
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        # Seed portfolio state so the BUY has cash/NAV to validate against.
        conn = init_db(str(project / "test.db"))
        PortfolioStateRepo(conn).upsert(TODAY, cash=10_000_000.0, nav=10_000_000.0)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "LLM 自己一致性判断" in content
        assert "不一致率: 0.0%" in content
        assert "BUY 7203.T" in content

        # unanimous across n_samples=3 -> one validated instruction persisted.
        conn2 = init_db(str(project / "test.db"))
        ticket = f"{TODAY.strftime('%Y%m%d')}-01"
        assert InstructionRepo(conn2).get_by_ticket_no(ticket) is not None
        assert fake.call_count == 3
        assert len(LlmCallRepo(conn2).list_all()) == 3

    def test_llm_buy_without_portfolio_is_rejected(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _FakeAnthropic(_valid_payload())
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        # NAV/cash default to 0 without portfolio state -> BUY rejected, no instruction.
        assert "拒否:" in content
        conn = init_db(str(project / "test.db"))
        assert InstructionRepo(conn).count_for_date(TODAY) == 0

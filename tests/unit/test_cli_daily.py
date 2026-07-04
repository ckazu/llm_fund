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
from llm_fund.store.repos import (
    InstructionRepo,
    InstrumentRepo,
    LlmCallRepo,
    PortfolioStateRepo,
    PositionRepo,
)

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


def _order_payload(
    *, action: str = "BUY", units: int = 100, symbol: str = "7203.T"
) -> dict[str, Any]:
    # entry near the stub close (100) so PriceBandSanity/TickSize pass.
    return {
        "symbol": symbol,
        "action": action,
        "units": units,
        "entry_price": 110.0,
        "tp_price": 130.0,
        "sl_price": 95.0,
        "valid_days": 3,
        "rationale": "MA25 上抜けの押し目",
    }


def _payload(*orders: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "market_view": "レンジ上限を試す",
        "no_trade": False,
        "orders": list(orders),
    }


def _valid_payload() -> dict[str, Any]:
    return _payload(_order_payload())


def _seed_open_position(
    db_path: str, *, symbol: str = "7203.T", units: int, avg_cost: float = 100.0
) -> None:
    """Seed one open virtual position so daily validation sees a prior-day holding."""
    conn = init_db(db_path)
    instrument_repo = InstrumentRepo(conn)
    record = instrument_repo.get_by_symbol(symbol)
    instrument_id = (
        record.id if record is not None else instrument_repo.add(symbol, symbol, "jp")
    )
    PositionRepo(conn).add(
        instrument_id=instrument_id,
        units=units,
        avg_cost=avg_cost,
        opened_at=date(2026, 1, 1),
        closed_at=None,
    )


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

    def test_llm_first_day_buy_validates_against_starting_capital(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On the first daily run portfolio_state is unseeded; the gate must fall back to
        # the configured starting_capital instead of nav=cash=0 (which would reject
        # every BUY and silently discard the first day's judgment).
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _FakeAnthropic(_valid_payload())
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "BUY 7203.T" in content
        conn = init_db(str(project / "test.db"))
        ticket = f"{TODAY.strftime('%Y%m%d')}-01"
        assert InstructionRepo(conn).get_by_ticket_no(ticket) is not None

    def test_llm_sell_of_held_position_validates(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A SELL/CLOSE must be able to validate once a position exists: the gate has to
        # read held units from `positions`, else ExitWithinHolding rejects every exit.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _FakeAnthropic(_payload(_order_payload(action="SELL", units=100)))
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        _seed_open_position(str(project / "test.db"), units=200)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "SELL 7203.T" in content
        conn = init_db(str(project / "test.db"))
        ticket = f"{TODAY.strftime('%Y%m%d')}-01"
        stored = InstructionRepo(conn).get_by_ticket_no(ticket)
        assert stored is not None
        assert stored.action == "SELL"

    def test_prior_holding_caps_second_day_buy(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A prior-day holding must count toward MaxPositionPct: a fresh 500-unit BUY that
        # would pass in isolation (~5.5% of NAV) is rejected because 1000 held units are
        # folded in ((1000+500)*110 = 165,000 > NAV*15% = 150,000).
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _FakeAnthropic(_payload(_order_payload(action="BUY", units=500)))
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        _seed_open_position(str(project / "test.db"), units=1000)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "拒否:" in content
        assert "MaxPositionPct" in content
        conn = init_db(str(project / "test.db"))
        assert InstructionRepo(conn).count_for_date(TODAY) == 0

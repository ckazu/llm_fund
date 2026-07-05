"""End-to-end tests for `fund daily` (technical-spec.md 8, 10章).

Covers the `--no-llm` template smoke path and the LLM judgment path with the
backend fully mocked (`FakeBackend` + a patched router; no subprocess / HTTP),
against a temporary DB. Verifies the trade line renders instructions /
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
from tests.factories import build_llm_config_dict
from tests.fakes import FAKE_MODEL_LABEL, FakeBackend, FakeRouter

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


def _write_config(tmp_path: Path, db_path: str, *, claude_command: str = "claude") -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "llm": build_llm_config_dict(command=claude_command),
                "limits": {
                    "max_position_pct": 15.0,
                    "max_turnover_pct": 30.0,
                    "max_instructions_per_day": 5,
                    "require_stop_loss": True,
                },
                "benchmark": {"index_symbol": "1306.T", "momentum_lookback_days": 120},
                "report": {"output_dir": "reports"},
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


# --- fake LLM backend (LLM path) ----------------------------------------------


def _patch_backend(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> FakeBackend:
    """Route every role to a FakeBackend that always returns `payload`."""
    fake = FakeBackend([payload], repeat_last=True)
    monkeypatch.setattr("llm_fund.cli.build_router", lambda settings: FakeRouter(fake))
    return fake


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
    def test_missing_claude_command_exits_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # claude コマンド不在はバックエンド解決失敗＝設定異常（exit 3）として扱う。
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, "test.db", claude_command="no-such-claude-cmd")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 3

    def test_undefined_judgment_role_exits_config_error(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = project / "config" / "default.yaml"
        conf = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        del conf["llm"]["roles"]["judgment"]
        config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 3

    def test_llm_path_validates_and_persists_instruction(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        fake = _patch_backend(monkeypatch, _valid_payload())

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
        calls = LlmCallRepo(conn2).list_all()
        assert len(calls) == 3
        # llm_calls.model には "backend:model" ラベルを記録する。
        assert all(call.model == FAKE_MODEL_LABEL for call in calls)

    def test_llm_first_day_buy_validates_against_starting_capital(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On the first daily run portfolio_state is unseeded; the gate must fall back to
        # the configured starting_capital instead of nav=cash=0 (which would reject
        # every BUY and silently discard the first day's judgment).
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        _patch_backend(monkeypatch, _valid_payload())

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
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        _patch_backend(monkeypatch, _payload(_order_payload(action="SELL", units=100)))

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
        monkeypatch.setattr("llm_fund.cli.YFinanceSource", lambda: _StubSource(TODAY))
        _patch_backend(monkeypatch, _payload(_order_payload(action="BUY", units=500)))

        _seed_open_position(str(project / "test.db"), units=1000)

        result = runner.invoke(app, ["daily"])

        assert result.exit_code == 0, result.output
        content = _daily_content(project)
        assert "拒否:" in content
        assert "MaxPositionPct" in content
        conn = init_db(str(project / "test.db"))
        assert InstructionRepo(conn).count_for_date(TODAY) == 0

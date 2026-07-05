"""CLI test for `fund benchmark` (S8: virtual fill + control benchmarks)."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_fund.cli import app
from llm_fund.domain.enums import Action, InstructionStatus
from llm_fund.domain.models import Candle
from llm_fund.store.db import init_db
from llm_fund.store.repos import (
    BenchmarkNavRepo,
    BriefingRepo,
    CandleRepo,
    InstructionRepo,
    InstrumentRepo,
    StrategyRepo,
    UniverseRepo,
    VirtualFillRepo,
)
from llm_fund.tracking.benchmark import STRATEGY_INDEX
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
                "tracking": {
                    "starting_capital": 1000000.0,
                    "commission_rate": 0.0,
                    "min_commission": 0.0,
                    "slippage_pct": 0.0,
                    "random_seed": 42,
                },
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
                        "instruments": [
                            {"symbol": "7203.T", "name": "トヨタ"},
                            {"symbol": "6758.T", "name": "ソニー"},
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _candle(symbol: str, day: int, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        trade_date=date(2026, 1, day),
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000,
        adj_close=close,
    )


def _seed_market(project: Path) -> None:
    conn = init_db(str(project / "test.db"))
    universe_id = UniverseRepo(conn).add("jp_stocks", "jp", trade_enabled=True)
    briefing_id = BriefingRepo(conn).add(universe_id, date(2026, 1, 5), "daily", "md", "{}")
    for symbol, name, closes in [
        ("1306.T", "TOPIX ETF", [1000.0, 1000.0, 1000.0, 1000.0, 1100.0]),
        ("7203.T", "トヨタ", [500.0, 500.0, 500.0, 500.0, 550.0]),
        ("6758.T", "ソニー", [200.0, 210.0, 220.0, 230.0, 260.0]),
    ]:
        iid = InstrumentRepo(conn).add(symbol, name, "jp")
        CandleRepo(conn).upsert_many(
            iid, [_candle(symbol, 5 + i, c) for i, c in enumerate(closes)]
        )
    # BUY 7203.T: entry 500, tp 560, sl 480, activation day6+, fills day6, TP day9(550<560? no)
    toyota = InstrumentRepo(conn).get_by_symbol("7203.T")
    assert toyota is not None
    InstructionRepo(conn).add(
        ticket_no="20260105-01",
        briefing_id=briefing_id,
        instrument_id=toyota.id,
        action=Action.BUY.value,
        units=100,
        entry_price=500.0,
        tp_price=540.0,
        sl_price=480.0,
        valid_until=date(2026, 1, 12),
        rationale="モメンタム継続を狙う指値エントリー",
        validator_result_json=None,
        status=InstructionStatus.PENDING.value,
    )
    conn.close()


class TestBenchmarkCommand:
    def test_benchmark_writes_navs_and_virtual_fill(self, project: Path) -> None:
        _seed_market(project)

        result = runner.invoke(app, ["benchmark", "--date", "2026-01-09"])

        assert result.exit_code == 0, result.output
        conn = init_db(str(project / "test.db"))
        # 仮想執行が指示を約定させ virtual_fills を書いている
        fills = VirtualFillRepo(conn).list_all()
        assert len(fills) == 1
        assert fills[0].fill_date is not None
        # 対照群の NAV が書かれている
        index = StrategyRepo(conn).get_by_code(STRATEGY_INDEX)
        assert index is not None
        assert BenchmarkNavRepo(conn).latest(index.id) is not None

    def test_benchmark_idempotent(self, project: Path) -> None:
        _seed_market(project)
        runner.invoke(app, ["benchmark", "--date", "2026-01-09"])
        runner.invoke(app, ["benchmark", "--date", "2026-01-09"])

        conn = init_db(str(project / "test.db"))
        assert len(VirtualFillRepo(conn).list_all()) == 1
        rows = conn.execute("SELECT COUNT(*) AS n FROM portfolio_state").fetchone()
        assert rows["n"] == 1

    def test_benchmark_unknown_config_exits_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["benchmark"])
        assert result.exit_code == 3

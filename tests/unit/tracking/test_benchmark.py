"""対照群ベンチマークの指標・NAV を手計算と突合（S8 / spec 7章 / FR-5）。"""

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.models import Candle
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    BenchmarkNavRepo,
    BriefingRepo,
    CandleRepo,
    InstructionRepo,
    InstrumentRepo,
    PortfolioStateRepo,
    StrategyRepo,
    UniverseRepo,
    VirtualFillRepo,
)
from llm_fund.tracking.benchmark import (
    STRATEGY_FUND,
    STRATEGY_INDEX,
    common_window,
    compute_metrics,
    run_benchmark,
    select_momentum,
    select_random,
    simulate_weighted_buy_hold,
)
from llm_fund.tracking.virtual_fill import CostModel

STARTING = 1_000_000.0
NO_COST = CostModel()


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _candle(symbol: str, day: int, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        trade_date=date(2026, 1, day),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000,
        adj_close=close,
    )


def _seed_symbol(conn: sqlite3.Connection, symbol: str, closes: dict[int, float]) -> None:
    instrument_id = InstrumentRepo(conn).add(symbol, symbol, "jp")
    CandleRepo(conn).upsert_many(
        instrument_id, [_candle(symbol, day, c) for day, c in closes.items()]
    )


def _series(closes: list[float]) -> list[tuple[date, float]]:
    return [(date(2026, 1, i + 1), c) for i, c in enumerate(closes)]


class TestMetrics:
    def test_max_drawdown(self) -> None:
        metrics = compute_metrics(
            _series([100, 120, 90, 150]),
            total_cost=0.0,
            deployed_notional=0.0,
            starting_capital=STARTING,
        )
        # peak 120 → 90: (90-120)/120 = -25%
        assert metrics["max_drawdown_pct"] == pytest.approx(25.0)

    def test_win_rate(self) -> None:
        metrics = compute_metrics(
            _series([100, 120, 90, 150]),
            total_cost=0.0,
            deployed_notional=0.0,
            starting_capital=STARTING,
        )
        # returns: +0.2, -0.25, +0.6667 → 2/3 positive
        assert metrics["win_rate"] == pytest.approx(2 / 3)

    def test_sharpe_zero_when_no_variance(self) -> None:
        metrics = compute_metrics(
            _series([100, 110, 121]),  # constant +10% returns → std 0
            total_cost=0.0,
            deployed_notional=0.0,
            starting_capital=STARTING,
        )
        assert metrics["sharpe"] == pytest.approx(0.0)

    def test_sharpe_zero_with_single_point(self) -> None:
        metrics = compute_metrics(
            _series([100]),
            total_cost=0.0,
            deployed_notional=0.0,
            starting_capital=STARTING,
        )
        assert metrics["sharpe"] == pytest.approx(0.0)

    def test_turnover_and_cost_ratio(self) -> None:
        metrics = compute_metrics(
            _series([100, 110]),
            total_cost=2_000.0,
            deployed_notional=998_000.0,
            starting_capital=STARTING,
        )
        assert metrics["turnover"] == pytest.approx(0.998)
        assert metrics["cost_ratio"] == pytest.approx(0.002)


class TestBuyHold:
    def test_no_cost_marks_to_market(self) -> None:
        candles = {"IDX": [_candle("IDX", 1, 1000.0), _candle("IDX", 2, 1100.0)]}
        run = simulate_weighted_buy_hold(
            {"IDX": 1.0},
            candles,
            [date(2026, 1, 1), date(2026, 1, 2)],
            starting_capital=STARTING,
            costs=NO_COST,
        )
        assert run.nav_series[0][1] == pytest.approx(1_000_000.0)
        assert run.nav_series[1][1] == pytest.approx(1_100_000.0)
        assert run.total_cost == pytest.approx(0.0)

    def test_entry_cost_reduces_initial_nav(self) -> None:
        candles = {"IDX": [_candle("IDX", 1, 1000.0), _candle("IDX", 2, 1100.0)]}
        costs = CostModel(commission_rate=0.001, slippage_pct=0.001)
        run = simulate_weighted_buy_hold(
            {"IDX": 1.0},
            candles,
            [date(2026, 1, 1), date(2026, 1, 2)],
            starting_capital=STARTING,
            costs=costs,
        )
        investable = STARTING / 1.002
        assert run.nav_series[0][1] == pytest.approx(investable)
        assert run.total_cost == pytest.approx(investable * 0.002)
        assert run.nav_series[1][1] == pytest.approx(investable * 1.1)


class TestSelection:
    def test_momentum_picks_highest_riser(self) -> None:
        candles = {
            "A": [_candle("A", 1, 100.0), _candle("A", 2, 200.0)],  # +100%
            "B": [_candle("B", 1, 100.0), _candle("B", 2, 110.0)],  # +10%
        }
        assert select_momentum(candles, date(2026, 1, 2), 5) == "A"

    def test_random_is_reproducible(self) -> None:
        symbols = ["A", "B", "C", "D"]
        assert select_random(symbols, 42) == select_random(symbols, 42)
        assert select_random(symbols, 42) in symbols

    def test_random_none_when_empty(self) -> None:
        assert select_random([], 42) is None


class TestCommonWindow:
    def test_intersection_of_available_dates(self) -> None:
        candles = {
            "A": [_candle("A", 1, 1.0), _candle("A", 2, 1.0), _candle("A", 3, 1.0)],
            "B": [_candle("B", 2, 1.0), _candle("B", 3, 1.0)],
        }
        assert common_window(candles, date(2026, 1, 3)) == [
            date(2026, 1, 2),
            date(2026, 1, 3),
        ]


class TestRunBenchmark:
    def _seed(self, conn: sqlite3.Connection) -> None:
        # days 1-2 = モメンタムのルックバック履歴, days 3-4 = fund 運用（比較ウィンドウ）
        _seed_symbol(conn, "1306.T", {1: 1000.0, 2: 1000.0, 3: 1000.0, 4: 1100.0})
        _seed_symbol(conn, "7203.T", {1: 500.0, 2: 500.0, 3: 500.0, 4: 550.0})
        _seed_symbol(conn, "6758.T", {1: 200.0, 2: 220.0, 3: 240.0, 4: 260.0})
        PortfolioStateRepo(conn).upsert(date(2026, 1, 3), 1_000_000.0, 1_000_000.0)
        PortfolioStateRepo(conn).upsert(date(2026, 1, 4), 1_000_000.0, 1_020_000.0)

    def test_writes_all_strategies(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        summary = run_benchmark(
            conn,
            date(2026, 1, 4),
            universe_symbols=["7203.T", "6758.T"],
            index_symbol="1306.T",
            momentum_lookback_days=120,
            random_seed=42,
            costs=NO_COST,
            starting_capital=STARTING,
        )
        assert set(summary.latest_nav) == {
            STRATEGY_FUND,
            STRATEGY_INDEX,
            "equal_weight",
            "momentum",
            "random",
        }
        # index buy&hold: 1_000_000 * 1100/1000 = 1_100_000
        assert summary.latest_nav[STRATEGY_INDEX] == pytest.approx(1_100_000.0)
        # fund は portfolio_state の最新 NAV
        assert summary.latest_nav[STRATEGY_FUND] == pytest.approx(1_020_000.0)

    def test_idempotent_rerun(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        kwargs = {
            "universe_symbols": ["7203.T", "6758.T"],
            "index_symbol": "1306.T",
            "momentum_lookback_days": 120,
            "random_seed": 42,
            "costs": NO_COST,
            "starting_capital": STARTING,
        }
        run_benchmark(conn, date(2026, 1, 4), **kwargs)  # type: ignore[arg-type]
        run_benchmark(conn, date(2026, 1, 4), **kwargs)  # type: ignore[arg-type]
        rows = conn.execute("SELECT COUNT(*) AS n FROM benchmark_navs").fetchone()
        # 5 strategies * 2 dates = 10（二重計上しない）
        assert rows["n"] == 10

    def test_fund_cost_and_turnover_reflect_virtual_fills(
        self, conn: sqlite3.Connection
    ) -> None:
        # fund の cost_ratio/turnover は virtual_fills の実コスト・実投下代金から算出する
        # （0 埋めだと対照群だけ有コスト・有回転に見え FR-5 の比較が不公平になる）。
        self._seed(conn)
        universe_id = UniverseRepo(conn).add("jp", "jp", trade_enabled=True)
        instrument_id = InstrumentRepo(conn).get_by_symbol("7203.T").id  # type: ignore[union-attr]
        briefing_id = BriefingRepo(conn).add(universe_id, date(2026, 1, 3), "daily", "md", "{}")
        instruction_id = InstructionRepo(conn).add(
            ticket_no="20260103-01",
            briefing_id=briefing_id,
            instrument_id=instrument_id,
            action="BUY",
            units=100,
            entry_price=500.0,
            tp_price=550.0,
            sl_price=480.0,
            valid_until=date(2026, 1, 4),
            rationale="test",
            validator_result_json=None,
            status="pending",
        )
        VirtualFillRepo(conn).upsert(
            instruction_id=instruction_id,
            fill_date=date(2026, 1, 3),
            fill_price=500.0,
            exit_date=None,
            exit_price=None,
            exit_reason=None,
            commission=250.0,
            slippage=500.0,
            pnl=None,
        )

        summary = run_benchmark(
            conn,
            date(2026, 1, 4),
            universe_symbols=["7203.T", "6758.T"],
            index_symbol="1306.T",
            momentum_lookback_days=120,
            random_seed=42,
            costs=NO_COST,
            starting_capital=STARTING,
        )

        fund_metrics = summary.metrics[STRATEGY_FUND]
        # deployed_notional = 500 * 100 = 50,000 -> turnover = 0.05
        assert fund_metrics["turnover"] == pytest.approx(50_000.0 / STARTING)
        # total_cost = 250 + 500 = 750 -> cost_ratio = 0.00075
        assert fund_metrics["cost_ratio"] == pytest.approx(750.0 / STARTING)

    def test_metrics_json_persisted(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        run_benchmark(
            conn,
            date(2026, 1, 4),
            universe_symbols=["7203.T", "6758.T"],
            index_symbol="1306.T",
            momentum_lookback_days=120,
            random_seed=42,
            costs=NO_COST,
            starting_capital=STARTING,
        )
        strategy_id = StrategyRepo(conn).get_by_code(STRATEGY_INDEX)
        assert strategy_id is not None
        latest = BenchmarkNavRepo(conn).latest(strategy_id.id)
        assert latest is not None
        assert latest.metrics_json is not None
        metrics = json.loads(latest.metrics_json)
        assert set(metrics) == {
            "max_drawdown_pct",
            "sharpe",
            "win_rate",
            "turnover",
            "cost_ratio",
        }

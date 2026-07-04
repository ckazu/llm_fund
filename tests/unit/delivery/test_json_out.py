"""Schema tests for `delivery/json_out.py` (S10: `--format json`)."""

from datetime import date

from llm_fund.delivery.json_out import (
    render_benchmark_json,
    render_daily_json,
    render_gate_decision_json,
    render_monthly_json,
    render_report_json,
    render_status_json,
    render_weekly_json,
)
from llm_fund.domain.enums import Action
from llm_fund.domain.models import OrderPlan, Rejection, ValidatedInstruction
from llm_fund.review.monthly import MonthlyReviewResult, UniverseChangeWire
from llm_fund.review.weekly import WeeklyPerformanceSummary, WeeklyReviewResult
from llm_fund.tracking.benchmark import BenchmarkSummary
from llm_fund.validator.gate import GateDecision

VALID_UNTIL = date(2026, 7, 5)


def _validated() -> ValidatedInstruction:
    return ValidatedInstruction(
        ticket_no="20260704-01",
        symbol="7203.T",
        action=Action.BUY,
        units=100,
        entry_price=2000.0,
        tp_price=2200.0,
        sl_price=1950.0,
        valid_until=VALID_UNTIL,
        rationale="上昇トレンド継続",
    )


def _rejection() -> Rejection:
    order = OrderPlan(
        symbol="6758.T",
        action=Action.BUY,
        units=100,
        entry_price=1000.0,
        tp_price=1100.0,
        sl_price=950.0,
        valid_until=VALID_UNTIL,
        rationale="test",
    )
    return Rejection(order=order, reasons=["max_position_pct超過"])


class TestRenderReportJson:
    def test_shape(self) -> None:
        payload = render_report_json(["jp_stocks"], "reports/2026-07-04-report.md", [])

        assert payload == {
            "universes": ["jp_stocks"],
            "report_path": "reports/2026-07-04-report.md",
            "stale_messages": [],
        }


class TestRenderGateDecisionJson:
    def test_includes_validated_and_rejections(self) -> None:
        decision = GateDecision(validated=[_validated()], rejections=[_rejection()])

        payload = render_gate_decision_json("jp_stocks", decision)

        assert payload["universe"] == "jp_stocks"
        assert payload["no_trade"] is False
        assert payload["instructions"][0]["ticket_no"] == "20260704-01"
        assert payload["instructions"][0]["action"] == "BUY"
        assert payload["rejections"][0]["symbol"] == "6758.T"
        assert payload["rejections"][0]["reasons"] == ["max_position_pct超過"]

    def test_no_trade_decision(self) -> None:
        decision = GateDecision(no_trade=True, no_trade_reason="鮮度違反")

        payload = render_gate_decision_json("jp_stocks", decision)

        assert payload["no_trade"] is True
        assert payload["no_trade_reason"] == "鮮度違反"
        assert payload["instructions"] == []
        assert payload["rejections"] == []


class TestRenderDailyJson:
    def test_combines_report_and_trade(self) -> None:
        decision = GateDecision(no_trade=True, no_trade_reason="鮮度違反")

        payload = render_daily_json(
            "reports/2026-07-04-daily.md", ["jp_stocks", "us_stocks"], {"jp_stocks": decision}
        )

        assert payload["report_universes"] == ["jp_stocks", "us_stocks"]
        assert len(payload["trade"]) == 1
        assert payload["trade"][0]["universe"] == "jp_stocks"


class TestRenderWeeklyJson:
    def test_shape(self) -> None:
        result = WeeklyReviewResult(
            performance=WeeklyPerformanceSummary(
                n_instructions=3,
                n_filled=2,
                win_rate=0.5,
                avg_pnl=1000.0,
                avg_tp_width_pct=1.2,
                avg_sl_width_pct=0.8,
                unexecuted_rate=0.1,
            ),
            no_change=False,
            rationale="勝率低下",
            criteria_id=7,
            diff="SL幅拡大",
        )

        payload = render_weekly_json(result)

        assert payload == {
            "no_change": False,
            "rationale": "勝率低下",
            "criteria_id": 7,
            "diff": "SL幅拡大",
        }


class TestRenderMonthlyJson:
    def test_includes_universe_changes(self) -> None:
        result = MonthlyReviewResult(
            no_change=False,
            rationale="対照群に劣後",
            policy_id=3,
            diff="ETF比率引き上げ",
            universe_changes=[
                UniverseChangeWire(
                    universe_code="jp_stocks",
                    symbol="9999.T",
                    action="remove",
                    reason="流動性低下",
                )
            ],
        )

        payload = render_monthly_json(result)

        assert payload["policy_id"] == 3
        assert payload["universe_changes"] == [
            {
                "universe_code": "jp_stocks",
                "symbol": "9999.T",
                "action": "remove",
                "reason": "流動性低下",
            }
        ]


class TestRenderBenchmarkJson:
    def test_shape(self) -> None:
        summary = BenchmarkSummary(
            latest_nav={"fund": 1_050_000.0},
            metrics={"fund": {"max_drawdown_pct": 2.0, "sharpe": 1.1}},
        )

        payload = render_benchmark_json(summary)

        assert payload == {
            "latest_nav": {"fund": 1_050_000.0},
            "metrics": {"fund": {"max_drawdown_pct": 2.0, "sharpe": 1.1}},
        }


class TestRenderStatusJson:
    def test_shape(self) -> None:
        payload = render_status_json(
            nav=1_000_000.0,
            nav_date="2026-07-04",
            cash=500_000.0,
            pending_instructions=[{"ticket_no": "20260704-01", "action": "BUY", "units": 100}],
            unrecorded_proposals=[{"kind": "criteria", "id": 7}],
            unexecuted_rate=0.1,
        )

        assert payload["nav"] == 1_000_000.0
        assert payload["pending_proposals"] == [{"kind": "criteria", "id": 7}]

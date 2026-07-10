"""Validation-boundary tests for frozen domain models."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from llm_fund.domain.enums import Action, ExecutionStatus, FillExitReason
from llm_fund.domain.models import (
    Candle,
    ExecutionRecord,
    JudgmentResult,
    OrderPlan,
    Rejection,
    ValidatedInstruction,
    VirtualFill,
)


def _candle(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "7203.T",
        "trade_date": date(2026, 7, 1),
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1000,
        "adj_close": 105.0,
    }
    base.update(overrides)
    return base


class TestCandle:
    def test_valid_candle_constructs(self) -> None:
        candle = Candle(**_candle())
        assert candle.symbol == "7203.T"

    def test_is_frozen(self) -> None:
        candle = Candle(**_candle())
        with pytest.raises(ValidationError):
            candle.close = 999.0  # type: ignore[misc]

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(**_candle(high=90.0, low=95.0))

    def test_high_below_open_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(**_candle(open=150.0, high=110.0))

    def test_low_above_close_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(**_candle(close=50.0, low=95.0))

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(**_candle(open=0))

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(**_candle(volume=-1))

    def test_zero_volume_allowed(self) -> None:
        candle = Candle(**_candle(volume=0))
        assert candle.volume == 0


def _order_plan(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "7203.T",
        "action": Action.BUY,
        "units": 100,
        "entry_price": 3120.0,
        "tp_price": 3320.0,
        "sl_price": 3020.0,
        "valid_until": date(2026, 7, 10),
        "rationale": "モメンタム継続を確認",
    }
    base.update(overrides)
    return base


class TestOrderPlan:
    def test_valid_order_plan_constructs(self) -> None:
        plan = OrderPlan(**_order_plan())
        assert plan.action == Action.BUY

    def test_zero_units_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderPlan(**_order_plan(units=0))

    def test_negative_units_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderPlan(**_order_plan(units=-100))

    def test_empty_rationale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderPlan(**_order_plan(rationale=""))

    def test_non_positive_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderPlan(**_order_plan(entry_price=0))


class TestJudgmentResult:
    def test_no_trade_with_orders_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgmentResult(
                no_trade=True,
                no_trade_reason="data stale",
                orders=[OrderPlan(**_order_plan())],
            )

    def test_no_trade_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgmentResult(no_trade=True, no_trade_reason=None)

    def test_no_trade_with_reason_and_no_orders_ok(self) -> None:
        result = JudgmentResult(no_trade=True, no_trade_reason="data stale")
        assert result.orders == []

    def test_trade_case_with_orders_ok(self) -> None:
        result = JudgmentResult(orders=[OrderPlan(**_order_plan())])
        assert len(result.orders) == 1


class TestValidatedInstruction:
    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "ticket_no": "20260704-01",
            "symbol": "7203.T",
            "action": Action.BUY,
            "units": 100,
            "entry_price": 3120.0,
            "tp_price": 3320.0,
            "sl_price": 3020.0,
            "valid_until": date(2026, 7, 10),
            "rationale": "モメンタム継続を確認",
        }
        base.update(overrides)
        return base

    def test_valid_ticket_no_format(self) -> None:
        instruction = ValidatedInstruction(**self._kwargs())
        assert instruction.ticket_no == "20260704-01"

    @pytest.mark.parametrize(
        "bad_ticket_no",
        ["2026-07-04-01", "20260704", "20260704-1", "abcdefgh-01"],
    )
    def test_invalid_ticket_no_format_rejected(self, bad_ticket_no: str) -> None:
        with pytest.raises(ValidationError):
            ValidatedInstruction(**self._kwargs(ticket_no=bad_ticket_no))


class TestRejection:
    def test_requires_at_least_one_reason(self) -> None:
        with pytest.raises(ValidationError):
            Rejection(order=OrderPlan(**_order_plan()), reasons=[])


class TestExecutionRecord:
    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "ticket_no": "20260704-01",
            "executed_at": datetime(2026, 7, 4, 10, 30),
            "side": Action.BUY,
            "order_type": "IFO",
            "actual_price": 3120.0,
            "actual_units": 100,
            "commission": 55.0,
            "status": ExecutionStatus.FILLED,
        }
        base.update(overrides)
        return base

    def test_filled_without_skip_reason_ok(self) -> None:
        record = ExecutionRecord(**self._kwargs())
        assert record.status == ExecutionStatus.FILLED

    def test_skipped_requires_skip_reason(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionRecord(**self._kwargs(status=ExecutionStatus.SKIPPED, skip_reason=None))

    def test_skipped_with_reason_ok(self) -> None:
        record = ExecutionRecord(
            **self._kwargs(status=ExecutionStatus.SKIPPED, skip_reason="市場休場のため見送り")
        )
        assert record.skip_reason == "市場休場のため見送り"


class TestVirtualFill:
    def test_exit_without_fill_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VirtualFill(
                ticket_no="20260704-01",
                fill_date=None,
                exit_date=date(2026, 7, 5),
                exit_reason=FillExitReason.TP,
            )

    def test_fill_without_exit_ok(self) -> None:
        fill = VirtualFill(ticket_no="20260704-01", fill_date=date(2026, 7, 4), fill_price=3120.0)
        assert fill.exit_date is None

    def test_fill_with_exit_ok(self) -> None:
        fill = VirtualFill(
            ticket_no="20260704-01",
            fill_date=date(2026, 7, 4),
            fill_price=3120.0,
            exit_date=date(2026, 7, 5),
            exit_price=3320.0,
            exit_reason=FillExitReason.TP,
            pnl=200.0,
        )
        assert fill.exit_reason == FillExitReason.TP

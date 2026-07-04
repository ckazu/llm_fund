"""Boundary tests for the LLM wire-format schemas (technical-spec.md 5章).

The wire schema is the trust boundary for LLM output: everything that reaches
the validator/tracking layers must first survive these checks. Coverage focuses
on the failure edges named in the spec (unknown/wrong action, non-positive
prices/units, lot-multiple violations, missing fields, schema_version mismatch,
and the no_trade consistency invariant) and on the valid_days → valid_until
conversion performed on the way to the domain model.
"""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from llm_fund.domain.enums import Action
from llm_fund.judgment.schemas import (
    DEFAULT_LOT_SIZE,
    MIN_VALID_DAYS,
    SCHEMA_VERSION,
    LlmJudgment,
    LlmOrderPlan,
)

AS_OF = date(2026, 7, 4)


def _order(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "7203.T",
        "action": "BUY",
        "units": 100,
        "entry_price": 3120.0,
        "tp_price": 3320.0,
        "sl_price": 3020.0,
        "valid_days": 3,
        "rationale": "MA25 上抜けの押し目",
    }
    base.update(overrides)
    return base


def _judgment(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "market_view": "レンジ上限を試す展開",
        "orders": [_order()],
    }
    base.update(overrides)
    return base


class TestLlmOrderPlan:
    def test_valid_order_constructs(self) -> None:
        order = LlmOrderPlan(**_order())
        assert order.symbol == "7203.T"
        assert order.action is Action.BUY

    def test_action_accepts_sell_and_close(self) -> None:
        assert LlmOrderPlan(**_order(action="SELL")).action is Action.SELL
        assert LlmOrderPlan(**_order(action="CLOSE")).action is Action.CLOSE

    def test_unknown_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(action="HODL"))

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(symbol=""))

    def test_zero_units_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(units=0))

    def test_negative_units_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(units=-100))

    def test_units_not_lot_multiple_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(units=DEFAULT_LOT_SIZE + 1))

    def test_units_exact_lot_multiple_accepted(self) -> None:
        order = LlmOrderPlan(**_order(units=DEFAULT_LOT_SIZE * 4))
        assert order.units == DEFAULT_LOT_SIZE * 4

    @pytest.mark.parametrize("field", ["entry_price", "tp_price", "sl_price"])
    def test_non_positive_prices_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(**{field: 0.0}))
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(**{field: -1.0}))

    def test_valid_days_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(valid_days=MIN_VALID_DAYS - 1))

    def test_valid_days_at_minimum_accepted(self) -> None:
        order = LlmOrderPlan(**_order(valid_days=MIN_VALID_DAYS))
        assert order.valid_days == MIN_VALID_DAYS

    def test_empty_rationale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(rationale=""))

    @pytest.mark.parametrize(
        "field",
        [
            "symbol",
            "action",
            "units",
            "entry_price",
            "tp_price",
            "sl_price",
            "valid_days",
            "rationale",
        ],
    )
    def test_missing_required_field_rejected(self, field: str) -> None:
        payload = _order()
        del payload[field]
        with pytest.raises(ValidationError):
            LlmOrderPlan(**payload)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmOrderPlan(**_order(valid_until="2026-07-07"))

    def test_to_order_plan_resolves_valid_until(self) -> None:
        order = LlmOrderPlan(**_order(valid_days=3))
        plan = order.to_order_plan(AS_OF)
        assert plan.valid_until == AS_OF + timedelta(days=3)
        assert plan.symbol == "7203.T"
        assert plan.units == 100


class TestLlmJudgment:
    def test_valid_judgment_constructs(self) -> None:
        judgment = LlmJudgment(**_judgment())
        assert judgment.schema_version == SCHEMA_VERSION
        assert len(judgment.orders) == 1

    def test_schema_version_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmJudgment(**_judgment(schema_version=SCHEMA_VERSION + 1))

    def test_missing_schema_version_rejected(self) -> None:
        payload = _judgment()
        del payload["schema_version"]
        with pytest.raises(ValidationError):
            LlmJudgment(**payload)

    def test_empty_market_view_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmJudgment(**_judgment(market_view=""))

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmJudgment(**_judgment(disagreement_rate=0.1))

    def test_no_trade_with_orders_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmJudgment(
                **_judgment(no_trade=True, no_trade_reason="様子見", orders=[_order()])
            )

    def test_no_trade_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LlmJudgment(**_judgment(no_trade=True, orders=[]))

    def test_no_trade_with_reason_and_no_orders_accepted(self) -> None:
        judgment = LlmJudgment(
            **_judgment(no_trade=True, no_trade_reason="鮮度不足", orders=[])
        )
        assert judgment.no_trade is True
        assert judgment.orders == []

    def test_orders_default_to_empty(self) -> None:
        payload = _judgment()
        del payload["orders"]
        judgment = LlmJudgment(**payload)
        assert judgment.orders == []

    def test_to_judgment_result_converts_all_orders(self) -> None:
        payload = _judgment(
            orders=[_order(symbol="7203.T"), _order(symbol="6758.T", valid_days=5)]
        )
        judgment = LlmJudgment(**payload)
        result = judgment.to_judgment_result(AS_OF)
        assert [o.symbol for o in result.orders] == ["7203.T", "6758.T"]
        assert result.orders[1].valid_until == AS_OF + timedelta(days=5)
        assert result.market_view == "レンジ上限を試す展開"
        assert result.no_trade is False

    def test_to_judgment_result_preserves_no_trade(self) -> None:
        judgment = LlmJudgment(
            **_judgment(no_trade=True, no_trade_reason="データ鮮度不足", orders=[])
        )
        result = judgment.to_judgment_result(AS_OF)
        assert result.no_trade is True
        assert result.no_trade_reason == "データ鮮度不足"
        assert result.orders == []

"""Tests for the deterministic --no-llm template judgment (technical-spec.md 5章).

The template must be deterministic (same output every call) and must never emit
trade instructions — it exists to smoke-test the pipeline without an LLM call.
"""

from llm_fund.domain.models import JudgmentResult
from llm_fund.judgment.template import (
    TEMPLATE_MARKET_VIEW,
    TEMPLATE_NO_TRADE_REASON,
    template_judgment,
)


class TestTemplateJudgment:
    def test_returns_judgment_result(self) -> None:
        assert isinstance(template_judgment(), JudgmentResult)

    def test_always_no_trade_with_no_orders(self) -> None:
        result = template_judgment()
        assert result.no_trade is True
        assert result.orders == []

    def test_carries_template_reason_and_market_view(self) -> None:
        result = template_judgment()
        assert result.no_trade_reason == TEMPLATE_NO_TRADE_REASON
        assert result.market_view == TEMPLATE_MARKET_VIEW

    def test_is_deterministic(self) -> None:
        assert template_judgment() == template_judgment()

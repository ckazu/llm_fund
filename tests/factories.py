"""Shared object factories for tests.

Single source for constructing the objects that several test modules previously
duplicated: the domain `OrderPlan`, the validator `InstrumentContext` /
`RiskLimits` / `ValidationContext`, and the LLM wire-format payload dicts.

Each test module keeps its own default *values* (e.g. test_rules uses a
3,000-yen instrument, test_gate a 2,000-yen one) by passing overrides; the field
wiring lives here so adding a field to any of these shapes touches one place
rather than every test module (the maintainability concern in the review).
"""

from datetime import date
from typing import Any

from llm_fund.domain.enums import Action
from llm_fund.domain.models import OrderPlan
from llm_fund.validator.rules import InstrumentContext, RiskLimits, ValidationContext

DEFAULT_VALID_UNTIL = date(2026, 7, 7)
DEFAULT_RATIONALE = "MA25 を上抜け出来高も伴い上昇トレンド継続と判断"


def build_order_plan(
    *,
    symbol: str = "7203.T",
    action: Action = Action.BUY,
    units: int = 100,
    entry_price: float = 2000.0,
    tp_price: float = 2200.0,
    sl_price: float = 1950.0,
    valid_until: date = DEFAULT_VALID_UNTIL,
    rationale: str = DEFAULT_RATIONALE,
) -> OrderPlan:
    return OrderPlan(
        symbol=symbol,
        action=action,
        units=units,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        valid_until=valid_until,
        rationale=rationale,
    )


def build_instrument_context(
    *,
    symbol: str = "7203.T",
    in_universe: bool = True,
    lot_size: int = 100,
    prev_close: float = 2000.0,
    current_units: int = 0,
) -> InstrumentContext:
    return InstrumentContext(
        symbol=symbol,
        in_universe=in_universe,
        lot_size=lot_size,
        prev_close=prev_close,
        current_units=current_units,
    )


def build_risk_limits(
    *,
    max_loss_per_trade_pct: float = 3.0,
    max_position_pct: float = 25.0,
    max_exposure_pct: float = 100.0,
    max_turnover_pct: float = 30.0,
    min_rationale_length: int = 20,
    require_stop_loss: bool = True,
    max_instructions_per_day: int = 5,
) -> RiskLimits:
    return RiskLimits(
        max_loss_per_trade_pct=max_loss_per_trade_pct,
        max_position_pct=max_position_pct,
        max_exposure_pct=max_exposure_pct,
        max_turnover_pct=max_turnover_pct,
        min_rationale_length=min_rationale_length,
        require_stop_loss=require_stop_loss,
        max_instructions_per_day=max_instructions_per_day,
    )


def build_validation_context(
    *,
    nav: float = 1_000_000.0,
    cash: float = 1_000_000.0,
    current_exposure: float = 0.0,
    instruments: dict[str, InstrumentContext],
    limits: RiskLimits,
    data_fresh: bool = True,
) -> ValidationContext:
    return ValidationContext(
        nav=nav,
        cash=cash,
        current_exposure=current_exposure,
        instruments=instruments,
        limits=limits,
        data_fresh=data_fresh,
    )


def build_llm_config_dict(
    *, command: str = "claude", n_samples: int = 3
) -> dict[str, Any]:
    """config/default.yaml の `llm` セクション相当（新形式: roles + backends）。"""
    return {
        "ratio_only": True,
        "judgment": {"n_samples": n_samples},
        "roles": {
            "judgment": {"backend": "claude_cli", "model": "sonnet"},
            "weekly_review": {"backend": "claude_cli", "model": "opus"},
            "monthly_review": {"backend": "claude_cli", "model": "opus"},
        },
        "backends": {"claude_cli": {"command": command, "timeout_seconds": 300}},
    }


def build_llm_order_payload(
    *,
    symbol: str = "7203.T",
    action: str = "BUY",
    units: int = 100,
    entry_price: float = 3000.0,
    tp_price: float = 3300.0,
    sl_price: float = 2900.0,
    valid_days: int = 3,
    rationale: str = "上昇トレンド継続と判断",
) -> dict[str, Any]:
    """LLM wire-format order dict (before schema validation)."""
    return {
        "symbol": symbol,
        "action": action,
        "units": units,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "valid_days": valid_days,
        "rationale": rationale,
    }


def build_llm_judgment_payload(
    orders: list[dict[str, Any]],
    *,
    schema_version: int = 1,
    market_view: str = "レンジ上限を試す展開",
    no_trade: bool = False,
    no_trade_reason: str | None = None,
) -> dict[str, Any]:
    """LLM wire-format judgment dict (before schema validation)."""
    return {
        "schema_version": schema_version,
        "market_view": market_view,
        "no_trade": no_trade,
        "no_trade_reason": no_trade_reason,
        "orders": orders,
    }

"""Boundary-focused tests for the individual validator rules (technical-spec.md 6章).

Each HARD rule is exercised at three points: comfortably inside the limit,
exactly on the limit (must pass — the limit is inclusive), and one unit past it
(must be rejected). The absolute caps are the safety core, so coverage here is
the thickest in the codebase (technical-spec.md 10章).
"""

from datetime import date

import pytest

from llm_fund.config import LimitsSettings
from llm_fund.domain.enums import Action
from llm_fund.domain.models import OrderPlan
from llm_fund.validator.rules import (
    ABSOLUTE_MAX_EXPOSURE_PCT,
    ABSOLUTE_MAX_LOSS_PER_TRADE_PCT,
    ABSOLUTE_MAX_POSITION_PCT,
    ABSOLUTE_MAX_TURNOVER_PCT,
    MIN_RATIONALE_LENGTH,
    InstrumentContext,
    RiskLimits,
    ValidationContext,
    rule_cash_sufficiency,
    rule_exit_within_holding,
    rule_lot_size,
    rule_max_exposure,
    rule_max_loss_per_trade,
    rule_max_position_pct,
    rule_price_band_sanity,
    rule_rationale_quality,
    rule_stop_loss_required,
    rule_tick_size,
    rule_universe_member,
    tse_price_band_width,
    tse_tick_size,
)

AS_OF = date(2026, 7, 4)
VALID_UNTIL = date(2026, 7, 7)
LONG_RATIONALE = "MA25 を上抜け出来高も伴い上昇トレンド継続と判断"


def _limits(
    *,
    max_position_pct: float = 15.0,
    max_turnover_pct: float = 30.0,
    max_exposure_pct: float = 100.0,
    max_loss_per_trade_pct: float = 3.0,
    min_rationale_length: int = MIN_RATIONALE_LENGTH,
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


def _instrument(
    symbol: str = "7203.T",
    *,
    in_universe: bool = True,
    lot_size: int = 100,
    prev_close: float = 3000.0,
    current_units: int = 0,
) -> InstrumentContext:
    return InstrumentContext(
        symbol=symbol,
        in_universe=in_universe,
        lot_size=lot_size,
        prev_close=prev_close,
        current_units=current_units,
    )


def _ctx(
    *,
    nav: float = 1_000_000.0,
    cash: float = 500_000.0,
    current_exposure: float = 0.0,
    instruments: dict[str, InstrumentContext] | None = None,
    limits: RiskLimits | None = None,
    data_fresh: bool = True,
) -> ValidationContext:
    if instruments is None:
        instruments = {"7203.T": _instrument()}
    return ValidationContext(
        nav=nav,
        cash=cash,
        current_exposure=current_exposure,
        instruments=instruments,
        limits=limits or _limits(),
        data_fresh=data_fresh,
    )


def _order(
    *,
    symbol: str = "7203.T",
    action: Action = Action.BUY,
    units: int = 100,
    entry_price: float = 3000.0,
    tp_price: float = 3300.0,
    sl_price: float = 2900.0,
    rationale: str = LONG_RATIONALE,
) -> OrderPlan:
    return OrderPlan(
        symbol=symbol,
        action=action,
        units=units,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        valid_until=VALID_UNTIL,
        rationale=rationale,
    )


class TestAbsoluteConstants:
    def test_absolute_caps_have_expected_values(self) -> None:
        # These are the safety contract (technical-spec.md 6章) — pin them.
        assert ABSOLUTE_MAX_LOSS_PER_TRADE_PCT == 3.0
        assert ABSOLUTE_MAX_POSITION_PCT == 25.0
        assert ABSOLUTE_MAX_TURNOVER_PCT == 50.0
        assert ABSOLUTE_MAX_EXPOSURE_PCT == 100.0


class TestRiskLimitsFromSettings:
    def test_uses_configured_values_when_within_caps(self) -> None:
        settings = LimitsSettings(
            max_position_pct=15.0,
            max_turnover_pct=30.0,
            max_instructions_per_day=5,
            require_stop_loss=True,
        )
        limits = RiskLimits.from_settings(settings)
        assert limits.max_position_pct == 15.0
        assert limits.max_turnover_pct == 30.0
        assert limits.require_stop_loss is True
        assert limits.max_instructions_per_day == 5
        # No config field exists for these — they fall back to the absolute cap.
        assert limits.max_loss_per_trade_pct == ABSOLUTE_MAX_LOSS_PER_TRADE_PCT
        assert limits.max_exposure_pct == ABSOLUTE_MAX_EXPOSURE_PCT

    def test_clamps_to_absolute_cap_even_if_config_slips_through(self) -> None:
        # Construct a settings object bypassing its own validator to prove the
        # limits layer independently clamps (config で緩和不可).
        settings = LimitsSettings.model_construct(
            max_position_pct=999.0,
            max_turnover_pct=999.0,
            max_instructions_per_day=5,
            require_stop_loss=True,
        )
        limits = RiskLimits.from_settings(settings)
        assert limits.max_position_pct == ABSOLUTE_MAX_POSITION_PCT
        assert limits.max_turnover_pct == ABSOLUTE_MAX_TURNOVER_PCT


class TestStopLossRequired:
    def test_buy_with_sl_below_entry_passes(self) -> None:
        assert rule_stop_loss_required(_order(sl_price=2900.0), _ctx()) is None

    def test_buy_with_sl_equal_entry_rejected(self) -> None:
        order = _order(entry_price=3000.0, sl_price=3000.0)
        assert rule_stop_loss_required(order, _ctx()) is not None

    def test_buy_with_sl_above_entry_rejected(self) -> None:
        order = _order(entry_price=3000.0, sl_price=3100.0)
        assert rule_stop_loss_required(order, _ctx()) is not None

    def test_sell_exit_has_no_stop_loss_requirement(self) -> None:
        order = _order(action=Action.SELL, sl_price=3100.0)
        assert rule_stop_loss_required(order, _ctx()) is None

    def test_close_exit_has_no_stop_loss_requirement(self) -> None:
        order = _order(action=Action.CLOSE, sl_price=3100.0)
        assert rule_stop_loss_required(order, _ctx()) is None


class TestMaxLossPerTrade:
    def test_loss_within_limit_passes(self) -> None:
        # (3000-2900)*100 = 10,000 <= 1,000,000*3% = 30,000
        assert rule_max_loss_per_trade(_order(), _ctx()) is None

    def test_loss_exactly_at_limit_passes(self) -> None:
        # (3300-3000)*100 = 30,000 == 30,000
        order = _order(entry_price=3300.0, sl_price=3000.0, tp_price=3600.0)
        assert rule_max_loss_per_trade(order, _ctx()) is None

    def test_loss_one_yen_over_limit_rejected(self) -> None:
        # (3301-3000)*100 = 30,100 > 30,000
        order = _order(entry_price=3301.0, sl_price=3000.0, tp_price=3600.0)
        assert rule_max_loss_per_trade(order, _ctx()) is not None

    def test_exit_order_does_not_risk_capital(self) -> None:
        order = _order(action=Action.SELL, entry_price=3301.0, sl_price=3000.0)
        assert rule_max_loss_per_trade(order, _ctx()) is None


class TestCashSufficiency:
    def test_cost_within_cash_passes(self) -> None:
        assert rule_cash_sufficiency(_order(), _ctx(cash=500_000.0)) is None

    def test_cost_exactly_equals_cash_passes(self) -> None:
        order = _order(entry_price=5000.0, units=100)  # 500,000
        assert rule_cash_sufficiency(order, _ctx(cash=500_000.0)) is None

    def test_cost_over_cash_rejected(self) -> None:
        order = _order(entry_price=5000.0, units=200)  # 1,000,000
        assert rule_cash_sufficiency(order, _ctx(cash=500_000.0)) is not None

    def test_exit_order_does_not_consume_cash(self) -> None:
        order = _order(action=Action.SELL, entry_price=5000.0, units=200)
        assert rule_cash_sufficiency(order, _ctx(cash=1.0)) is None


class TestMaxPositionPct:
    def test_position_within_limit_passes(self) -> None:
        order = _order(entry_price=1000.0, units=100)  # 100,000 = 10%
        assert rule_max_position_pct(order, _ctx()) is None

    def test_position_exactly_at_limit_passes(self) -> None:
        order = _order(entry_price=1500.0, units=100)  # 150,000 = 15%
        assert rule_max_position_pct(order, _ctx()) is None

    def test_position_over_limit_rejected(self) -> None:
        order = _order(entry_price=1500.0, units=200)  # 300,000 = 30%
        assert rule_max_position_pct(order, _ctx()) is not None

    def test_existing_holding_counts_toward_position(self) -> None:
        instruments = {"7203.T": _instrument(current_units=100)}
        order = _order(entry_price=1000.0, units=100)  # (100+100)*1000 = 200,000 = 20%
        assert rule_max_position_pct(order, _ctx(instruments=instruments)) is not None

    def test_exit_order_reduces_position(self) -> None:
        order = _order(action=Action.CLOSE, entry_price=1500.0, units=200)
        assert rule_max_position_pct(order, _ctx()) is None


class TestMaxExposure:
    def test_exposure_within_limit_passes(self) -> None:
        order = _order(entry_price=1000.0, units=100)
        assert rule_max_exposure(order, _ctx(current_exposure=0.0)) is None

    def test_exposure_exactly_at_limit_passes(self) -> None:
        order = _order(entry_price=1000.0, units=100)  # +100,000
        assert rule_max_exposure(order, _ctx(current_exposure=900_000.0)) is None

    def test_exposure_over_limit_rejected(self) -> None:
        order = _order(entry_price=1000.0, units=100)  # +100,000 -> 1,050,000
        assert rule_max_exposure(order, _ctx(current_exposure=950_000.0)) is not None

    def test_exit_order_does_not_add_exposure(self) -> None:
        order = _order(action=Action.SELL, entry_price=1000.0, units=100)
        assert rule_max_exposure(order, _ctx(current_exposure=950_000.0)) is None


class TestExitWithinHolding:
    def test_sell_within_holding_passes(self) -> None:
        instruments = {"7203.T": _instrument(current_units=200)}
        order = _order(action=Action.SELL, units=100)
        assert rule_exit_within_holding(order, _ctx(instruments=instruments)) is None

    def test_sell_exactly_all_holding_passes(self) -> None:
        instruments = {"7203.T": _instrument(current_units=100)}
        order = _order(action=Action.CLOSE, units=100)
        assert rule_exit_within_holding(order, _ctx(instruments=instruments)) is None

    def test_sell_over_holding_rejected(self) -> None:
        instruments = {"7203.T": _instrument(current_units=100)}
        order = _order(action=Action.SELL, units=10000)
        assert rule_exit_within_holding(order, _ctx(instruments=instruments)) is not None

    def test_sell_with_no_holding_rejected(self) -> None:
        instruments = {"7203.T": _instrument(current_units=0)}
        order = _order(action=Action.CLOSE, units=100)
        assert rule_exit_within_holding(order, _ctx(instruments=instruments)) is not None

    def test_buy_is_not_subject_to_holding_check(self) -> None:
        instruments = {"7203.T": _instrument(current_units=0)}
        order = _order(action=Action.BUY, units=100)
        assert rule_exit_within_holding(order, _ctx(instruments=instruments)) is None

    def test_missing_instrument_defers_to_universe_rule(self) -> None:
        order = _order(symbol="9999.T", action=Action.SELL, units=100)
        assert rule_exit_within_holding(order, _ctx()) is None


class TestUniverseMember:
    def test_member_in_universe_passes(self) -> None:
        assert rule_universe_member(_order(), _ctx()) is None

    def test_unknown_symbol_rejected(self) -> None:
        assert rule_universe_member(_order(symbol="9999.T"), _ctx()) is not None

    def test_known_symbol_flagged_not_in_universe_rejected(self) -> None:
        instruments = {"7203.T": _instrument(in_universe=False)}
        assert rule_universe_member(_order(), _ctx(instruments=instruments)) is not None


class TestLotSize:
    def test_exact_multiple_passes(self) -> None:
        assert rule_lot_size(_order(units=200), _ctx()) is None

    def test_single_lot_passes(self) -> None:
        assert rule_lot_size(_order(units=100), _ctx()) is None

    def test_non_multiple_rejected(self) -> None:
        assert rule_lot_size(_order(units=150), _ctx()) is not None

    def test_below_one_lot_rejected(self) -> None:
        assert rule_lot_size(_order(units=50), _ctx()) is not None

    def test_lot_size_one_allows_any_units(self) -> None:
        instruments = {"7203.T": _instrument(lot_size=1)}
        assert rule_lot_size(_order(units=7), _ctx(instruments=instruments)) is None

    def test_missing_instrument_defers_to_universe_rule(self) -> None:
        assert rule_lot_size(_order(symbol="9999.T", units=150), _ctx()) is None


class TestTickSize:
    def test_all_prices_tick_aligned_passes(self) -> None:
        order = _order(entry_price=3000.0, tp_price=3300.0, sl_price=2900.0)
        assert rule_tick_size(order, _ctx()) is None

    def test_entry_off_tick_rejected(self) -> None:
        # 3001 is above 3,000 so tick becomes 5; 3001 is not a multiple of 5.
        assert rule_tick_size(_order(entry_price=3001.0), _ctx()) is not None

    def test_fractional_price_off_tick_rejected(self) -> None:
        assert rule_tick_size(_order(entry_price=3122.5), _ctx()) is not None

    def test_higher_bracket_tick_enforced(self) -> None:
        # 35,010 in the 10-50k bracket (tick 50) is not a multiple of 50.
        order = _order(entry_price=35_010.0, tp_price=36_000.0, sl_price=34_000.0)
        assert rule_tick_size(order, _ctx()) is not None

    def test_missing_instrument_still_checks_tick(self) -> None:
        # Tick only depends on price, so it applies even without instrument ctx.
        assert rule_tick_size(_order(symbol="9999.T", entry_price=3001.0), _ctx()) is not None


class TestPriceBandSanity:
    def test_entry_at_prev_close_passes(self) -> None:
        assert rule_price_band_sanity(_order(entry_price=3000.0), _ctx()) is None

    def test_entry_at_upper_band_passes(self) -> None:
        # prev_close 3000 -> width 700; 3000+700 = 3700 is inclusive.
        assert rule_price_band_sanity(_order(entry_price=3700.0), _ctx()) is None

    def test_entry_above_upper_band_rejected(self) -> None:
        assert rule_price_band_sanity(_order(entry_price=3705.0), _ctx()) is not None

    def test_entry_at_lower_band_passes(self) -> None:
        assert rule_price_band_sanity(_order(entry_price=2300.0), _ctx()) is None

    def test_entry_below_lower_band_rejected(self) -> None:
        assert rule_price_band_sanity(_order(entry_price=2295.0), _ctx()) is not None

    def test_missing_instrument_defers_to_universe_rule(self) -> None:
        assert rule_price_band_sanity(_order(symbol="9999.T", entry_price=9999.0), _ctx()) is None


class TestRationaleQuality:
    def test_sufficient_rationale_no_warning(self) -> None:
        assert rule_rationale_quality(_order(rationale=LONG_RATIONALE), _ctx()) is None

    def test_rationale_exactly_at_minimum_no_warning(self) -> None:
        text = "x" * MIN_RATIONALE_LENGTH
        assert rule_rationale_quality(_order(rationale=text), _ctx()) is None

    def test_short_rationale_warns(self) -> None:
        assert rule_rationale_quality(_order(rationale="買い"), _ctx()) is not None

    def test_whitespace_padded_short_rationale_warns(self) -> None:
        assert rule_rationale_quality(_order(rationale="  買い  "), _ctx()) is not None


class TestTseTickSize:
    @pytest.mark.parametrize(
        ("price", "expected"),
        [
            (2999.0, 1.0),
            (3000.0, 1.0),
            (3001.0, 5.0),
            (5000.0, 5.0),
            (5001.0, 10.0),
            (30_000.0, 10.0),
            (30_001.0, 50.0),
            (50_000.0, 50.0),
            (50_001.0, 100.0),
            (300_000.0, 100.0),
            (300_001.0, 500.0),
            (500_001.0, 1_000.0),
            (3_000_001.0, 5_000.0),
            (60_000_000.0, 100_000.0),
        ],
    )
    def test_tick_brackets(self, price: float, expected: float) -> None:
        assert tse_tick_size(price) == expected


class TestTsePriceBandWidth:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (50.0, 30.0),
            (99.0, 30.0),
            (100.0, 50.0),
            (199.0, 50.0),
            (200.0, 80.0),
            (499.0, 80.0),
            (500.0, 100.0),
            (3000.0, 700.0),
            (10_000.0, 3_000.0),
        ],
    )
    def test_band_brackets(self, base: float, expected: float) -> None:
        assert tse_price_band_width(base) == expected

    def test_extreme_price_uses_fallback_width(self) -> None:
        assert tse_price_band_width(50_000_000.0) > 0

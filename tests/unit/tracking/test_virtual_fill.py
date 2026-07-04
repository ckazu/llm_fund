"""固定 OHLC フィクスチャで保守的仮想約定の全分岐を手計算と突合（S8 / spec 7章）。

ここは偽陽性を出すと検証全体が無意味になる核心なので、SL 優先・ギャップ・期限切れ・
部分観察（未決済保有）を網羅する。
"""

from datetime import date

import pytest

from llm_fund.domain.enums import Action, FillExitReason
from llm_fund.domain.models import Candle
from llm_fund.tracking.virtual_fill import (
    CostModel,
    commission_for,
    simulate_fill,
    slippage_for,
)

TICKET = "20260105-01"
UNITS = 100
ENTRY = 1000.0
TP = 1100.0
SL = 900.0
# 有効期限は十分先に置き、エントリー探索が期限で切れないようにする（期限テストを除く）。
VALID_UNTIL = date(2026, 1, 30)


def _c(day: int, o: float, h: float, low: float, c: float) -> Candle:
    return Candle(
        symbol="7203.T",
        trade_date=date(2026, 1, day),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1_000,
        adj_close=c,
    )


def _fill(candles: list[Candle], **kw: object) -> object:
    params: dict[str, object] = {
        "ticket_no": TICKET,
        "action": Action.BUY,
        "units": UNITS,
        "entry_price": ENTRY,
        "tp_price": TP,
        "sl_price": SL,
        "valid_until": VALID_UNTIL,
        "candles": candles,
    }
    params.update(kw)
    return simulate_fill(**params)  # type: ignore[arg-type]


class TestEntry:
    def test_fills_at_entry_when_open_above_limit(self) -> None:
        # 寄付き 1005 > 指値。ザラ場で low=1000 まで下押しして約定 → fill=min(1000,1005)=1000
        candles = [_c(5, 1005, 1010, 1000, 1005)]
        vf = _fill(candles)
        assert vf.fill_date == date(2026, 1, 5)
        assert vf.fill_price == pytest.approx(1000.0)

    def test_fills_at_open_when_open_below_limit(self) -> None:
        # 寄付き 980 <= 指値。板寄せで寄付き約定 → fill=min(1000,980)=980（有利側）
        candles = [_c(5, 980, 1005, 970, 1000)]
        vf = _fill(candles)
        assert vf.fill_price == pytest.approx(980.0)

    def test_no_fill_when_low_above_entry(self) -> None:
        # low=1050 > 指値 1000 → 未約定。期限未到達なので保留（EXPIRY ではない）。
        candles = [_c(5, 1080, 1090, 1050, 1080)]
        vf = _fill(candles, valid_until=date(2026, 1, 30))
        assert vf.fill_date is None
        assert vf.exit_reason is None


class TestExit:
    def test_take_profit(self) -> None:
        candles = [
            _c(5, 1000, 1010, 995, 1005),  # fill at 1000
            _c(6, 1050, 1120, 1040, 1100),  # high>=tp, no gap, no sl → TP at 1100
        ]
        vf = _fill(candles)
        assert vf.exit_reason is FillExitReason.TP
        assert vf.exit_price == pytest.approx(1100.0)
        assert vf.pnl == pytest.approx((1100.0 - 1000.0) * UNITS)

    def test_stop_loss(self) -> None:
        candles = [
            _c(5, 1000, 1010, 995, 1005),
            _c(6, 950, 980, 880, 890),  # low<=sl, no gap, no tp → SL at 900
        ]
        vf = _fill(candles)
        assert vf.exit_reason is FillExitReason.SL
        assert vf.exit_price == pytest.approx(900.0)
        assert vf.pnl == pytest.approx((900.0 - 1000.0) * UNITS)

    def test_both_tp_and_sl_same_day_prefers_sl(self) -> None:
        # 同一足で high>=tp かつ low<=sl。保守側で SL を採用する（核心）。
        candles = [
            _c(5, 1000, 1010, 995, 1005),
            _c(6, 1000, 1150, 850, 1000),  # both touched → SL priority at 900
        ]
        vf = _fill(candles)
        assert vf.exit_reason is FillExitReason.SL
        assert vf.exit_price == pytest.approx(900.0)

    def test_gap_down_through_stop_exits_at_open(self) -> None:
        # 下方ギャップ open=850 <= sl → 寄付きで決済（sl より不利なスリッページの現実化）
        candles = [
            _c(5, 1000, 1010, 995, 1005),
            _c(6, 850, 870, 840, 860),
        ]
        vf = _fill(candles)
        assert vf.exit_reason is FillExitReason.SL
        assert vf.exit_price == pytest.approx(850.0)
        assert vf.pnl == pytest.approx((850.0 - 1000.0) * UNITS)

    def test_gap_up_through_target_caps_at_tp(self) -> None:
        # 上方ギャップ open=1200 >= tp。利益を過大評価しないため TP 価格で頭打ち。
        candles = [
            _c(5, 1000, 1010, 995, 1005),
            _c(6, 1200, 1250, 1190, 1220),
        ]
        vf = _fill(candles)
        assert vf.exit_reason is FillExitReason.TP
        assert vf.exit_price == pytest.approx(1100.0)
        assert vf.pnl == pytest.approx((1100.0 - 1000.0) * UNITS)

    def test_fill_day_moves_not_evaluated_for_exit(self) -> None:
        # 約定当日に tp/sl 両方をつけても決済しない（翌日以降のみ評価）。翌日は無反応。
        candles = [
            _c(5, 1000, 1200, 850, 1000),  # fill (low<=entry) but no same-day exit
            _c(6, 1000, 1050, 980, 1000),  # neither tp nor sl
        ]
        vf = _fill(candles)
        assert vf.fill_date == date(2026, 1, 5)
        assert vf.exit_date is None
        assert vf.pnl is None

    def test_filled_but_open_position(self) -> None:
        candles = [
            _c(5, 1000, 1010, 995, 1005),
            _c(6, 1010, 1050, 960, 1020),  # neither tp nor sl → open
        ]
        vf = _fill(candles)
        assert vf.fill_date == date(2026, 1, 5)
        assert vf.exit_date is None
        assert vf.pnl is None
        # 未決済でもエントリー側の手数料は計上される（コスト0なので0）。
        assert vf.commission == pytest.approx(0.0)


class TestExpiry:
    def test_expired_unfilled_when_period_reached(self) -> None:
        candles = [
            _c(5, 1080, 1090, 1050, 1080),  # low>entry
            _c(6, 1080, 1090, 1060, 1080),  # low>entry, date==valid_until
        ]
        vf = _fill(candles, valid_until=date(2026, 1, 6))
        assert vf.fill_date is None
        assert vf.exit_reason is FillExitReason.EXPIRY

    def test_entry_after_expiry_is_ignored(self) -> None:
        candles = [
            _c(5, 1080, 1090, 1050, 1080),  # no fill, date==valid_until
            _c(7, 990, 1000, 980, 995),  # would fill but date>valid_until
        ]
        vf = _fill(candles, valid_until=date(2026, 1, 5))
        assert vf.fill_date is None
        assert vf.exit_reason is FillExitReason.EXPIRY

    def test_pending_when_period_not_reached(self) -> None:
        candles = [_c(5, 1080, 1090, 1050, 1080)]  # no fill, valid_until far away
        vf = _fill(candles, valid_until=date(2026, 1, 30))
        assert vf.fill_date is None
        assert vf.exit_reason is None


class TestCosts:
    def test_round_trip_costs_applied_to_both_sides(self) -> None:
        costs = CostModel(commission_rate=0.001, min_commission=100.0, slippage_pct=0.002)
        candles = [
            _c(5, 1000, 1010, 995, 1005),  # fill at 1000
            _c(6, 1050, 1120, 1040, 1100),  # TP at 1100
        ]
        vf = _fill(candles, costs=costs)
        entry_notional = 1000.0 * UNITS  # 100_000
        exit_notional = 1100.0 * UNITS  # 110_000
        # commission: max(100_000*0.001,100)=100 ; max(110_000*0.001,100)=110
        expected_commission = 100.0 + 110.0
        # slippage: 100_000*0.002=200 ; 110_000*0.002=220
        expected_slippage = 200.0 + 220.0
        assert vf.commission == pytest.approx(expected_commission)
        assert vf.slippage == pytest.approx(expected_slippage)
        gross = (1100.0 - 1000.0) * UNITS  # 10_000
        assert vf.pnl == pytest.approx(gross - expected_commission - expected_slippage)
        assert commission_for(entry_notional, costs) == pytest.approx(100.0)
        assert slippage_for(exit_notional, costs) == pytest.approx(220.0)


class TestGuards:
    def test_non_buy_action_rejected(self) -> None:
        with pytest.raises(ValueError, match="BUY only"):
            _fill([_c(5, 1000, 1010, 995, 1005)], action=Action.SELL)

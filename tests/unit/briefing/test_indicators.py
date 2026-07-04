"""Hand-computed indicator checks against a fixed synthetic candle fixture (S4)."""

from datetime import date, timedelta

import pytest

from llm_fund.briefing.indicators import (
    compute_atr_pct,
    compute_indicator_row,
    compute_ma_deviation_pct,
    compute_return_pct,
    compute_volume_ratio,
)
from llm_fund.domain.models import Candle

BASE_CLOSE = 100.0
DEFAULT_VOLUME = 1000
SPIKE_VOLUME = 3000


def _linear_candles(
    days: int, symbol: str = "TEST", spike_last_volume: bool = True
) -> list[Candle]:
    """`days` candles with close rising by 1/day (100, 101, ...), constant volume.

    high/low are always `close +/- 1`, so every day's true range is a
    constant 2 (see module docstring test below for the derivation) and the
    only "surprise" data point is the last day's volume when `spike_last_volume`.
    """
    start = date(2026, 1, 1)
    candles = []
    for i in range(days):
        close = BASE_CLOSE + i
        volume = DEFAULT_VOLUME
        if spike_last_volume and i == days - 1:
            volume = SPIKE_VOLUME
        candles.append(
            Candle(
                symbol=symbol,
                trade_date=start + timedelta(days=i),
                open=close - 0.5,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=volume,
                adj_close=close,
            )
        )
    return candles


class TestComputeReturnPct:
    def test_matches_hand_calculation(self) -> None:
        closes = [100.0 + i for i in range(80)]  # 100..179

        assert compute_return_pct(closes, 1) == pytest.approx((179 - 178) / 178 * 100)
        assert compute_return_pct(closes, 5) == pytest.approx((179 - 174) / 174 * 100)
        assert compute_return_pct(closes, 20) == pytest.approx((179 - 159) / 159 * 100)
        assert compute_return_pct(closes, 60) == pytest.approx((179 - 119) / 119 * 100)

    def test_insufficient_history_returns_none(self) -> None:
        closes = [100.0, 101.0, 102.0]

        assert compute_return_pct(closes, 5) is None


class TestComputeMaDeviationPct:
    def test_matches_hand_calculation(self) -> None:
        closes = [100.0 + i for i in range(80)]  # 100..179

        # MA25 over the last 25 closes (155..179) = 167.0
        assert compute_ma_deviation_pct(closes, 25) == pytest.approx((179 - 167.0) / 167.0 * 100)
        # MA75 over the last 75 closes (105..179) = 142.0
        assert compute_ma_deviation_pct(closes, 75) == pytest.approx((179 - 142.0) / 142.0 * 100)

    def test_insufficient_history_returns_none(self) -> None:
        closes = [100.0 + i for i in range(10)]

        assert compute_ma_deviation_pct(closes, 25) is None


class TestComputeAtrPct:
    def test_matches_hand_calculation(self) -> None:
        candles = _linear_candles(80, spike_last_volume=False)

        # Every day's true range is a constant 2 (see `_linear_candles` docstring),
        # so ATR14 = 2, expressed as % of the latest close (179).
        assert compute_atr_pct(candles, 14) == pytest.approx(2 / 179 * 100)

    def test_insufficient_history_returns_none(self) -> None:
        candles = _linear_candles(10)

        assert compute_atr_pct(candles, 14) is None


class TestComputeVolumeRatio:
    def test_matches_hand_calculation(self) -> None:
        candles = _linear_candles(80, spike_last_volume=True)

        # 19 days at 1000 + 1 day at 3000 = 22000 / 20 = 1100 average.
        assert compute_volume_ratio(candles, 20) == pytest.approx(3000 / 1100)

    def test_insufficient_history_returns_none(self) -> None:
        candles = _linear_candles(10)

        assert compute_volume_ratio(candles, 20) is None

    def test_zero_average_volume_returns_none(self) -> None:
        candles = _linear_candles(20, spike_last_volume=False)
        candles = [c.model_copy(update={"volume": 0}) for c in candles]

        assert compute_volume_ratio(candles, 20) is None


class TestComputeIndicatorRow:
    def test_full_history_populates_all_fields(self) -> None:
        candles = _linear_candles(80, spike_last_volume=True)

        row = compute_indicator_row(candles)

        assert row.symbol == "TEST"
        assert row.trade_date == candles[-1].trade_date
        assert row.close == pytest.approx(179.0)
        assert row.return_1d_pct == pytest.approx((179 - 178) / 178 * 100)
        assert row.return_60d_pct == pytest.approx((179 - 119) / 119 * 100)
        assert row.ma_deviation_25d_pct == pytest.approx((179 - 167.0) / 167.0 * 100)
        assert row.ma_deviation_75d_pct == pytest.approx((179 - 142.0) / 142.0 * 100)
        assert row.atr_pct_14 == pytest.approx(2 / 179 * 100)
        assert row.volume_ratio_20d == pytest.approx(3000 / 1100)

    def test_short_history_leaves_long_window_fields_none(self) -> None:
        candles = _linear_candles(10)

        row = compute_indicator_row(candles)

        assert row.return_1d_pct is not None
        assert row.return_60d_pct is None
        assert row.ma_deviation_25d_pct is None
        assert row.ma_deviation_75d_pct is None
        assert row.atr_pct_14 is None
        assert row.volume_ratio_20d is None

    def test_empty_candles_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_indicator_row([])

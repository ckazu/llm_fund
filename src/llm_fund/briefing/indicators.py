"""Technical indicator calculations for LLM briefings (technical-spec.md 2, 5, 10章).

Every indicator is a ratio/normalized value (% return, % MA deviation, ATR as
% of price, volume vs its own moving average) rather than a raw price level or
volume count. This is deliberate: LLMs are known to over-react to absolute
values and the most recent single data point while missing proportional
change over a window (technical-spec.md 主要リスク表), so the briefing feeds
only proportional/normalized numbers.

Indicator calculation uses `adj_close` (split/dividend adjusted), consistent
with `Candle`'s convention that `close` is reserved for order-price validation
and `adj_close` for indicator math.
"""

from statistics import mean

from llm_fund.domain.models import Candle, IndicatorRow

RETURN_WINDOWS_DAYS = (1, 5, 20, 60)
MA_WINDOWS_DAYS = (25, 75)
ATR_WINDOW_DAYS = 14
VOLUME_WINDOW_DAYS = 20


def _pct_change(current: float, base: float) -> float:
    return (current - base) / base * 100.0


def compute_return_pct(closes: list[float], window_days: int) -> float | None:
    """% change from `window_days` bars ago to the latest bar.

    Returns `None` if `closes` doesn't contain enough history for `window_days`.
    """
    if len(closes) <= window_days:
        return None
    return _pct_change(closes[-1], closes[-1 - window_days])


def compute_ma_deviation_pct(closes: list[float], window_days: int) -> float | None:
    """% deviation of the latest close from its `window_days`-bar moving average.

    Returns `None` if `closes` doesn't contain enough history for `window_days`.
    """
    if len(closes) < window_days:
        return None
    moving_average = mean(closes[-window_days:])
    return _pct_change(closes[-1], moving_average)


def compute_atr_pct(candles: list[Candle], window_days: int = ATR_WINDOW_DAYS) -> float | None:
    """ATR(`window_days`) expressed as a % of the latest close.

    Expressed as a percentage (rather than a raw price-unit ATR) so volatility
    is comparable across instruments trading at very different price levels.
    Returns `None` if there isn't enough history (needs `window_days + 1` bars,
    since the first true-range value requires a previous close).
    """
    if len(candles) < window_days + 1:
        return None
    window = candles[-window_days:]
    true_ranges = []
    for i, candle in enumerate(window):
        # high/low を調整係数 (adj_close/close) でスケールし、株式分割・配当による
        # 価格の不連続をならす。Candle は adj_high/adj_low を持たないため close 比で
        # 近似する（他指標が adj_close を使うのと整合させ、分割ギャップが値幅を汚さない）。
        factor = candle.adj_close / candle.close
        adj_high = candle.high * factor
        adj_low = candle.low * factor
        prev_adj_close = candles[len(candles) - window_days + i - 1].adj_close
        true_ranges.append(
            max(
                adj_high - adj_low,
                abs(adj_high - prev_adj_close),
                abs(adj_low - prev_adj_close),
            )
        )
    atr = mean(true_ranges)
    return atr / candles[-1].adj_close * 100.0


def compute_volume_ratio(
    candles: list[Candle], window_days: int = VOLUME_WINDOW_DAYS
) -> float | None:
    """Latest volume divided by its `window_days`-bar moving average (1.0 = average day).

    Returns `None` if there isn't enough history, or the average volume is 0.
    """
    if len(candles) < window_days:
        return None
    volumes = [c.volume for c in candles[-window_days:]]
    average_volume = mean(volumes)
    if average_volume == 0:
        return None
    return candles[-1].volume / average_volume


def compute_indicator_row(candles: list[Candle]) -> IndicatorRow:
    """Compute the latest-day `IndicatorRow` for one instrument.

    `candles` must be sorted ascending by `trade_date` and non-empty (as
    returned by `CandleRepo.get_range`/`PriceLoader.load_fresh`).
    """
    if not candles:
        raise ValueError("candles must not be empty")

    latest = candles[-1]
    closes = [c.adj_close for c in candles]

    return IndicatorRow(
        symbol=latest.symbol,
        trade_date=latest.trade_date,
        close=latest.close,
        return_1d_pct=compute_return_pct(closes, RETURN_WINDOWS_DAYS[0]),
        return_5d_pct=compute_return_pct(closes, RETURN_WINDOWS_DAYS[1]),
        return_20d_pct=compute_return_pct(closes, RETURN_WINDOWS_DAYS[2]),
        return_60d_pct=compute_return_pct(closes, RETURN_WINDOWS_DAYS[3]),
        ma_deviation_25d_pct=compute_ma_deviation_pct(closes, MA_WINDOWS_DAYS[0]),
        ma_deviation_75d_pct=compute_ma_deviation_pct(closes, MA_WINDOWS_DAYS[1]),
        atr_pct_14=compute_atr_pct(candles, ATR_WINDOW_DAYS),
        volume_ratio_20d=compute_volume_ratio(candles, VOLUME_WINDOW_DAYS),
    )

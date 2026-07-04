"""Price data sources (technical-spec.md 2, S0 PoC notes in docs/data-source-notes.md).

`YFinanceSource` fetches with `auto_adjust=False` so both the unadjusted
`close` (used for order price validation, e.g. `PriceBandSanity`) and the
adjusted `adj_close` (used for indicator calculations, e.g. returns/MA) are
available and kept in separate columns. Fetching only with `auto_adjust=True`
would collapse them into a single adjusted value and lose the unadjusted
price needed for order validation.
"""

from datetime import date, timedelta
from typing import Protocol

import yfinance as yf

from llm_fund.domain.models import Candle


class PriceSource(Protocol):
    """Abstraction over a daily OHLCV data provider (technical-spec.md 2章)."""

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        """Return daily candles for `symbol` within `[start, end]` inclusive.

        Sorted ascending by `trade_date`. Returns an empty list if no bars are
        available in the range (e.g. holidays only, or a fetch failure).
        """
        ...


class YFinanceSource:
    """`PriceSource` backed by `yfinance.Ticker.history` (unofficial, EOD only).

    `Ticker.history` is used rather than `yf.download` because it returns a
    single-symbol frame with a market-local tz-aware index (see
    docs/data-source-notes.md 2-3節), which lets us read the correct local
    trading day directly via `.date()`.
    """

    def fetch(self, symbol: str, start: date, end: date) -> list[Candle]:
        ticker = yf.Ticker(symbol)
        # yfinance の `end` は非包含なので、`end` 当日を含めるため1日足す。
        history = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
        )
        if history.empty:
            return []
        return [self._row_to_candle(symbol, index, row) for index, row in history.iterrows()]

    @staticmethod
    def _row_to_candle(symbol: str, index: object, row: object) -> Candle:
        return Candle(
            symbol=symbol,
            trade_date=index.date(),  # type: ignore[attr-defined]
            open=float(row["Open"]),  # type: ignore[index]
            high=float(row["High"]),  # type: ignore[index]
            low=float(row["Low"]),  # type: ignore[index]
            close=float(row["Close"]),  # type: ignore[index]
            volume=int(row["Volume"]),  # type: ignore[index]
            adj_close=float(row["Adj Close"]),  # type: ignore[index]
        )

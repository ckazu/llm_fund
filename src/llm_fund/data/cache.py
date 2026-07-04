"""Diff-based caching of candle data into the `candles` table (technical-spec.md 2章).

`CandleCache.sync` only requests the gap after the latest cached bar from the
`PriceSource`, so repeated calls (e.g. daily cron runs) do not re-fetch
history already stored.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from llm_fund.data.prices import PriceSource
from llm_fund.domain.models import Candle
from llm_fund.store.repos import CandleRepo


@dataclass
class CandleCache:
    """Wraps a `CandleRepo` + `PriceSource` pair to provide diff sync/read."""

    candle_repo: CandleRepo
    source: PriceSource

    def sync(self, instrument_id: int, symbol: str, start: date, end: date) -> int:
        """Fetch and persist candles missing between `start` and `end` (inclusive).

        Returns the number of candles written (0 if the cache is already
        up to date through `end`).
        """
        fetch_start = start
        latest = self.candle_repo.latest(instrument_id)
        if latest is not None:
            day_after_latest = latest.trade_date + timedelta(days=1)
            fetch_start = max(start, day_after_latest)
        if fetch_start > end:
            return 0

        candles = self.source.fetch(symbol, fetch_start, end)
        if not candles:
            return 0
        return self.candle_repo.upsert_many(instrument_id, candles)

    def read(
        self, instrument_id: int, start: date | None = None, end: date | None = None
    ) -> list[Candle]:
        """Read cached candles for `instrument_id`, optionally bounded by date."""
        return self.candle_repo.get_range(instrument_id, start, end)

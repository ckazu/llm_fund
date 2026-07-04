"""Unified data access API with a freshness/gap gate (technical-spec.md 2, 6, 10章).

`PriceLoader.load_fresh` is the single entry point downstream layers
(briefing/judgment) should use to read price data for a trading decision: it
enforces that the most recent cached bar is not stale, raising
`DataFreshnessError` otherwise. `validator/gate.py` (S5) maps that error to
the HARD `DataFreshness` rule, forcing `NO_TRADE`.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from llm_fund.data.cache import CandleCache
from llm_fund.data.prices import PriceSource
from llm_fund.domain.models import Candle
from llm_fund.store.repos import CandleRepo, InstrumentRecord

# 営業日カレンダーを持たないため、土日＋短い祝日を跨いでも許容できる日数で近似する既定値。
# config/default.yaml の `data.max_staleness_days` で上書き可能（絶対上限はなし）。
DEFAULT_MAX_STALENESS_DAYS = 4


class DataFreshnessError(Exception):
    """Raised when an instrument has no cached data, or its latest bar is stale.

    Callers must treat this as a forced NO_TRADE (technical-spec.md 6章 DataFreshness)."""


@dataclass
class PriceLoader:
    """Combines `CandleCache` sync with the freshness gate for a single instrument."""

    candle_repo: CandleRepo
    source: PriceSource
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS

    def fetch_and_cache(
        self, instrument: InstrumentRecord, as_of: date, lookback_days: int
    ) -> int:
        """Sync the cache for `instrument` up to `as_of`. Returns candles written."""
        cache = CandleCache(self.candle_repo, self.source)
        start = as_of - timedelta(days=lookback_days)
        return cache.sync(instrument.id, instrument.symbol, start, as_of)

    def load_fresh(
        self, instrument: InstrumentRecord, as_of: date, lookback_days: int
    ) -> list[Candle]:
        """Return cached candles for `instrument`, enforcing the freshness gate.

        Raises `DataFreshnessError` if no candles are cached, or the latest
        cached bar is more than `max_staleness_days` before `as_of`.
        """
        start = as_of - timedelta(days=lookback_days)
        candles = self.candle_repo.get_range(instrument.id, start, as_of)
        if not candles:
            raise DataFreshnessError(f"{instrument.symbol}: no cached candles found")

        latest = candles[-1]
        staleness_days = (as_of - latest.trade_date).days
        if staleness_days > self.max_staleness_days:
            raise DataFreshnessError(
                f"{instrument.symbol}: latest bar {latest.trade_date} is "
                f"{staleness_days} days old (max {self.max_staleness_days})"
            )
        return candles

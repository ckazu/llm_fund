"""Tests for data/prices.py: YFinanceSource column mapping (yfinance fully mocked)."""

from datetime import date

import pandas as pd
import pytest

from llm_fund.data.prices import YFinanceSource


class _FakeTicker:
    def __init__(self, history_df: pd.DataFrame) -> None:
        self._df = history_df
        self.calls: list[dict[str, object]] = []

    def history(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self._df


def _make_history_df() -> pd.DataFrame:
    index = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"]).tz_localize("Asia/Tokyo")
    return pd.DataFrame(
        {
            "Open": [3100.0, 3110.0, 3120.0],
            "High": [3150.0, 3140.0, 3160.0],
            "Low": [3080.0, 3090.0, 3100.0],
            "Close": [3120.0, 3115.0, 3140.0],
            "Volume": [1_000_000, 900_000, 1_100_000],
            "Adj Close": [3080.0, 3075.0, 3100.0],
        },
        index=index,
    )


class TestYFinanceSource:
    def test_fetch_maps_columns_to_candles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTicker(_make_history_df())
        monkeypatch.setattr("llm_fund.data.prices.yf.Ticker", lambda symbol: fake)

        candles = YFinanceSource().fetch("7203.T", date(2026, 6, 30), date(2026, 7, 2))

        assert len(candles) == 3
        first = candles[0]
        assert first.symbol == "7203.T"
        assert first.trade_date == date(2026, 6, 30)
        assert first.close == 3120.0
        assert first.adj_close == 3080.0
        # 配当調整で close と adj_close が乖離するケース（docs/data-source-notes.md 4節）
        assert first.close != first.adj_close
        assert candles[-1].trade_date == date(2026, 7, 2)

    def test_fetch_requests_unadjusted_and_adjusted_via_auto_adjust_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTicker(_make_history_df())
        monkeypatch.setattr("llm_fund.data.prices.yf.Ticker", lambda symbol: fake)

        YFinanceSource().fetch("7203.T", date(2026, 6, 30), date(2026, 7, 2))

        assert fake.calls[0]["auto_adjust"] is False

    def test_fetch_end_date_is_inclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeTicker(_make_history_df())
        monkeypatch.setattr("llm_fund.data.prices.yf.Ticker", lambda symbol: fake)

        YFinanceSource().fetch("7203.T", date(2026, 6, 30), date(2026, 7, 2))

        # yfinance の end は非包含のため、渡す end は呼出し元の end + 1日でなければならない
        assert fake.calls[0]["end"] == "2026-07-03"

    def test_fetch_empty_history_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeTicker(pd.DataFrame())
        monkeypatch.setattr("llm_fund.data.prices.yf.Ticker", lambda symbol: fake)

        candles = YFinanceSource().fetch("NODATA", date(2026, 1, 1), date(2026, 1, 2))

        assert candles == []

"""Universe-level Markdown briefing table + data snapshot (technical-spec.md 2, 3, 5章).

Builds a `Briefing` (Markdown table + JSON data snapshot) from one universe's
per-instrument candle history, and persists it to the `briefings` table.
Chart images/raw time series are deliberately never included here (only the
normalized `IndicatorRow` values) — technical-spec.md 5章.
"""

import json
from datetime import date

from llm_fund.briefing.indicators import compute_indicator_row
from llm_fund.domain.models import Briefing, Candle, IndicatorRow
from llm_fund.store.repos import BriefingRepo

_TABLE_HEADER = (
    "| Symbol | Close | 1d% | 5d% | 20d% | 60d% | MA25乖離% | MA75乖離% | ATR14% | 出来高比20d |\n"
)
_TABLE_SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|\n"
_NO_DATA_MESSAGE = "対象データなし\n"


def _fmt(value: float | None, digits: int = 2) -> str:
    """Render a possibly-missing indicator value; missing data is `N/A`, never 0."""
    return "N/A" if value is None else f"{value:.{digits}f}"


def _format_row(row: IndicatorRow) -> str:
    return (
        f"| {row.symbol} | {row.close:.2f} | {_fmt(row.return_1d_pct)} | "
        f"{_fmt(row.return_5d_pct)} | {_fmt(row.return_20d_pct)} | {_fmt(row.return_60d_pct)} | "
        f"{_fmt(row.ma_deviation_25d_pct)} | {_fmt(row.ma_deviation_75d_pct)} | "
        f"{_fmt(row.atr_pct_14)} | {_fmt(row.volume_ratio_20d)} |\n"
    )


def render_markdown_table(
    universe_code: str, kind: str, briefing_date: date, rows: list[IndicatorRow]
) -> str:
    """Render the Markdown section for one universe's indicator rows."""
    header = f"## {universe_code} ({kind}, {briefing_date.isoformat()})\n\n"
    if not rows:
        return header + _NO_DATA_MESSAGE
    body = "".join(_format_row(row) for row in rows)
    return header + _TABLE_HEADER + _TABLE_SEPARATOR + body


def build_universe_briefing(
    universe_code: str,
    kind: str,
    briefing_date: date,
    instrument_candles: dict[str, list[Candle]],
) -> Briefing:
    """Compute indicators for every instrument in `instrument_candles` and build a `Briefing`.

    Instruments with no candles (e.g. dropped by the freshness gate) are
    skipped rather than raising, so a partial universe can still be reported.
    Rows are sorted by symbol for deterministic output (snapshot-testable).
    """
    rows = sorted(
        (compute_indicator_row(candles) for candles in instrument_candles.values() if candles),
        key=lambda row: row.symbol,
    )
    content_md = render_markdown_table(universe_code, kind, briefing_date, rows)
    data_snapshot: dict[str, object] = {
        "universe": universe_code,
        "kind": kind,
        "date": briefing_date.isoformat(),
        "indicators": [row.model_dump(mode="json") for row in rows],
    }
    return Briefing(
        universe_code=universe_code,
        briefing_date=briefing_date,
        kind=kind,
        content_md=content_md,
        data_snapshot=data_snapshot,
    )


def save_briefing(repo: BriefingRepo, universe_id: int, briefing: Briefing) -> int:
    """Persist `briefing` to the `briefings` table. Returns the new row's internal id."""
    return repo.add(
        universe_id=universe_id,
        briefing_date=briefing.briefing_date,
        kind=briefing.kind,
        content_md=briefing.content_md,
        data_snapshot_json=json.dumps(briefing.data_snapshot, ensure_ascii=False),
    )

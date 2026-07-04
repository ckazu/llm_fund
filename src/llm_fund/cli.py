"""CLI entry point for llm_fund."""

import sqlite3
from datetime import date

import typer

from llm_fund.briefing.builder import build_universe_briefing, save_briefing
from llm_fund.config import AppSettings, ConfigError, load_settings
from llm_fund.data.loader import DataFreshnessError, PriceLoader
from llm_fund.data.prices import YFinanceSource
from llm_fund.delivery.report import write_report
from llm_fund.domain.models import Candle
from llm_fund.store.db import init_db
from llm_fund.store.repos import (
    BriefingRepo,
    CandleRepo,
    InstrumentRepo,
    UniverseRecord,
    UniverseRepo,
)

DEFAULT_FETCH_LOOKBACK_DAYS = 365
REPORT_KIND = "report"
DAILY_KIND = "daily"
# S4 では判断ライン（judgment/）が未実装のため、trade ユニバースは常に NO_TRADE 固定。
# S6 で LLM/テンプレート判断が入るまでこの理由文言を使う。
NO_TRADE_REASON_JUDGMENT_NOT_IMPLEMENTED = (
    "判断ライン未実装（S6 で実装予定）。現状は常に NO_TRADE。"
)

app = typer.Typer(help="LLM-powered investment fund manager CLI")


def _resolve_as_of(date_str: str | None) -> date:
    """Parse `--date` (YYYY-MM-DD), defaulting to today when omitted."""
    return date.fromisoformat(date_str) if date_str else date.today()


def _sync_and_load_universe(
    settings: AppSettings,
    conn: sqlite3.Connection,
    code: str,
    as_of: date,
    lookback_days: int,
) -> tuple[UniverseRecord, dict[str, list[Candle]], list[str]]:
    """Fetch+cache every instrument in universe `code`, then load fresh candles.

    Returns `(universe_record, {symbol: candles}, stale_messages)`. Instruments
    that fail the freshness gate are omitted from the candles dict and
    reported in `stale_messages` rather than raising, so the caller can decide
    whether a partially-stale universe still blocks the report.
    """
    universe_conf = settings.universes[code]
    instrument_repo = InstrumentRepo(conn)
    universe_repo = UniverseRepo(conn)
    if universe_repo.get_by_code(code) is None:
        universe_repo.add(code, universe_conf.market, universe_conf.report, universe_conf.trade)
    universe_record = universe_repo.get_by_code(code)
    assert universe_record is not None

    loader = PriceLoader(CandleRepo(conn), YFinanceSource(), settings.data.max_staleness_days)
    stale_messages: list[str] = []
    candles_by_symbol: dict[str, list[Candle]] = {}
    for inst in universe_conf.instruments:
        record = instrument_repo.get_by_symbol(inst.symbol)
        if record is None:
            instrument_id = instrument_repo.add(inst.symbol, inst.name, universe_conf.market)
            record = instrument_repo.get_by_id(instrument_id)
        assert record is not None

        loader.fetch_and_cache(record, as_of, lookback_days)
        try:
            candles_by_symbol[inst.symbol] = loader.load_fresh(record, as_of, lookback_days)
        except DataFreshnessError as exc:
            stale_messages.append(str(exc))

    return universe_record, candles_by_symbol, stale_messages


def _build_report_markdown(
    settings: AppSettings,
    conn: sqlite3.Connection,
    codes: list[str],
    as_of: date,
    kind: str,
    lookback_days: int,
) -> tuple[str, list[str]]:
    """Build combined Markdown for `codes`, saving one `briefings` row per universe.

    Returns `(content_md, stale_messages)`.
    """
    briefing_repo = BriefingRepo(conn)
    sections: list[str] = []
    stale_messages: list[str] = []
    for code in codes:
        universe_record, candles_by_symbol, stale = _sync_and_load_universe(
            settings, conn, code, as_of, lookback_days
        )
        stale_messages.extend(stale)
        briefing = build_universe_briefing(code, kind, as_of, candles_by_symbol)
        save_briefing(briefing_repo, universe_record.id, briefing)
        sections.append(briefing.content_md)
    return "\n".join(sections), stale_messages


def _render_no_trade_section(trade_codes: list[str]) -> str:
    """Fixed NO_TRADE section for trade-enabled universes (judgment line lands in S6)."""
    lines = ["## 売買判断（Trade Line）\n\n"]
    if not trade_codes:
        lines.append("対象ユニバースなし\n")
        return "".join(lines)
    for code in trade_codes:
        lines.append(f"- {code}: NO_TRADE — {NO_TRADE_REASON_JUDGMENT_NOT_IMPLEMENTED}\n")
    return "".join(lines)


@app.command()
def report(
    universe: str | None = typer.Argument(
        None,
        help="Universe code from config/universes.yaml. Omit to report all report universes.",
    ),
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Report line only: fetch -> briefing -> Markdown report file. No trading judgment.

    Aborts with exit code 2 if any instrument's latest bar fails the
    freshness gate, or exit code 3 if configuration is invalid/the universe
    is unknown.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[report] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    if universe is not None and universe not in settings.universes:
        typer.echo(f"[report] unknown universe: {universe}", err=True)
        raise typer.Exit(code=3)

    codes = (
        [universe]
        if universe is not None
        else [code for code, conf in settings.universes.items() if conf.report]
    )

    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)
    content_md, stale_messages = _build_report_markdown(
        settings, conn, codes, as_of, REPORT_KIND, DEFAULT_FETCH_LOOKBACK_DAYS
    )

    if stale_messages:
        for message in stale_messages:
            typer.echo(f"[report] stale: {message}", err=True)
        raise typer.Exit(code=2)

    out_path = write_report(settings.report.output_dir, as_of, REPORT_KIND, content_md)
    typer.echo(f"[report] universes={codes} format={format_} written to {out_path}")


@app.command()
def daily(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    no_llm: bool = typer.Option(
        False, help="Use template judgment instead of LLM (for testing)."
    ),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Generate the daily briefing for all report universes, plus trading instructions.

    Trading judgment (LLM/template) lands in S6 (validator in S5); until then
    trade-enabled universes always report NO_TRADE.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[daily] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    report_codes = [code for code, conf in settings.universes.items() if conf.report]
    trade_codes = [code for code, conf in settings.universes.items() if conf.trade]

    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)
    content_md, stale_messages = _build_report_markdown(
        settings, conn, report_codes, as_of, DAILY_KIND, DEFAULT_FETCH_LOOKBACK_DAYS
    )

    if stale_messages:
        for message in stale_messages:
            typer.echo(f"[daily] stale: {message}", err=True)
        raise typer.Exit(code=2)

    full_content = content_md + "\n" + _render_no_trade_section(trade_codes)
    out_path = write_report(settings.report.output_dir, as_of, DAILY_KIND, full_content)
    typer.echo(
        f"[daily] date={as_of.isoformat()}, no_llm={no_llm}, format={format_} -> {out_path}"
    )


@app.command()
def weekly(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Review past week's trades and propose criteria adjustments.

    Analyzes execution performance and recommends updates to IFO parameters.
    """
    typer.echo(f"[weekly] date={date}, format={format_} (not yet implemented)")


@app.command()
def monthly(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Review past month's performance and propose policy/universe changes.

    Recommends updates to monthly policy and universe membership.
    """
    typer.echo(f"[monthly] date={date}, format={format_} (not yet implemented)")


@app.command()
def record(
    ticket_no: str = typer.Argument(..., help="Instruction ticket ID (e.g., 20260704-01)"),
    executed_at: str = typer.Option(
        ..., help="Execution timestamp (ISO 8601, e.g., 2026-07-04T10:30:00+09:00)"
    ),
    actual_price: float = typer.Option(..., help="Actual execution price"),
    actual_units: int = typer.Option(..., help="Actual executed units"),
    status: str = typer.Option(
        "filled", help="Execution status: filled, partial, skipped"
    ),
    skip_reason: str | None = typer.Option(None, help="Reason if skipped"),
) -> None:
    """Record manual execution result for a trading instruction.

    Updates the instruction status and calculates deviation from expected execution.
    """
    typer.echo(
        f"[record] ticket_no={ticket_no}, executed_at={executed_at}, "
        f"actual_price={actual_price}, actual_units={actual_units}, "
        f"status={status} (not yet implemented)"
    )


@app.command()
def benchmark(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Calculate and compare performance across LLM, index, momentum, and random benchmarks.

    Uses virtual execution to measure pure judgment quality independent of manual execution.
    """
    typer.echo(f"[benchmark] date={date}, format={format_} (not yet implemented)")


@app.command()
def fetch(
    universe: str | None = typer.Argument(
        None, help="Universe code from config/universes.yaml. Omit to fetch all universes."
    ),
    days: int = typer.Option(
        DEFAULT_FETCH_LOOKBACK_DAYS, help="Number of historical days to fetch (default: 1 year)"
    ),
) -> None:
    """Fetch or refresh price data for a universe (or all universes) from yfinance.

    Caches candles locally in SQLite (diff sync). Aborts with exit code 2 if
    any fetched instrument's latest bar fails the freshness gate, or exit
    code 3 if configuration is invalid/the universe is unknown.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[fetch] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    if universe is not None and universe not in settings.universes:
        typer.echo(f"[fetch] unknown universe: {universe}", err=True)
        raise typer.Exit(code=3)

    target_codes = [universe] if universe is not None else list(settings.universes)

    conn = init_db(settings.db_path)
    instrument_repo = InstrumentRepo(conn)
    universe_repo = UniverseRepo(conn)
    loader = PriceLoader(CandleRepo(conn), YFinanceSource(), settings.data.max_staleness_days)

    as_of = date.today()
    stale_messages: list[str] = []
    for code in target_codes:
        universe_conf = settings.universes[code]
        if universe_repo.get_by_code(code) is None:
            universe_repo.add(code, universe_conf.market, universe_conf.report, universe_conf.trade)

        for inst in universe_conf.instruments:
            record = instrument_repo.get_by_symbol(inst.symbol)
            if record is None:
                instrument_id = instrument_repo.add(inst.symbol, inst.name, universe_conf.market)
                record = instrument_repo.get_by_id(instrument_id)
            assert record is not None

            loader.fetch_and_cache(record, as_of, days)
            try:
                loader.load_fresh(record, as_of, days)
            except DataFreshnessError as exc:
                stale_messages.append(str(exc))

    if stale_messages:
        for message in stale_messages:
            typer.echo(f"[fetch] stale: {message}", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"[fetch] universes={target_codes} days={days} completed")


@app.command()
def status() -> None:
    """Display current system status (DB, cache freshness, fund NAV, positions).

    Quick health check before running daily/weekly/monthly cycles.
    """
    typer.echo("[status] (not yet implemented)")


if __name__ == "__main__":
    app()

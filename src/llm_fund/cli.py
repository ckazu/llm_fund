"""CLI entry point for llm_fund."""

from datetime import date

import typer

from llm_fund.config import ConfigError, load_settings
from llm_fund.data.loader import DataFreshnessError, PriceLoader
from llm_fund.data.prices import YFinanceSource
from llm_fund.store.db import init_db
from llm_fund.store.repos import CandleRepo, InstrumentRepo, UniverseRepo

DEFAULT_FETCH_LOOKBACK_DAYS = 365

app = typer.Typer(help="LLM-powered investment fund manager CLI")


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
    """Generate daily briefing and trading instructions (if configured).

    For trade-enabled universes, generates structured trading instructions
    based on LLM judgment or template rules.
    """
    typer.echo(
        f"[daily] date={date}, no_llm={no_llm}, format={format_} (not yet implemented)"
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

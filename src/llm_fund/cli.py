"""CLI entry point for llm_fund."""

import typer

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
    symbols: str = typer.Option(
        "",
        help="Comma-separated symbols to fetch (e.g., 7203.T,AAPL). "
        "Empty = all configured instruments.",
    ),
    days: int = typer.Option(
        365, help="Number of historical days to fetch (default: 1 year)"
    ),
) -> None:
    """Fetch or refresh price data for configured instruments from yfinance.

    Caches data locally in SQLite and validates freshness for trading.
    """
    typer.echo(
        f"[fetch] symbols={symbols if symbols else 'all configured'}, "
        f"days={days} (not yet implemented)"
    )


@app.command()
def status() -> None:
    """Display current system status (DB, cache freshness, fund NAV, positions).

    Quick health check before running daily/weekly/monthly cycles.
    """
    typer.echo("[status] (not yet implemented)")


if __name__ == "__main__":
    app()

"""CLI entry point for llm_fund."""

import sqlite3
from datetime import date

import anthropic
import typer

from llm_fund.briefing.builder import build_universe_briefing, save_briefing
from llm_fund.config import AppSettings, ConfigError, load_settings
from llm_fund.data.loader import DataFreshnessError, PriceLoader
from llm_fund.data.prices import YFinanceSource
from llm_fund.delivery.report import write_report
from llm_fund.domain.models import Candle, JudgmentResult
from llm_fund.judgment import prompts
from llm_fund.judgment.client import (
    LlmConfig,
    gather_consistent_judgment,
)
from llm_fund.judgment.template import template_judgment
from llm_fund.store.db import init_db
from llm_fund.store.repos import (
    AuditEventRepo,
    BriefingRepo,
    CandleRepo,
    InstructionRepo,
    InstrumentRepo,
    LlmCallRepo,
    PortfolioStateRepo,
    UniverseRecord,
    UniverseRepo,
)
from llm_fund.validator.gate import GateDecision, apply_gate, persist_gate_result
from llm_fund.validator.rules import (
    InstrumentContext,
    RiskLimits,
    ValidationContext,
)

DEFAULT_FETCH_LOOKBACK_DAYS = 365
REPORT_KIND = "report"
DAILY_KIND = "daily"

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
) -> tuple[str, list[str], dict[str, int], dict[str, dict[str, list[Candle]]]]:
    """Build combined Markdown for `codes`, saving one `briefings` row per universe.

    Returns `(content_md, stale_messages, briefing_ids, candles_by_code)`. The latter
    two let the trade line reuse the briefing row and candles already fetched here,
    avoiding a second yfinance round trip for universes that are both report and trade.
    """
    briefing_repo = BriefingRepo(conn)
    sections: list[str] = []
    stale_messages: list[str] = []
    briefing_ids: dict[str, int] = {}
    candles_by_code: dict[str, dict[str, list[Candle]]] = {}
    for code in codes:
        universe_record, candles_by_symbol, stale = _sync_and_load_universe(
            settings, conn, code, as_of, lookback_days
        )
        stale_messages.extend(stale)
        candles_by_code[code] = candles_by_symbol
        briefing = build_universe_briefing(code, kind, as_of, candles_by_symbol)
        briefing_ids[code] = save_briefing(briefing_repo, universe_record.id, briefing)
        sections.append(briefing.content_md)
    return "\n".join(sections), stale_messages, briefing_ids, candles_by_code


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
    content_md, stale_messages, _, _ = _build_report_markdown(
        settings, conn, codes, as_of, REPORT_KIND, DEFAULT_FETCH_LOOKBACK_DAYS
    )

    if stale_messages:
        for message in stale_messages:
            typer.echo(f"[report] stale: {message}", err=True)
        raise typer.Exit(code=2)

    out_path = write_report(settings.report.output_dir, as_of, REPORT_KIND, content_md)
    typer.echo(f"[report] universes={codes} format={format_} written to {out_path}")


def _build_validation_context(
    settings: AppSettings,
    conn: sqlite3.Connection,
    candles_by_symbol: dict[str, list[Candle]],
    data_fresh: bool,
) -> tuple[ValidationContext, dict[str, int]]:
    """Build a `ValidationContext` + symbol->instrument_id map for the gate.

    NAV/cash come from `portfolio_state` (latest snapshot; 0 when uninitialised, which
    conservatively rejects every BUY). Positions are unavailable until S8, so held units
    and current exposure are 0. `prev_close` uses the latest non-adjusted close.
    """
    portfolio = PortfolioStateRepo(conn).latest()
    nav = portfolio.nav if portfolio is not None else 0.0
    cash = portfolio.cash if portfolio is not None else 0.0

    instrument_repo = InstrumentRepo(conn)
    instruments: dict[str, InstrumentContext] = {}
    instrument_id_by_symbol: dict[str, int] = {}
    for symbol, candles in candles_by_symbol.items():
        if not candles:
            continue
        record = instrument_repo.get_by_symbol(symbol)
        if record is None:
            continue
        instrument_id_by_symbol[symbol] = record.id
        instruments[symbol] = InstrumentContext(
            symbol=symbol,
            in_universe=True,
            lot_size=record.lot_size,
            prev_close=candles[-1].close,
            current_units=0,
        )

    ctx = ValidationContext(
        nav=nav,
        cash=cash,
        current_exposure=0.0,
        instruments=instruments,
        limits=RiskLimits.from_settings(settings.limits),
        data_fresh=data_fresh,
    )
    return ctx, instrument_id_by_symbol


def _run_judgment(
    settings: AppSettings,
    conn: sqlite3.Connection,
    *,
    briefing_md: str,
    as_of: date,
    briefing_id: int,
) -> tuple[JudgmentResult, float]:
    """Run the LLM self-consistency gate and return `(judgment, disagreement_rate)`."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    config = LlmConfig(
        model=settings.llm.model,
        temperature=settings.judgment.temperature,
        max_tokens=settings.judgment.max_tokens,
        n_samples=settings.judgment.n_samples,
    )
    portfolio = PortfolioStateRepo(conn).latest()
    nav = portfolio.nav if portfolio is not None else 0.0
    cash = portfolio.cash if portfolio is not None else 0.0
    user_prompt = prompts.build_user_prompt(
        briefing_md=briefing_md,
        portfolio_summary=prompts.format_portfolio_ratio(nav, cash, {}),
    )
    decision = gather_consistent_judgment(
        client,
        config,
        system=prompts.SYSTEM_PROMPT,
        user_prompt=user_prompt,
        as_of=as_of,
        prompt_version=prompts.PROMPT_VERSION,
        briefing_id=briefing_id,
        llm_call_sink=LlmCallRepo(conn),
        audit_sink=AuditEventRepo(conn),
    )
    return decision.judgment, decision.disagreement_rate


def _run_trade_line(
    settings: AppSettings,
    conn: sqlite3.Connection,
    code: str,
    as_of: date,
    *,
    no_llm: bool,
    briefing_id: int,
    briefing_md: str,
    candles_by_symbol: dict[str, list[Candle]],
) -> str:
    """Judge -> validate -> persist -> render one trade universe's Markdown section."""
    expected = {inst.symbol for inst in settings.universes[code].instruments}
    present = {symbol for symbol, candles in candles_by_symbol.items() if candles}
    data_fresh = expected.issubset(present)

    disagreement: float | None = None
    if no_llm:
        judgment: JudgmentResult | None = template_judgment()
    elif not data_fresh:
        # 鮮度違反時は課金前に LLM をスキップし、ゲートに NO_TRADE を強制させる。
        judgment = None
    else:
        judgment, disagreement = _run_judgment(
            settings, conn, briefing_md=briefing_md, as_of=as_of, briefing_id=briefing_id
        )

    ctx, instrument_id_by_symbol = _build_validation_context(
        settings, conn, candles_by_symbol, data_fresh
    )
    instruction_repo = InstructionRepo(conn)
    if judgment is None:
        gate_decision = apply_gate([], ctx, as_of, instruction_repo.next_sequence(as_of))
    elif judgment.no_trade:
        gate_decision = GateDecision(no_trade=True, no_trade_reason=judgment.no_trade_reason)
    else:
        gate_decision = apply_gate(
            judgment.orders, ctx, as_of, instruction_repo.next_sequence(as_of)
        )

    persist_gate_result(
        gate_decision,
        instruction_repo,
        AuditEventRepo(conn),
        briefing_id=briefing_id,
        instrument_id_by_symbol=instrument_id_by_symbol,
    )
    return _render_trade_section(code, gate_decision, disagreement, no_llm=no_llm)


def _render_trade_section(
    code: str,
    decision: GateDecision,
    disagreement_rate: float | None,
    *,
    no_llm: bool,
) -> str:
    """Render validated instructions, rejections, warnings and disagreement for one universe."""
    mode = "テンプレート判断（--no-llm）" if no_llm else "LLM 自己一致性判断"
    lines = [f"### {code}\n", f"- モード: {mode}\n"]
    if disagreement_rate is not None:
        lines.append(f"- 不一致率: {disagreement_rate * 100:.1f}%\n")
    if decision.no_trade:
        lines.append(f"- {code}: NO_TRADE — {decision.no_trade_reason}\n")
    if decision.validated:
        lines.append("- 指示:\n")
        for vi in decision.validated:
            warn = f"（警告: {'; '.join(vi.warnings)}）" if vi.warnings else ""
            lines.append(
                f"  - {vi.ticket_no} {vi.action.value} {vi.symbol} {vi.units}株 "
                f"entry={vi.entry_price:.1f} tp={vi.tp_price:.1f} "
                f"sl={vi.sl_price:.1f}{warn}\n"
            )
    elif not decision.no_trade:
        lines.append(f"- {code}: 承認された指示なし\n")
    if decision.rejections:
        lines.append("- 拒否:\n")
        for rej in decision.rejections:
            lines.append(
                f"  - {rej.order.symbol} {rej.order.action.value} — "
                f"{'; '.join(rej.reasons)}\n"
            )
    return "".join(lines)


def _render_trade_line(
    settings: AppSettings,
    conn: sqlite3.Connection,
    trade_codes: list[str],
    as_of: date,
    *,
    no_llm: bool,
    briefing_ids: dict[str, int],
    candles_by_code: dict[str, dict[str, list[Candle]]],
) -> str:
    """Run the trade line for every trade-enabled universe and combine their sections."""
    header = "## 売買判断（Trade Line）\n\n"
    if not trade_codes:
        return header + "対象ユニバースなし\n"

    sections: list[str] = []
    for code in trade_codes:
        candles_by_symbol = candles_by_code.get(code)
        briefing_id = briefing_ids.get(code)
        if candles_by_symbol is None or briefing_id is None:
            # trade-only ユニバース（report 対象外）はここで初めて取得しブリーフィングを保存する。
            universe_record, candles_by_symbol, _ = _sync_and_load_universe(
                settings, conn, code, as_of, DEFAULT_FETCH_LOOKBACK_DAYS
            )
            briefing = build_universe_briefing(code, DAILY_KIND, as_of, candles_by_symbol)
            briefing_id = save_briefing(BriefingRepo(conn), universe_record.id, briefing)
        briefing_md = build_universe_briefing(
            code, DAILY_KIND, as_of, candles_by_symbol
        ).content_md
        sections.append(
            _run_trade_line(
                settings,
                conn,
                code,
                as_of,
                no_llm=no_llm,
                briefing_id=briefing_id,
                briefing_md=briefing_md,
                candles_by_symbol=candles_by_symbol,
            )
        )
    return header + "\n".join(sections)


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
    """Report all report universes, then judge -> validate -> record for trade universes.

    Aborts with exit code 2 if any report universe's data is stale, or 3 if config is
    invalid (or the LLM path is requested without an API key). NO_TRADE is exit 0.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[daily] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    report_codes = [code for code, conf in settings.universes.items() if conf.report]
    trade_codes = [code for code, conf in settings.universes.items() if conf.trade]

    if not no_llm and trade_codes and settings.anthropic_api_key is None:
        typer.echo(
            "[daily] ANTHROPIC_API_KEY 未設定。--no-llm を使うか .env に設定してください。",
            err=True,
        )
        raise typer.Exit(code=3)

    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)
    content_md, stale_messages, briefing_ids, candles_by_code = _build_report_markdown(
        settings, conn, report_codes, as_of, DAILY_KIND, DEFAULT_FETCH_LOOKBACK_DAYS
    )

    if stale_messages:
        for message in stale_messages:
            typer.echo(f"[daily] stale: {message}", err=True)
        raise typer.Exit(code=2)

    trade_md = _render_trade_line(
        settings,
        conn,
        trade_codes,
        as_of,
        no_llm=no_llm,
        briefing_ids=briefing_ids,
        candles_by_code=candles_by_code,
    )
    full_content = content_md + "\n" + trade_md
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

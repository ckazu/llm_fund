"""CLI entry point for llm_fund."""

import json
import sqlite3
from datetime import UTC, date, datetime

import anthropic
import typer

from llm_fund.briefing.builder import build_universe_briefing, save_briefing
from llm_fund.config import AppSettings, ConfigError, load_settings
from llm_fund.data.loader import DataFreshnessError, PriceLoader
from llm_fund.data.prices import YFinanceSource
from llm_fund.delivery.report import write_report
from llm_fund.domain.enums import ExecutionStatus, InstructionStatus
from llm_fund.domain.models import Candle, JudgmentResult
from llm_fund.judgment import prompts
from llm_fund.judgment.client import (
    LlmConfig,
    gather_consistent_judgment,
)
from llm_fund.judgment.template import template_judgment
from llm_fund.review.monthly import (
    AUDIT_KIND_POLICY_APPROVED,
    render_benchmark_performance_md,
    render_monthly_result_md,
    run_monthly_review,
)
from llm_fund.review.monthly import PROMPT_VERSION as MONTHLY_PROMPT_VERSION
from llm_fund.review.weekly import (
    AUDIT_KIND_CRITERIA_APPROVED,
    render_weekly_result_md,
    run_weekly_review,
)
from llm_fund.review.weekly import PROMPT_VERSION as WEEKLY_PROMPT_VERSION
from llm_fund.store.db import init_db
from llm_fund.store.repos import (
    AuditEventRepo,
    BriefingRepo,
    CandleRepo,
    CriteriaRepo,
    ExecutionRepo,
    InstructionRepo,
    InstrumentRepo,
    LlmCallRepo,
    PolicyRepo,
    PortfolioStateRepo,
    UniverseRecord,
    UniverseRepo,
    price_deviation_pct,
)
from llm_fund.tracking import benchmark as benchmark_mod
from llm_fund.tracking.virtual_fill import CostModel, run_virtual_fills
from llm_fund.validator.gate import GateDecision, apply_gate, persist_gate_result
from llm_fund.validator.rules import (
    InstrumentContext,
    RiskLimits,
    ValidationContext,
)

DEFAULT_FETCH_LOOKBACK_DAYS = 365
REPORT_KIND = "report"
DAILY_KIND = "daily"
# ブローカーへの注文形態。本システムはサーバーサイド IFO 注文のみを扱う（technical-spec.md 2章）。
ORDER_TYPE_IFO = "IFO"

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
    # 仮想執行でポートフォリオ状態を as_of まで更新し、対照群 NAV を日次更新する（FR-5）。
    summary = _run_tracking(settings, conn, as_of)
    benchmark_md = _render_benchmark_section(summary)
    full_content = content_md + "\n" + trade_md + "\n" + benchmark_md
    out_path = write_report(settings.report.output_dir, as_of, DAILY_KIND, full_content)
    typer.echo(
        f"[daily] date={as_of.isoformat()}, no_llm={no_llm}, format={format_} -> {out_path}"
    )


def _require_llm_settings(settings: AppSettings, command: str) -> anthropic.Anthropic:
    """Abort with exit code 3 if no API key is configured, else return an anthropic client."""
    if settings.anthropic_api_key is None:
        typer.echo(
            f"[{command}] ANTHROPIC_API_KEY 未設定。.env に設定してください。", err=True
        )
        raise typer.Exit(code=3)
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _review_llm_config(settings: AppSettings) -> LlmConfig:
    """`weekly`/`monthly` は自己一致性チェックを行わない単発呼び出し（n_samples=1）。"""
    return LlmConfig(
        model=settings.llm.model,
        temperature=settings.judgment.temperature,
        max_tokens=settings.judgment.max_tokens,
        n_samples=1,
    )


@app.command()
def weekly(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Review past week's trades and propose criteria adjustments (FR-6).

    Analyzes virtual-fill performance and IFO width, and proposes a criteria
    change (diff + rationale) saved as `status=draft`. Requires human approval
    via `fund approve criteria:<id>` before it takes effect.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[weekly] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    client = _require_llm_settings(settings, "weekly")
    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)

    result = run_weekly_review(
        conn,
        client,
        _review_llm_config(settings),
        as_of=as_of,
        prompt_version=WEEKLY_PROMPT_VERSION,
        llm_call_sink=LlmCallRepo(conn),
        audit_sink=AuditEventRepo(conn),
    )
    typer.echo(render_weekly_result_md(result))
    typer.echo(f"[weekly] date={as_of.isoformat()} format={format_} no_change={result.no_change}")


@app.command()
def monthly(
    date: str | None = typer.Option(None, help="Target date (YYYY-MM-DD). Defaults to today."),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown, json"
    ),
) -> None:
    """Review past month's performance and propose policy/universe changes (FR-6).

    Compares LLM vs. control-group NAV and proposes a policy change (diff +
    rationale) saved as `status=draft`, plus any universe change suggestions
    recorded to `audit_events` for manual reflection into
    `config/universes.yaml`. Requires human approval via `fund approve
    policy:<id>` before the policy change takes effect.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[monthly] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    client = _require_llm_settings(settings, "monthly")
    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)

    summary = _run_tracking(settings, conn, as_of)
    performance_md = render_benchmark_performance_md(summary)
    result = run_monthly_review(
        conn,
        client,
        _review_llm_config(settings),
        as_of=as_of,
        prompt_version=MONTHLY_PROMPT_VERSION,
        performance_summary=performance_md,
        llm_call_sink=LlmCallRepo(conn),
        audit_sink=AuditEventRepo(conn),
    )
    typer.echo(render_monthly_result_md(result, performance_md))
    typer.echo(
        f"[monthly] date={as_of.isoformat()} format={format_} no_change={result.no_change}"
    )


@app.command()
def approve(
    proposal: str = typer.Argument(
        ..., help="Proposal id as 'criteria:<id>' or 'policy:<id>' (from fund weekly/monthly)."
    ),
) -> None:
    """Approve a weekly criteria or monthly policy proposal (FR-6).

    Activates the given proposal and marks the previously active version of
    the same kind as superseded. Approving a superseded criteria/policy again
    reactivates it and supersedes the current one -- this is the rollback path.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[approve] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    kind, sep, id_str = proposal.partition(":")
    if not sep or not id_str.isdigit():
        typer.echo(
            f"[approve] invalid proposal: {proposal!r} "
            "(expected 'criteria:<id>' or 'policy:<id>')",
            err=True,
        )
        raise typer.Exit(code=1)
    proposal_id = int(id_str)

    conn = init_db(settings.db_path)
    audit_repo = AuditEventRepo(conn)

    if kind == "criteria":
        criteria_repo = CriteriaRepo(conn)
        if criteria_repo.get_by_id(proposal_id) is None:
            typer.echo(f"[approve] unknown criteria id: {proposal_id}", err=True)
            raise typer.Exit(code=1)
        criteria_repo.approve(proposal_id)
        audit_repo.add(
            AUDIT_KIND_CRITERIA_APPROVED,
            json.dumps({"criteria_id": proposal_id}, ensure_ascii=False),
        )
        typer.echo(f"[approve] criteria:{proposal_id} activated")
    elif kind == "policy":
        policy_repo = PolicyRepo(conn)
        if policy_repo.get_by_id(proposal_id) is None:
            typer.echo(f"[approve] unknown policy id: {proposal_id}", err=True)
            raise typer.Exit(code=1)
        policy_repo.approve(proposal_id)
        audit_repo.add(
            AUDIT_KIND_POLICY_APPROVED,
            json.dumps({"policy_id": proposal_id}, ensure_ascii=False),
        )
        typer.echo(f"[approve] policy:{proposal_id} activated")
    else:
        typer.echo(
            f"[approve] unknown proposal kind: {kind!r} (expected 'criteria' or 'policy')",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def record(
    ticket_no: str = typer.Argument(..., help="Instruction ticket ID (e.g., 20260704-01)"),
    price: float = typer.Option(..., "--price", help="Actual execution price"),
    units: int | None = typer.Option(
        None, "--units", help="Actual executed units (default: the instructed units)"
    ),
    skipped: str | None = typer.Option(
        None, "--skipped", help="Reason the order was not executed (marks status=skipped)"
    ),
    commission: float = typer.Option(0.0, "--commission", help="Actual commission paid"),
) -> None:
    """Record the human execution result for one instruction (FR-4).

    Resolves `ticket_no` to the internal instruction (PK never exposed), writes an
    `executions` row, and moves the instruction to filled/partial/skipped. Aborts with
    exit code 1 if `ticket_no` is unknown or the units/price combination is invalid.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[record] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    conn = init_db(settings.db_path)
    instruction_repo = InstructionRepo(conn)
    instruction = instruction_repo.get_by_ticket_no(ticket_no)
    if instruction is None:
        typer.echo(f"[record] unknown ticket_no: {ticket_no}", err=True)
        raise typer.Exit(code=1)

    if skipped is not None:
        actual_units = 0
        exec_status = ExecutionStatus.SKIPPED
        deviation_note = None
    else:
        actual_units = units if units is not None else instruction.units
        if actual_units <= 0:
            typer.echo(
                "[record] --units must be > 0 unless --skipped is given", err=True
            )
            raise typer.Exit(code=1)
        exec_status = (
            ExecutionStatus.FILLED
            if actual_units >= instruction.units
            else ExecutionStatus.PARTIAL
        )
        deviation_pct = price_deviation_pct(price, instruction.entry_price)
        deviation_note = f"価格乖離: {deviation_pct:+.2f}%"

    ExecutionRepo(conn).add(
        instruction_id=instruction.id,
        executed_at=datetime.now(UTC),
        side=instruction.action,
        order_type=ORDER_TYPE_IFO,
        actual_price=price,
        actual_units=actual_units,
        commission=commission,
        status=exec_status.value,
        skip_reason=skipped,
        deviation_note=deviation_note,
    )
    instruction_repo.update_status(instruction.id, exec_status.value)

    typer.echo(
        f"[record] ticket_no={ticket_no} status={exec_status.value} "
        f"price={price} units={actual_units}"
        + (f" skip_reason={skipped}" if skipped else f" {deviation_note}")
    )


def _cost_model(settings: AppSettings) -> CostModel:
    """Build the shared `CostModel` from `tracking` config (same for fund and benchmarks)."""
    return CostModel(
        commission_rate=settings.tracking.commission_rate,
        min_commission=settings.tracking.min_commission,
        slippage_pct=settings.tracking.slippage_pct,
    )


def _trade_universe_symbols(settings: AppSettings) -> list[str]:
    """All instrument symbols across trade-enabled universes (benchmark opportunity set)."""
    symbols: list[str] = []
    for conf in settings.universes.values():
        if conf.trade:
            symbols.extend(inst.symbol for inst in conf.instruments)
    return symbols


def _run_tracking(
    settings: AppSettings, conn: sqlite3.Connection, as_of: date
) -> benchmark_mod.BenchmarkSummary:
    """Run the virtual-fill engine, then update the control benchmarks (daily / on demand).

    Virtual fill runs first so `portfolio_state` (the fund NAV series) is current before the
    benchmark reads it. Both are deterministic full recomputations, so re-running is idempotent.
    """
    costs = _cost_model(settings)
    run_virtual_fills(
        conn, as_of, costs=costs, starting_capital=settings.tracking.starting_capital
    )
    return benchmark_mod.run_benchmark(
        conn,
        as_of,
        universe_symbols=_trade_universe_symbols(settings),
        index_symbol=settings.benchmark.index_symbol,
        momentum_lookback_days=settings.benchmark.momentum_lookback_days,
        random_seed=settings.tracking.random_seed,
        costs=costs,
        starting_capital=settings.tracking.starting_capital,
    )


def _render_benchmark_section(summary: benchmark_mod.BenchmarkSummary) -> str:
    """Render the LLM-vs-control NAV comparison as a Markdown section."""
    lines = ["## ベンチマーク比較（対照群）\n"]
    if not summary.latest_nav:
        return lines[0] + "\n比較可能なデータがありません\n"
    for code in (
        benchmark_mod.STRATEGY_FUND,
        benchmark_mod.STRATEGY_INDEX,
        benchmark_mod.STRATEGY_EQUAL_WEIGHT,
        benchmark_mod.STRATEGY_MOMENTUM,
        benchmark_mod.STRATEGY_RANDOM,
    ):
        if code not in summary.latest_nav:
            continue
        name = benchmark_mod.STRATEGY_NAMES[code]
        nav = summary.latest_nav[code]
        metrics = summary.metrics.get(code, {})
        max_dd = metrics.get("max_drawdown_pct", 0.0)
        sharpe = metrics.get("sharpe", 0.0)
        lines.append(
            f"- {name}: NAV={nav:,.0f} (MaxDD={max_dd:.1f}%, Sharpe={sharpe:.2f})\n"
        )
    return "".join(lines)


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
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[benchmark] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    as_of = _resolve_as_of(date)
    conn = init_db(settings.db_path)
    summary = _run_tracking(settings, conn, as_of)
    typer.echo(_render_benchmark_section(summary))
    typer.echo(
        f"[benchmark] date={as_of.isoformat()} format={format_} "
        f"strategies={len(summary.latest_nav)}"
    )


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
    """Display NAV, unrecorded instructions, and instruction/execution deviation (FR-4).

    Quick health check before running daily/weekly/monthly cycles.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(f"[status] config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    conn = init_db(settings.db_path)
    lines = ["[status]"]

    portfolio = PortfolioStateRepo(conn).latest()
    if portfolio is not None:
        lines.append(
            f"- NAV: {portfolio.nav:,.0f} (date={portfolio.state_date.isoformat()}, "
            f"cash={portfolio.cash:,.0f})"
        )
    else:
        lines.append("- NAV: 未初期化")

    instruction_repo = InstructionRepo(conn)
    pending = instruction_repo.list_by_status(InstructionStatus.PENDING.value)
    lines.append(f"- 未記録の指示: {len(pending)}件")
    for inst in pending:
        lines.append(f"  - {inst.ticket_no} {inst.action} {inst.units}株")

    deviations = ExecutionRepo(conn).list_deviations()
    if deviations:
        lines.append("- 指示と執行の乖離:")
        for dev in deviations:
            if dev.deviation_pct is None:
                lines.append(f"  - {dev.ticket_no}: 未執行（{dev.status}）")
            else:
                lines.append(f"  - {dev.ticket_no}: 価格乖離 {dev.deviation_pct:+.2f}%")

    unexecuted_rate = ExecutionRepo(conn).unexecuted_rate()
    if unexecuted_rate is not None:
        lines.append(f"- 未執行率: {unexecuted_rate * 100:.1f}%")

    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()

"""Serial validation gate: turns LLM `OrderPlan`s into `ValidatedInstruction`s
or `Rejection`s and records every outcome (technical-spec.md 6章).

`apply_gate` is a pure function (no I/O) so the money-risk logic is exhaustively
unit-testable. It applies the per-order HARD rules from `rules.py`, plus two
concerns that a single order cannot judge on its own:

* **DataFreshness** — a precondition. If the upstream loader could not provide
  fresh data (`ctx.data_fresh is False`) every order is rejected and NO_TRADE is
  forced (technical-spec.md 6章「全指示拒否・NO_TRADE 強制」).
* **MaxTurnover** — cumulative across the batch. Orders are processed in order;
  an order is rejected if accepting it would push the day's accumulated notional
  past the turnover cap. Rejected orders never consume the turnover budget nor a
  ticket_no sequence number.

`persist_gate_result` is the thin I/O wrapper: validated instructions land in
`instructions` (with `validator_result_json`) and every outcome — validation,
warning, rejection, NO_TRADE — is appended to `audit_events`.
"""

import json
from dataclasses import dataclass, field
from datetime import date

from llm_fund.domain.models import OrderPlan, Rejection, ValidatedInstruction
from llm_fund.store.repos import AuditEventRepo, InstructionRepo
from llm_fund.validator.rules import (
    HARD_RULES,
    SOFT_RULES,
    ValidationContext,
    notional,
)

_PCT_DIVISOR = 100.0

AUDIT_KIND_VALIDATED = "instruction_validated"
AUDIT_KIND_REJECTION = "instruction_rejected"
AUDIT_KIND_NO_TRADE = "no_trade"

_TICKET_FMT = "%Y%m%d"
NO_TRADE_REASON_STALE_DATA = (
    "データ鮮度違反により全指示を拒否し NO_TRADE を強制（technical-spec.md 6章）"
)
_RULE_DATA_FRESHNESS = "DataFreshness"
_RULE_MAX_TURNOVER = "MaxTurnover"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Outcome of validating a batch of orders for one trading day."""

    validated: list[ValidatedInstruction] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    no_trade: bool = False
    no_trade_reason: str | None = None


def format_ticket_no(as_of: date, sequence: int) -> str:
    """Human-facing instruction id `YYYYMMDD-NN` (technical-spec.md 3章)."""
    return f"{as_of.strftime(_TICKET_FMT)}-{sequence:02d}"


def _turnover_reason(
    accumulated: float, this_notional: float, ctx: ValidationContext
) -> str | None:
    limit = ctx.nav * ctx.limits.max_turnover_pct / _PCT_DIVISOR
    if accumulated + this_notional > limit:
        return (
            f"{_RULE_MAX_TURNOVER}: 当日回転率が上限 {limit:.0f}"
            f"（NAV×{ctx.limits.max_turnover_pct}%）を超過"
        )
    return None


def apply_gate(
    orders: list[OrderPlan],
    ctx: ValidationContext,
    as_of: date,
    ticket_seq_start: int = 1,
) -> GateDecision:
    """Validate `orders` serially, issuing ticket_no from `ticket_seq_start`.

    Pure: performs no I/O. `ticket_seq_start` is the next sequence number for
    `as_of` (1-based), supplied by the caller from `InstructionRepo.next_sequence`
    so re-runs on the same day keep numbering monotonic.
    """
    if not ctx.data_fresh:
        stale_reason = f"{_RULE_DATA_FRESHNESS}: {NO_TRADE_REASON_STALE_DATA}"
        stale_rejections = [Rejection(order=order, reasons=[stale_reason]) for order in orders]
        return GateDecision(
            validated=[],
            rejections=stale_rejections,
            no_trade=True,
            no_trade_reason=NO_TRADE_REASON_STALE_DATA,
        )

    validated: list[ValidatedInstruction] = []
    rejections: list[Rejection] = []
    accumulated_notional = 0.0
    sequence = ticket_seq_start

    for order in orders:
        reasons = [
            f"{name}: {reason}"
            for name, rule in HARD_RULES
            if (reason := rule(order, ctx)) is not None
        ]
        this_notional = notional(order)
        turnover_reason = _turnover_reason(accumulated_notional, this_notional, ctx)
        if turnover_reason is not None:
            reasons.append(turnover_reason)

        if reasons:
            rejections.append(Rejection(order=order, reasons=reasons))
            continue

        warnings = [
            f"{name}: {warning}"
            for name, rule in SOFT_RULES
            if (warning := rule(order, ctx)) is not None
        ]
        accumulated_notional += this_notional
        validated.append(
            ValidatedInstruction(
                ticket_no=format_ticket_no(as_of, sequence),
                symbol=order.symbol,
                action=order.action,
                units=order.units,
                entry_price=order.entry_price,
                tp_price=order.tp_price,
                sl_price=order.sl_price,
                valid_until=order.valid_until,
                rationale=order.rationale,
                warnings=warnings,
            )
        )
        sequence += 1

    return GateDecision(validated=validated, rejections=rejections)


def persist_gate_result(
    decision: GateDecision,
    instruction_repo: InstructionRepo,
    audit_repo: AuditEventRepo,
    *,
    briefing_id: int,
    instrument_id_by_symbol: dict[str, int],
) -> None:
    """Write validated instructions and audit-log every outcome.

    Rejected orders are audit-only: they never received a ticket_no, so they must
    not be written to `instructions` (its ticket_no is NOT NULL UNIQUE).
    """
    for vi in decision.validated:
        validator_json = json.dumps(
            {"result": "validated", "warnings": vi.warnings},
            ensure_ascii=False,
        )
        instruction_repo.add(
            ticket_no=vi.ticket_no,
            briefing_id=briefing_id,
            instrument_id=instrument_id_by_symbol[vi.symbol],
            action=vi.action.value,
            units=vi.units,
            entry_price=vi.entry_price,
            tp_price=vi.tp_price,
            sl_price=vi.sl_price,
            valid_until=vi.valid_until,
            rationale=vi.rationale,
            validator_result_json=validator_json,
            status=vi.status.value,
        )
        audit_repo.add(
            AUDIT_KIND_VALIDATED,
            json.dumps(
                {"ticket_no": vi.ticket_no, "symbol": vi.symbol, "warnings": vi.warnings},
                ensure_ascii=False,
            ),
        )

    for rej in decision.rejections:
        audit_repo.add(
            AUDIT_KIND_REJECTION,
            json.dumps(
                {
                    "symbol": rej.order.symbol,
                    "action": rej.order.action.value,
                    "reasons": rej.reasons,
                },
                ensure_ascii=False,
            ),
        )

    if decision.no_trade:
        audit_repo.add(
            AUDIT_KIND_NO_TRADE,
            json.dumps({"reason": decision.no_trade_reason}, ensure_ascii=False),
        )

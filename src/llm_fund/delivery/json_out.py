"""`--format json` structured output for `llm_company` integration (technical-spec.md 2章).

Each `render_*_json` function returns a JSON-serialisable `dict` built from the
same result objects the Markdown renderers in `cli.py` consume, so the two
formats never drift. Callers do `json.dumps(render_x_json(...), ensure_ascii=False)`.
"""

from pathlib import Path
from typing import Any

from llm_fund.review.monthly import MonthlyReviewResult
from llm_fund.review.weekly import WeeklyReviewResult
from llm_fund.tracking.benchmark import BenchmarkSummary
from llm_fund.validator.gate import GateDecision


def render_report_json(
    codes: list[str], out_path: str | Path, stale_messages: list[str]
) -> dict[str, Any]:
    """`fund report` result: which universes were reported and where."""
    return {
        "universes": codes,
        "report_path": str(out_path),
        "stale_messages": stale_messages,
    }


def render_gate_decision_json(code: str, decision: GateDecision) -> dict[str, Any]:
    """One trade universe's validated/rejected instructions for `fund daily`."""
    return {
        "universe": code,
        "no_trade": decision.no_trade,
        "no_trade_reason": decision.no_trade_reason,
        "instructions": [
            {
                "ticket_no": vi.ticket_no,
                "action": vi.action.value,
                "symbol": vi.symbol,
                "units": vi.units,
                "entry_price": vi.entry_price,
                "tp_price": vi.tp_price,
                "sl_price": vi.sl_price,
                "warnings": vi.warnings,
            }
            for vi in decision.validated
        ],
        "rejections": [
            {
                "symbol": rej.order.symbol,
                "action": rej.order.action.value,
                "reasons": rej.reasons,
            }
            for rej in decision.rejections
        ],
    }


def render_daily_json(
    out_path: str | Path,
    report_codes: list[str],
    trade_decisions: dict[str, GateDecision],
) -> dict[str, Any]:
    """`fund daily` result: report universes + per-universe trade decisions."""
    return {
        "report_universes": report_codes,
        "report_path": str(out_path),
        "trade": [render_gate_decision_json(code, d) for code, d in trade_decisions.items()],
    }


def render_weekly_json(result: WeeklyReviewResult) -> dict[str, Any]:
    """`fund weekly` result: criteria proposal (or no-change) + performance summary."""
    return {
        "no_change": result.no_change,
        "rationale": result.rationale,
        "criteria_id": result.criteria_id,
        "diff": result.diff,
    }


def render_monthly_json(result: MonthlyReviewResult) -> dict[str, Any]:
    """`fund monthly` result: policy proposal + any universe-change suggestions."""
    return {
        "no_change": result.no_change,
        "rationale": result.rationale,
        "policy_id": result.policy_id,
        "diff": result.diff,
        "universe_changes": [
            {
                "universe_code": uc.universe_code,
                "symbol": uc.symbol,
                "action": uc.action,
                "reason": uc.reason,
            }
            for uc in result.universe_changes
        ],
    }


def render_benchmark_json(summary: BenchmarkSummary) -> dict[str, Any]:
    """`fund benchmark` result: latest NAV and cost metrics per strategy."""
    return {
        "latest_nav": summary.latest_nav,
        "metrics": summary.metrics,
    }


def render_status_json(
    *,
    nav: float | None,
    nav_date: str | None,
    cash: float | None,
    pending_instructions: list[dict[str, Any]],
    unrecorded_proposals: list[dict[str, Any]],
    unexecuted_rate: float | None,
) -> dict[str, Any]:
    """`fund status` result: data freshness / NAV / pending items in one payload."""
    return {
        "nav": nav,
        "nav_date": nav_date,
        "cash": cash,
        "pending_instructions": pending_instructions,
        "pending_proposals": unrecorded_proposals,
        "unexecuted_rate": unexecuted_rate,
    }

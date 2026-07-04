"""Tests for the serial validation gate (validator/gate.py, technical-spec.md 6章).

Covers ticket_no issuance (YYYYMMDD-NN), HARD rejection vs SOFT warning,
cumulative turnover across a batch, and the forced-NO_TRADE-on-stale-data path.
Persistence into instructions/audit_events is exercised end-to-end against a
migrated in-memory SQLite DB.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import Action, InstructionStatus
from llm_fund.domain.models import OrderPlan
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import (
    AuditEventRepo,
    BriefingRepo,
    InstructionRepo,
    InstrumentRepo,
    UniverseRepo,
)
from llm_fund.validator.gate import (
    AUDIT_KIND_NO_TRADE,
    AUDIT_KIND_REJECTION,
    AUDIT_KIND_VALIDATED,
    apply_gate,
    format_ticket_no,
    persist_gate_result,
)
from llm_fund.validator.rules import (
    InstrumentContext,
    RiskLimits,
    ValidationContext,
)

AS_OF = date(2026, 7, 4)
VALID_UNTIL = date(2026, 7, 7)
LONG_RATIONALE = "MA25 を上抜け出来高も伴い上昇トレンド継続と判断"


def _ctx(
    *,
    nav: float = 1_000_000.0,
    cash: float = 1_000_000.0,
    current_exposure: float = 0.0,
    instruments: dict[str, InstrumentContext] | None = None,
    max_turnover_pct: float = 30.0,
    data_fresh: bool = True,
) -> ValidationContext:
    if instruments is None:
        instruments = {
            "7203.T": InstrumentContext(
                symbol="7203.T", in_universe=True, lot_size=100, prev_close=2000.0
            ),
            "6758.T": InstrumentContext(
                symbol="6758.T", in_universe=True, lot_size=100, prev_close=2000.0
            ),
        }
    limits = RiskLimits(
        max_loss_per_trade_pct=3.0,
        max_position_pct=25.0,
        max_exposure_pct=100.0,
        max_turnover_pct=max_turnover_pct,
        min_rationale_length=20,
        require_stop_loss=True,
    )
    return ValidationContext(
        nav=nav,
        cash=cash,
        current_exposure=current_exposure,
        instruments=instruments,
        limits=limits,
        data_fresh=data_fresh,
    )


def _order(
    *,
    symbol: str = "7203.T",
    action: Action = Action.BUY,
    units: int = 100,
    entry_price: float = 2000.0,
    tp_price: float = 2200.0,
    sl_price: float = 1950.0,
    rationale: str = LONG_RATIONALE,
) -> OrderPlan:
    return OrderPlan(
        symbol=symbol,
        action=action,
        units=units,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        valid_until=VALID_UNTIL,
        rationale=rationale,
    )


class TestFormatTicketNo:
    def test_format_pads_sequence_to_two_digits(self) -> None:
        assert format_ticket_no(AS_OF, 1) == "20260704-01"
        assert format_ticket_no(AS_OF, 12) == "20260704-12"


class TestApplyGateHappyPath:
    def test_single_valid_order_is_validated(self) -> None:
        decision = apply_gate([_order()], _ctx(), AS_OF)
        assert decision.no_trade is False
        assert decision.rejections == []
        assert len(decision.validated) == 1
        vi = decision.validated[0]
        assert vi.ticket_no == "20260704-01"
        assert vi.symbol == "7203.T"
        assert vi.status == InstructionStatus.PENDING
        assert vi.warnings == []

    def test_ticket_sequence_starts_from_offset(self) -> None:
        decision = apply_gate([_order()], _ctx(), AS_OF, ticket_seq_start=3)
        assert decision.validated[0].ticket_no == "20260704-03"

    def test_two_valid_orders_get_sequential_tickets(self) -> None:
        orders = [_order(), _order(symbol="6758.T")]
        # Raise the turnover cap so both 200k orders fit; turnover is covered separately.
        decision = apply_gate(orders, _ctx(max_turnover_pct=100.0), AS_OF)
        assert [vi.ticket_no for vi in decision.validated] == ["20260704-01", "20260704-02"]


class TestApplyGateRejection:
    def test_hard_violation_is_rejected_with_reasons(self) -> None:
        # SL above entry (StopLossRequired) — a HARD violation.
        decision = apply_gate([_order(sl_price=3100.0)], _ctx(), AS_OF)
        assert decision.validated == []
        assert len(decision.rejections) == 1
        assert decision.rejections[0].reasons  # non-empty
        assert any("StopLossRequired" in r for r in decision.rejections[0].reasons)

    def test_multiple_hard_violations_collected(self) -> None:
        # Off-tick entry AND unknown universe member -> at least two reasons.
        order = _order(symbol="9999.T", entry_price=3001.0)
        decision = apply_gate([order], _ctx(), AS_OF)
        assert len(decision.rejections) == 1
        assert len(decision.rejections[0].reasons) >= 2

    def test_rejected_order_does_not_consume_ticket_sequence(self) -> None:
        orders = [_order(sl_price=3100.0), _order()]  # first rejected, second valid
        decision = apply_gate(orders, _ctx(), AS_OF)
        assert len(decision.rejections) == 1
        assert len(decision.validated) == 1
        assert decision.validated[0].ticket_no == "20260704-01"


class TestApplyGateWarnings:
    def test_short_rationale_produces_warning_but_validates(self) -> None:
        decision = apply_gate([_order(rationale="買い")], _ctx(), AS_OF)
        assert len(decision.validated) == 1
        assert decision.validated[0].warnings  # non-empty


class TestApplyGateTurnover:
    def test_cumulative_turnover_within_limit_accepts_all(self) -> None:
        # nav 1,000,000, limit 30% = 300,000. 200,000 + 100,000 = 300,000 (==).
        orders = [
            _order(entry_price=2000.0, units=100, sl_price=1950.0),  # 200,000
            _order(symbol="6758.T", entry_price=2000.0, units=50, sl_price=1950.0),  # 100,000
        ]
        # units=50 is not a lot multiple; use lot_size 1 instrument to isolate turnover.
        instruments = {
            "7203.T": InstrumentContext(
                symbol="7203.T", in_universe=True, lot_size=100, prev_close=2000.0
            ),
            "6758.T": InstrumentContext(
                symbol="6758.T", in_universe=True, lot_size=1, prev_close=2000.0
            ),
        }
        decision = apply_gate(orders, _ctx(instruments=instruments), AS_OF)
        assert len(decision.validated) == 2
        assert decision.rejections == []

    def test_order_breaching_cumulative_turnover_is_rejected(self) -> None:
        orders = [
            _order(entry_price=2000.0, units=100, sl_price=1950.0),  # 200,000 accepted
            _order(symbol="6758.T", entry_price=2000.0, units=100, sl_price=1950.0),  # +200,000
        ]
        decision = apply_gate(orders, _ctx(), AS_OF)  # limit 300,000
        assert len(decision.validated) == 1
        assert decision.validated[0].symbol == "7203.T"
        assert len(decision.rejections) == 1
        assert any("Turnover" in r for r in decision.rejections[0].reasons)

    def test_single_order_over_turnover_rejected(self) -> None:
        order = _order(entry_price=4000.0, units=100, sl_price=3900.0)  # 400,000 > 300,000
        decision = apply_gate([order], _ctx(), AS_OF)
        assert decision.validated == []
        assert len(decision.rejections) == 1


class TestApplyGateDataFreshness:
    def test_stale_data_forces_no_trade_and_rejects_all(self) -> None:
        orders = [_order(), _order(symbol="6758.T", entry_price=2000.0, sl_price=1950.0)]
        decision = apply_gate(orders, _ctx(data_fresh=False), AS_OF)
        assert decision.no_trade is True
        assert decision.no_trade_reason is not None
        assert decision.validated == []
        assert len(decision.rejections) == 2
        for rej in decision.rejections:
            assert any("DataFreshness" in r for r in rej.reasons)

    def test_stale_data_with_no_orders_still_no_trade(self) -> None:
        decision = apply_gate([], _ctx(data_fresh=False), AS_OF)
        assert decision.no_trade is True
        assert decision.rejections == []

    def test_fresh_data_with_no_orders_is_not_forced_no_trade(self) -> None:
        decision = apply_gate([], _ctx(data_fresh=True), AS_OF)
        assert decision.no_trade is False
        assert decision.validated == []


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "gate.db")
    apply_migrations(c)
    return c


def _seed_briefing_and_instrument(conn: sqlite3.Connection) -> tuple[int, dict[str, int]]:
    universe_id = UniverseRepo(conn).add("jp_stocks", "jp", report_enabled=True, trade_enabled=True)
    instrument_id = InstrumentRepo(conn).add("7203.T", "トヨタ自動車", "jp")
    briefing_id = BriefingRepo(conn).add(universe_id, AS_OF, "daily", "md", "{}")
    return briefing_id, {"7203.T": instrument_id}


class TestPersistGateResult:
    def test_validated_instruction_and_audit_recorded(self, conn: sqlite3.Connection) -> None:
        briefing_id, id_by_symbol = _seed_briefing_and_instrument(conn)
        decision = apply_gate([_order()], _ctx(), AS_OF)
        persist_gate_result(
            decision,
            InstructionRepo(conn),
            AuditEventRepo(conn),
            briefing_id=briefing_id,
            instrument_id_by_symbol=id_by_symbol,
        )

        stored = InstructionRepo(conn).get_by_ticket_no("20260704-01")
        assert stored is not None
        assert stored.status == InstructionStatus.PENDING.value
        assert stored.validator_result_json is not None
        payload = json.loads(stored.validator_result_json)
        assert payload["result"] == "validated"

        audits = AuditEventRepo(conn).list_by_kind(AUDIT_KIND_VALIDATED)
        assert len(audits) == 1

    def test_rejection_recorded_only_in_audit(self, conn: sqlite3.Connection) -> None:
        briefing_id, id_by_symbol = _seed_briefing_and_instrument(conn)
        decision = apply_gate([_order(sl_price=3100.0)], _ctx(), AS_OF)
        persist_gate_result(
            decision,
            InstructionRepo(conn),
            AuditEventRepo(conn),
            briefing_id=briefing_id,
            instrument_id_by_symbol=id_by_symbol,
        )
        # No instruction row (rejected orders never get a ticket_no).
        assert conn.execute("SELECT COUNT(*) AS n FROM instructions").fetchone()["n"] == 0
        rej_audits = AuditEventRepo(conn).list_by_kind(AUDIT_KIND_REJECTION)
        assert len(rej_audits) == 1
        detail = json.loads(rej_audits[0].detail_json)
        assert detail["reasons"]

    def test_no_trade_recorded_in_audit(self, conn: sqlite3.Connection) -> None:
        briefing_id, id_by_symbol = _seed_briefing_and_instrument(conn)
        decision = apply_gate([_order()], _ctx(data_fresh=False), AS_OF)
        persist_gate_result(
            decision,
            InstructionRepo(conn),
            AuditEventRepo(conn),
            briefing_id=briefing_id,
            instrument_id_by_symbol=id_by_symbol,
        )
        assert len(AuditEventRepo(conn).list_by_kind(AUDIT_KIND_NO_TRADE)) == 1

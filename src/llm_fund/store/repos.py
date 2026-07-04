"""Repository layer over the SQLite store.

S2 added instruments/universes/candles/portfolio_state. S4 adds `BriefingRepo`
(briefings table) for the report line. S5 adds `InstructionRepo`/`AuditEventRepo`.
S7 adds `ExecutionRepo` (executions table; FR-4 執行記録・乖離). Tracking
repositories are still added in later steps as their modules are built.

Records returned here are plain frozen dataclasses rather than the pydantic
domain models in `domain/models.py`: they mirror DB rows (including the
internal surrogate `id`) and are not part of the LLM/validator data flow that
`domain/models.py` covers, except for `Candle` which is reused directly.
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime

from llm_fund.domain.enums import ExecutionStatus, ProposalStatus
from llm_fund.domain.models import Candle

# 東証の一般的な売買単位（100株）。instruments.lot_size の既定値。
DEFAULT_LOT_SIZE = 100


@dataclass(frozen=True, slots=True)
class InstrumentRecord:
    id: int
    symbol: str
    name: str
    market: str
    lot_size: int
    active: bool


@dataclass(frozen=True, slots=True)
class UniverseRecord:
    id: int
    code: str
    market: str
    report_enabled: bool
    trade_enabled: bool


@dataclass(frozen=True, slots=True)
class PortfolioStateRecord:
    id: int
    state_date: date
    cash: float
    nav: float
    note: str | None


class InstrumentRepo:
    """CRUD for `instruments` (symbol is the natural key, id is internal)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        symbol: str,
        name: str,
        market: str,
        lot_size: int = DEFAULT_LOT_SIZE,
        active: bool = True,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO instruments (symbol, name, market, lot_size, active) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, name, market, lot_size, int(active)),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_symbol(self, symbol: str) -> InstrumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM instruments WHERE symbol = ?", (symbol,)
        ).fetchone()
        return self._to_record(row) if row else None

    def get_by_id(self, instrument_id: int) -> InstrumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM instruments WHERE id = ?", (instrument_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def list_active(self) -> list[InstrumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM instruments WHERE active = 1 ORDER BY symbol"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> InstrumentRecord:
        return InstrumentRecord(
            id=row["id"],
            symbol=row["symbol"],
            name=row["name"],
            market=row["market"],
            lot_size=row["lot_size"],
            active=bool(row["active"]),
        )


class UniverseRepo:
    """CRUD for `universes` (code is the natural key, id is internal)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        code: str,
        market: str,
        report_enabled: bool = True,
        trade_enabled: bool = False,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO universes (code, market, report_enabled, trade_enabled) "
            "VALUES (?, ?, ?, ?)",
            (code, market, int(report_enabled), int(trade_enabled)),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_code(self, code: str) -> UniverseRecord | None:
        row = self._conn.execute(
            "SELECT * FROM universes WHERE code = ?", (code,)
        ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[UniverseRecord]:
        rows = self._conn.execute("SELECT * FROM universes ORDER BY code").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> UniverseRecord:
        return UniverseRecord(
            id=row["id"],
            code=row["code"],
            market=row["market"],
            report_enabled=bool(row["report_enabled"]),
            trade_enabled=bool(row["trade_enabled"]),
        )


class CandleRepo:
    """Read/write for `candles`, keyed by (instrument_id, date)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_many(self, instrument_id: int, candles: list[Candle]) -> int:
        """Insert or replace candles for `instrument_id`. Returns the count written."""
        self._conn.executemany(
            "INSERT INTO candles "
            "(instrument_id, date, open, high, low, close, volume, adj_close) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(instrument_id, date) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume, adj_close=excluded.adj_close",
            [
                (
                    instrument_id,
                    c.trade_date.isoformat(),
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                    c.adj_close,
                )
                for c in candles
            ],
        )
        self._conn.commit()
        return len(candles)

    def get_range(
        self,
        instrument_id: int,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Candle]:
        query = "SELECT c.*, i.symbol FROM candles c JOIN instruments i ON i.id = c.instrument_id"
        query += " WHERE c.instrument_id = ?"
        params: list[str | int] = [instrument_id]
        if start is not None:
            query += " AND c.date >= ?"
            params.append(start.isoformat())
        if end is not None:
            query += " AND c.date <= ?"
            params.append(end.isoformat())
        query += " ORDER BY c.date"
        rows = self._conn.execute(query, params).fetchall()
        return [self._to_candle(row) for row in rows]

    def latest(self, instrument_id: int) -> Candle | None:
        row = self._conn.execute(
            "SELECT c.*, i.symbol FROM candles c JOIN instruments i ON i.id = c.instrument_id "
            "WHERE c.instrument_id = ? ORDER BY c.date DESC LIMIT 1",
            (instrument_id,),
        ).fetchone()
        return self._to_candle(row) if row else None

    @staticmethod
    def _to_candle(row: sqlite3.Row) -> Candle:
        return Candle(
            symbol=row["symbol"],
            trade_date=date.fromisoformat(row["date"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            adj_close=row["adj_close"],
        )


class PortfolioStateRepo:
    """Read/write for `portfolio_state` (the source of truth for NAV/cash)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, state_date: date, cash: float, nav: float, note: str | None = None) -> int:
        self._conn.execute(
            "INSERT INTO portfolio_state (date, cash, nav, note) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET cash=excluded.cash, nav=excluded.nav, "
            "note=excluded.note",
            (state_date.isoformat(), cash, nav, note),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM portfolio_state WHERE date = ?", (state_date.isoformat(),)
        ).fetchone()
        return int(row["id"])

    def get_by_date(self, state_date: date) -> PortfolioStateRecord | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_state WHERE date = ?", (state_date.isoformat(),)
        ).fetchone()
        return self._to_record(row) if row else None

    def latest(self) -> PortfolioStateRecord | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_state ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[PortfolioStateRecord]:
        rows = self._conn.execute(
            "SELECT * FROM portfolio_state ORDER BY date"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> PortfolioStateRecord:
        return PortfolioStateRecord(
            id=row["id"],
            state_date=date.fromisoformat(row["date"]),
            cash=row["cash"],
            nav=row["nav"],
            note=row["note"],
        )


@dataclass(frozen=True, slots=True)
class BriefingRecord:
    id: int
    universe_id: int
    briefing_date: date
    kind: str
    content_md: str
    data_snapshot_json: str
    created_at: datetime


class BriefingRepo:
    """Write/read for `briefings` (one row per universe/date/kind)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        universe_id: int,
        briefing_date: date,
        kind: str,
        content_md: str,
        data_snapshot_json: str,
    ) -> int:
        created_at = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO briefings "
            "(universe_id, date, kind, content_md, data_snapshot_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                universe_id,
                briefing_date.isoformat(),
                kind,
                content_md,
                data_snapshot_json,
                created_at,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_id(self, briefing_id: int) -> BriefingRecord | None:
        row = self._conn.execute(
            "SELECT * FROM briefings WHERE id = ?", (briefing_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def latest_for_universe(self, universe_id: int, kind: str) -> BriefingRecord | None:
        row = self._conn.execute(
            "SELECT * FROM briefings WHERE universe_id = ? AND kind = ? "
            "ORDER BY date DESC LIMIT 1",
            (universe_id, kind),
        ).fetchone()
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: sqlite3.Row) -> BriefingRecord:
        return BriefingRecord(
            id=row["id"],
            universe_id=row["universe_id"],
            briefing_date=date.fromisoformat(row["date"]),
            kind=row["kind"],
            content_md=row["content_md"],
            data_snapshot_json=row["data_snapshot_json"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class InstructionRecord:
    id: int
    ticket_no: str
    briefing_id: int
    instrument_id: int
    action: str
    units: int
    entry_price: float
    tp_price: float
    sl_price: float
    valid_until: date
    rationale: str
    validator_result_json: str | None
    status: str


class InstructionRepo:
    """Write/read for `instructions` (validator output; ticket_no is the natural key)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        ticket_no: str,
        briefing_id: int,
        instrument_id: int,
        action: str,
        units: int,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        valid_until: date,
        rationale: str,
        validator_result_json: str | None,
        status: str,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO instructions "
            "(ticket_no, briefing_id, instrument_id, action, units, entry_price, "
            "tp_price, sl_price, valid_until, rationale, validator_result_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticket_no,
                briefing_id,
                instrument_id,
                action,
                units,
                entry_price,
                tp_price,
                sl_price,
                valid_until.isoformat(),
                rationale,
                validator_result_json,
                status,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_ticket_no(self, ticket_no: str) -> InstructionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM instructions WHERE ticket_no = ?", (ticket_no,)
        ).fetchone()
        return self._to_record(row) if row else None

    def count_for_date(self, as_of: date) -> int:
        """Count instructions whose ticket_no belongs to `as_of` (YYYYMMDD-*)."""
        prefix = f"{as_of.strftime('%Y%m%d')}-%"
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM instructions WHERE ticket_no LIKE ?", (prefix,)
        ).fetchone()
        return int(row["n"])

    def next_sequence(self, as_of: date) -> int:
        """Next 1-based ticket_no sequence for `as_of` (idempotent re-runs stay monotonic)."""
        return self.count_for_date(as_of) + 1

    def update_status(self, instruction_id: int, status: str) -> None:
        """Move an instruction to `status` (S7: filled/partial/skipped after `fund record`)."""
        self._conn.execute(
            "UPDATE instructions SET status = ? WHERE id = ?", (status, instruction_id)
        )
        self._conn.commit()

    def list_by_status(self, status: str) -> list[InstructionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM instructions WHERE status = ? ORDER BY ticket_no", (status,)
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_all(self) -> list[InstructionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM instructions ORDER BY ticket_no"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> InstructionRecord:
        return InstructionRecord(
            id=row["id"],
            ticket_no=row["ticket_no"],
            briefing_id=row["briefing_id"],
            instrument_id=row["instrument_id"],
            action=row["action"],
            units=row["units"],
            entry_price=row["entry_price"],
            tp_price=row["tp_price"],
            sl_price=row["sl_price"],
            valid_until=date.fromisoformat(row["valid_until"]),
            rationale=row["rationale"],
            validator_result_json=row["validator_result_json"],
            status=row["status"],
        )


@dataclass(frozen=True, slots=True)
class LlmCallRecord:
    id: int
    ts: datetime
    kind: str
    model: str
    temperature: float
    prompt_version: str
    schema_version: int
    sample_index: int
    briefing_id: int | None
    policy_id: int | None
    criteria_id: int | None
    prompt: str
    response: str | None
    token_usage_json: str | None


class LlmCallRepo:
    """Append-only audit log of every LLM API call (`llm_calls`, technical-spec.md 3, 5章).

    Records prompt/response/model/temperature/prompt_version/sample_index/token usage so
    each judgment is reproducible and benchmark aggregation can be partitioned by
    prompt_version + model + temperature (評価プロトコルの固定).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        kind: str,
        model: str,
        temperature: float,
        prompt_version: str,
        schema_version: int,
        sample_index: int,
        prompt: str,
        response: str | None,
        token_usage_json: str | None,
        briefing_id: int | None = None,
        policy_id: int | None = None,
        criteria_id: int | None = None,
    ) -> int:
        ts = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO llm_calls "
            "(ts, kind, model, temperature, prompt_version, schema_version, sample_index, "
            "policy_id, criteria_id, briefing_id, prompt, response, token_usage_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                kind,
                model,
                temperature,
                prompt_version,
                schema_version,
                sample_index,
                policy_id,
                criteria_id,
                briefing_id,
                prompt,
                response,
                token_usage_json,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_all(self) -> list[LlmCallRecord]:
        rows = self._conn.execute("SELECT * FROM llm_calls ORDER BY id").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> LlmCallRecord:
        return LlmCallRecord(
            id=row["id"],
            ts=datetime.fromisoformat(row["ts"]),
            kind=row["kind"],
            model=row["model"],
            temperature=row["temperature"],
            prompt_version=row["prompt_version"],
            schema_version=row["schema_version"],
            sample_index=row["sample_index"],
            briefing_id=row["briefing_id"],
            policy_id=row["policy_id"],
            criteria_id=row["criteria_id"],
            prompt=row["prompt"],
            response=row["response"],
            token_usage_json=row["token_usage_json"],
        )


@dataclass(frozen=True, slots=True)
class AuditEventRecord:
    id: int
    ts: datetime
    kind: str
    detail_json: str


class AuditEventRepo:
    """Append-only log for validator/judgment outcomes (`audit_events`)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, kind: str, detail_json: str) -> int:
        ts = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO audit_events (ts, kind, detail_json) VALUES (?, ?, ?)",
            (ts, kind, detail_json),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_by_kind(self, kind: str) -> list[AuditEventRecord]:
        rows = self._conn.execute(
            "SELECT * FROM audit_events WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_all(self) -> list[AuditEventRecord]:
        rows = self._conn.execute("SELECT * FROM audit_events ORDER BY id").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> AuditEventRecord:
        return AuditEventRecord(
            id=row["id"],
            ts=datetime.fromisoformat(row["ts"]),
            kind=row["kind"],
            detail_json=row["detail_json"],
        )


# --- S9 review（週次/月次レビューサイクル）------------------------------------
# `policies`/`criteria` は月次方針・週次基準の承認ライフサイクルを保持する
# （technical-spec.md 3章）。両テーブルとも新規提案は status=draft で書かれ、
# `approve()` が承認・旧版の supersede を1トランザクション相当（同一 conn）で行う。
# criteria のみ `superseded_by` を持つため、superseded になった行を再度 approve()
# すればそれが再度 active になり、直前の active 版が supersede される
# （= ロールバック。FR-6「人間承認で有効化、ロールバック可能」）。


@dataclass(frozen=True, slots=True)
class PolicyRecord:
    id: int
    effective_from: date
    content: str
    status: str
    approved_at: datetime | None


class PolicyRepo:
    """月次方針の承認ライフサイクル（`policies`）。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        effective_from: date,
        content: str,
        status: str = ProposalStatus.DRAFT.value,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO policies (effective_from, content, status, approved_at) "
            "VALUES (?, ?, ?, NULL)",
            (effective_from.isoformat(), content, status),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_id(self, policy_id: int) -> PolicyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM policies WHERE id = ?", (policy_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def get_active(self) -> PolicyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM policies WHERE status = ? ORDER BY approved_at DESC LIMIT 1",
            (ProposalStatus.ACTIVE.value,),
        ).fetchone()
        return self._to_record(row) if row else None

    def list_by_status(self, status: str) -> list[PolicyRecord]:
        rows = self._conn.execute(
            "SELECT * FROM policies WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def approve(self, policy_id: int) -> None:
        """`policy_id` を active 化し、既存の active 版を superseded にする。

        ロールバックは既に superseded になった旧版を再度 approve() することで行う
        （承認とロールバックで同じメソッドを共用する）。
        """
        target = self.get_by_id(policy_id)
        if target is None:
            raise ValueError(f"unknown policy id: {policy_id}")

        current_active = self.get_active()
        if current_active is not None and current_active.id != policy_id:
            self._conn.execute(
                "UPDATE policies SET status = ? WHERE id = ?",
                (ProposalStatus.SUPERSEDED.value, current_active.id),
            )
        approved_at = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE policies SET status = ?, approved_at = ? WHERE id = ?",
            (ProposalStatus.ACTIVE.value, approved_at, policy_id),
        )
        self._conn.commit()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> PolicyRecord:
        return PolicyRecord(
            id=row["id"],
            effective_from=date.fromisoformat(row["effective_from"]),
            content=row["content"],
            status=row["status"],
            approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
        )


@dataclass(frozen=True, slots=True)
class CriteriaRecord:
    id: int
    effective_from: date
    content: str
    diff: str | None
    rationale: str | None
    status: str
    approved_at: datetime | None
    superseded_by: int | None


class CriteriaRepo:
    """週次基準の承認ライフサイクル（`criteria`。ロールバック可能な `superseded_by` 付き）。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        effective_from: date,
        content: str,
        diff: str | None,
        rationale: str | None,
        status: str = ProposalStatus.DRAFT.value,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO criteria "
            "(effective_from, content, diff, rationale, status, approved_at, superseded_by) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL)",
            (effective_from.isoformat(), content, diff, rationale, status),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_id(self, criteria_id: int) -> CriteriaRecord | None:
        row = self._conn.execute(
            "SELECT * FROM criteria WHERE id = ?", (criteria_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def get_active(self) -> CriteriaRecord | None:
        row = self._conn.execute(
            "SELECT * FROM criteria WHERE status = ? ORDER BY approved_at DESC LIMIT 1",
            (ProposalStatus.ACTIVE.value,),
        ).fetchone()
        return self._to_record(row) if row else None

    def list_by_status(self, status: str) -> list[CriteriaRecord]:
        rows = self._conn.execute(
            "SELECT * FROM criteria WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def approve(self, criteria_id: int) -> None:
        """`criteria_id` を active 化する（承認/ロールバック共用）。

        既存の active 版があれば `status=superseded` にし `superseded_by=criteria_id`
        を記録する。superseded になった版を改めて approve() すればそれが active に
        戻り、直前の active 版が supersede される（ロールバック）。
        """
        target = self.get_by_id(criteria_id)
        if target is None:
            raise ValueError(f"unknown criteria id: {criteria_id}")

        current_active = self.get_active()
        if current_active is not None and current_active.id != criteria_id:
            self._conn.execute(
                "UPDATE criteria SET status = ?, superseded_by = ? WHERE id = ?",
                (ProposalStatus.SUPERSEDED.value, criteria_id, current_active.id),
            )
        approved_at = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE criteria SET status = ?, approved_at = ?, superseded_by = NULL WHERE id = ?",
            (ProposalStatus.ACTIVE.value, approved_at, criteria_id),
        )
        self._conn.commit()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> CriteriaRecord:
        return CriteriaRecord(
            id=row["id"],
            effective_from=date.fromisoformat(row["effective_from"]),
            content=row["content"],
            diff=row["diff"],
            rationale=row["rationale"],
            status=row["status"],
            approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
            superseded_by=row["superseded_by"],
        )


def price_deviation_pct(actual_price: float, expected_price: float) -> float:
    """指示価格 (expected) からの実行価格 (actual) の乖離率（符号付き%）。"""
    return (actual_price - expected_price) / expected_price * 100


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    id: int
    instruction_id: int
    executed_at: datetime
    side: str
    order_type: str
    actual_price: float
    actual_units: int
    commission: float
    status: str
    skip_reason: str | None
    deviation_note: str | None


@dataclass(frozen=True, slots=True)
class ExecutionDeviation:
    """1件の執行の指示との乖離（`fund status` の可視化用）。skipped は乖離%を持たない。"""

    ticket_no: str
    status: str
    entry_price: float
    actual_price: float
    deviation_pct: float | None


class ExecutionRepo:
    """Write/read for `executions`（FR-4: 人間の執行結果と指示の乖離）。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        instruction_id: int,
        executed_at: datetime,
        side: str,
        order_type: str,
        actual_price: float,
        actual_units: int,
        commission: float,
        status: str,
        skip_reason: str | None = None,
        deviation_note: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO executions "
            "(instruction_id, executed_at, side, order_type, actual_price, actual_units, "
            "commission, status, skip_reason, deviation_note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                instruction_id,
                executed_at.isoformat(),
                side,
                order_type,
                actual_price,
                actual_units,
                commission,
                status,
                skip_reason,
                deviation_note,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_instruction_id(self, instruction_id: int) -> ExecutionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM executions WHERE instruction_id = ? ORDER BY id DESC LIMIT 1",
            (instruction_id,),
        ).fetchone()
        return self._to_record(row) if row else None

    def list_deviations(self) -> list[ExecutionDeviation]:
        """全執行を指示と結合し、価格乖離%を算出する（`status=skipped` は None）。"""
        rows = self._conn.execute(
            "SELECT e.status AS status, e.actual_price AS actual_price, "
            "i.entry_price AS entry_price, i.ticket_no AS ticket_no "
            "FROM executions e JOIN instructions i ON i.id = e.instruction_id "
            "ORDER BY e.id"
        ).fetchall()
        result: list[ExecutionDeviation] = []
        for row in rows:
            is_skipped = row["status"] == ExecutionStatus.SKIPPED.value
            deviation_pct = (
                None
                if is_skipped
                else price_deviation_pct(row["actual_price"], row["entry_price"])
            )
            result.append(
                ExecutionDeviation(
                    ticket_no=row["ticket_no"],
                    status=row["status"],
                    entry_price=row["entry_price"],
                    actual_price=row["actual_price"],
                    deviation_pct=deviation_pct,
                )
            )
        return result

    def unexecuted_rate(self) -> float | None:
        """`skipped` 件数 / 全執行記録件数。記録が1件も無ければ None。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS skipped FROM executions",
            (ExecutionStatus.SKIPPED.value,),
        ).fetchone()
        if row["n"] == 0:
            return None
        return int(row["skipped"]) / int(row["n"])

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            id=row["id"],
            instruction_id=row["instruction_id"],
            executed_at=datetime.fromisoformat(row["executed_at"]),
            side=row["side"],
            order_type=row["order_type"],
            actual_price=row["actual_price"],
            actual_units=row["actual_units"],
            commission=row["commission"],
            status=row["status"],
            skip_reason=row["skip_reason"],
            deviation_note=row["deviation_note"],
        )


# --- S8 tracking（仮想執行・ベンチマーク）---------------------------------------
# これらのリポジトリが書き込む positions / pending_orders / portfolio_state /
# virtual_fills / benchmark_navs は tracking レイヤ（virtual_fill.py / benchmark.py）
# だけが更新する。仮想執行エンジンがポートフォリオ状態の唯一の書き手である
# （technical-spec.md 7章。真実の源泉）。


@dataclass(frozen=True, slots=True)
class VirtualFillRecord:
    id: int
    instruction_id: int
    fill_date: date | None
    fill_price: float | None
    exit_date: date | None
    exit_price: float | None
    exit_reason: str | None
    commission: float
    slippage: float
    pnl: float | None


class VirtualFillRepo:
    """Ledger of the conservative virtual-fill engine (`virtual_fills`).

    Keyed logically by `instruction_id` (each instruction has at most one fill
    record). `upsert` replaces any existing row for the instruction so re-running
    the engine on a later date can extend a fill with its exit without duplicating.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        *,
        instruction_id: int,
        fill_date: date | None,
        fill_price: float | None,
        exit_date: date | None,
        exit_price: float | None,
        exit_reason: str | None,
        commission: float,
        slippage: float,
        pnl: float | None,
    ) -> int:
        self._conn.execute(
            "DELETE FROM virtual_fills WHERE instruction_id = ?", (instruction_id,)
        )
        cur = self._conn.execute(
            "INSERT INTO virtual_fills "
            "(instruction_id, fill_date, fill_price, exit_date, exit_price, exit_reason, "
            "commission, slippage, pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                instruction_id,
                fill_date.isoformat() if fill_date else None,
                fill_price,
                exit_date.isoformat() if exit_date else None,
                exit_price,
                exit_reason,
                commission,
                slippage,
                pnl,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_instruction_id(self, instruction_id: int) -> VirtualFillRecord | None:
        row = self._conn.execute(
            "SELECT * FROM virtual_fills WHERE instruction_id = ?", (instruction_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def list_all(self) -> list[VirtualFillRecord]:
        rows = self._conn.execute(
            "SELECT * FROM virtual_fills ORDER BY id"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> VirtualFillRecord:
        return VirtualFillRecord(
            id=row["id"],
            instruction_id=row["instruction_id"],
            fill_date=date.fromisoformat(row["fill_date"]) if row["fill_date"] else None,
            fill_price=row["fill_price"],
            exit_date=date.fromisoformat(row["exit_date"]) if row["exit_date"] else None,
            exit_price=row["exit_price"],
            exit_reason=row["exit_reason"],
            commission=row["commission"],
            slippage=row["slippage"],
            pnl=row["pnl"],
        )


@dataclass(frozen=True, slots=True)
class PositionRecord:
    id: int
    instrument_id: int
    units: int
    avg_cost: float
    opened_at: date
    closed_at: date | None


class PositionRepo:
    """Open/closed virtual positions (`positions`). Rebuilt from `virtual_fills`.

    The virtual-fill engine owns this table wholesale: each run clears it and
    rebuilds from the current fills, so the table is always a deterministic
    projection of the ledger (idempotent re-runs).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def clear(self) -> None:
        self._conn.execute("DELETE FROM positions")
        self._conn.commit()

    def add(
        self,
        *,
        instrument_id: int,
        units: int,
        avg_cost: float,
        opened_at: date,
        closed_at: date | None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO positions (instrument_id, units, avg_cost, opened_at, closed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                instrument_id,
                units,
                avg_cost,
                opened_at.isoformat(),
                closed_at.isoformat() if closed_at else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_open(self) -> list[PositionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE closed_at IS NULL ORDER BY id"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_all(self) -> list[PositionRecord]:
        rows = self._conn.execute("SELECT * FROM positions ORDER BY id").fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> PositionRecord:
        return PositionRecord(
            id=row["id"],
            instrument_id=row["instrument_id"],
            units=row["units"],
            avg_cost=row["avg_cost"],
            opened_at=date.fromisoformat(row["opened_at"]),
            closed_at=date.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        )


@dataclass(frozen=True, slots=True)
class PendingOrderRecord:
    id: int
    instruction_id: int
    expires_at: date
    status: str


class PendingOrderRepo:
    """Virtual IFO order lifecycle (`pending_orders`). Rebuilt each engine run."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def clear(self) -> None:
        self._conn.execute("DELETE FROM pending_orders")
        self._conn.commit()

    def add(self, *, instruction_id: int, expires_at: date, status: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO pending_orders (instruction_id, expires_at, status) "
            "VALUES (?, ?, ?)",
            (instruction_id, expires_at.isoformat(), status),
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def list_all(self) -> list[PendingOrderRecord]:
        rows = self._conn.execute(
            "SELECT * FROM pending_orders ORDER BY id"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_by_status(self, status: str) -> list[PendingOrderRecord]:
        rows = self._conn.execute(
            "SELECT * FROM pending_orders WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> PendingOrderRecord:
        return PendingOrderRecord(
            id=row["id"],
            instruction_id=row["instruction_id"],
            expires_at=date.fromisoformat(row["expires_at"]),
            status=row["status"],
        )


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    id: int
    code: str
    name: str


class StrategyRepo:
    """Benchmark strategy registry (`strategies`; code is the natural key)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(self, code: str, name: str) -> int:
        row = self._conn.execute(
            "SELECT id FROM strategies WHERE code = ?", (code,)
        ).fetchone()
        if row is not None:
            return int(row["id"])
        cur = self._conn.execute(
            "INSERT INTO strategies (code, name) VALUES (?, ?)", (code, name)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_by_code(self, code: str) -> StrategyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE code = ?", (code,)
        ).fetchone()
        if row is None:
            return None
        return StrategyRecord(id=row["id"], code=row["code"], name=row["name"])


@dataclass(frozen=True, slots=True)
class BenchmarkNavRecord:
    id: int
    strategy_id: int
    nav_date: date
    nav: float
    metrics_json: str | None


class BenchmarkNavRepo:
    """Daily NAV + metrics per strategy (`benchmark_navs`; UNIQUE strategy_id+date)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        *,
        strategy_id: int,
        nav_date: date,
        nav: float,
        metrics_json: str | None,
    ) -> int:
        self._conn.execute(
            "INSERT INTO benchmark_navs (strategy_id, date, nav, metrics_json) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(strategy_id, date) DO UPDATE SET "
            "nav=excluded.nav, metrics_json=excluded.metrics_json",
            (strategy_id, nav_date.isoformat(), nav, metrics_json),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM benchmark_navs WHERE strategy_id = ? AND date = ?",
            (strategy_id, nav_date.isoformat()),
        ).fetchone()
        return int(row["id"])

    def list_by_strategy(self, strategy_id: int) -> list[BenchmarkNavRecord]:
        rows = self._conn.execute(
            "SELECT * FROM benchmark_navs WHERE strategy_id = ? ORDER BY date",
            (strategy_id,),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def latest(self, strategy_id: int) -> BenchmarkNavRecord | None:
        row = self._conn.execute(
            "SELECT * FROM benchmark_navs WHERE strategy_id = ? ORDER BY date DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: sqlite3.Row) -> BenchmarkNavRecord:
        return BenchmarkNavRecord(
            id=row["id"],
            strategy_id=row["strategy_id"],
            nav_date=date.fromisoformat(row["date"]),
            nav=row["nav"],
            metrics_json=row["metrics_json"],
        )

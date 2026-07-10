"""Unit tests for `review/monthly.py` (S9: FR-6 月次レビュー)。

Mirrors test_weekly.py's mocking style (deterministic `FakeBackend`). Covers:
policy proposal persists a draft `policies` row + audit event, universe-change
suggestions are recorded to `audit_events` (not a DB-backed approvable table),
and no_change persists nothing.
"""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import ProposalStatus
from llm_fund.judgment.client import LlmConfig
from llm_fund.review.monthly import (
    AUDIT_KIND_POLICY_PROPOSED,
    AUDIT_KIND_UNIVERSE_CHANGE_PROPOSED,
    run_monthly_review,
)
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import AuditEventRepo, LlmCallRepo, PolicyRepo
from tests.fakes import FAKE_MODEL_LABEL, FakeBackend

PROMPT_VERSION = "test.1"


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _config() -> LlmConfig:
    return LlmConfig(model=FAKE_MODEL_LABEL, temperature=0.2, max_tokens=1024, n_samples=1)


class TestRunMonthlyReview:
    def test_no_change_persists_nothing(self, conn: sqlite3.Connection) -> None:
        fake = FakeBackend(
            [{"schema_version": 1, "no_change": True, "rationale": "対照群との有意差なし"}],
            repeat_last=True,
        )

        result = run_monthly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 1),
            prompt_version=PROMPT_VERSION,
            performance_summary="（比較可能なベンチマークデータがありません）",
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is True
        assert result.policy_id is None
        assert PolicyRepo(conn).list_by_status(ProposalStatus.DRAFT.value) == []

    def test_change_proposal_persists_draft_policy_and_audit_events(
        self, conn: sqlite3.Connection
    ) -> None:
        fake = FakeBackend(
            [
                {
                    "schema_version": 1,
                    "no_change": False,
                    "new_policy": "TOPIX比10%超過を目標。ETF比率を引き上げる",
                    "diff": "個別株比率60%->40%、ETF比率40%->60%",
                    "rationale": "個別株が対照群に対し継続劣後",
                    "universe_changes": [
                        {
                            "universe_code": "jp_stocks",
                            "symbol": "1306.T",
                            "action": "add",
                            "reason": "ETF比率引き上げのため",
                        }
                    ],
                }
            ],
            repeat_last=True,
        )

        result = run_monthly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 1),
            prompt_version=PROMPT_VERSION,
            performance_summary="fund NAV=980,000 (index NAV=1,020,000)",
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is False
        assert result.policy_id is not None
        draft = PolicyRepo(conn).get_by_id(result.policy_id)
        assert draft is not None
        assert draft.status == ProposalStatus.DRAFT.value

        assert len(AuditEventRepo(conn).list_by_kind(AUDIT_KIND_POLICY_PROPOSED)) == 1
        universe_events = AuditEventRepo(conn).list_by_kind(AUDIT_KIND_UNIVERSE_CHANGE_PROPOSED)
        assert len(universe_events) == 1
        assert "1306.T" in universe_events[0].detail_json

    def test_invalid_schema_falls_back_to_no_change(self, conn: sqlite3.Connection) -> None:
        fake = FakeBackend(
            [{"schema_version": 1, "no_change": True, "new_policy": "x"}], repeat_last=True
        )

        result = run_monthly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 1),
            prompt_version=PROMPT_VERSION,
            performance_summary="n/a",
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is True
        assert fake.call_count == 2

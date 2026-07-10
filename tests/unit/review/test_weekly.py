"""Unit tests for `review/weekly.py` (S9: FR-6 週次レビュー)。

The LLM backend is a deterministic `FakeBackend` (no subprocess / HTTP). Covers:
proposal generation persists a draft `criteria` row + audit event, no_change
short-circuits persistence, and schema-validation failure falls back to
no_change without raising.
"""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import ProposalStatus
from llm_fund.judgment.client import LlmConfig
from llm_fund.review.weekly import (
    AUDIT_KIND_CRITERIA_PROPOSED,
    collect_weekly_performance,
    run_weekly_review,
)
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import AuditEventRepo, CriteriaRepo, LlmCallRepo
from tests.fakes import FAKE_MODEL_LABEL, FakeBackend

PROMPT_VERSION = "test.1"


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


def _config() -> LlmConfig:
    return LlmConfig(model=FAKE_MODEL_LABEL, temperature=0.2, max_tokens=1024, n_samples=1)


class TestRunWeeklyReview:
    def test_no_change_response_persists_nothing(self, conn: sqlite3.Connection) -> None:
        fake = FakeBackend(
            [
                {
                    "schema_version": 1,
                    "no_change": True,
                    "rationale": "十分な件数がなく判断材料が乏しい",
                }
            ],
            repeat_last=True,
        )

        result = run_weekly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 4),
            prompt_version=PROMPT_VERSION,
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is True
        assert result.criteria_id is None
        assert CriteriaRepo(conn).list_by_status(ProposalStatus.DRAFT.value) == []
        assert fake.call_count == 1
        assert len(LlmCallRepo(conn).list_all()) == 1

    def test_change_proposal_persists_draft_criteria_and_audit_event(
        self, conn: sqlite3.Connection
    ) -> None:
        fake = FakeBackend(
            [
                {
                    "schema_version": 1,
                    "no_change": False,
                    "new_criteria": "TP幅3.5%, SL幅2%",
                    "diff": "SL幅を1.5%から2%へ拡大",
                    "rationale": "直近1週間の勝率が低く損切りが浅すぎる",
                }
            ],
            repeat_last=True,
        )

        result = run_weekly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 4),
            prompt_version=PROMPT_VERSION,
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is False
        assert result.criteria_id is not None
        draft = CriteriaRepo(conn).get_by_id(result.criteria_id)
        assert draft is not None
        assert draft.status == ProposalStatus.DRAFT.value
        assert draft.diff == "SL幅を1.5%から2%へ拡大"

        audit = AuditEventRepo(conn).list_by_kind(AUDIT_KIND_CRITERIA_PROPOSED)
        assert len(audit) == 1

    def test_invalid_schema_falls_back_to_no_change_after_one_retry(
        self, conn: sqlite3.Connection
    ) -> None:
        # Missing required `rationale` -> pydantic validation fails on both attempts.
        fake = FakeBackend([{"schema_version": 1, "no_change": True}], repeat_last=True)

        result = run_weekly_review(
            conn,
            fake,
            _config(),
            as_of=date(2026, 7, 4),
            prompt_version=PROMPT_VERSION,
            llm_call_sink=LlmCallRepo(conn),
            audit_sink=AuditEventRepo(conn),
        )

        assert result.no_change is True
        assert fake.call_count == 2  # original + one correction retry
        assert CriteriaRepo(conn).list_by_status(ProposalStatus.DRAFT.value) == []


class TestCollectWeeklyPerformance:
    def test_empty_db_reports_no_data(self, conn: sqlite3.Connection) -> None:
        summary = collect_weekly_performance(conn, date(2026, 7, 4))
        assert summary.n_instructions == 0
        assert summary.win_rate is None
        assert summary.unexecuted_rate is None

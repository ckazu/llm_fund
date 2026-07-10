"""CRUD tests for LlmCallRepo (llm_calls audit log, technical-spec.md 3, 5章)."""

import sqlite3
from pathlib import Path

import pytest

from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import LlmCallRepo


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


class TestLlmCallRepo:
    def test_add_and_list_all(self, conn: sqlite3.Connection) -> None:
        repo = LlmCallRepo(conn)
        repo.add(
            kind="daily_judgment",
            model="claude-sonnet-5",
            temperature=1.0,
            prompt_version="2026-07-04.1",
            schema_version=1,
            sample_index=1,
            prompt="<briefing>...</briefing>",
            response='{"schema_version": 1}',
            token_usage_json='{"input_tokens": 10, "output_tokens": 5}',
        )
        records = repo.list_all()
        assert len(records) == 1
        record = records[0]
        assert record.model == "claude-sonnet-5"
        assert record.sample_index == 1
        assert record.schema_version == 1
        assert record.briefing_id is None
        assert record.token_usage_json == '{"input_tokens": 10, "output_tokens": 5}'

    def test_records_are_ordered_and_keep_sample_index(self, conn: sqlite3.Connection) -> None:
        repo = LlmCallRepo(conn)
        for index in (1, 2, 3):
            repo.add(
                kind="daily_judgment",
                model="m",
                temperature=1.0,
                prompt_version="v",
                schema_version=1,
                sample_index=index,
                prompt="p",
                response=None,
                token_usage_json=None,
            )
        assert [r.sample_index for r in repo.list_all()] == [1, 2, 3]

    def test_nullable_response_persisted(self, conn: sqlite3.Connection) -> None:
        repo = LlmCallRepo(conn)
        repo.add(
            kind="daily_judgment",
            model="m",
            temperature=0.5,
            prompt_version="v",
            schema_version=1,
            sample_index=1,
            prompt="p",
            response=None,
            token_usage_json=None,
        )
        record = repo.list_all()[0]
        assert record.response is None
        assert record.temperature == 0.5

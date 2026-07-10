"""Approval lifecycle tests for `PolicyRepo`/`CriteriaRepo` (S9: FR-6 レビューサイクル)。"""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_fund.domain.enums import ProposalStatus
from llm_fund.store.db import apply_migrations, connect
from llm_fund.store.repos import CriteriaRepo, PolicyRepo


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    apply_migrations(c)
    return c


class TestCriteriaRepo:
    def test_add_defaults_to_draft(self, conn: sqlite3.Connection) -> None:
        repo = CriteriaRepo(conn)
        cid = repo.add(
            effective_from=date(2026, 7, 4),
            content="TP幅3%, SL幅2%",
            diff="SL幅を1.5%から2%へ拡大",
            rationale="直近の勝率低下",
        )
        record = repo.get_by_id(cid)
        assert record is not None
        assert record.status == ProposalStatus.DRAFT.value
        assert record.approved_at is None
        assert record.superseded_by is None
        assert repo.get_active() is None

    def test_approve_activates_and_has_no_predecessor_to_supersede(
        self, conn: sqlite3.Connection
    ) -> None:
        repo = CriteriaRepo(conn)
        cid = repo.add(
            effective_from=date(2026, 7, 4), content="v1", diff="d1", rationale="r1"
        )
        repo.approve(cid)

        active = repo.get_active()
        assert active is not None
        assert active.id == cid
        assert active.status == ProposalStatus.ACTIVE.value
        assert active.approved_at is not None

    def test_approving_new_proposal_supersedes_previous_active(
        self, conn: sqlite3.Connection
    ) -> None:
        repo = CriteriaRepo(conn)
        old_id = repo.add(effective_from=date(2026, 7, 4), content="v1", diff="d1", rationale="r1")
        repo.approve(old_id)

        new_id = repo.add(effective_from=date(2026, 7, 11), content="v2", diff="d2", rationale="r2")
        repo.approve(new_id)

        old = repo.get_by_id(old_id)
        new = repo.get_by_id(new_id)
        assert old is not None and new is not None
        assert old.status == ProposalStatus.SUPERSEDED.value
        assert old.superseded_by == new_id
        assert new.status == ProposalStatus.ACTIVE.value
        active = repo.get_active()
        assert active is not None
        assert active.id == new_id

    def test_reapproving_superseded_version_rolls_back(self, conn: sqlite3.Connection) -> None:
        repo = CriteriaRepo(conn)
        old_id = repo.add(effective_from=date(2026, 7, 4), content="v1", diff="d1", rationale="r1")
        repo.approve(old_id)
        new_id = repo.add(effective_from=date(2026, 7, 11), content="v2", diff="d2", rationale="r2")
        repo.approve(new_id)

        # Rollback: re-approve the superseded old version.
        repo.approve(old_id)

        old = repo.get_by_id(old_id)
        new = repo.get_by_id(new_id)
        assert old is not None and new is not None
        assert old.status == ProposalStatus.ACTIVE.value
        assert old.superseded_by is None
        assert new.status == ProposalStatus.SUPERSEDED.value
        assert new.superseded_by == old_id
        active = repo.get_active()
        assert active is not None
        assert active.id == old_id

    def test_approve_unknown_id_raises(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError):
            CriteriaRepo(conn).approve(999)


class TestPolicyRepo:
    def test_add_defaults_to_draft(self, conn: sqlite3.Connection) -> None:
        repo = PolicyRepo(conn)
        pid = repo.add(effective_from=date(2026, 7, 1), content="現物ロングオンリー継続")
        record = repo.get_by_id(pid)
        assert record is not None
        assert record.status == ProposalStatus.DRAFT.value
        assert repo.get_active() is None

    def test_approve_supersedes_previous_active(self, conn: sqlite3.Connection) -> None:
        repo = PolicyRepo(conn)
        old_id = repo.add(effective_from=date(2026, 7, 1), content="policy v1")
        repo.approve(old_id)
        new_id = repo.add(effective_from=date(2026, 8, 1), content="policy v2")
        repo.approve(new_id)

        old = repo.get_by_id(old_id)
        new = repo.get_by_id(new_id)
        assert old is not None and new is not None
        assert old.status == ProposalStatus.SUPERSEDED.value
        assert new.status == ProposalStatus.ACTIVE.value
        active = repo.get_active()
        assert active is not None
        assert active.id == new_id

    def test_reapproving_superseded_policy_rolls_back(self, conn: sqlite3.Connection) -> None:
        repo = PolicyRepo(conn)
        old_id = repo.add(effective_from=date(2026, 7, 1), content="policy v1")
        repo.approve(old_id)
        new_id = repo.add(effective_from=date(2026, 8, 1), content="policy v2")
        repo.approve(new_id)

        repo.approve(old_id)

        active = repo.get_active()
        assert active is not None
        assert active.id == old_id
        new = repo.get_by_id(new_id)
        assert new is not None
        assert new.status == ProposalStatus.SUPERSEDED.value

    def test_approve_unknown_id_raises(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError):
            PolicyRepo(conn).approve(999)

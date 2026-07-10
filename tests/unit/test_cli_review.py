"""CLI tests for `fund weekly` / `fund monthly` / `fund approve` (S9: FR-6).

Covers the end-to-end state transition with the LLM backend fully mocked
(`FakeBackend` + a patched router; no subprocess / HTTP): proposal generation ->
`fund approve` activates it -> a second proposal supersedes the first ->
re-approving the first rolls back (its `superseded_by` link flips).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from llm_fund.cli import app
from llm_fund.domain.enums import ProposalStatus
from llm_fund.store.db import init_db
from llm_fund.store.repos import CriteriaRepo, PolicyRepo
from tests.factories import build_llm_config_dict
from tests.fakes import FakeBackend, FakeRouter

runner = CliRunner()


def _write_config(tmp_path: Path, *, claude_command: str = "claude") -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "llm": build_llm_config_dict(command=claude_command),
                "limits": {
                    "max_position_pct": 15.0,
                    "max_turnover_pct": 30.0,
                    "max_instructions_per_day": 5,
                    "require_stop_loss": True,
                },
                "benchmark": {"index_symbol": "1306.T", "momentum_lookback_days": 120},
                "report": {"output_dir": "reports"},
                "db_path": "test.db",
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "universes.yaml").write_text(
        yaml.safe_dump({"universes": {}}), encoding="utf-8"
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    return tmp_path


def _patch_backend(
    monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any]]
) -> FakeBackend:
    """Route every role to a FakeBackend returning `payloads` in order."""
    fake = FakeBackend(payloads, repeat_last=True)
    monkeypatch.setattr("llm_fund.cli.build_router", lambda settings: FakeRouter(fake))
    return fake


def _criteria_payload(diff: str, rationale: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "no_change": False,
        "new_criteria": f"基準: {diff}",
        "diff": diff,
        "rationale": rationale,
    }


class TestWeeklyProposeApproveRollback:
    def test_full_lifecycle(self, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _patch_backend(monkeypatch, [_criteria_payload("SL幅拡大", "勝率低下")])

        result = runner.invoke(app, ["weekly"])
        assert result.exit_code == 0, result.output
        assert "基準変更を提案しました" in result.output

        conn = init_db(str(project / "test.db"))
        drafts = CriteriaRepo(conn).list_by_status(ProposalStatus.DRAFT.value)
        assert len(drafts) == 1
        old_id = drafts[0].id

        approve_result = runner.invoke(app, ["approve", f"criteria:{old_id}"])
        assert approve_result.exit_code == 0, approve_result.output

        conn2 = init_db(str(project / "test.db"))
        active = CriteriaRepo(conn2).get_active()
        assert active is not None
        assert active.id == old_id

        # A second weekly run proposes and approves a new version, superseding the first.
        fake._script[:] = [_criteria_payload("SL幅さらに拡大", "継続的な損切り過小")]
        result2 = runner.invoke(app, ["weekly"])
        assert result2.exit_code == 0, result2.output

        conn3 = init_db(str(project / "test.db"))
        new_drafts = [
            c
            for c in CriteriaRepo(conn3).list_by_status(ProposalStatus.DRAFT.value)
            if c.id != old_id
        ]
        assert len(new_drafts) == 1
        new_id = new_drafts[0].id

        runner.invoke(app, ["approve", f"criteria:{new_id}"])
        conn4 = init_db(str(project / "test.db"))
        old_after = CriteriaRepo(conn4).get_by_id(old_id)
        assert old_after is not None
        assert old_after.status == ProposalStatus.SUPERSEDED.value
        assert old_after.superseded_by == new_id

        # Rollback: re-approve the old (now superseded) version.
        rollback_result = runner.invoke(app, ["approve", f"criteria:{old_id}"])
        assert rollback_result.exit_code == 0, rollback_result.output

        conn5 = init_db(str(project / "test.db"))
        active_after_rollback = CriteriaRepo(conn5).get_active()
        assert active_after_rollback is not None
        assert active_after_rollback.id == old_id
        new_after = CriteriaRepo(conn5).get_by_id(new_id)
        assert new_after is not None
        assert new_after.status == ProposalStatus.SUPERSEDED.value
        assert new_after.superseded_by == old_id


class TestApproveErrors:
    def test_unknown_kind_exits_1(self, project: Path) -> None:
        result = runner.invoke(app, ["approve", "bogus:1"])
        assert result.exit_code == 1

    def test_malformed_id_exits_1(self, project: Path) -> None:
        result = runner.invoke(app, ["approve", "criteria:abc"])
        assert result.exit_code == 1

    def test_unknown_criteria_id_exits_1(self, project: Path) -> None:
        result = runner.invoke(app, ["approve", "criteria:999"])
        assert result.exit_code == 1

    def test_missing_claude_command_exits_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, claude_command="no-such-claude-cmd")
        result = runner.invoke(app, ["weekly"])
        assert result.exit_code == 3


class TestMonthlyProposeApprove:
    def test_policy_proposal_activates(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_backend(
            monkeypatch,
            [
                {
                    "schema_version": 1,
                    "no_change": False,
                    "new_policy": "ETF比率を引き上げる",
                    "diff": "個別株60%->40%",
                    "rationale": "対照群に継続劣後",
                    "universe_changes": [],
                }
            ],
        )

        result = runner.invoke(app, ["monthly"])
        assert result.exit_code == 0, result.output
        assert "方針変更を提案しました" in result.output

        conn = init_db(str(project / "test.db"))
        drafts = PolicyRepo(conn).list_by_status(ProposalStatus.DRAFT.value)
        assert len(drafts) == 1
        pid = drafts[0].id

        approve_result = runner.invoke(app, ["approve", f"policy:{pid}"])
        assert approve_result.exit_code == 0, approve_result.output

        conn2 = init_db(str(project / "test.db"))
        active = PolicyRepo(conn2).get_active()
        assert active is not None
        assert active.id == pid

    def test_no_change_reports_and_persists_nothing(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_backend(
            monkeypatch,
            [{"schema_version": 1, "no_change": True, "rationale": "有意な劣後なし"}],
        )

        result = runner.invoke(app, ["monthly"])
        assert result.exit_code == 0, result.output
        assert "方針変更の提案: なし" in result.output

        conn = init_db(str(project / "test.db"))
        assert PolicyRepo(conn).list_by_status(ProposalStatus.DRAFT.value) == []

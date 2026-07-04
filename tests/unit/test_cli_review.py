"""CLI tests for `fund weekly` / `fund monthly` / `fund approve` (S9: FR-6).

Covers the end-to-end state transition with anthropic fully mocked: proposal
generation -> `fund approve` activates it -> a second proposal supersedes the
first -> re-approving the first rolls back (its `superseded_by` link flips).
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

runner = CliRunner()


def _write_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "llm": {"model": "claude-sonnet-5"},
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _write_config(tmp_path)
    return tmp_path


class _Msg:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [type("B", (), {"type": "tool_use", "input": payload})()]
        self.usage = type("U", (), {"input_tokens": 5, "output_tokens": 5})()


class _FakeAnthropic:
    """Returns each payload in `payloads` in order (one per `.messages.create()` call)."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.call_count = 0

        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _Msg:
                payload = outer._payloads[min(outer.call_count, len(outer._payloads) - 1)]
                outer.call_count += 1
                return _Msg(payload)

        self.messages = _Messages()


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
        fake = _FakeAnthropic([_criteria_payload("SL幅拡大", "勝率低下")])
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

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
        fake.call_count = 0
        fake._payloads = [_criteria_payload("SL幅さらに拡大", "継続的な損切り過小")]
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

    def test_missing_api_key_exits_config_error(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["weekly"])
        assert result.exit_code == 3


class TestMonthlyProposeApprove:
    def test_policy_proposal_activates(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeAnthropic(
            [
                {
                    "schema_version": 1,
                    "no_change": False,
                    "new_policy": "ETF比率を引き上げる",
                    "diff": "個別株60%->40%",
                    "rationale": "対照群に継続劣後",
                    "universe_changes": [],
                }
            ]
        )
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

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
        fake = _FakeAnthropic(
            [{"schema_version": 1, "no_change": True, "rationale": "有意な劣後なし"}]
        )
        monkeypatch.setattr("llm_fund.cli.anthropic.Anthropic", lambda **kwargs: fake)

        result = runner.invoke(app, ["monthly"])
        assert result.exit_code == 0, result.output
        assert "方針変更の提案: なし" in result.output

        conn = init_db(str(project / "test.db"))
        assert PolicyRepo(conn).list_by_status(ProposalStatus.DRAFT.value) == []

"""Tests for the role -> backend/model router (llm/router.py)."""

import pytest

from llm_fund.config import (
    AppSettings,
    BenchmarkSettings,
    LimitsSettings,
    LlmBackendSettings,
    LlmRoleSettings,
    LLMSettings,
    ReportSettings,
)
from llm_fund.llm.claude_cli import ClaudeCliBackend
from llm_fund.llm.openai_compat import OpenAiCompatBackend
from llm_fund.llm.router import (
    ROLE_JUDGMENT,
    ROLE_WEEKLY_REVIEW,
    LlmRouter,
    LlmRouterError,
    RoleRoute,
    build_router,
)


def _settings(*, local_llm_api_key: str | None = None) -> AppSettings:
    """役割別に別バックエンド/別モデルを割り当てた最小の AppSettings。"""
    return AppSettings(
        local_llm_api_key=local_llm_api_key,
        llm=LLMSettings(
            roles={
                "judgment": LlmRoleSettings(backend="claude_cli", model="sonnet"),
                "weekly_review": LlmRoleSettings(backend="local", model="qwen3-8b"),
            },
            backends={
                "claude_cli": LlmBackendSettings(command="claude", timeout_seconds=120),
                "local": LlmBackendSettings(
                    base_url="http://127.0.0.1:8080/v1", timeout_seconds=60
                ),
            },
        ),
        limits=LimitsSettings(
            max_position_pct=15.0, max_turnover_pct=30.0, max_instructions_per_day=5
        ),
        benchmark=BenchmarkSettings(index_symbol="1306.T", momentum_lookback_days=120),
        report=ReportSettings(output_dir="reports"),
        universes={},
    )


class TestBuildRouter:
    def test_claude_cli_role_resolves_to_cli_backend(self) -> None:
        backend, model = build_router(_settings()).for_role(ROLE_JUDGMENT)
        assert isinstance(backend, ClaudeCliBackend)
        assert model == "sonnet"
        assert backend.model == "sonnet"
        assert backend.command == "claude"
        assert backend.timeout_seconds == 120
        assert backend.name == "claude_cli"

    def test_local_role_resolves_to_openai_compat_backend(self) -> None:
        backend, model = build_router(
            _settings(local_llm_api_key="secret")
        ).for_role(ROLE_WEEKLY_REVIEW)
        assert isinstance(backend, OpenAiCompatBackend)
        assert model == "qwen3-8b"
        assert backend.base_url == "http://127.0.0.1:8080/v1"
        assert backend.timeout_seconds == 60
        assert backend.api_key == "secret"
        assert backend.name == "local"

    def test_roles_can_use_different_backends_and_models(self) -> None:
        # 用途別のバックエンド/モデル使い分けがこの設計のポイント。
        router = build_router(_settings())
        judgment_backend, judgment_model = router.for_role(ROLE_JUDGMENT)
        weekly_backend, weekly_model = router.for_role(ROLE_WEEKLY_REVIEW)
        assert type(judgment_backend) is not type(weekly_backend)
        assert judgment_model != weekly_model

    def test_label_for_is_backend_colon_model(self) -> None:
        router = build_router(_settings())
        assert router.label_for(ROLE_JUDGMENT) == "claude_cli:sonnet"
        assert router.label_for(ROLE_WEEKLY_REVIEW) == "local:qwen3-8b"


class TestRouterErrors:
    def test_undefined_role_raises(self) -> None:
        router = build_router(_settings())
        with pytest.raises(LlmRouterError, match="未定義の role"):
            router.for_role("monthly_review")

    def test_undefined_backend_raises(self) -> None:
        router = LlmRouter(
            roles={"judgment": RoleRoute(backend="missing", model="m")}, factories={}
        )
        with pytest.raises(LlmRouterError, match="未定義の backend"):
            router.for_role("judgment")

    def test_label_for_undefined_role_raises(self) -> None:
        router = build_router(_settings())
        with pytest.raises(LlmRouterError):
            router.label_for("nope")

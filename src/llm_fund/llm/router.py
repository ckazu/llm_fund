"""用途（role）→ バックエンド/モデルのルーティング（technical-spec.md 9章）。

config の `llm.roles`（role 名 → backend 名 + model）と `llm.backends`（backend 名 →
接続設定）から、role ごとの `LlmBackend` 実装とモデル名を解決する。日次判断と
週次/月次レビューで別バックエンド・別モデルを使い分けられるのがポイント。

未定義 role / 未定義 backend の解決は `LlmRouterError`（設定エラー）とし、
cli.py が終了コード 3（設定異常）へ変換する。roles が参照する backend 名の存在は
config 読込時にも検証される（`config.LLMSettings`）ため、通常はここまで届かない。
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_fund.llm.backend import LlmBackend
from llm_fund.llm.claude_cli import DEFAULT_CLAUDE_COMMAND, ClaudeCliBackend
from llm_fund.llm.openai_compat import OpenAiCompatBackend

if TYPE_CHECKING:
    from llm_fund.config import AppSettings

# llm.roles で使う role 名（呼び出し箇所と config を結ぶ公開名）。
ROLE_JUDGMENT = "judgment"
ROLE_WEEKLY_REVIEW = "weekly_review"
ROLE_MONTHLY_REVIEW = "monthly_review"

# llm_calls.model に記録する "backend:model" ラベルの区切り文字。
_LABEL_SEPARATOR = ":"

# model 名 → 構築済みバックエンド（backend 名ごとに接続設定を束縛したファクトリ）。
BackendFactory = Callable[[str], LlmBackend]


class LlmRouterError(Exception):
    """role / backend の解決失敗（設定エラー。cli では終了コード 3 に写像）。"""


@dataclass(frozen=True, slots=True)
class RoleRoute:
    """1 role 分のルーティング（backend 名 + モデル名）。"""

    backend: str
    model: str


class LlmRouter:
    """role 名から `(LlmBackend, model)` を解決するルーター。"""

    def __init__(
        self, roles: Mapping[str, RoleRoute], factories: Mapping[str, BackendFactory]
    ) -> None:
        self._roles = dict(roles)
        self._factories = dict(factories)

    def _route(self, role: str) -> RoleRoute:
        route = self._roles.get(role)
        if route is None:
            raise LlmRouterError(
                f"未定義の role: {role!r}（llm.roles に定義してください。"
                f"定義済み: {sorted(self._roles)}）"
            )
        if route.backend not in self._factories:
            raise LlmRouterError(
                f"role {role!r} が未定義の backend {route.backend!r} を参照している"
                f"（定義済み: {sorted(self._factories)}）"
            )
        return route

    def for_role(self, role: str) -> tuple[LlmBackend, str]:
        """role のバックエンド実装とモデル名を返す。未定義は `LlmRouterError`。"""
        route = self._route(role)
        return self._factories[route.backend](route.model), route.model

    def label_for(self, role: str) -> str:
        """llm_calls.model に記録する "backend:model" ラベル（例 "claude_cli:sonnet"）。"""
        route = self._route(role)
        return f"{route.backend}{_LABEL_SEPARATOR}{route.model}"


def _claude_cli_factory(
    name: str, command: str, timeout_seconds: int
) -> BackendFactory:
    def factory(model: str) -> LlmBackend:
        return ClaudeCliBackend(
            model=model, command=command, timeout_seconds=timeout_seconds, name=name
        )

    return factory


def _openai_compat_factory(
    name: str, base_url: str, timeout_seconds: int, api_key: str | None
) -> BackendFactory:
    def factory(model: str) -> LlmBackend:
        return OpenAiCompatBackend(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            name=name,
        )

    return factory


def build_router(settings: "AppSettings") -> LlmRouter:
    """検証済み設定から `LlmRouter` を構築する。

    backend 種別は接続設定から推定する（`base_url` があれば OpenAI 互換、
    無ければ claude CLI。排他性は `config.LlmBackendSettings` が検証済み）。
    OpenAI 互換バックエンドには .env の `LOCAL_LLM_API_KEY`（任意）を渡す。
    """
    factories: dict[str, BackendFactory] = {}
    for name, backend_conf in settings.llm.backends.items():
        if backend_conf.base_url is not None:
            factories[name] = _openai_compat_factory(
                name,
                backend_conf.base_url,
                backend_conf.timeout_seconds,
                settings.local_llm_api_key,
            )
        else:
            factories[name] = _claude_cli_factory(
                name,
                backend_conf.command or DEFAULT_CLAUDE_COMMAND,
                backend_conf.timeout_seconds,
            )
    roles = {
        role: RoleRoute(backend=conf.backend, model=conf.model)
        for role, conf in settings.llm.roles.items()
    }
    return LlmRouter(roles, factories)

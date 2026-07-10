"""LLM バックエンドの共通契約（Protocol・応答型・例外）。

バックエンド実装（claude_cli / openai_compat）はこのモジュールにのみ依存し、
呼び出し側（judgment / review / cli）は `LlmBackend` Protocol 越しにのみ実装を扱う。
これにより API 課金型 SDK への依存を持たず、設定でバックエンドを差し替えられる。
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


class LlmBackendError(Exception):
    """バックエンド障害（プロセス失敗・タイムアウト・HTTP エラー・不正応答）。

    呼び出し側はこの例外だけを扱えばよく、subprocess / httpx の例外型には触れない。
    """


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """1回の補完呼び出しの結果。

    `backend` は設定上のバックエンド名（例 "claude_cli", "local"）、`model` は
    実際に使ったモデル名。`usage` はバックエンドが返すトークン使用量（形式は
    バックエンド依存。無ければ空 dict）。
    """

    text: str
    model: str
    backend: str
    usage: dict[str, Any] = field(default_factory=dict)


class LlmBackend(Protocol):
    """テキスト補完バックエンドの契約。

    実装は `system`（システムプロンプト）と `prompt`（ユーザープロンプト）から
    1回の補完を行う。障害は全て `LlmBackendError` に変換して送出する。
    パラメータをネイティブに指定できないバックエンド（claude CLI）は
    `max_tokens` / `temperature` を無視してよい（実装側で明記する）。
    """

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, temperature: float
    ) -> LlmResponse: ...

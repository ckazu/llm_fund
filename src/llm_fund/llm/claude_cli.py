"""Claude Code CLI（`claude -p`）バックエンド。

サブスクリプション認証済みの `claude` コマンドをサブプロセスとして実行する
（API 課金の anthropic SDK は使用しない）。実フラグは `claude --help` で確認済み:

    claude -p --model <model> --output-format json --append-system-prompt <system>

プロンプト本文は引数ではなく stdin で渡す（ブリーフィング Markdown を含む長文で
ARG_MAX を超えるリスクを避けるため。`-p` は stdin をプロンプトとして読む）。
`--output-format json` の出力は `result`（応答テキスト）と `usage`（トークン使用量）を
含む JSON オブジェクトで、これを `LlmResponse` に変換する。

claude CLI は max_tokens / temperature をフラグとして受け付けないため、この
バックエンドは両パラメータを無視する（評価プロトコル上は llm_calls に記録される
設定値がそのまま残るが、実挙動は CLI 側の既定に従う）。
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from llm_fund.llm.backend import LlmBackendError, LlmResponse

# config/default.yaml `llm.backends.claude_cli` の既定値。
DEFAULT_CLAUDE_COMMAND = "claude"
DEFAULT_TIMEOUT_SECONDS = 300

# 設定でバックエンド名を与えない場合の既定名（LlmResponse.backend / llm_calls 記録用）。
DEFAULT_BACKEND_NAME = "claude_cli"


@dataclass(frozen=True, slots=True)
class ClaudeCliBackend:
    """`claude -p` をサブプロセス実行する `LlmBackend` 実装。"""

    model: str
    command: str = DEFAULT_CLAUDE_COMMAND
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    name: str = DEFAULT_BACKEND_NAME

    def is_available(self) -> bool:
        """`command` が PATH 上に存在するか（cli.py が起動前チェックに使う）。"""
        return shutil.which(self.command) is not None

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, temperature: float
    ) -> LlmResponse:
        """1回の補完。非ゼロ終了・タイムアウト・JSON 不正は `LlmBackendError`。

        `max_tokens` / `temperature` は claude CLI が受け付けないため無視する
        （モジュール docstring 参照）。
        """
        args = [
            self.command,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--append-system-prompt",
            system,
        ]
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise LlmBackendError(f"claude コマンドが見つからない: {self.command}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LlmBackendError(
                f"claude CLI がタイムアウト（{self.timeout_seconds}秒）"
            ) from exc

        if proc.returncode != 0:
            raise LlmBackendError(
                f"claude CLI が非ゼロ終了（code={proc.returncode}）: {proc.stderr.strip()}"
            )
        return self._parse_output(proc.stdout)

    def _parse_output(self, stdout: str) -> LlmResponse:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmBackendError(f"claude CLI の出力が JSON として不正: {exc}") from exc
        # CLI バージョンにより単一オブジェクトまたはイベント配列（type=result 要素が最終結果）が返る
        if isinstance(payload, list):
            results = [e for e in payload if isinstance(e, dict) and e.get("type") == "result"]
            if not results:
                raise LlmBackendError("claude CLI のイベント配列に type=result 要素が無い")
            payload = results[-1]
        if not isinstance(payload, dict):
            raise LlmBackendError("claude CLI の出力が JSON オブジェクトではない")
        if payload.get("is_error"):
            raise LlmBackendError(f"claude CLI がエラー応答を返した: {payload.get('result')}")

        text = payload.get("result")
        if not isinstance(text, str):
            raise LlmBackendError("claude CLI の出力に result（文字列）が無い")

        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return LlmResponse(text=text, model=self.model, backend=self.name, usage=usage)

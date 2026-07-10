"""OpenAI 互換 HTTP サーバ（mlx_lm.server / Ollama / LM Studio 等）バックエンド。

`{base_url}/chat/completions` に system/user の2メッセージを POST し、
`choices[0].message.content` と `usage` を `LlmResponse` に変換する。
接続失敗・HTTP エラー・応答形式の不正は全て `LlmBackendError` に変換する。

ローカルサーバでも API キーを要求する構成（LM Studio の認証設定等）に備え、
`api_key`（.env の `LOCAL_LLM_API_KEY`、任意）が設定されていれば
Authorization: Bearer ヘッダを付ける。
"""

from dataclasses import dataclass
from typing import Any

import httpx

from llm_fund.llm.backend import LlmBackendError, LlmResponse

# config/default.yaml `llm.backends.<name>` の既定タイムアウト。
DEFAULT_TIMEOUT_SECONDS = 300

# 設定でバックエンド名を与えない場合の既定名（LlmResponse.backend / llm_calls 記録用）。
DEFAULT_BACKEND_NAME = "openai_compat"

_CHAT_COMPLETIONS_PATH = "/chat/completions"


@dataclass(frozen=True, slots=True)
class OpenAiCompatBackend:
    """OpenAI 互換 chat/completions を呼ぶ `LlmBackend` 実装。"""

    base_url: str
    model: str
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    api_key: str | None = None
    name: str = DEFAULT_BACKEND_NAME

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, temperature: float
    ) -> LlmResponse:
        """1回の補完。接続失敗・HTTP エラー・不正応答は `LlmBackendError`。"""
        url = self.base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = httpx.post(
                url, json=body, headers=headers, timeout=self.timeout_seconds
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlmBackendError(f"OpenAI 互換サーバ呼び出しに失敗: {exc}") from exc
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> LlmResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlmBackendError(f"OpenAI 互換サーバの応答が JSON として不正: {exc}") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmBackendError(
                "OpenAI 互換サーバの応答に choices[0].message.content が無い"
            ) from exc
        if not isinstance(text, str):
            raise LlmBackendError("OpenAI 互換サーバの content が文字列ではない")

        raw_usage = payload.get("usage") if isinstance(payload, dict) else None
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return LlmResponse(text=text, model=self.model, backend=self.name, usage=usage)

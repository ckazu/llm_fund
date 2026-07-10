"""Tests for the OpenAI-compatible HTTP backend (httpx fully mocked via respx)."""

import json
from typing import Any

import httpx
import pytest
import respx

from llm_fund.llm.backend import LlmBackendError
from llm_fund.llm.openai_compat import OpenAiCompatBackend

BASE_URL = "http://127.0.0.1:8080/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"
MODEL = "qwen3-8b"


def _backend(**overrides: Any) -> OpenAiCompatBackend:
    defaults: dict[str, Any] = {
        "base_url": BASE_URL,
        "model": MODEL,
        "timeout_seconds": 30,
        "name": "local",
    }
    defaults.update(overrides)
    return OpenAiCompatBackend(**defaults)


def _complete(backend: OpenAiCompatBackend) -> Any:
    return backend.complete("sys", "user prompt", max_tokens=1024, temperature=0.2)


def _chat_response(content: str = "こんにちは") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }


class TestOpenAiCompatBackend:
    @respx.mock
    def test_success_parses_content_and_usage(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=_chat_response("様子見です"))
        )

        response = _complete(_backend())

        assert response.text == "様子見です"
        assert response.model == MODEL
        assert response.backend == "local"
        assert response.usage == {"prompt_tokens": 12, "completion_tokens": 34}
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == MODEL
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user prompt"},
        ]
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.2
        # API キー未設定なら Authorization ヘッダは付けない。
        assert "authorization" not in route.calls.last.request.headers

    @respx.mock
    def test_api_key_sent_as_bearer(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        _complete(_backend(api_key="secret-key"))
        assert route.calls.last.request.headers["authorization"] == "Bearer secret-key"

    @respx.mock
    def test_http_error_raises_backend_error(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(LlmBackendError, match="失敗"):
            _complete(_backend())

    @respx.mock
    def test_connect_error_raises_backend_error(self) -> None:
        respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(LlmBackendError, match="失敗"):
            _complete(_backend())

    @respx.mock
    def test_non_json_body_raises_backend_error(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="not-json"))
        with pytest.raises(LlmBackendError, match="JSON として不正"):
            _complete(_backend())

    @respx.mock
    def test_missing_choices_raises_backend_error(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"choices": []}))
        with pytest.raises(LlmBackendError, match="choices"):
            _complete(_backend())

    @respx.mock
    def test_missing_usage_defaults_to_empty_dict(self) -> None:
        payload = {"choices": [{"message": {"content": "ok"}}]}
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
        assert _complete(_backend()).usage == {}

    @respx.mock
    def test_trailing_slash_in_base_url_normalised(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_chat_response()))
        response = _complete(_backend(base_url=BASE_URL + "/"))
        assert response.text == "こんにちは"

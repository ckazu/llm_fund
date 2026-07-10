"""Tests for the claude CLI backend (subprocess fully mocked)."""

import json
import subprocess
from typing import Any

import pytest

from llm_fund.llm.backend import LlmBackendError
from llm_fund.llm.claude_cli import ClaudeCliBackend

MODEL = "sonnet"


def _cli_output(result: str = "こんにちは", **extra: Any) -> str:
    payload: dict[str, Any] = {
        "type": "result",
        "is_error": False,
        "result": result,
        "usage": {"input_tokens": 12, "output_tokens": 34},
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _backend(**overrides: Any) -> ClaudeCliBackend:
    defaults: dict[str, Any] = {"model": MODEL, "timeout_seconds": 30, "name": "claude_cli"}
    defaults.update(overrides)
    return ClaudeCliBackend(**defaults)


def _complete(backend: ClaudeCliBackend) -> Any:
    return backend.complete("sys", "user prompt", max_tokens=1024, temperature=0.2)


class TestClaudeCliBackend:
    def test_success_parses_result_and_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _completed(_cli_output("本日は様子見です"))

        monkeypatch.setattr("llm_fund.llm.claude_cli.subprocess.run", fake_run)

        response = _complete(_backend())

        assert response.text == "本日は様子見です"
        assert response.model == MODEL
        assert response.backend == "claude_cli"
        assert response.usage == {"input_tokens": 12, "output_tokens": 34}
        # claude --help で確認済みの実フラグで呼び出すこと。
        args = captured["args"]
        assert args[0] == "claude"
        assert "-p" in args
        assert args[args.index("--model") + 1] == MODEL
        assert args[args.index("--output-format") + 1] == "json"
        assert args[args.index("--append-system-prompt") + 1] == "sys"
        # プロンプト本文は stdin 渡し（ARG_MAX 回避）。
        assert captured["kwargs"]["input"] == "user prompt"
        assert captured["kwargs"]["timeout"] == 30

    def test_event_array_output_uses_last_result_element(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 新しい CLI はイベント配列を返す。type=result 要素の result/usage を採用する。
        events = json.dumps(
            [
                {"type": "system", "subtype": "init"},
                {"type": "assistant", "message": {"content": "..."}},
                {
                    "type": "result",
                    "is_error": False,
                    "result": '{"answer": 2}',
                    "usage": {"input_tokens": 5, "output_tokens": 7},
                },
            ],
            ensure_ascii=False,
        )
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed(events),
        )

        response = _complete(_backend())

        assert response.text == '{"answer": 2}'
        assert response.usage == {"input_tokens": 5, "output_tokens": 7}

    def test_event_array_without_result_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = json.dumps([{"type": "system", "subtype": "init"}])
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed(events),
        )
        with pytest.raises(LlmBackendError, match="type=result"):
            _complete(_backend())

    def test_timeout_raises_backend_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=args, timeout=30)

        monkeypatch.setattr("llm_fund.llm.claude_cli.subprocess.run", fake_run)
        with pytest.raises(LlmBackendError, match="タイムアウト"):
            _complete(_backend())

    def test_missing_command_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError(args[0])

        monkeypatch.setattr("llm_fund.llm.claude_cli.subprocess.run", fake_run)
        with pytest.raises(LlmBackendError, match="見つからない"):
            _complete(_backend())

    def test_nonzero_exit_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed("", returncode=1, stderr="boom"),
        )
        with pytest.raises(LlmBackendError, match="非ゼロ終了"):
            _complete(_backend())

    def test_invalid_json_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed("not-json"),
        )
        with pytest.raises(LlmBackendError, match="JSON として不正"):
            _complete(_backend())

    def test_missing_result_field_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed(json.dumps({"type": "result"})),
        )
        with pytest.raises(LlmBackendError, match="result"):
            _complete(_backend())

    def test_is_error_payload_raises_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run",
            lambda *a, **k: _completed(_cli_output("credit exhausted", is_error=True)),
        )
        with pytest.raises(LlmBackendError, match="エラー応答"):
            _complete(_backend())

    def test_missing_usage_defaults_to_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"type": "result", "is_error": False, "result": "ok"})
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.subprocess.run", lambda *a, **k: _completed(payload)
        )
        assert _complete(_backend()).usage == {}

    def test_is_available_uses_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "llm_fund.llm.claude_cli.shutil.which", lambda cmd: "/usr/local/bin/claude"
        )
        assert _backend().is_available() is True
        monkeypatch.setattr("llm_fund.llm.claude_cli.shutil.which", lambda cmd: None)
        assert _backend().is_available() is False

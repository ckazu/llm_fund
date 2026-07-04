"""Tests for the anthropic wrapper + self-consistency gate (technical-spec.md 5章).

anthropic is fully mocked: a fake client returns a scripted sequence of tool-use
messages (or raises), so schema validation, the one-shot correction retry, the
tenacity API retry, and the unanimity-based discard logic are all exercised
without any network call.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anthropic
import httpx
import pytest

from llm_fund.domain.enums import Action
from llm_fund.judgment.client import (
    AUDIT_KIND_DISAGREEMENT,
    NO_TRADE_ALL_SAMPLES_FAILED,
    LlmConfig,
    gather_consistent_judgment,
    request_judgment,
)
from tests.factories import build_llm_judgment_payload, build_llm_order_payload

AS_OF = date(2026, 7, 4)


# --- fakes -------------------------------------------------------------------


@dataclass
class _Block:
    type: str
    input: dict[str, Any] | None = None


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class _Message:
    content: list[_Block]
    usage: _Usage = field(default_factory=_Usage)


class _FakeMessages:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.messages = _FakeMessages(script)


class _FakeLlmSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, **kwargs: Any) -> int:
        self.records.append(kwargs)
        return len(self.records)


class _FakeAudit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def add(self, kind: str, detail_json: str) -> int:
        self.events.append((kind, detail_json))
        return len(self.events)


def _order(symbol: str = "7203.T", action: str = "BUY") -> dict[str, Any]:
    return build_llm_order_payload(symbol=symbol, action=action)


def _payload(orders: list[dict[str, Any]], *, no_trade: bool = False) -> dict[str, Any]:
    return build_llm_judgment_payload(
        orders, no_trade=no_trade, no_trade_reason="様子見" if no_trade else None
    )


def _tool_msg(payload: dict[str, Any]) -> _Message:
    return _Message(content=[_Block(type="tool_use", input=payload)])


def _retryable() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))


def _config(n_samples: int = 3) -> LlmConfig:
    return LlmConfig(model="claude-sonnet-5", n_samples=n_samples)


def _run_single(client: _FakeClient, sink: _FakeLlmSink, sample_index: int = 1) -> Any:
    return request_judgment(
        client,
        _config(),
        system="sys",
        user_prompt="user",
        as_of=AS_OF,
        sample_index=sample_index,
        prompt_version="v1",
        briefing_id=7,
        llm_call_sink=sink,
    )


# --- request_judgment (single sample) ---------------------------------------


class TestRequestJudgment:
    def test_valid_response_parsed(self) -> None:
        sink = _FakeLlmSink()
        client = _FakeClient([_tool_msg(_payload([_order()]))])
        result = _run_single(client, sink)
        assert result is not None
        assert [o.symbol for o in result.orders] == ["7203.T"]
        assert result.orders[0].valid_until == date(2026, 7, 7)
        assert len(sink.records) == 1
        assert sink.records[0]["sample_index"] == 1
        assert sink.records[0]["briefing_id"] == 7

    def test_invalid_then_valid_uses_correction_retry(self) -> None:
        sink = _FakeLlmSink()
        # first payload violates the schema (unknown action); second is valid.
        client = _FakeClient(
            [_tool_msg(_payload([_order(action="HODL")])), _tool_msg(_payload([_order()]))]
        )
        result = _run_single(client, sink)
        assert result is not None
        assert len(sink.records) == 2  # original + one correction attempt

    def test_all_invalid_returns_none(self) -> None:
        sink = _FakeLlmSink()
        bad = _tool_msg(_payload([_order(action="HODL")]))
        client = _FakeClient([bad, bad])
        assert _run_single(client, sink) is None
        assert len(sink.records) == 2

    def test_missing_tool_block_returns_none(self) -> None:
        sink = _FakeLlmSink()
        text_only = _Message(content=[_Block(type="text")])
        client = _FakeClient([text_only, text_only])
        assert _run_single(client, sink) is None

    def test_api_error_retried_then_succeeds(self) -> None:
        sink = _FakeLlmSink()
        client = _FakeClient([_retryable(), _tool_msg(_payload([_order()]))])
        result = _run_single(client, sink)
        assert result is not None
        # two physical create() calls (retry), one llm_calls row (single logical attempt).
        assert len(client.messages.calls) == 2
        assert len(sink.records) == 1

    def test_persistent_api_error_records_error_and_returns_none(self) -> None:
        sink = _FakeLlmSink()
        # each _call_and_parse retries once: 2 creates; correction is a 2nd _call_and_parse.
        client = _FakeClient([_retryable(), _retryable(), _retryable(), _retryable()])
        assert _run_single(client, sink) is None
        assert len(sink.records) == 2
        assert all(r["response"].startswith("API_ERROR") for r in sink.records)


# --- gather_consistent_judgment (self-consistency gate) ----------------------


class TestGatherConsistentJudgment:
    def _gather(self, script: list[Any], n_samples: int = 3) -> tuple[Any, _FakeAudit]:
        sink = _FakeLlmSink()
        audit = _FakeAudit()
        client = _FakeClient(script)
        decision = gather_consistent_judgment(
            client,
            _config(n_samples),
            system="sys",
            user_prompt="user",
            as_of=AS_OF,
            prompt_version="v1",
            briefing_id=7,
            llm_call_sink=sink,
            audit_sink=audit,
        )
        return decision, audit

    def test_unanimous_symbol_kept(self) -> None:
        script = [_tool_msg(_payload([_order()])) for _ in range(3)]
        decision, audit = self._gather(script)
        assert decision.disagreement_rate == 0.0
        assert [o.symbol for o in decision.judgment.orders] == ["7203.T"]
        assert decision.judgment.no_trade is False
        assert decision.n_successful == 3
        assert audit.events == []

    def test_split_action_discarded(self) -> None:
        script = [
            _tool_msg(_payload([_order(action="BUY")])),
            _tool_msg(_payload([_order(action="SELL")])),
            _tool_msg(_payload([_order(action="BUY")])),
        ]
        decision, audit = self._gather(script)
        assert decision.discarded_symbols == ["7203.T"]
        assert decision.disagreement_rate == 1.0
        assert decision.judgment.no_trade is True
        assert len(audit.events) == 1
        assert audit.events[0][0] == AUDIT_KIND_DISAGREEMENT

    def test_partial_presence_discarded(self) -> None:
        # 7203.T only appears in 2 of 3 samples -> not unanimous.
        script = [
            _tool_msg(_payload([_order()])),
            _tool_msg(_payload([_order()])),
            _tool_msg(_payload([], no_trade=True)),
        ]
        decision, _ = self._gather(script)
        assert decision.discarded_symbols == ["7203.T"]
        assert decision.judgment.no_trade is True

    def test_mixed_keeps_agreed_drops_split(self) -> None:
        # 7203.T unanimous BUY; 6758.T split -> one kept, one discarded, rate 0.5.
        script = [
            _tool_msg(_payload([_order("7203.T", "BUY"), _order("6758.T", "BUY")])),
            _tool_msg(_payload([_order("7203.T", "BUY"), _order("6758.T", "SELL")])),
            _tool_msg(_payload([_order("7203.T", "BUY"), _order("6758.T", "BUY")])),
        ]
        decision, audit = self._gather(script)
        assert [o.symbol for o in decision.judgment.orders] == ["7203.T"]
        assert decision.discarded_symbols == ["6758.T"]
        assert decision.disagreement_rate == pytest.approx(0.5)
        assert len(audit.events) == 1

    def test_all_samples_failed_forces_no_trade(self) -> None:
        bad = _tool_msg(_payload([_order(action="HODL")]))
        # 3 samples, each with an original + correction attempt that both fail.
        decision, _ = self._gather([bad] * 6)
        assert decision.judgment.no_trade is True
        assert decision.judgment.no_trade_reason == NO_TRADE_ALL_SAMPLES_FAILED
        assert decision.n_successful == 0
        assert decision.disagreement_rate == 1.0

    def test_all_abstain_is_no_trade_with_zero_disagreement(self) -> None:
        script = [_tool_msg(_payload([], no_trade=True)) for _ in range(3)]
        decision, _ = self._gather(script)
        assert decision.judgment.no_trade is True
        assert decision.disagreement_rate == 0.0
        assert decision.discarded_symbols == []

    def test_kept_order_action_is_domain_enum(self) -> None:
        script = [_tool_msg(_payload([_order()])) for _ in range(3)]
        decision, _ = self._gather(script)
        assert decision.judgment.orders[0].action is Action.BUY

"""anthropic SDK ラッパと自己一致性ゲート（technical-spec.md 5章）。

構造化出力は tool use（`tool_choice` でツールを強制）で担保する。API 呼び出しは tenacity で
1回リトライし、スキーマ検証に失敗したら1回だけ修正リトライ、それでも失敗した場合はその
サンプルを None（＝現状維持相当）として扱う（technical-spec.md 5章のフォールバック契約）。

自己一致性ゲート（`gather_consistent_judgment`）は同一入力で n_samples 回判断させ、
銘柄ごとに全サンプル全会一致でない指示を破棄する（保守側）。不一致率を算出し、破棄は
audit_events に記録する。

監査記録・破棄記録はハーネス側の責務であり、LLM に発注・記録系の能力を渡さないという
インジェクション防御（technical-spec.md 5章）と両立させるため、この層は store を直接
import せず、狭い書き込み口 Protocol（`LlmCallSink` / `AuditSink`）越しにのみ記録する。
判断層は validator/tracking を一切 import しない（technical-spec.md 2章の一方向依存）。
"""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import anthropic
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_fund.domain.models import JudgmentResult, OrderPlan
from llm_fund.judgment.schemas import (
    JUDGMENT_TOOL_DESCRIPTION,
    JUDGMENT_TOOL_NAME,
    SCHEMA_VERSION,
    LlmJudgment,
    judgment_tool_schema,
)

# llm_calls.kind の値（呼び出し種別。週次/月次で別値を使う想定）。
LLM_CALL_KIND_DAILY = "daily_judgment"

# audit_events.kind: 自己一致性ゲートで破棄した銘柄の記録。
AUDIT_KIND_DISAGREEMENT = "judgment_disagreement"

# 既定値（config/default.yaml の judgment セクションで上書き可能）。
DEFAULT_N_SAMPLES = 3
DEFAULT_MAX_TOKENS = 4096
# 1.0 は anthropic API の既定 temperature。Sonnet 5 等は非既定の temperature を 400 で拒否する
# ため、既定値を使うことで tool_choice 強制と両立させつつサンプル間の自然な揺らぎを得る。
DEFAULT_TEMPERATURE = 1.0

# API リトライ回数（合計試行回数）。1回リトライ = 2回試行（technical-spec.md 5章）。
_API_MAX_ATTEMPTS = 2

# 全サンプル失敗時 / 全銘柄破棄時 / 全サンプル現状維持時の NO_TRADE 理由。
NO_TRADE_ALL_SAMPLES_FAILED = "全サンプルの LLM 応答がスキーマ検証に失敗したため NO_TRADE"
NO_TRADE_NO_CONSENSUS = "全銘柄で判断が割れた（全会一致に至らなかった）ため NO_TRADE"
NO_TRADE_ALL_ABSTAIN = "全サンプルが現状維持（指示なし）だったため NO_TRADE"

_CORRECTION_INSTRUCTION = (
    "直前の応答は submit_judgment ツールのスキーマ検証に失敗しました。"
    "エラー: {error}\n"
    "スキーマに厳密に従い、submit_judgment ツールで再度提出してください。"
)


class LlmCallSink(Protocol):
    """llm_calls への追記口（`store.repos.LlmCallRepo.add` を構造的に満たす）。"""

    def add(
        self,
        *,
        kind: str,
        model: str,
        temperature: float,
        prompt_version: str,
        schema_version: int,
        sample_index: int,
        prompt: str,
        response: str | None,
        token_usage_json: str | None,
        briefing_id: int | None = ...,
        policy_id: int | None = ...,
        criteria_id: int | None = ...,
    ) -> int: ...


class AuditSink(Protocol):
    """audit_events への追記口（`store.repos.AuditEventRepo.add` を構造的に満たす）。"""

    def add(self, kind: str, detail_json: str) -> int: ...


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """LLM 呼び出しの評価プロトコル（llm_calls に記録される固定パラメータ）。"""

    model: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    n_samples: int = DEFAULT_N_SAMPLES


@dataclass(frozen=True, slots=True)
class ConsistencyDecision:
    """自己一致性ゲートの結果。"""

    judgment: JudgmentResult
    disagreement_rate: float
    discarded_symbols: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_successful: int = 0


@retry(
    retry=retry_if_exception_type(anthropic.APIError),
    stop=stop_after_attempt(_API_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def _create_message(
    client: Any, config: LlmConfig, system: str, messages: list[dict[str, Any]]
) -> Any:
    """anthropic messages.create を tool 強制で呼ぶ（tenacity で API エラーを1回リトライ）。

    tool_choice でツールを強制するため thinking は無効化する（強制ツール選択と拡張思考は併用不可）。
    """
    return client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        thinking={"type": "disabled"},
        system=system,
        messages=messages,
        tools=[
            {
                "name": JUDGMENT_TOOL_NAME,
                "description": JUDGMENT_TOOL_DESCRIPTION,
                "input_schema": judgment_tool_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": JUDGMENT_TOOL_NAME},
    )


def _extract_tool_input(message: Any) -> dict[str, Any] | None:
    """応答から最初の tool_use ブロックの input（dict）を取り出す。無ければ None。"""
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


def _usage_json(message: Any) -> str | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    return json.dumps(
        {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        },
        ensure_ascii=False,
    )


def _call_and_parse(
    client: Any,
    config: LlmConfig,
    system: str,
    messages: list[dict[str, Any]],
    *,
    sample_index: int,
    prompt_version: str,
    briefing_id: int,
    llm_call_sink: LlmCallSink,
    kind: str,
) -> tuple[LlmJudgment | None, str | None]:
    """1回の物理 API 呼び出し + 検証。(判断, エラー文字列) を返し、llm_calls へ1行記録する。"""
    prompt_dump = json.dumps(messages, ensure_ascii=False)
    try:
        message = _create_message(client, config, system, messages)
    except anthropic.APIError as exc:
        llm_call_sink.add(
            kind=kind,
            model=config.model,
            temperature=config.temperature,
            prompt_version=prompt_version,
            schema_version=SCHEMA_VERSION,
            sample_index=sample_index,
            prompt=prompt_dump,
            response=f"API_ERROR: {exc}",
            token_usage_json=None,
            briefing_id=briefing_id,
        )
        return None, f"api_error: {exc}"

    raw = _extract_tool_input(message)
    response_text = (
        json.dumps(raw, ensure_ascii=False) if raw is not None else "（tool_use ブロックなし）"
    )
    llm_call_sink.add(
        kind=kind,
        model=config.model,
        temperature=config.temperature,
        prompt_version=prompt_version,
        schema_version=SCHEMA_VERSION,
        sample_index=sample_index,
        prompt=prompt_dump,
        response=response_text,
        token_usage_json=_usage_json(message),
        briefing_id=briefing_id,
    )
    if raw is None:
        return None, "no tool_use block in response"
    try:
        return LlmJudgment.model_validate(raw), None
    except ValidationError as exc:
        return None, str(exc)


def request_judgment(
    client: Any,
    config: LlmConfig,
    *,
    system: str,
    user_prompt: str,
    as_of: date,
    sample_index: int,
    prompt_version: str,
    briefing_id: int,
    llm_call_sink: LlmCallSink,
    kind: str = LLM_CALL_KIND_DAILY,
) -> JudgmentResult | None:
    """1サンプル分の判断を取得する。スキーマ検証失敗時は1回だけ修正リトライし、再失敗なら None。"""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    judgment, error = _call_and_parse(
        client,
        config,
        system,
        messages,
        sample_index=sample_index,
        prompt_version=prompt_version,
        briefing_id=briefing_id,
        llm_call_sink=llm_call_sink,
        kind=kind,
    )
    if judgment is not None:
        return judgment.to_judgment_result(as_of)

    correction: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": _CORRECTION_INSTRUCTION.format(error=error)},
    ]
    judgment, _ = _call_and_parse(
        client,
        config,
        system,
        correction,
        sample_index=sample_index,
        prompt_version=prompt_version,
        briefing_id=briefing_id,
        llm_call_sink=llm_call_sink,
        kind=kind,
    )
    return judgment.to_judgment_result(as_of) if judgment is not None else None


def _order_for(result: JudgmentResult, symbol: str) -> OrderPlan | None:
    return next((order for order in result.orders if order.symbol == symbol), None)


def gather_consistent_judgment(
    client: Any,
    config: LlmConfig,
    *,
    system: str,
    user_prompt: str,
    as_of: date,
    prompt_version: str,
    briefing_id: int,
    llm_call_sink: LlmCallSink,
    audit_sink: AuditSink,
    kind: str = LLM_CALL_KIND_DAILY,
) -> ConsistencyDecision:
    """n_samples 回判断させ、全会一致でない銘柄を破棄した合意判断を返す（自己一致性ゲート）。"""
    results = [
        request_judgment(
            client,
            config,
            system=system,
            user_prompt=user_prompt,
            as_of=as_of,
            sample_index=index,
            prompt_version=prompt_version,
            briefing_id=briefing_id,
            llm_call_sink=llm_call_sink,
            kind=kind,
        )
        for index in range(1, config.n_samples + 1)
    ]
    successful = [result for result in results if result is not None]
    if not successful:
        return ConsistencyDecision(
            judgment=JudgmentResult(
                orders=[],
                market_view="",
                no_trade=True,
                no_trade_reason=NO_TRADE_ALL_SAMPLES_FAILED,
            ),
            disagreement_rate=1.0,
            discarded_symbols=[],
            n_samples=config.n_samples,
            n_successful=0,
        )

    market_view = successful[0].market_view
    # 出現順を保った銘柄の重複なしリスト（決定論的な処理順）。
    distinct: list[str] = []
    for result in successful:
        for order in result.orders:
            if order.symbol not in distinct:
                distinct.append(order.symbol)

    agreed: list[OrderPlan] = []
    discarded: list[str] = []
    for symbol in distinct:
        orders = [_order_for(result, symbol) for result in successful]
        present = [order for order in orders if order is not None]
        actions = {order.action for order in present}
        # 全サンプルに同一 action で出現した場合のみ合意（欠損＝そのサンプルの現状維持＝不一致）。
        unanimous = len(present) == len(successful) and len(actions) == 1
        if unanimous:
            agreed.append(present[0])
        else:
            discarded.append(symbol)
            audit_sink.add(
                AUDIT_KIND_DISAGREEMENT,
                json.dumps(
                    {
                        "symbol": symbol,
                        "actions": sorted(action.value for action in actions),
                        "present": len(present),
                        "samples": len(successful),
                    },
                    ensure_ascii=False,
                ),
            )

    disagreement_rate = len(discarded) / len(distinct) if distinct else 0.0
    if agreed:
        judgment = JudgmentResult(orders=agreed, market_view=market_view, no_trade=False)
    else:
        reason = NO_TRADE_ALL_ABSTAIN if not distinct else NO_TRADE_NO_CONSENSUS
        judgment = JudgmentResult(
            orders=[], market_view=market_view, no_trade=True, no_trade_reason=reason
        )

    return ConsistencyDecision(
        judgment=judgment,
        disagreement_rate=disagreement_rate,
        discarded_symbols=discarded,
        n_samples=config.n_samples,
        n_successful=len(successful),
    )

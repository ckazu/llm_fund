"""LLM バックエンドラッパと自己一致性ゲート（technical-spec.md 5章）。

LLM 呼び出しは `llm.backend.LlmBackend` Protocol 越しに行い、特定 SDK には依存しない
（バックエンドは claude CLI / OpenAI 互換ローカルサーバを config で切替可能）。
構造化出力は tool use が使えないため、システムプロンプトに JSON スキーマを明示して
「JSON のみを出力せよ」と指示し、応答から JSON を抽出（```json フェンス対応）した上で
pydantic 検証する。呼び出しは tenacity で1回リトライし、スキーマ検証に失敗したら1回だけ
修正リトライ、それでも失敗した場合はそのサンプルを None（＝現状維持相当）として扱う
（technical-spec.md 5章のフォールバック契約）。

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
from typing import Protocol

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_fund.domain.models import JudgmentResult, OrderPlan
from llm_fund.judgment.schemas import (
    SCHEMA_VERSION,
    LlmJudgment,
    judgment_json_schema,
)
from llm_fund.llm.backend import LlmBackend, LlmBackendError, LlmResponse
from llm_fund.llm.structured import extract_json, json_output_instruction

# llm_calls.kind の値（呼び出し種別。週次/月次で別値を使う想定）。
LLM_CALL_KIND_DAILY = "daily_judgment"

# audit_events.kind: 自己一致性ゲートで破棄した銘柄の記録。
AUDIT_KIND_DISAGREEMENT = "judgment_disagreement"

# 既定値（config/default.yaml の llm.judgment セクションで上書き可能）。
DEFAULT_N_SAMPLES = 3
DEFAULT_MAX_TOKENS = 4096
# サンプル間の自然な揺らぎを保ちつつ判断を安定させる保守的な既定値。
# claude CLI バックエンドは temperature を無視する（llm/claude_cli.py 参照）。
DEFAULT_TEMPERATURE = 0.2

# バックエンドのリトライ回数（合計試行回数）。1回リトライ = 2回試行（technical-spec.md 5章）。
_BACKEND_MAX_ATTEMPTS = 2

# 全サンプル失敗時 / 全銘柄破棄時 / 全サンプル現状維持時の NO_TRADE 理由。
NO_TRADE_ALL_SAMPLES_FAILED = "全サンプルの LLM 応答がスキーマ検証に失敗したため NO_TRADE"
NO_TRADE_NO_CONSENSUS = "全銘柄で判断が割れた（全会一致に至らなかった）ため NO_TRADE"
NO_TRADE_ALL_ABSTAIN = "全サンプルが現状維持（指示なし）だったため NO_TRADE"

_CORRECTION_INSTRUCTION = (
    "直前の応答は投資判断 JSON のスキーマ検証に失敗しました。"
    "エラー: {error}\n"
    "システムプロンプトの JSON スキーマに厳密に従い、JSON のみを再度出力してください。"
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
    """LLM 呼び出しの評価プロトコル（llm_calls に記録される固定パラメータ）。

    `model` は "backend:model" 形式のラベル（例 "claude_cli:sonnet"。
    `llm.router.LlmRouter.label_for` が生成）。実際のバックエンド/モデル解決は
    router が行い、この値は llm_calls の期間分離キーとしてのみ使う。
    """

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
    retry=retry_if_exception_type(LlmBackendError),
    stop=stop_after_attempt(_BACKEND_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def _complete(
    backend: LlmBackend, config: LlmConfig, system: str, prompt: str
) -> LlmResponse:
    """バックエンド補完を1回呼ぶ（tenacity でバックエンド障害を1回リトライ）。"""
    return backend.complete(
        system, prompt, max_tokens=config.max_tokens, temperature=config.temperature
    )


def _usage_json(response: LlmResponse) -> str | None:
    if not response.usage:
        return None
    return json.dumps(response.usage, ensure_ascii=False)


def _call_and_parse(
    backend: LlmBackend,
    config: LlmConfig,
    system: str,
    prompt: str,
    *,
    sample_index: int,
    prompt_version: str,
    briefing_id: int,
    llm_call_sink: LlmCallSink,
    kind: str,
) -> tuple[LlmJudgment | None, str | None]:
    """1回の論理呼び出し + 検証。(判断, エラー文字列) を返し、llm_calls へ1行記録する。

    システムプロンプトには JSON スキーマ明示の出力指示を追記する（tool use の代替）。
    """
    full_system = f"{system}\n\n{json_output_instruction(judgment_json_schema())}"
    try:
        response = _complete(backend, config, full_system, prompt)
    except LlmBackendError as exc:
        llm_call_sink.add(
            kind=kind,
            model=config.model,
            temperature=config.temperature,
            prompt_version=prompt_version,
            schema_version=SCHEMA_VERSION,
            sample_index=sample_index,
            prompt=prompt,
            response=f"BACKEND_ERROR: {exc}",
            token_usage_json=None,
            briefing_id=briefing_id,
        )
        return None, f"backend_error: {exc}"

    llm_call_sink.add(
        kind=kind,
        model=config.model,
        temperature=config.temperature,
        prompt_version=prompt_version,
        schema_version=SCHEMA_VERSION,
        sample_index=sample_index,
        prompt=prompt,
        response=response.text,
        token_usage_json=_usage_json(response),
        briefing_id=briefing_id,
    )
    try:
        raw = extract_json(response.text)
    except ValueError as exc:
        return None, str(exc)
    try:
        return LlmJudgment.model_validate(raw), None
    except ValidationError as exc:
        return None, str(exc)


def request_judgment(
    backend: LlmBackend,
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
    judgment, error = _call_and_parse(
        backend,
        config,
        system,
        user_prompt,
        sample_index=sample_index,
        prompt_version=prompt_version,
        briefing_id=briefing_id,
        llm_call_sink=llm_call_sink,
        kind=kind,
    )
    if judgment is not None:
        return judgment.to_judgment_result(as_of)

    correction_prompt = (
        f"{user_prompt}\n\n{_CORRECTION_INSTRUCTION.format(error=error)}"
    )
    judgment, _ = _call_and_parse(
        backend,
        config,
        system,
        correction_prompt,
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
    backend: LlmBackend,
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
            backend,
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

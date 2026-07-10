"""LLM 入出力の pydantic スキーマ（technical-spec.md 5章）。

judgment/ は判断層であり、LLM が返す JSON を受け取る「ワイヤ形式」をここで検証する。
検証を通ったワイヤ形式は `to_judgment_result` で domain の `JudgmentResult` /
`OrderPlan`（frozen モデル）へ変換され、以降の validator / tracking はこの domain
モデルだけを扱う。ワイヤ形式と domain モデルを分けているのは:

* ワイヤ形式は LLM に相対日数 `valid_days` を出させる（絶対日付より LLM が扱いやすく、
  カレンダー依存を判断層の外に押し出せる）。domain 側は絶対日付 `valid_until` を持つ。
* `extra="forbid"` で未知フィールドを弾き、`schema_version` の不一致を検知することで、
  スキーマ検証失敗 → 1回だけ修正リトライ → 再失敗なら NO_TRADE、という
  technical-spec.md 5章のフォールバック契約を構造的に成立させる。

lot_size の倍数検証はここでは既定単位（100株）に対する一次チェックにとどめる。
銘柄ごとの実 lot_size に対する厳密な検証は validator 層（rule_lot_size）が行う。
"""

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_fund.domain.enums import Action
from llm_fund.domain.models import JudgmentResult, OrderPlan

# 現行スキーマ版。LLM 出力の schema_version はこの値と一致しなければならない
# （不一致は破壊的変更の可能性 → スキーマ検証失敗として扱う）。
SCHEMA_VERSION = 1

# ワイヤ段階での lot 一次チェックに使う東証標準の売買単位。銘柄別の実 lot は
# validator 層が instruments.lot_size で厳密に再検証する。
DEFAULT_LOT_SIZE = 100

# valid_days（注文有効日数）の下限。当日限りでも最低1日。
MIN_VALID_DAYS = 1


class LlmOrderPlan(BaseModel):
    """LLM が出す1指示のワイヤ形式（technical-spec.md 5章の orders[] 要素）。"""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    action: Action
    units: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    tp_price: float = Field(gt=0)
    sl_price: float = Field(gt=0)
    valid_days: int = Field(ge=MIN_VALID_DAYS)
    rationale: str = Field(min_length=1)

    @field_validator("units")
    @classmethod
    def _units_are_lot_multiple(cls, value: int) -> int:
        if value % DEFAULT_LOT_SIZE != 0:
            raise ValueError(
                f"units={value} は売買単位 {DEFAULT_LOT_SIZE} の倍数でなければならない"
            )
        return value

    def to_order_plan(self, as_of: date) -> OrderPlan:
        """相対 `valid_days` を `as_of` からの絶対日付に変換し domain モデル化する。

        営業日カレンダーを持たないため暦日で加算する（data/loader.py の鮮度判定と
        同じ近似方針）。
        """
        return OrderPlan(
            symbol=self.symbol,
            action=self.action,
            units=self.units,
            entry_price=self.entry_price,
            tp_price=self.tp_price,
            sl_price=self.sl_price,
            valid_until=as_of + timedelta(days=self.valid_days),
            rationale=self.rationale,
        )


class LlmJudgment(BaseModel):
    """LLM 判断1回分のワイヤ形式（technical-spec.md 5章の出力ルート）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    market_view: str = Field(min_length=1)
    portfolio_view: str = ""
    no_trade: bool = False
    no_trade_reason: str | None = None
    orders: list[LlmOrderPlan] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_matches(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={value} は非対応（対応版 {SCHEMA_VERSION}）"
            )
        return value

    @model_validator(mode="after")
    def _check_no_trade_consistency(self) -> "LlmJudgment":
        # domain の JudgmentResult と同じ不変条件をワイヤ段階でも強制し、矛盾した
        # 出力（NO_TRADE なのに注文がある等）を早期に検証失敗させる。
        if self.no_trade and self.orders:
            raise ValueError("no_trade=True must not carry any orders")
        if self.no_trade and not self.no_trade_reason:
            raise ValueError("no_trade=True requires no_trade_reason")
        return self

    def to_judgment_result(self, as_of: date) -> JudgmentResult:
        """domain の `JudgmentResult` へ変換する（valid_days → valid_until 解決）。"""
        return JudgmentResult(
            orders=[order.to_order_plan(as_of) for order in self.orders],
            portfolio_view=self.portfolio_view,
            market_view=self.market_view,
            no_trade=self.no_trade,
            no_trade_reason=self.no_trade_reason,
        )


def judgment_json_schema() -> dict[str, Any]:
    """`LlmJudgment` の JSON Schema を返す（プロンプトに明示する出力スキーマ）。

    ワイヤ形式（pydantic モデル）を唯一の真実としてスキーマを導出し、検証側と
    プロンプト側でのスキーマのずれを防ぐ。`extra="forbid"` により
    `additionalProperties: false` が付与され、未知フィールドは検証段階で弾かれる。
    """
    return LlmJudgment.model_json_schema()

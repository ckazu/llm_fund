"""Frozen domain models (technical-spec.md 4章).

All models are immutable (`frozen=True`): once constructed, a value represents
a fact that happened (a candle, a judgment, a fill) and must not be mutated in
place. Callers build a new instance instead of patching fields.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_fund.domain.enums import (
    Action,
    ExecutionStatus,
    FillExitReason,
    InstructionStatus,
)

# 人間向け識別子 ticket_no のフォーマット: "YYYYMMDD-NN"（同日の連番2桁）
TICKET_NO_PATTERN = r"^\d{8}-\d{2}$"
# 連番は2桁なので1日あたり 01〜99 の 99 件が上限。日次指示数の設定はこれを超えられない。
MAX_DAILY_TICKET_SEQUENCE = 99


class _Frozen(BaseModel):
    """Base class wiring the shared frozen/strict pydantic config."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Candle(_Frozen):
    """1銘柄1日分の OHLCV。close は非調整（注文価格用）、adj_close は調整後（指標計算用）。"""

    symbol: str = Field(min_length=1)
    trade_date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    adj_close: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> "Candle":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self


class IndicatorRow(_Frozen):
    """1銘柄1日分の指標（リターン・移動平均乖離・ATR・出来高比）。

    割合変化・正規化値のみを持つ（technical-spec.md 5章: LLM は絶対値に過剰反応
    するため）。算出に必要な日数分のヒストリーが無い項目は `None`（欠損を明示し、
    無理に0や直近値で埋めない）。
    """

    symbol: str = Field(min_length=1)
    trade_date: date
    close: float = Field(gt=0)
    return_1d_pct: float | None = None
    return_5d_pct: float | None = None
    return_20d_pct: float | None = None
    return_60d_pct: float | None = None
    ma_deviation_25d_pct: float | None = None
    ma_deviation_75d_pct: float | None = None
    atr_pct_14: float | None = None
    volume_ratio_20d: float | None = None


class Briefing(_Frozen):
    """日次/週次/月次ブリーフィング（指標テーブル＋データスナップショット参照）。"""

    universe_code: str = Field(min_length=1)
    briefing_date: date
    kind: str = Field(min_length=1)
    content_md: str = Field(min_length=1)
    data_snapshot: dict[str, object] = Field(default_factory=dict)


class OrderPlan(_Frozen):
    """LLM が出す1指示（validator 通過前）。"""

    symbol: str = Field(min_length=1)
    action: Action
    units: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    tp_price: float = Field(gt=0)
    sl_price: float = Field(gt=0)
    valid_until: date
    rationale: str = Field(min_length=1)


class JudgmentResult(_Frozen):
    """LLM 判断の構造化出力（judgment/schemas.py で受け取る内容の内部表現）。"""

    orders: list[OrderPlan] = Field(default_factory=list)
    portfolio_view: str = ""
    market_view: str = ""
    no_trade: bool = False
    no_trade_reason: str | None = None

    @model_validator(mode="after")
    def _check_no_trade_consistency(self) -> "JudgmentResult":
        if self.no_trade and self.orders:
            raise ValueError("no_trade=True must not carry any orders")
        if self.no_trade and not self.no_trade_reason:
            raise ValueError("no_trade=True requires no_trade_reason")
        return self


class ValidatedInstruction(_Frozen):
    """validator 通過後の指示。ticket_no 発番済み。"""

    ticket_no: str = Field(pattern=TICKET_NO_PATTERN)
    symbol: str = Field(min_length=1)
    action: Action
    units: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    tp_price: float = Field(gt=0)
    sl_price: float = Field(gt=0)
    valid_until: date
    rationale: str = Field(min_length=1)
    status: InstructionStatus = InstructionStatus.PENDING
    warnings: list[str] = Field(default_factory=list)


class Rejection(_Frozen):
    """拒否されたプラン＋理由（監査用）。"""

    order: OrderPlan
    reasons: list[str] = Field(min_length=1)


class ExecutionRecord(_Frozen):
    """人間が記録した執行結果。"""

    ticket_no: str = Field(pattern=TICKET_NO_PATTERN)
    executed_at: datetime
    side: Action
    order_type: str = Field(min_length=1)
    actual_price: float = Field(gt=0)
    actual_units: int = Field(ge=0)
    commission: float = Field(ge=0)
    status: ExecutionStatus
    skip_reason: str | None = None
    deviation_note: str | None = None

    @model_validator(mode="after")
    def _check_skip_reason(self) -> "ExecutionRecord":
        if self.status == ExecutionStatus.SKIPPED and not self.skip_reason:
            raise ValueError("status=skipped requires skip_reason")
        return self


class VirtualFill(_Frozen):
    """保守的仮想約定エンジンの結果（tracking/virtual_fill.py）。"""

    ticket_no: str = Field(pattern=TICKET_NO_PATTERN)
    fill_date: date | None = None
    fill_price: float | None = None
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: FillExitReason | None = None
    commission: float = Field(ge=0, default=0.0)
    slippage: float = Field(ge=0, default=0.0)
    pnl: float | None = None

    @model_validator(mode="after")
    def _check_exit_requires_fill(self) -> "VirtualFill":
        if self.exit_date is not None and self.fill_date is None:
            raise ValueError("exit_date requires fill_date (must be filled before exit)")
        return self


class BenchmarkRow(_Frozen):
    """対照群戦略の日次 NAV・指標（tracking/benchmark.py）。"""

    strategy_code: str = Field(min_length=1)
    nav_date: date
    nav: float = Field(gt=0)
    metrics: dict[str, float] = Field(default_factory=dict)

"""Individual validator rules and the absolute risk caps (technical-spec.md 6章).

This is the compliance layer that stands between the (fallible) LLM judgment and
real capital. The absolute caps below are the safety contract: they are defined
in code and are *never* relaxable via `config/*.yaml`. `config.py` validates the
configured `limits.*` against them at startup, and `RiskLimits.from_settings`
additionally clamps every effective limit to its absolute cap — so even a config
that slips past validation cannot loosen the money-risk rules.

Rules are pure functions with a uniform signature `(order, ctx) -> str | None`:
they return a human-readable reason string when the order violates the rule, or
`None` when it passes. `gate.py` applies them serially. Aggregate/precondition
concerns that cannot be judged from a single order in isolation — cumulative
turnover and data freshness — live in `gate.py` instead.

Long-only cash semantics (requirements.md 現物ロングオンリー): only BUY orders
introduce new capital risk, so the capital-risk rules (StopLossRequired,
MaxLossPerTrade, CashSufficiency, MaxPositionPct, MaxExposure) pass exit orders
(SELL/CLOSE) through. Exit orders instead face ExitWithinHolding, which forbids
selling more units than are held (no naked short in a spot long-only book).
Structural rules (UniverseMember, LotSize, TickSize, PriceBandSanity) apply to
every order regardless of side.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_fund.domain.enums import Action
from llm_fund.domain.models import OrderPlan

if TYPE_CHECKING:
    # config.py imports the absolute caps from this module; importing
    # LimitsSettings only for typing avoids a circular import at runtime.
    from llm_fund.config import LimitsSettings

# --- 絶対上限（config で緩和不可。technical-spec.md 6章）---------------------
ABSOLUTE_MAX_LOSS_PER_TRADE_PCT = 3.0
ABSOLUTE_MAX_POSITION_PCT = 25.0
ABSOLUTE_MAX_TURNOVER_PCT = 50.0
# 現物ロングオンリー（信用・レバレッジなし）のため総エクスポージャーは満額投資が上限。
ABSOLUTE_MAX_EXPOSURE_PCT = 100.0

# SOFT: これ未満の根拠文は「短すぎる」と警告する（拒否はしない）。
MIN_RATIONALE_LENGTH = 20

# 浮動小数の価格を呼値の整数倍かどうか判定する際の許容誤差。
_TICK_EPSILON = 1e-9
_PCT_DIVISOR = 100.0

# --- 東証 呼値テーブル（標準・TOPIX100 以外）---------------------------------
# (価格の上限[この値以下], 呼値)。先頭から最初に price <= 上限 を満たす行の呼値を使う。
TSE_TICK_TABLE: tuple[tuple[float, float], ...] = (
    (3_000.0, 1.0),
    (5_000.0, 5.0),
    (30_000.0, 10.0),
    (50_000.0, 50.0),
    (300_000.0, 100.0),
    (500_000.0, 500.0),
    (3_000_000.0, 1_000.0),
    (5_000_000.0, 5_000.0),
    (30_000_000.0, 10_000.0),
    (50_000_000.0, 50_000.0),
)
# 上表の最大価格を超える場合の呼値。
TSE_TICK_ABOVE = 100_000.0

# --- 東証 値幅制限テーブル ----------------------------------------------------
# (基準値段の上限[この値未満], 制限値幅)。先頭から最初に base < 上限 を満たす行を使う。
PRICE_BAND_TABLE: tuple[tuple[float, float], ...] = (
    (100.0, 30.0),
    (200.0, 50.0),
    (500.0, 80.0),
    (700.0, 100.0),
    (1_000.0, 150.0),
    (1_500.0, 300.0),
    (2_000.0, 400.0),
    (3_000.0, 500.0),
    (5_000.0, 700.0),
    (7_000.0, 1_000.0),
    (10_000.0, 1_500.0),
    (15_000.0, 3_000.0),
    (20_000.0, 4_000.0),
    (30_000.0, 5_000.0),
    (50_000.0, 7_000.0),
    (70_000.0, 10_000.0),
    (100_000.0, 15_000.0),
    (150_000.0, 30_000.0),
    (200_000.0, 40_000.0),
    (300_000.0, 50_000.0),
    (500_000.0, 70_000.0),
    (700_000.0, 100_000.0),
    (1_000_000.0, 150_000.0),
    (1_500_000.0, 300_000.0),
    (2_000_000.0, 400_000.0),
    (3_000_000.0, 500_000.0),
    (5_000_000.0, 700_000.0),
    (7_000_000.0, 1_000_000.0),
    (10_000_000.0, 1_500_000.0),
)
# 上表の最大基準値段以上の場合の制限値幅（現実的な株価域外の保守的フォールバック）。
PRICE_BAND_ABOVE = 3_000_000.0


def tse_tick_size(price: float) -> float:
    """Return the TSE tick size (呼値) applicable at `price`."""
    for upper, tick in TSE_TICK_TABLE:
        if price <= upper:
            return tick
    return TSE_TICK_ABOVE


def tse_price_band_width(base_price: float) -> float:
    """Return the daily price-limit width (制限値幅) for a base (prev-close) price."""
    for upper, width in PRICE_BAND_TABLE:
        if base_price < upper:
            return width
    return PRICE_BAND_ABOVE


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Effective risk limits used by the rules, already clamped to absolute caps.

    Build with `from_settings` so the clamping is applied; direct construction is
    used only in tests. `max_loss_per_trade_pct` and `max_exposure_pct` have no
    `config` field and always take their absolute cap.
    """

    max_loss_per_trade_pct: float
    max_position_pct: float
    max_exposure_pct: float
    max_turnover_pct: float
    min_rationale_length: int
    require_stop_loss: bool
    max_instructions_per_day: int

    @classmethod
    def from_settings(cls, settings: "LimitsSettings") -> "RiskLimits":
        return cls(
            max_loss_per_trade_pct=ABSOLUTE_MAX_LOSS_PER_TRADE_PCT,
            max_position_pct=min(settings.max_position_pct, ABSOLUTE_MAX_POSITION_PCT),
            max_exposure_pct=ABSOLUTE_MAX_EXPOSURE_PCT,
            max_turnover_pct=min(settings.max_turnover_pct, ABSOLUTE_MAX_TURNOVER_PCT),
            min_rationale_length=MIN_RATIONALE_LENGTH,
            require_stop_loss=settings.require_stop_loss,
            max_instructions_per_day=settings.max_instructions_per_day,
        )


@dataclass(frozen=True, slots=True)
class InstrumentContext:
    """Per-instrument reference data a rule needs (resolved from repos upstream)."""

    symbol: str
    in_universe: bool
    lot_size: int
    prev_close: float  # 前日終値（非調整）: tick / price-band の基準
    current_units: int = 0  # 現在保有ロング株数（約定後の組入計算に使用）


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Everything the rule set needs beyond a single `OrderPlan`."""

    nav: float
    cash: float
    current_exposure: float  # 現在の全保有時価総額（MaxExposure 用）
    instruments: dict[str, InstrumentContext]
    limits: RiskLimits
    data_fresh: bool = True


def _instrument(order: OrderPlan, ctx: ValidationContext) -> InstrumentContext | None:
    return ctx.instruments.get(order.symbol)


def _notional(order: OrderPlan) -> float:
    """Order value at the entry price (used for cash / exposure / turnover)."""
    return order.entry_price * order.units


def notional(order: OrderPlan) -> float:
    """Public alias of the order notional, reused by `gate.py` for turnover."""
    return _notional(order)


def _is_tick_aligned(price: float, tick: float) -> bool:
    ratio = price / tick
    return abs(ratio - round(ratio)) < _TICK_EPSILON


# --- HARD rules --------------------------------------------------------------


def rule_stop_loss_required(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """BUY の損切りが entry 以上（＝損切りにならない）指示を拒否する。

    OrderPlan は sl_price > 0 を必須にするため「未指定」は構造的に発生しない。
    exit(SELL/CLOSE) は新規の下方リスクを負わないため対象外。
    """
    if order.action is not Action.BUY:
        return None
    if order.sl_price >= order.entry_price:
        return (
            f"BUY の sl_price={order.sl_price} は entry_price={order.entry_price} 未満で"
            "なければならない（損切りとして機能しない）"
        )
    return None


def rule_max_loss_per_trade(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """想定損失 (entry-sl)*units が NAV×上限% を超える指示を拒否する。"""
    if order.action is not Action.BUY:
        return None
    loss = (order.entry_price - order.sl_price) * order.units
    limit = ctx.nav * ctx.limits.max_loss_per_trade_pct / _PCT_DIVISOR
    if loss > limit:
        return (
            f"想定損失 {loss:.0f} が上限 {limit:.0f}"
            f"（NAV×{ctx.limits.max_loss_per_trade_pct}%）を超過"
        )
    return None


def rule_cash_sufficiency(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """現金余力を超える BUY を拒否する。"""
    if order.action is not Action.BUY:
        return None
    cost = _notional(order)
    if cost > ctx.cash:
        return f"必要資金 {cost:.0f} が現金余力 {ctx.cash:.0f} を超過"
    return None


def rule_max_position_pct(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """約定後の1銘柄組入比率が上限% を超える指示を拒否する。"""
    if order.action is not Action.BUY:
        return None
    inst = _instrument(order, ctx)
    if inst is None:
        return None
    post_units = inst.current_units + order.units
    post_value = post_units * order.entry_price
    limit = ctx.nav * ctx.limits.max_position_pct / _PCT_DIVISOR
    if post_value > limit:
        return (
            f"約定後の組入 {post_value:.0f} が上限 {limit:.0f}"
            f"（NAV×{ctx.limits.max_position_pct}%）を超過"
        )
    return None


def rule_max_exposure(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """約定後の総エクスポージャーが上限% を超える指示を拒否する。"""
    if order.action is not Action.BUY:
        return None
    post_exposure = ctx.current_exposure + _notional(order)
    limit = ctx.nav * ctx.limits.max_exposure_pct / _PCT_DIVISOR
    if post_exposure > limit:
        return (
            f"約定後の総エクスポージャー {post_exposure:.0f} が上限 {limit:.0f}"
            f"（NAV×{ctx.limits.max_exposure_pct}%）を超過"
        )
    return None


def rule_exit_within_holding(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """SELL/CLOSE が保有株数を超える指示を拒否する。

    現物ロングオンリー（requirements.md）では空売りを持たないため、保有株数を超える
    決済は事実上の裸ショートになる。BUY は新規建てなので対象外（保有制約は
    MaxPositionPct が見る）。
    """
    if order.action is Action.BUY:
        return None
    inst = _instrument(order, ctx)
    if inst is None:  # 未登録銘柄は UniverseMember が拒否する
        return None
    if order.units > inst.current_units:
        return (
            f"決済株数 {order.units} が保有株数 {inst.current_units} を超過"
            "（現物ロングオンリーのため空売りは不可）"
        )
    return None


def rule_universe_member(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """ユニバース外（未登録 or in_universe=False）の銘柄を拒否する。"""
    inst = _instrument(order, ctx)
    if inst is None or not inst.in_universe:
        return f"銘柄 {order.symbol} は取引ユニバースに含まれない"
    return None


def rule_lot_size(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """売買単位（lot_size、通常100株）の倍数でない株数を拒否する。"""
    inst = _instrument(order, ctx)
    if inst is None:  # 未登録銘柄は UniverseMember が拒否する
        return None
    if order.units % inst.lot_size != 0:
        return f"株数 {order.units} が売買単位 {inst.lot_size} の倍数でない"
    return None


def rule_tick_size(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """entry/tp/sl が各価格帯の呼値の整数倍でない指示を拒否する。"""
    offending: list[str] = []
    for label, price in (
        ("entry", order.entry_price),
        ("tp", order.tp_price),
        ("sl", order.sl_price),
    ):
        tick = tse_tick_size(price)
        if not _is_tick_aligned(price, tick):
            offending.append(f"{label}={price}（呼値 {tick}）")
    if offending:
        return "呼値に合致しない価格: " + ", ".join(offending)
    return None


def rule_price_band_sanity(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """entry が前日終値から値幅制限を超えて乖離している指示を拒否する。"""
    inst = _instrument(order, ctx)
    if inst is None:  # 未登録銘柄は UniverseMember が拒否する
        return None
    width = tse_price_band_width(inst.prev_close)
    if abs(order.entry_price - inst.prev_close) > width:
        return (
            f"entry_price={order.entry_price} が前日終値 {inst.prev_close} の"
            f"値幅制限 ±{width} を超えて乖離"
        )
    return None


# --- SOFT rules --------------------------------------------------------------


def rule_rationale_quality(order: OrderPlan, ctx: ValidationContext) -> str | None:
    """根拠文が短すぎる指示に警告を付与する（SOFT: 拒否しない）。"""
    if len(order.rationale.strip()) < ctx.limits.min_rationale_length:
        return f"根拠文が短い（{ctx.limits.min_rationale_length}文字未満）: 判断品質を確認"
    return None


# 個別ルールの共通シグネチャ。違反理由文字列 or None（合格）を返す。
Rule = Callable[[OrderPlan, ValidationContext], str | None]

# 直列適用の順序。gate.py が (name, func) を順に評価する。
HARD_RULES: tuple[tuple[str, Rule], ...] = (
    ("StopLossRequired", rule_stop_loss_required),
    ("MaxLossPerTrade", rule_max_loss_per_trade),
    ("CashSufficiency", rule_cash_sufficiency),
    ("MaxPositionPct", rule_max_position_pct),
    ("MaxExposure", rule_max_exposure),
    ("ExitWithinHolding", rule_exit_within_holding),
    ("UniverseMember", rule_universe_member),
    ("LotSize", rule_lot_size),
    ("TickSize", rule_tick_size),
    ("PriceBandSanity", rule_price_band_sanity),
)

SOFT_RULES: tuple[tuple[str, Rule], ...] = (("RationaleQuality", rule_rationale_quality),)

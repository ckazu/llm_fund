"""保守的仮想約定エンジン（technical-spec.md 7章 / requirements FR-5）。

日足 OHLC のみで IFO（If-Done + OCO）を**保守的に**判定する。ここが本プロジェクト
の核心で、判定が甘いと「LLM が有効に見える」偽陽性を生み検証全体が無意味になる。
そのため曖昧なケースは常に戦略に不利な側へ倒す:

1. エントリー判定（指値 BUY）: 有効期間内の日足で ``low <= entry`` なら約定。約定
   価格は ``min(entry, open)``（寄付きが指値より有利なら寄付き＝現実の板寄せ）。
2. 決済判定（約定の**翌営業日以降**、日足ごとに）:
   * 寄付きが SL を割込む下方ギャップ（``open <= sl``）→ 寄付きで損切り（スリッページ
     の現実化。SL 価格より不利な約定を realize する）。
   * 寄付きが TP を上抜く上方ギャップ（``open >= tp``）→ **TP 価格で利確**（寄付きの
     より高い価格ではなく指値で頭打ち。利益を過大評価しない保守側）。
   * 日中に TP と SL の**両方**に到達 → **SL を優先**（保守側。同一足内の到達順序は
     日足からは判別不能なため不利側を採用）。
   * どちらか一方のみ → その指値価格で決済。
3. 有効期限切れ: 期間内に entry 未到達 → 未約定として記録（未約定も成績の一部）。
4. コスト: 手数料（率＋最低額）とスリッページ（率）を全約定に適用。往復（entry+exit）
   の両サイドに課す。

**決済の起点を約定の翌日**とするのは、日足では約定当日のザラ場の値動き順序（entry
到達と TP/SL 到達の前後関係）を復元できないため。当日決済を捏造すると有利にも不利
にも恣意が入るので、確定情報のある翌日から評価する（既知の制約として明記）。

このモジュールは ``portfolio_state`` / ``positions`` / ``pending_orders`` /
``virtual_fills`` を更新する**唯一の書き手**。``run_virtual_fills`` は指示・日足・
基準日 ``as_of`` の決定論的な関数として毎回全再計算するため、同日2回実行しても二重
計上しない（冪等）。現物ロングオンリーのため対象は BUY エントリーのみ。
"""

import sqlite3
from dataclasses import dataclass
from datetime import date

from llm_fund.domain.enums import Action, FillExitReason, PendingOrderStatus
from llm_fund.domain.models import Candle, VirtualFill
from llm_fund.store.repos import (
    BriefingRepo,
    CandleRepo,
    InstructionRepo,
    PendingOrderRepo,
    PortfolioStateRepo,
    PositionRepo,
    VirtualFillRepo,
)


@dataclass(frozen=True, slots=True)
class CostModel:
    """全約定に課すコスト条件（config.tracking と対応。ベンチマークにも同一適用）。"""

    commission_rate: float = 0.0
    min_commission: float = 0.0
    slippage_pct: float = 0.0


# コスト0の既定モデル（純粋な価格判定ロジックを検証する際の既定引数用）。
_ZERO_COST = CostModel()


def commission_for(notional: float, costs: CostModel) -> float:
    """約定代金 ``notional`` に対する手数料（率 × 代金、ただし最低額を下限とする）。"""
    return max(notional * costs.commission_rate, costs.min_commission)


def slippage_for(notional: float, costs: CostModel) -> float:
    """約定代金 ``notional`` に対するスリッページコスト（率 × 代金）。"""
    return notional * costs.slippage_pct


@dataclass(frozen=True, slots=True)
class _EntryResult:
    index: int
    fill_price: float
    fill_date: date


def _find_entry(
    candles: list[Candle], entry_price: float, valid_until: date
) -> _EntryResult | None:
    """有効期限内で指値 BUY が最初に約定する日足を探す（``low <= entry``）。"""
    for index, candle in enumerate(candles):
        if candle.trade_date > valid_until:
            break
        if candle.low <= entry_price:
            return _EntryResult(
                index=index,
                fill_price=min(entry_price, candle.open),
                fill_date=candle.trade_date,
            )
    return None


@dataclass(frozen=True, slots=True)
class _ExitResult:
    exit_price: float
    exit_date: date
    reason: FillExitReason


def _find_exit(
    candles: list[Candle], tp_price: float, sl_price: float
) -> _ExitResult | None:
    """約定翌日以降の日足から決済を判定する（保守側: ギャップ優先・両到達は SL 優先）。

    ``candles`` には既に約定日の翌日以降のみが渡される前提。
    """
    for candle in candles:
        if candle.open <= sl_price:  # 下方ギャップ: 寄付きで損切り（不利側を realize）
            return _ExitResult(candle.open, candle.trade_date, FillExitReason.SL)
        if candle.open >= tp_price:  # 上方ギャップ: TP 価格で頭打ち（利益を過大評価しない）
            return _ExitResult(tp_price, candle.trade_date, FillExitReason.TP)
        hit_tp = candle.high >= tp_price
        hit_sl = candle.low <= sl_price
        if hit_sl:  # 両到達（hit_tp and hit_sl）も SL を優先する
            return _ExitResult(sl_price, candle.trade_date, FillExitReason.SL)
        if hit_tp:
            return _ExitResult(tp_price, candle.trade_date, FillExitReason.TP)
    return None


def simulate_fill(
    *,
    ticket_no: str,
    action: Action,
    units: int,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    valid_until: date,
    candles: list[Candle],
    costs: CostModel = _ZERO_COST,
) -> VirtualFill:
    """1指示を保守的ルールで仮想約定させ ``VirtualFill`` を返す（純関数・I/O なし）。

    ``candles`` は活性化日以降の昇順日足（``candles[0]`` が最初のエントリー候補日）。
    決済判定は約定日の**翌日**以降のみを走査する。未約定かつ有効期限に到達している
    場合は ``exit_reason=EXPIRY``（未約定失効）としてマークする。
    """
    if action is not Action.BUY:
        # 現物ロングオンリー。SELL/CLOSE（既存ポジションの裁量決済）は原 BUY の
        # OCO(tp/sl) で決済されるため、本エンジンの対象外（申し送り事項）。
        raise ValueError(f"simulate_fill supports BUY only, got {action}")

    entry = _find_entry(candles, entry_price, valid_until)
    if entry is None:
        # 有効期限日に到達していれば失効。まだ届いていなければ未約定のまま保留。
        expired = bool(candles) and candles[-1].trade_date >= valid_until
        return VirtualFill(
            ticket_no=ticket_no,
            exit_reason=FillExitReason.EXPIRY if expired else None,
        )

    entry_notional = entry.fill_price * units
    entry_commission = commission_for(entry_notional, costs)
    entry_slippage = slippage_for(entry_notional, costs)

    exit_ = _find_exit(candles[entry.index + 1 :], tp_price, sl_price)
    if exit_ is None:
        # 約定済みだが未決済（ポジション保有中）。損益は未確定。
        return VirtualFill(
            ticket_no=ticket_no,
            fill_date=entry.fill_date,
            fill_price=entry.fill_price,
            commission=entry_commission,
            slippage=entry_slippage,
        )

    exit_notional = exit_.exit_price * units
    exit_commission = commission_for(exit_notional, costs)
    exit_slippage = slippage_for(exit_notional, costs)
    gross_pnl = (exit_.exit_price - entry.fill_price) * units
    total_commission = entry_commission + exit_commission
    total_slippage = entry_slippage + exit_slippage
    return VirtualFill(
        ticket_no=ticket_no,
        fill_date=entry.fill_date,
        fill_price=entry.fill_price,
        exit_date=exit_.exit_date,
        exit_price=exit_.exit_price,
        exit_reason=exit_.reason,
        commission=total_commission,
        slippage=total_slippage,
        pnl=gross_pnl - total_commission - total_slippage,
    )


@dataclass(frozen=True, slots=True)
class EngineResult:
    """``run_virtual_fills`` の結果サマリ（レポート/テスト用）。"""

    as_of: date
    cash: float
    nav: float
    open_positions: int
    filled: int
    exited: int
    expired: int


def _close_price_as_of(candles: list[Candle], as_of: date) -> float | None:
    """``as_of`` 以前で最新の非調整終値（保有ポジションの時価評価用）。"""
    eligible = [c for c in candles if c.trade_date <= as_of]
    return eligible[-1].close if eligible else None


def run_virtual_fills(
    conn: sqlite3.Connection,
    as_of: date,
    *,
    costs: CostModel,
    starting_capital: float,
) -> EngineResult:
    """全 BUY 指示を仮想約定させ、ポートフォリオ状態を ``as_of`` まで再計算する。

    決定論的に全再計算し ``virtual_fills`` を upsert、``positions`` /
    ``pending_orders`` を全消去して再構築、``portfolio_state`` を ``as_of`` で upsert
    する。指示・日足が同じなら何度実行しても同一結果（冪等）。
    """
    instruction_repo = InstructionRepo(conn)
    briefing_repo = BriefingRepo(conn)
    candle_repo = CandleRepo(conn)
    vf_repo = VirtualFillRepo(conn)
    position_repo = PositionRepo(conn)
    pending_repo = PendingOrderRepo(conn)
    state_repo = PortfolioStateRepo(conn)

    position_repo.clear()
    pending_repo.clear()

    cash = starting_capital
    nav_positions = 0.0
    open_positions = 0
    filled = 0
    exited = 0
    expired = 0

    for instruction in instruction_repo.list_all():
        if instruction.action != Action.BUY.value:
            continue  # ロングオンリー: 本エンジンは BUY エントリーのみを扱う
        briefing = briefing_repo.get_by_id(instruction.briefing_id)
        if briefing is None:
            continue
        # 活性化は判断当日の翌営業日以降（当日終値で判断しているため当日約定は不可）。
        candles = [
            c
            for c in candle_repo.get_range(
                instruction.instrument_id, start=briefing.briefing_date, end=as_of
            )
            if c.trade_date > briefing.briefing_date
        ]
        vf = simulate_fill(
            ticket_no=instruction.ticket_no,
            action=Action.BUY,
            units=instruction.units,
            entry_price=instruction.entry_price,
            tp_price=instruction.tp_price,
            sl_price=instruction.sl_price,
            valid_until=instruction.valid_until,
            candles=candles,
            costs=costs,
        )
        vf_repo.upsert(
            instruction_id=instruction.id,
            fill_date=vf.fill_date,
            fill_price=vf.fill_price,
            exit_date=vf.exit_date,
            exit_price=vf.exit_price,
            exit_reason=vf.exit_reason.value if vf.exit_reason else None,
            commission=vf.commission,
            slippage=vf.slippage,
            pnl=vf.pnl,
        )

        if vf.fill_date is None:
            status = (
                PendingOrderStatus.EXPIRED
                if vf.exit_reason is FillExitReason.EXPIRY
                else PendingOrderStatus.PENDING
            )
            if status is PendingOrderStatus.EXPIRED:
                expired += 1
            pending_repo.add(
                instruction_id=instruction.id,
                expires_at=instruction.valid_until,
                status=status.value,
            )
            continue

        filled += 1
        assert vf.fill_price is not None
        entry_notional = vf.fill_price * instruction.units
        cash -= entry_notional + commission_for(entry_notional, costs) + slippage_for(
            entry_notional, costs
        )
        pending_repo.add(
            instruction_id=instruction.id,
            expires_at=instruction.valid_until,
            status=PendingOrderStatus.FILLED.value,
        )

        if vf.exit_date is not None and vf.exit_price is not None:
            exited += 1
            exit_notional = vf.exit_price * instruction.units
            cash += exit_notional - commission_for(exit_notional, costs) - slippage_for(
                exit_notional, costs
            )
            position_repo.add(
                instrument_id=instruction.instrument_id,
                units=instruction.units,
                avg_cost=vf.fill_price,
                opened_at=vf.fill_date,
                closed_at=vf.exit_date,
            )
        else:
            open_positions += 1
            position_repo.add(
                instrument_id=instruction.instrument_id,
                units=instruction.units,
                avg_cost=vf.fill_price,
                opened_at=vf.fill_date,
                closed_at=None,
            )
            candles_all = candle_repo.get_range(instruction.instrument_id, end=as_of)
            close = _close_price_as_of(candles_all, as_of)
            if close is not None:
                nav_positions += close * instruction.units

    nav = cash + nav_positions
    state_repo.upsert(as_of, cash, nav, note="virtual_fill")
    return EngineResult(
        as_of=as_of,
        cash=cash,
        nav=nav,
        open_positions=open_positions,
        filled=filled,
        exited=exited,
        expired=expired,
    )

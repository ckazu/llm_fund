"""対照群ベンチマーク（technical-spec.md 7章 / requirements FR-5）。

LLM 運用（``fund``）を、**同一の開始資本・手数料・スリッページ条件**で常時 4 つの
単純戦略と比較する。目的は「LLM 判断がインデックス積立や単純ルール、さらには
ランダム判断と有意に区別できるか」を測ること。比較の公平性が崩れると検証が無意味
になるため、全戦略に同じ ``CostModel`` と ``starting_capital`` を課す。

戦略（``strategies`` テーブル）:

* ``fund``        — LLM 運用の仮想 NAV（``portfolio_state`` を参照。virtual_fill.py が真実の源泉）
* ``index``       — インデックス積立の代理（TOPIX 連動 ETF を buy&hold）
* ``equal_weight``— 売買ユニバースの等金額 buy&hold
* ``momentum``    — 単純モメンタム（ルックバック期間の上昇率が最大の1銘柄を buy&hold）
* ``random``      — シード固定のランダム1銘柄 buy&hold（「もっともらしいだけのランダム」でない
                    ことの検定用。同一シードで再現可能）

各対照群は共通ウィンドウ（対象銘柄すべてに日足が揃う日付の共通部分）の初日終値で
建て、以降は日次で時価評価する（買い直しなし）。日次 NAV と指標（MaxDD/Sharpe/
勝率/回転率/コスト比率）を ``benchmark_navs`` に upsert する（UNIQUE(strategy_id,date)
により同日2回実行で二重計上しない）。

簡略化（申し送り）: ベンチマークは端株を許容し（正確な資金配分のため）開始時の
一括投資（積立ではなく lump-sum）とする。最低手数料はポートフォリオ単位では適用せず
``commission_rate``＋``slippage_pct`` を投下代金に課す。「積立」化と最低手数料の厳密化
は後続ステップの拡張余地。
"""

import json
import math
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date

from llm_fund.domain.models import Candle
from llm_fund.store.repos import (
    BenchmarkNavRepo,
    CandleRepo,
    InstrumentRepo,
    PortfolioStateRepo,
    StrategyRepo,
)
from llm_fund.tracking.virtual_fill import CostModel

# 戦略コード（strategies.code。行追加のみで対照群を増やせる）。
STRATEGY_FUND = "fund"
STRATEGY_INDEX = "index"
STRATEGY_EQUAL_WEIGHT = "equal_weight"
STRATEGY_MOMENTUM = "momentum"
STRATEGY_RANDOM = "random"

STRATEGY_NAMES: dict[str, str] = {
    STRATEGY_FUND: "LLM 運用",
    STRATEGY_INDEX: "インデックス積立",
    STRATEGY_EQUAL_WEIGHT: "等金額ポートフォリオ",
    STRATEGY_MOMENTUM: "単純モメンタム",
    STRATEGY_RANDOM: "ランダム判断",
}

# Sharpe 年率換算の営業日数。
TRADING_DAYS_PER_YEAR = 252
_PCT = 100.0


@dataclass(frozen=True, slots=True)
class StrategyRun:
    """1戦略のシミュレーション結果（NAV 系列＋コスト/回転の集計）。"""

    nav_series: list[tuple[date, float]] = field(default_factory=list)
    total_cost: float = 0.0
    deployed_notional: float = 0.0


def _close_by_date(candles: list[Candle]) -> dict[date, float]:
    return {c.trade_date: c.close for c in candles}


def common_window(
    candles_by_symbol: dict[str, list[Candle]], as_of: date
) -> list[date]:
    """対象銘柄すべてに日足が存在する日付の共通部分（``as_of`` 以前、昇順）。"""
    if not candles_by_symbol:
        return []
    date_sets = [
        {c.trade_date for c in candles if c.trade_date <= as_of}
        for candles in candles_by_symbol.values()
    ]
    common = set.intersection(*date_sets) if date_sets else set()
    return sorted(common)


def simulate_weighted_buy_hold(
    weights: dict[str, float],
    candles_by_symbol: dict[str, list[Candle]],
    window: list[date],
    *,
    starting_capital: float,
    costs: CostModel,
) -> StrategyRun:
    """``weights``（合計1）で初日に建て、以降 buy&hold で時価評価した NAV 系列を返す。

    投下代金 = ``starting_capital / (1 + commission_rate + slippage_pct)`` とし、残りを
    エントリーコストに充てる（現金残 0）。初日 NAV は投下代金（コスト控除後）に一致する。
    """
    if not window or not weights:
        return StrategyRun()
    cost_rate = costs.commission_rate + costs.slippage_pct
    investable = starting_capital / (1.0 + cost_rate)
    total_cost = investable * cost_rate
    closes = {sym: _close_by_date(candles_by_symbol[sym]) for sym in weights}
    first = window[0]
    shares: dict[str, float] = {}
    for sym, weight in weights.items():
        price0 = closes[sym][first]
        shares[sym] = investable * weight / price0
    nav_series: list[tuple[date, float]] = []
    for day in window:
        nav = sum(shares[sym] * closes[sym][day] for sym in weights)
        nav_series.append((day, nav))
    return StrategyRun(
        nav_series=nav_series, total_cost=total_cost, deployed_notional=investable
    )


def select_momentum(
    candles_by_symbol: dict[str, list[Candle]],
    window_start: date,
    lookback_days: int,
) -> str | None:
    """ルックバック期間の上昇率が最大の銘柄を選ぶ（単純モメンタム）。

    上昇率 = ``close(window_start) / close(window_start 以前で lookback 日前以内の最古) - 1``。
    データ不足の銘柄は除外し、算出可能な中で最大の銘柄コードを返す。
    """
    best_symbol: str | None = None
    best_return = -math.inf
    for symbol, candles in candles_by_symbol.items():
        prior = [c for c in candles if c.trade_date <= window_start]
        if len(prior) < 2:
            continue
        end_close = prior[-1].close
        cutoff_index = max(0, len(prior) - 1 - lookback_days)
        start_close = prior[cutoff_index].close
        momentum = end_close / start_close - 1.0
        if momentum > best_return:
            best_return = momentum
            best_symbol = symbol
    return best_symbol


def select_random(symbols: list[str], seed: int) -> str | None:
    """シード固定でランダムに1銘柄を選ぶ（再現可能。ソート順に依存を排除して決定論化）。"""
    if not symbols:
        return None
    rng = random.Random(seed)
    return rng.choice(sorted(symbols))


def compute_metrics(
    nav_series: list[tuple[date, float]],
    *,
    total_cost: float,
    deployed_notional: float,
    starting_capital: float,
) -> dict[str, float]:
    """NAV 系列から MaxDD/Sharpe/勝率/回転率/コスト比率を算出する（純関数）。"""
    navs = [nav for _, nav in nav_series]
    returns = [
        navs[i] / navs[i - 1] - 1.0
        for i in range(1, len(navs))
        if navs[i - 1] != 0.0
    ]
    return {
        "max_drawdown_pct": _max_drawdown_pct(navs),
        "sharpe": _sharpe(returns),
        "win_rate": _win_rate(returns),
        "turnover": deployed_notional / starting_capital if starting_capital else 0.0,
        "cost_ratio": total_cost / starting_capital if starting_capital else 0.0,
    }


def _max_drawdown_pct(navs: list[float]) -> float:
    """ピークからの最大下落率（正の%）。"""
    peak = -math.inf
    max_dd = 0.0
    for nav in navs:
        peak = max(peak, nav)
        if peak > 0:
            drawdown = (nav - peak) / peak
            max_dd = min(max_dd, drawdown)
    return abs(max_dd) * _PCT


def _sharpe(returns: list[float]) -> float:
    """年率換算 Sharpe（無リスク金利0）。サンプルが不足/無分散なら 0。"""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _win_rate(returns: list[float]) -> float:
    """プラスリターン日の割合。"""
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def _load_candles(
    conn: sqlite3.Connection, symbols: list[str], as_of: date
) -> dict[str, list[Candle]]:
    instrument_repo = InstrumentRepo(conn)
    candle_repo = CandleRepo(conn)
    result: dict[str, list[Candle]] = {}
    for symbol in symbols:
        record = instrument_repo.get_by_symbol(symbol)
        if record is None:
            continue
        candles = [
            c for c in candle_repo.get_range(record.id, end=as_of) if c.trade_date <= as_of
        ]
        if candles:
            result[symbol] = candles
    return result


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """``run_benchmark`` の結果（レポート用の最新 NAV とコスト比率）。"""

    latest_nav: dict[str, float]
    metrics: dict[str, dict[str, float]]


def _persist_series(
    repo: BenchmarkNavRepo,
    strategy_repo: StrategyRepo,
    code: str,
    run: StrategyRun,
    starting_capital: float,
) -> dict[str, float]:
    """1戦略の NAV 系列と（起点からその日までの）累積指標を upsert し、最終指標を返す。"""
    strategy_id = strategy_repo.get_or_create(code, STRATEGY_NAMES[code])
    latest_metrics: dict[str, float] = {}
    for index, (nav_date, nav) in enumerate(run.nav_series):
        metrics = compute_metrics(
            run.nav_series[: index + 1],
            total_cost=run.total_cost,
            deployed_notional=run.deployed_notional,
            starting_capital=starting_capital,
        )
        repo.upsert(
            strategy_id=strategy_id,
            nav_date=nav_date,
            nav=nav,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )
        latest_metrics = metrics
    return latest_metrics


def run_benchmark(
    conn: sqlite3.Connection,
    as_of: date,
    *,
    universe_symbols: list[str],
    index_symbol: str,
    momentum_lookback_days: int,
    random_seed: int,
    costs: CostModel,
    starting_capital: float,
) -> BenchmarkSummary:
    """4 つの対照群＋LLM 運用（fund）の NAV を更新し比較サマリを返す。

    決定論的な全再計算＋upsert により冪等（同日2回実行で benchmark_navs は増えない）。
    """
    strategy_repo = StrategyRepo(conn)
    nav_repo = BenchmarkNavRepo(conn)

    all_symbols = sorted(set(universe_symbols) | {index_symbol})
    candles_by_symbol = _load_candles(conn, all_symbols, as_of)
    universe_available = [s for s in universe_symbols if s in candles_by_symbol]

    latest_nav: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}

    fund_states = PortfolioStateRepo(conn).list_all()
    fund_series = [(s.state_date, s.nav) for s in fund_states if s.state_date <= as_of]

    # 比較ウィンドウは fund（LLM 運用）の運用開始日から始める。全戦略を同一時点で建て、
    # それ以前の日足はモメンタムのルックバック履歴に充てる（look-ahead を避ける前向き比較）。
    # fund 未運用時は共通日付全体を使う（モメンタムはルックバック不足なら選定を見送る）。
    common_full = common_window(candles_by_symbol, as_of)
    start_date = fund_series[0][0] if fund_series else (common_full[0] if common_full else None)

    # --- 対照群（共通ウィンドウで buy&hold）---
    window = [d for d in common_full if start_date is not None and d >= start_date]
    if window:
        window_start = window[0]
        runs: dict[str, StrategyRun] = {}
        if index_symbol in candles_by_symbol:
            runs[STRATEGY_INDEX] = simulate_weighted_buy_hold(
                {index_symbol: 1.0},
                candles_by_symbol,
                window,
                starting_capital=starting_capital,
                costs=costs,
            )
        if universe_available:
            equal = 1.0 / len(universe_available)
            runs[STRATEGY_EQUAL_WEIGHT] = simulate_weighted_buy_hold(
                {s: equal for s in universe_available},
                candles_by_symbol,
                window,
                starting_capital=starting_capital,
                costs=costs,
            )
            momentum_symbol = select_momentum(
                {s: candles_by_symbol[s] for s in universe_available},
                window_start,
                momentum_lookback_days,
            )
            if momentum_symbol is not None:
                runs[STRATEGY_MOMENTUM] = simulate_weighted_buy_hold(
                    {momentum_symbol: 1.0},
                    candles_by_symbol,
                    window,
                    starting_capital=starting_capital,
                    costs=costs,
                )
            random_symbol = select_random(universe_available, random_seed)
            if random_symbol is not None:
                runs[STRATEGY_RANDOM] = simulate_weighted_buy_hold(
                    {random_symbol: 1.0},
                    candles_by_symbol,
                    window,
                    starting_capital=starting_capital,
                    costs=costs,
                )
        for code, run in runs.items():
            metrics[code] = _persist_series(
                nav_repo, strategy_repo, code, run, starting_capital
            )
            if run.nav_series:
                latest_nav[code] = run.nav_series[-1][1]

    # --- fund（仮想執行が維持する portfolio_state を参照）---
    if fund_series:
        fund_run = StrategyRun(
            nav_series=fund_series, total_cost=0.0, deployed_notional=0.0
        )
        metrics[STRATEGY_FUND] = _persist_series(
            nav_repo, strategy_repo, STRATEGY_FUND, fund_run, starting_capital
        )
        latest_nav[STRATEGY_FUND] = fund_series[-1][1]

    return BenchmarkSummary(latest_nav=latest_nav, metrics=metrics)

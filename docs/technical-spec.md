# llm_fund 技術仕様案

作成日: 2026-07-04
ステータス: ドラフト（承認待ち）
前提: [requirements.md](requirements.md)

## 1. 技術スタック

| 項目 | 採用 | 備考 |
|---|---|---|
| 言語 / パッケージ管理 | Python 3.12 / uv | llm_company と統一 |
| CLI | typer（コマンド名 `fund`） | cron / llm_company trigger から起動 |
| LLM | Claude API（anthropic SDK） | モデルは config で切替（既定 claude-sonnet-5） |
| 価格データ | yfinance（EOD） | 非公式のためデータ層で抽象化し将来差替可能に |
| 永続化 | SQLite | 単一プロセス・個人利用のため十分 |
| 検証・型 | pytest / ruff / mypy(strict) | TDD、外部 API は全モック |

## 2. アーキテクチャ

```
[cron / llm_company trigger]
        ▼
cli.py (typer) ── fund daily | weekly | monthly | record | benchmark | fetch | status
        ▼
data/       価格取得(yfinance) + SQLiteキャッシュ + 鮮度/欠損ゲート
        ▼
briefing/   指標計算 → LLM向け構造化テーブル生成
        ▼
judgment/   Claude API 構造化出力（--no-llm でテンプレート判断に差替）
        ▼
validator/  ハード拒否ルール + 警告付与（コンプライアンス層）
        ▼
delivery/   Markdownレポート + JSON出力 + Webhook通知
        ▼
[人間が IFO 注文を執行] → fund record → tracking/（仮想執行・3者ベンチマーク）
```

モジュール間依存は上から下への一方向のみ。`judgment/`（LLM）は `validator/` `tracking/` を import しない（判断層は測定層に触れない）。

### ディレクトリ構成

```
src/llm_fund/
├── cli.py               # typer エントリポイント
├── config.py            # pydantic-settings: .env(秘匿) + config/*.yaml(非秘匿)
├── domain/
│   ├── models.py        # frozen モデル（下記 4 章）
│   └── enums.py         # Action, InstructionStatus, FillResult 等
├── data/
│   ├── prices.py        # PriceSource プロトコル + YFinanceSource
│   ├── cache.py         # SQLite キャッシュ（差分取得）
│   └── loader.py        # 統一 API + 鮮度/欠損ゲート
├── briefing/
│   ├── indicators.py    # リターン, MA乖離, ATR, 出来高比
│   └── builder.py       # LLM 向けテーブル/Markdown 生成
├── judgment/
│   ├── schemas.py       # LLM 入出力 pydantic スキーマ（schema_version 付き）
│   ├── client.py        # anthropic ラッパ（リトライ、監査ログ記録）
│   ├── prompts.py       # 方針・基準・過去成績を注入するプロンプト組立
│   └── template.py      # --no-llm 用テンプレート判断
├── validator/
│   ├── rules.py         # 個別ルール（絶対上限定数を含む）
│   └── gate.py          # 直列適用。HARD 違反=拒否 / SOFT 違反=警告付与
├── tracking/
│   ├── virtual_fill.py  # 保守的仮想約定エンジン
│   └── benchmark.py     # 3者比較 NAV・指標算出
├── review/
│   ├── weekly.py        # 振り返り → 基準変更提案（差分形式）
│   └── monthly.py       # 方針・ユニバース入替提案
├── delivery/
│   ├── report.py        # Markdown レポート
│   ├── webhook.py       # Discord/Slack 通知（任意）
│   └── json_out.py      # --format json（llm_company 連携用）
└── store/
    ├── db.py            # 接続・マイグレーション適用
    ├── migrations/      # 番号付き SQL（0001_init.sql, ...）
    └── repos.py         # Repository 群
```

## 3. DB スキーマ（SQLite）

設計原則: 全テーブル `id INTEGER PRIMARY KEY AUTOINCREMENT`（サロゲートキー）。ナチュラルキーは UNIQUE 制約。人間に露出する識別子は `ticket_no` 等の外部識別子で、内部 PK は露出しない。

```sql
instruments(id, symbol TEXT UNIQUE, name, market, lot_size INTEGER DEFAULT 100, active)
candles(id, instrument_id FK, date TEXT, open, high, low, close, volume,
        adj_close,                 -- 指標計算は調整後、注文価格は非調整を使用
        UNIQUE(instrument_id, date))

-- ポートフォリオ状態（仮想執行が維持する。真実の源泉）
portfolio_state(id, date TEXT UNIQUE, cash, nav, note)
positions(id, instrument_id FK, units, avg_cost, opened_at, closed_at NULL)
pending_orders(id, instruction_id FK, expires_at, status)   -- 未約定 IFO・失効管理

-- 判断サイクル
policies(id, effective_from, content, status, approved_at)   -- 月次方針
criteria(id, effective_from, content, diff, rationale, status, approved_at,
         superseded_by FK NULL)                               -- 週次基準（ロールバック可能）
briefings(id, date, kind, content_md, data_snapshot_json, created_at)
instructions(id, ticket_no TEXT UNIQUE,        -- 例: "20260704-01"（人間向け識別子）
             briefing_id FK, instrument_id FK, action, units,
             entry_price, tp_price, sl_price, valid_until,
             rationale, validator_result_json, status)
executions(id, instruction_id FK, executed_at, side, order_type,
           actual_price, actual_units, commission, status,   -- filled/partial/skipped
           skip_reason, deviation_note)

-- 測定
virtual_fills(id, instruction_id FK, fill_date, fill_price, exit_date, exit_price,
              exit_reason,            -- tp/sl/expiry/manual
              commission, slippage, pnl)
benchmark_snapshots(id, date TEXT UNIQUE, fund_nav, index_nav, momentum_nav,
                    metrics_json)     -- MaxDD, Sharpe, 勝率, 回転率, コスト比率

-- 監査
llm_calls(id, ts, kind, model, schema_version, policy_id FK, criteria_id FK,
          briefing_id FK, prompt TEXT, response TEXT, token_usage_json)
audit_events(id, ts, kind, detail_json)   -- 拒否/警告/NO_TRADE/承認/ロールバック
```

## 4. ドメインモデル（domain/models.py、全て frozen）

- `Candle`, `IndicatorRow`（1銘柄1日分の指標）
- `Briefing`（日付・テーブル・データスナップショット参照）
- `OrderPlan`: LLM が出す1指示。`symbol, action(BUY/SELL/CLOSE), units, entry, tp, sl, valid_until, rationale`
- `JudgmentResult`: `orders: list[OrderPlan], portfolio_view, market_view, no_trade: bool, no_trade_reason`
- `ValidatedInstruction`: validator 通過後。`ticket_no` 発番済み、`warnings: list[str]`
- `Rejection`: 拒否されたプラン＋理由（監査用）
- `ExecutionRecord`, `VirtualFill`, `BenchmarkRow`

## 5. LLM 入出力仕様（judgment/schemas.py）

### 入力（プロンプトに注入）
1. 月次方針（active な policies.content）
2. 週次基準（active な criteria.content）— IFO 幅の目安、エントリー条件等
3. 日次ブリーフィング（指標テーブル Markdown）
4. 現在ポートフォリオ（**比率ベース**: 各銘柄の組入%、現金%。実額は既定で送らない）
5. 直近 N 件の指示とその仮想成績（自己修正の材料）

### 出力（tool use / structured output で強制）

```json
{
  "schema_version": 1,
  "market_view": "…（市況の要約）",
  "no_trade": false,
  "orders": [
    {
      "symbol": "7203.T",
      "action": "BUY",
      "units": 100,
      "entry_price": 3120,
      "tp_price": 3320,
      "sl_price": 3020,
      "valid_days": 3,
      "rationale": "…"
    }
  ]
}
```

- pydantic でスキーマ検証。失敗時は1回だけ修正リトライ、再失敗なら `NO_TRADE` として記録
- `units` は lot_size（100株）の倍数のみ許可

## 6. バリデーター仕様（validator/）

コード内絶対上限（設定で緩和不可）:

```python
ABSOLUTE_MAX_LOSS_PER_TRADE_PCT = 3.0    # (entry-sl)*units ≤ NAV*3%
ABSOLUTE_MAX_POSITION_PCT = 25.0
ABSOLUTE_MAX_TURNOVER_PCT = 50.0
```

| ルール | 種別 | 内容 |
|---|---|---|
| StopLossRequired | HARD | sl 未指定 / BUY で sl ≥ entry は拒否 |
| MaxLossPerTrade | HARD | 想定損失が NAV×設定%（≤絶対上限）以下 |
| CashSufficiency | HARD | 現金余力を超える BUY を拒否 |
| MaxPositionPct | HARD | 約定後の1銘柄組入が上限以下 |
| MaxExposure | HARD | 総エクスポージャー上限 |
| MaxTurnover | HARD | 当日指示合計の回転率上限 |
| UniverseMember | HARD | ユニバース外銘柄を拒否 |
| DataFreshness | HARD | 最新バーが古い/欠損 → 全指示拒否・NO_TRADE 強制 |
| LotSize / TickSize | HARD | 売買単位・呼値の検証（呼値は東証の価格帯別テーブル） |
| PriceBandSanity | HARD | entry が前日終値から値幅制限を超えて乖離していたら拒否 |
| RationaleQuality | SOFT | 根拠文の欠落・短すぎは警告付与 |

全結果（承認・警告・拒否）を `audit_events` と `instructions.validator_result_json` に記録。

## 7. 仮想執行エンジン（tracking/virtual_fill.py）

日足 OHLC のみで IFO を保守的に判定する:

1. **エントリー判定**: 指値 entry に対し、有効期間内の日足で `low ≤ entry`（BUY）なら約定。約定価格は `min(entry, open)` … open が有利なら open（現実の板寄せ）
2. **決済判定**（約定後、日足ごとに）:
   - その日の high ≥ tp かつ low ≤ sl の**両方**に到達 → **SL 約定として扱う**（保守側）
   - どちらか一方のみ到達 → その価格で決済
   - ギャップ（open が sl より不利）→ open で決済（スリッページの現実化）
3. **有効期限切れ**: 期間内に entry 未到達 → 未約定として記録（未約定も成績の一部）
4. **コスト**: 手数料（設定値、例: 約定代金の0.05%＋最低額）とスリッページ（設定値）を全約定に適用

ベンチマーク側（積立・モメンタム）にも同一の手数料・キャッシュフロー条件を適用し、比較の公平性を保つ。

## 8. CLI 仕様

| コマンド | 動作 | 起動元 |
|---|---|---|
| `fund fetch` | ユニバースの価格を取得しキャッシュ更新 | daily 内からも呼ばれる |
| `fund daily [--no-llm] [--format json]` | fetch → briefing → 判断 → validation → レポート/通知 | cron（毎営業日朝）/ llm_company |
| `fund weekly` | 振り返り＋基準変更提案（承認待ち状態で保存） | cron（週末）/ llm_company |
| `fund monthly` | 方針・ユニバース入替提案 | cron（月初）/ llm_company |
| `fund approve <proposal>` | 週次/月次提案の承認・有効化 | 人間 |
| `fund record <ticket_no> --price N [--skipped 理由]` | 執行記録 | 人間 |
| `fund benchmark` | 仮想 NAV 更新＋3者比較レポート | daily 内 / 手動 |
| `fund status` | データ鮮度・最新 NAV・未記録指示・承認待ち提案の一覧 | 人間 / llm_company |

- 終了コード: 正常 0 / NO_TRADE 0（レポートに明記）/ データ異常 2 / 設定異常 3（cron 監視用）
- `--format json` は全コマンド共通で、llm_company worker 統合（第2段階）の連携面

## 9. 設定

- `.env`: `ANTHROPIC_API_KEY`, `NOTIFY_WEBHOOK_URL`（秘匿）
- `config/default.yaml`: モデル名、制限値（絶対上限以下のみ有効）、ベンチマーク設定、手数料・スリッページ、出力先
- `config/universe.yaml`: 監視銘柄（月次提案→`fund approve` で更新）
- 起動時に設定値が絶対上限を超えていたら終了コード 3 で abort

## 10. テスト方針

1. **validator/**: 境界値網羅（上限ちょうど/1単位超過、SL欠落、呼値・売買単位、鮮度違反）。最優先・カバレッジ最厚
2. **tracking/**: 固定 OHLC フィクスチャで仮想約定の全分岐（SL優先、ギャップ、期限切れ、コスト）を手計算と突合
3. **judgment/**: モックレスポンスでスキーマ検証・修正リトライ・NO_TRADE フォールバック
4. **briefing/**: 固定価格データ → 生成テーブルのスナップショット回帰
5. **store/**: マイグレーションの前進適用テスト、Repository の CRUD
6. **CLI 結合**: `fund daily --no-llm` を一時 DB で end-to-end（外部 API 全モック）
7. 市場休日・タイムゾーン（JST/UTC）・株式分割（adj_close と close の分離）・API 失敗時 NO_TRADE・再実行冪等性（同日2回実行で重複指示が出ない）

## 11. 実装ステップ

計画ファイル（S0〜S10）に準拠。S0 = yfinance 日本株 PoC（auto_adjust 検証）・IFO 仕様確認・評価基準定義 → S1 骨格 → S2 domain+store → S3 data → S4 briefing（--no-llm で daily 貫通）→ S5 validator → S6 judgment（4分割）→ S7 record → S8 tracking/benchmark → S9 weekly/monthly → S10 delivery/cron/runbook。

## 12. 第2段階（本仕様のスコープ外）

- llm_company worker 統合: scheduler から `fund daily --format json` を実行し Discord に配信、承認/執行ボタン → `fund record` 書き戻し
- ニュース・決算データの組込み（FR-1 拡張）
- データソースの J-Quants 等への差替

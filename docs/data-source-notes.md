# データソース検証メモ（S0 yfinance PoC）

作成日: 2026-07-04
検証環境: ネットワーク到達可能（`uv run python` から yfinance 経由で実データ取得を確認）。
検証コードは `scratch/yfinance_poc.py`（本番コードには含めない一時スクリプト）。
バージョン: `yfinance>=0.2.50`（pyproject.toml 指定。実際に解決されたバージョンは `uv.lock` 参照）

## 1. 取得可否

対象銘柄すべてで日足 OHLCV の取得に成功した。

| 銘柄 | 種別 | 取得 | 備考 |
|---|---|---|---|
| `7203.T` | 日本個別株（トヨタ） | OK | `.T` サフィックスで東証銘柄を指定 |
| `AAPL` | 米国個別株 | OK | サフィックスなし |
| `1306.T` | ETF（TOPIX連動） | OK | 個別株と同じ経路で取得可能。特別扱い不要 |

`yf.download(symbol, period=..., interval="1d", auto_adjust=...)` と `yf.Ticker(symbol).history(...)` の両方で取得確認。

## 2. API 選択: `download()` vs `Ticker().history()`

- `download()`: 複数銘柄一括向け。返る `DataFrame` の列が `(field, ticker)` の MultiIndex になる（単一銘柄でも）。**index は tz-naive**。
- `Ticker(symbol).history()`: 単一銘柄向け。列はフラット。**index は市場ローカルの tz-aware（下記3節）**。配当・株式分割の列（`Dividends`, `Stock Splits`）も同時に取得できる。

→ `data/prices.py` の `YFinanceSource` は **`Ticker().history()` を採用**し、銘柄ごとに個別取得してから DB へ正規化して格納する方針とする（tz 情報を確実に得るため、かつ列構造をシンプルに保つため）。複数銘柄の一括取得による高速化が必要になった場合は `download()` を再検討するが、その場合も保存前に tz-naive → 市場ローカル tz を明示的に付与する変換が必須。

## 3. タイムゾーン

`Ticker().history()` の index は銘柄の上場市場のローカルタイムゾーンで tz-aware になる。

| 銘柄 | index.tz |
|---|---|
| `7203.T` | `Asia/Tokyo` |
| `1306.T` | `Asia/Tokyo` |
| `AAPL` | `America/New_York` |

一方 `download()` の index は **tz-naive**（内部的にはおそらく市場ローカルの日付だが tz 情報が失われる）。

**注意点**:
- 日本株・米国株を同一 DB テーブル（`candles.date TEXT`）に格納する際、tz 付き datetime をそのまま文字列化すると市場によって表現が揺れる。`data/loader.py` で「市場ローカルの暦日（YYYY-MM-DD、tzは保持せず解釈上は各市場の現地日付）」に正規化してから `candles.date` に格納する方針とする。米国市場の日足は日本時間では日付が一日ずれて見える点を FR-1 の「ユニバース別レポート生成タイミング」で吸収する設計（技術仕様2章のとおり）と整合させる。
- cron 実行時刻と米国市場のクローズ時刻（日本時間早朝）の関係に注意。米国ユニバースのレポート生成は日本時間の早朝以降に実行する必要がある。

## 4. `auto_adjust=True` / `False` の差

- `auto_adjust=True`（`download()` の既定）: `Close` 列がすでに配当・株式分割調整後の値になる。`Adj Close` 列は出力されない。
- `auto_adjust=False`: `Close`（非調整の実際の取引価格）と `Adj Close`（調整後）の両方が出力される。

実データで確認（`7203.T` 過去1年、`auto_adjust=False`）:
- 配当・株式分割イベントは `actions`（`Dividends`, `Stock Splits`）で個別に取得可能。過去1年で配当が2回（2025-09-29, 2026-03-30）発生しており、それ以降の期間で `Close != Adj Close` の行が多数（178/約250営業日）存在することを確認した。株式分割は該当期間内では未発生（2021-10-01の5分割は取得期間より過去）。
- → 配当があるだけで `Close` と `Adj Close` は乖離する。技術仕様のスキーマどおり **両方を保存し用途で使い分ける**（`candles.adj_close` を追加保持、指標計算は調整後・注文価格は非調整）方針が妥当と確認できた。

**実装方針**: `YFinanceSource` は `auto_adjust=False` で取得し、`close`（非調整・実際の注文価格の参照に使用）と `adj_close`（調整後・リターン/移動平均等の指標計算に使用）の両方を `candles` テーブルに保存する。`auto_adjust=True` のみで取得すると非調整値が失われ、注文価格の妥当性検証（`PriceBandSanity` 等）に使う「実際の前日終値」が得られなくなるため不採用。

## 5. その他の注意点

- 出来高（`Volume`）は調整の影響を受けない生値。
- `history()` は配当・分割情報を含むため、将来 `virtual_fill.py` や指標計算で調整イベントを参照する際の追加取得コストは不要（同一呼び出しで取得可能）。
- yfinance は非公式スクレイピングベースであり、レート制限・スキーマ変更・一時的な取得失敗のリスクがある（要件定義書 6章のリスクどおり）。`data/loader.py` の鮮度・欠損ゲートで異常時は `NO_TRADE` を強制する設計を維持する。
- 日本個別株・ETFともに `.T` サフィックスで同一経路から取得できるため、`PriceSource` プロトコルの実装は銘柄種別で分岐不要（ユニバース設定の `market` はレポートタイミング・呼値テーブル選択等の用途に限定）。
- 今回のネットワーク環境では取得に成功したため「未検証」項目はない。ただし本番運用環境（cron 実行環境）でも同様に到達可能かは別途確認が必要（本 PoC の対象外）。

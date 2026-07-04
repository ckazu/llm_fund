# llm_fund

LLM がファンドマネージャー役として売買指示（IFO 発注案）を生成し、人間が実際の発注を執行する
投資判断支援 CLI。詳細仕様は [docs/technical-spec.md](docs/technical-spec.md)、要件は
[docs/requirements.md](docs/requirements.md) を参照。

## アーキテクチャ: 2ライン構成

`fund` は「レポートライン」と「トレードライン」の2系統に分かれている。

- **レポートライン**（`fund report`）: 価格取得 → 指標計算 → Markdown ブリーフィング生成のみ。
  LLM 判断を伴わず、`report: true` な全ユニバース（日本株・米国株どちらも）を対象にできる。
  市場クローズ後の状況共有目的で、市場ごとに個別実行することを想定。
- **トレードライン**（`fund daily` に内包）: レポートラインに加え、`trade: true` なユニバースに
  対して LLM 自己一致性判断（`n_samples` 回サンプリングし多数決/不一致率を算出）→ validator の
  ハード拒否ルール適用 → `instructions` テーブルへの記録まで行う。実発注はしない
  （常に人間が IFO 注文として執行し、`fund record` で結果を記録する）。

判断層（`judgment/`）は記録・配信層（`store/`, `delivery/`）を import しない一方向依存で、
LLM に発注・DB書込み・通知の権限を構造的に持たせていない。

## セットアップ

```bash
uv sync
cp .env.example .env   # なければ新規作成: ANTHROPIC_API_KEY, NOTIFY_WEBHOOK_URL(任意) を設定
```

`config/default.yaml` と `config/universes.yaml` を環境に合わせて用意する（キー詳細は
[docs/technical-spec.md](docs/technical-spec.md) 9章）。

## コマンド一覧

| コマンド | 説明 |
|---|---|
| `fund report [universe] [--date] [--format json]` | レポートラインのみ実行。省略時は `report: true` の全ユニバース |
| `fund daily [--no-llm] [--date] [--format json]` | レポート + トレードライン（判断→検証→記録）。`--no-llm` はテンプレート判断（動作確認用） |
| `fund weekly [--date] [--format json]` | 直近1週間の振り返り＋基準変更の提案（`status=draft` で保存、要承認） |
| `fund monthly [--date] [--format json]` | 直近1ヶ月の方針/ユニバース入替の提案（`status=draft` で保存、要承認） |
| `fund approve <criteria:id \| policy:id>` | 週次/月次の提案を承認し有効化（再承認でロールバック） |
| `fund record <ticket_no> --price P [--units N] [--skipped REASON] [--commission C]` | 人間が実際に執行した結果を記録 |
| `fund benchmark [--date] [--format json]` | LLM判断 vs. 指数/等金額/モメンタム/ランダム対照群のNAV比較 |
| `fund fetch [universe] [--days N]` | 価格データの取得/更新（ローカルSQLiteキャッシュへの差分同期） |
| `fund status [--format json]` | データ鮮度・最新NAV・未記録の指示・承認待ち提案の一括確認 |

全コマンドは `--format json` で `llm_company` 等の外部連携向けに構造化 JSON を標準出力へ
出力できる（既定は Markdown）。終了コードは 正常=0 / NO_TRADE=0 / データ異常=2 /
設定異常=3（cron監視向け、詳細は [docs/runbook.md](docs/runbook.md)）。

## 通知

`.env` に `NOTIFY_WEBHOOK_URL`（Discord/Slack 互換 Incoming Webhook）を設定すると、
`report`/`daily`/`weekly`/`monthly` の実行結果（指示・NO_TRADE・エラー）を自動通知する。
未設定時は通知をスキップし、Webhook 自体の失敗も本処理には影響しない（best-effort）。

## 開発

```bash
uv run pytest          # テスト（外部API=yfinance/anthropic/webhookは全モック）
uv run ruff check src tests
uv run mypy src
```

TDD・小さく安全なコミットを基本方針とする。運用（cron設定・障害対応・承認フロー）の詳細は
[docs/runbook.md](docs/runbook.md) を参照。

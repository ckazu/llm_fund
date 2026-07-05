# llm_fund 運用 Runbook

対象: `fund` CLI（[technical-spec.md](technical-spec.md) 準拠）を cron 等から定期実行する運用担当者。

## 1. 前提

- LLM 判断を使う `daily`/`weekly`/`monthly` には、`config/default.yaml` の `llm.backends` に
  応じて `claude` コマンド（Claude Code CLI、ログイン済み）または OpenAI 互換ローカル LLM
  サーバ（mlx_lm.server / Ollama / LM Studio 等）が利用可能であること。`.env` には任意で
  `NOTIFY_WEBHOOK_URL`（Discord/Slack 互換 Incoming Webhook URL）と `LOCAL_LLM_API_KEY`
  （ローカルサーバが認証を要求する場合）を設定する。
- `config/default.yaml` / `config/universes.yaml` を環境に合わせて配置する。
- 実行前に `uv sync` 済みであること。cron からは `uv run fund ...` で起動する。

## 2. cron 設定例

市場別のタイムゾーンに合わせ、レポートラインとトレードラインを分離してスケジュールする。
`crontab -e` の例（サーバーの TZ が `Asia/Tokyo` である前提。異なる場合は `TZ=` を各行に付与）:

```cron
# --- レポートライン（fund report）: 市場クローズ後、その市場のみ対象に実行 ---
# 日本株市場（東証クローズ後 15:30 JST）
30 15 * * 1-5 cd /opt/llm_fund && uv run fund report jp_stocks >> logs/report_jp.log 2>&1
# 米国株市場（米国市場クローズ後 = 翌日 06:00 JST、サマータイム時は 05:00 JST）
0 6 * * 2-6 cd /opt/llm_fund && uv run fund report us_stocks >> logs/report_us.log 2>&1

# --- トレードライン（fund daily）: 日本の平日朝、寄り付き前に判断・記録 ---
0 8 * * 1-5 cd /opt/llm_fund && uv run fund daily >> logs/daily.log 2>&1

# --- 振り返り（fund weekly）: 週末（土曜朝、当週の全営業日データが揃ってから） ---
0 9 * * 6 cd /opt/llm_fund && uv run fund weekly >> logs/weekly.log 2>&1

# --- 月次レビュー（fund monthly）: 月初（前月の全営業日データが確定した1日の朝） ---
0 9 1 * * cd /opt/llm_fund && uv run fund monthly >> logs/monthly.log 2>&1
```

補足:
- `fund report` は判断を行わないためレート制限やコストの心配がなく、市場ごとに個別実行してよい。
- `fund daily` は `report` 対象の全ユニバースを内包して実行するため、`report` の cron 行と時間が
  重複しても機能上の問題はない（レポートは冪等に上書きされる）。両方仕込むのは通知チャンネルを
  「市場クローズ速報」と「翌朝の売買判断」で分けたい場合の運用上の理由による。
- 祝日はカレンダーを持たないため、休場日に実行してもデータ鮮度ゲート（終了コード2）で
  空振りするだけで実害はない。

## 3. 終了コードと監視

`technical-spec.md` 2章の規約:

| 終了コード | 意味 | cron 監視での扱い |
|---|---|---|
| 0 | 正常（NO_TRADE を含む） | ログのみ確認 |
| 2 | データ異常（鮮度違反等） | アラート。yfinance 障害/休場日を確認 |
| 3 | 設定異常（`.env`/config 不備、APIキー未設定） | アラート。設定ファイルを確認、即修正 |

cron のジョブランナー（cronjob 監視ツール、`||` での通知コマンド連結等）でこれらを検知する。
`NOTIFY_WEBHOOK_URL` を設定していれば、`report`/`daily`/`weekly`/`monthly` は成功時のサマリ、
データ鮮度異常時はエラーメッセージを Discord/Slack に自動通知する（Webhook 自体の失敗は
本処理を止めない best-effort）。

## 4. 障害対応

### 4.1 データ異常（終了コード2: 鮮度違反）

1. `logs/*.log` で `stale:` メッセージの対象銘柄・日付を確認する。
2. yfinance 側の障害か、対象市場が休場日かを確認する（休場なら対応不要、翌営業日に再実行）。
3. yfinance が復旧した場合は `uv run fund fetch <universe>` を手動実行してキャッシュを更新し、
   その後 `fund report`/`fund daily` を再実行する。
4. 恒常的に特定銘柄が取得できない場合は `config/universes.yaml` からの除外を検討し、
   `fund monthly` のユニバース入替提案フローに合流させる。

### 4.2 設定異常（終了コード3）

1. `claude` コマンドの不在・未ログイン、`llm.roles` が参照する backend 名の定義漏れ、
   `config/default.yaml`/`config/universes.yaml` の欠落・YAML構文エラーを確認する。
2. `limits.*` が `validator/rules.py` の絶対上限を超えていないか確認する
   （超過時は `ConfigError` として起動時に検出される）。
3. 修正後、該当コマンドを手動で再実行して終了コード0を確認する。

### 4.3 Webhook 通知が届かない

- `NOTIFY_WEBHOOK_URL` 未設定なら通知はそもそも送られない（仕様どおりの no-op）。
- Webhook 送信失敗は本処理を止めないため、CLI 自体は正常終了する。届かない場合は
  Webhook URL の有効期限切れ/権限変更を疑い、Discord/Slack 側で URL を再発行する。
- 通知経路が使えない間も `reports/YYYY-MM-DD-<kind>.md` ファイルと DB
  （`instructions`/`audit_events` 等）には結果が残っているため、`fund status` で代替確認できる。

### 4.4 LLM 呼び出し障害（レート制限・タイムアウト等）

- `daily`/`weekly`/`monthly` は `llm_calls` テーブルに全呼び出しを記録する
  （リクエスト/レスポンス/コスト）。異常時はそこで原因を追える。
- `daily` は `--no-llm` でテンプレート判断に切り替えて運用を継続できる（暫定回避）。
- `weekly`/`monthly` にテンプレート回避はない。API キー/レート制限が復旧してから再実行する
  （既存の draft 提案は重複して溜まるだけで実害はないが、不要な draft は承認しなければ
  実行中の方針に影響しない）。

## 5. 承認フロー手順（FR-6）

1. `fund weekly`（基準変更）または `fund monthly`（方針変更）が提案を生成すると、
   標準出力に提案内容（diff + rationale）が表示され、`criteria`/`policies` テーブルに
   `status=draft` で保存される。
2. `fund status` で承認待ちの提案一覧（`fund approve <kind>:<id>` の形の実行例つき）を確認する。
3. 内容を人間がレビューし、妥当なら承認する:
   ```
   uv run fund approve criteria:<id>   # 週次基準変更
   uv run fund approve policy:<id>     # 月次方針変更
   ```
   承認すると当該提案が `status=active` になり、直前の active 版は `status=superseded` に
   遷移する。
4. ロールバックしたい場合は、superseded になった過去の提案 id に対して再度
   `fund approve <kind>:<id>` を実行する。これによりその版が再び active になり、
   現在の active 版が superseded に戻る。
5. `fund monthly` が生成するユニバース入替提案（`config/universes.yaml` の追加/除外候補）は
   `audit_events`（`kind=universe_change_proposed`）に記録されるのみで自動反映されない。
   人間が内容を読み、妥当なら `config/universes.yaml` を手動編集して反映する。

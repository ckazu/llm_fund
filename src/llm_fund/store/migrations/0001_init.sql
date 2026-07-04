-- 初期スキーマ（technical-spec.md 3章）
-- 設計原則: 全テーブル INTEGER PRIMARY KEY AUTOINCREMENT のサロゲートキー。
-- ナチュラルキーは UNIQUE 制約のみ。人間に露出する識別子は ticket_no 等の外部識別子。

CREATE TABLE universes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT NOT NULL UNIQUE,
    market          TEXT NOT NULL,
    report_enabled  INTEGER NOT NULL DEFAULT 1,
    trade_enabled   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE instruments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    market      TEXT NOT NULL,
    lot_size    INTEGER NOT NULL DEFAULT 100,
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE universe_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    universe_id     INTEGER NOT NULL REFERENCES universes(id),
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id),
    UNIQUE(universe_id, instrument_id)
);

-- 指標計算は adj_close（調整後）、注文価格は close（非調整）を使用
CREATE TABLE candles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id),
    date            TEXT NOT NULL,
    open            REAL NOT NULL,
    high            REAL NOT NULL,
    low             REAL NOT NULL,
    close           REAL NOT NULL,
    volume          INTEGER NOT NULL,
    adj_close       REAL NOT NULL,
    UNIQUE(instrument_id, date)
);

-- ポートフォリオ状態（仮想執行が維持する。真実の源泉）
CREATE TABLE portfolio_state (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL UNIQUE,
    cash    REAL NOT NULL,
    nav     REAL NOT NULL,
    note    TEXT
);

CREATE TABLE positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id),
    units           INTEGER NOT NULL,
    avg_cost        REAL NOT NULL,
    opened_at       TEXT NOT NULL,
    closed_at       TEXT
);

-- 月次方針
CREATE TABLE policies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    effective_from  TEXT NOT NULL,
    content         TEXT NOT NULL,
    status          TEXT NOT NULL,
    approved_at     TEXT
);

-- 週次基準（ロールバック可能）
CREATE TABLE criteria (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    effective_from  TEXT NOT NULL,
    content         TEXT NOT NULL,
    diff            TEXT,
    rationale       TEXT,
    status          TEXT NOT NULL,
    approved_at     TEXT,
    superseded_by   INTEGER REFERENCES criteria(id)
);

CREATE TABLE briefings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    universe_id         INTEGER NOT NULL REFERENCES universes(id),
    date                TEXT NOT NULL,
    kind                TEXT NOT NULL,
    content_md          TEXT NOT NULL,
    data_snapshot_json  TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

-- ticket_no は人間向け識別子（例: "20260704-01"）。内部 PK (id) は露出しない
CREATE TABLE instructions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no               TEXT NOT NULL UNIQUE,
    briefing_id             INTEGER NOT NULL REFERENCES briefings(id),
    instrument_id           INTEGER NOT NULL REFERENCES instruments(id),
    action                  TEXT NOT NULL,
    units                   INTEGER NOT NULL,
    entry_price             REAL NOT NULL,
    tp_price                REAL NOT NULL,
    sl_price                REAL NOT NULL,
    valid_until             TEXT NOT NULL,
    rationale               TEXT NOT NULL,
    validator_result_json   TEXT,
    status                  TEXT NOT NULL
);

-- 未約定 IFO・失効管理
CREATE TABLE pending_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id  INTEGER NOT NULL REFERENCES instructions(id),
    expires_at      TEXT NOT NULL,
    status          TEXT NOT NULL
);

CREATE TABLE executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id  INTEGER NOT NULL REFERENCES instructions(id),
    executed_at     TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    actual_price    REAL NOT NULL,
    actual_units    INTEGER NOT NULL,
    commission      REAL NOT NULL,
    status          TEXT NOT NULL,
    skip_reason     TEXT,
    deviation_note  TEXT
);

CREATE TABLE virtual_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id  INTEGER NOT NULL REFERENCES instructions(id),
    fill_date       TEXT,
    fill_price      REAL,
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    commission      REAL NOT NULL DEFAULT 0,
    slippage        REAL NOT NULL DEFAULT 0,
    pnl             REAL
);

-- fund / index / equal_weight / momentum / random
CREATE TABLE strategies (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    TEXT NOT NULL UNIQUE,
    name    TEXT NOT NULL
);

-- metrics_json: MaxDD, Sharpe, 勝率, 回転率, コスト比率
-- 対照群の追加（ニュースシャッフル版等）は行追加のみで対応
CREATE TABLE benchmark_navs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    date            TEXT NOT NULL,
    nav             REAL NOT NULL,
    metrics_json    TEXT,
    UNIQUE(strategy_id, date)
);

-- 監査: 全 LLM 呼び出し
CREATE TABLE llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    kind                TEXT NOT NULL,
    model               TEXT NOT NULL,
    temperature         REAL NOT NULL,
    prompt_version      TEXT NOT NULL,
    schema_version      INTEGER NOT NULL,
    sample_index        INTEGER NOT NULL,
    policy_id           INTEGER REFERENCES policies(id),
    criteria_id         INTEGER REFERENCES criteria(id),
    briefing_id         INTEGER REFERENCES briefings(id),
    prompt              TEXT NOT NULL,
    response            TEXT,
    token_usage_json    TEXT
);

-- 拒否/警告/NO_TRADE/承認/ロールバック
CREATE TABLE audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    detail_json TEXT NOT NULL
);

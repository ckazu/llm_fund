"""Discord/Slack-compatible webhook notification (technical-spec.md 2, 8章).

Both platforms accept the same minimal payload shape (`{"content": "..."}`) on
their incoming-webhook endpoints, so a single client covers both without
platform-specific branching. Notification is best-effort: `NOTIFY_WEBHOOK_URL`
is optional, and any failure (network, non-2xx, timeout) is swallowed so a
broken webhook never blocks or crashes report/daily/weekly/monthly (those own
the durable output — the Markdown file / DB rows already succeeded before
notification is attempted).
"""

import httpx

# Discord/Slack はメッセージ本文の長さ上限を持つため、詳細はレポートファイル/DBに
# 委ね、通知には要約のみを送る（技術仕様2章）。
MAX_CONTENT_LENGTH = 1800
WEBHOOK_TIMEOUT_SECONDS = 10.0


def send_webhook_notification(webhook_url: str | None, content: str) -> bool:
    """POST `content` to `webhook_url` as `{"content": ...}`. Returns success as bool.

    No-op (returns False) when `webhook_url` is None/empty. Never raises --
    callers should not need to handle webhook failures.
    """
    if not webhook_url:
        return False

    truncated = (
        content
        if len(content) <= MAX_CONTENT_LENGTH
        else content[: MAX_CONTENT_LENGTH - 1] + "…"
    )
    try:
        response = httpx.post(
            webhook_url, json={"content": truncated}, timeout=WEBHOOK_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False

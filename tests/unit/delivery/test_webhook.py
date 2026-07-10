"""Tests for `delivery/webhook.py` (S10): best-effort Discord/Slack notification."""

import json

import httpx
import respx

from llm_fund.delivery.webhook import MAX_CONTENT_LENGTH, send_webhook_notification

WEBHOOK_URL = "https://discord.com/api/webhooks/123/abc"


class TestSendWebhookNotification:
    def test_no_url_is_a_noop(self) -> None:
        assert send_webhook_notification(None, "hello") is False

    def test_empty_url_is_a_noop(self) -> None:
        assert send_webhook_notification("", "hello") is False

    @respx.mock
    def test_posts_content_json_on_success(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(204))

        result = send_webhook_notification(WEBHOOK_URL, "こんにちは")

        assert result is True
        assert route.called
        sent = json.loads(route.calls.last.request.content)
        assert sent == {"content": "こんにちは"}

    @respx.mock
    def test_truncates_long_content(self) -> None:
        respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(204))
        long_content = "x" * (MAX_CONTENT_LENGTH + 500)

        send_webhook_notification(WEBHOOK_URL, long_content)

        sent = json.loads(respx.calls.last.request.content)
        assert len(sent["content"]) == MAX_CONTENT_LENGTH

    @respx.mock
    def test_returns_false_on_http_error_without_raising(self) -> None:
        respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(500))

        result = send_webhook_notification(WEBHOOK_URL, "hello")

        assert result is False

    @respx.mock
    def test_returns_false_on_network_error_without_raising(self) -> None:
        respx.post(WEBHOOK_URL).mock(side_effect=httpx.ConnectError("boom"))

        result = send_webhook_notification(WEBHOOK_URL, "hello")

        assert result is False

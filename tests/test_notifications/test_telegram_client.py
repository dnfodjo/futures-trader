"""Tests for the TelegramClient — async Telegram Bot API wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.notifications.telegram_client import TelegramClient


@pytest.fixture
def client():
    return TelegramClient(
        bot_token="123:FAKE",
        chat_id="456",
        throttle_sec=0.0,  # No throttle for tests
        max_retries=2,
        retry_delay_sec=0.01,
    )


@pytest.fixture
def unconfigured_client():
    return TelegramClient(bot_token="", chat_id="")


# ── Configuration ────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_configured(self, client):
        assert client.is_configured is True

    def test_not_configured_no_token(self, unconfigured_client):
        assert unconfigured_client.is_configured is False

    def test_not_configured_no_chat_id(self):
        c = TelegramClient(bot_token="123:FAKE", chat_id="")
        assert c.is_configured is False


# ── Send: Not Configured ────────────────────────────────────────────────────


class TestSendNotConfigured:
    @pytest.mark.asyncio
    async def test_send_returns_error(self, unconfigured_client):
        result = await unconfigured_client.send("test")
        assert result["ok"] is False
        assert result["error"] == "not_configured"


# ── Send: Success ────────────────────────────────────────────────────────────


class TestSendSuccess:
    @pytest.mark.asyncio
    async def test_send_success(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 42}}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("Hello!")

        assert result["ok"] is True
        assert result["message_id"] == 42
        assert client.stats["messages_sent"] == 1

    @pytest.mark.asyncio
    async def test_send_truncates_long_message(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        long_text = "A" * 5000
        await client.send(long_text)

        # Verify the payload was truncated
        call_args = mock_session.post.call_args
        sent_text = call_args.kwargs.get("json", call_args[1].get("json", {}))["text"]
        assert len(sent_text) <= 4096
        assert sent_text.endswith("... (truncated)")


# ── Send: API Error ──────────────────────────────────────────────────────────


class TestSendAPIError:
    @pytest.mark.asyncio
    async def test_api_error(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.json = AsyncMock(
            return_value={"ok": False, "description": "Bad Request: chat not found"}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("test")

        assert result["ok"] is False
        assert "chat not found" in result["error"]
        assert client.stats["messages_failed"] == 1

    @pytest.mark.asyncio
    async def test_rate_limited(self, client):
        """When Telegram returns 429, client should wait and retry."""
        # First call: 429, second call: 200
        mock_resp_429 = AsyncMock()
        mock_resp_429.status = 429
        mock_resp_429.json = AsyncMock(
            return_value={
                "ok": False,
                "parameters": {"retry_after": 0.01},
                "description": "Too Many Requests",
            }
        )
        mock_resp_429.__aenter__ = AsyncMock(return_value=mock_resp_429)
        mock_resp_429.__aexit__ = AsyncMock(return_value=False)

        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 99}}
        )
        mock_resp_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
        mock_resp_200.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        # Return 429 first, then 200
        mock_session.post = MagicMock(
            side_effect=[mock_resp_429, mock_resp_200]
        )
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("test")

        assert result["ok"] is True
        assert client.stats["retries_attempted"] == 1


# ── Send: Network Error ─────────────────────────────────────────────────────


class TestSendNetworkError:
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, client):
        """Network errors should trigger retry with backoff."""
        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
        mock_resp_200.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        # First call raises, second succeeds
        mock_session.post = MagicMock(
            side_effect=[
                aiohttp.ClientError("Connection refused"),
                mock_resp_200,
            ]
        )
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("test")

        assert result["ok"] is True
        assert client.stats["retries_attempted"] == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, client):
        """After all retries fail, return error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("test")

        assert result["ok"] is False
        assert "Connection refused" in result["error"]
        assert client.stats["messages_failed"] == 1

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self, client):
        """Timeout errors should also trigger retry."""
        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
        mock_resp_200.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=[asyncio.TimeoutError(), mock_resp_200]
        )
        mock_session.closed = False

        client._session = mock_session

        result = await client.send("test")

        assert result["ok"] is True


# ── Throttle ─────────────────────────────────────────────────────────────────


class TestThrottle:
    @pytest.mark.asyncio
    async def test_throttle_delays_second_message(self):
        """Second message within throttle window should be delayed."""
        client = TelegramClient(
            bot_token="123:FAKE",
            chat_id="456",
            throttle_sec=0.05,  # 50ms throttle
            max_retries=0,
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        await client.send("msg 1")
        await client.send("msg 2")  # Should be throttled

        assert client.stats["messages_sent"] == 2
        assert client.stats["messages_throttled"] == 1

    @pytest.mark.asyncio
    async def test_priority_bypasses_throttle(self):
        """Priority messages should bypass throttle."""
        client = TelegramClient(
            bot_token="123:FAKE",
            chat_id="456",
            throttle_sec=10.0,  # Long throttle
            max_retries=0,
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        await client.send("msg 1")
        await client.send("priority!", priority=True)  # Should NOT be throttled

        assert client.stats["messages_sent"] == 2
        assert client.stats["messages_throttled"] == 0


# ── Close ────────────────────────────────────────────────────────────────────


class TestClose:
    @pytest.mark.asyncio
    async def test_close_session(self, client):
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False
        client._session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_no_session(self, client):
        await client.close()  # No error when no session


# ── Message Queue ────────────────────────────────────────────────────────────


class TestMessageQueue:
    def test_enqueue(self, client):
        client.enqueue("msg 1")
        client.enqueue("msg 2")
        assert client.queue_size == 2
        assert client.stats["messages_queued"] == 2

    @pytest.mark.asyncio
    async def test_flush_empty_queue(self, client):
        result = await client.flush_queue()
        assert result["ok"] is True
        assert result["flushed"] == 0

    @pytest.mark.asyncio
    async def test_flush_combines_messages(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 1}}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        client.enqueue("Update A")
        client.enqueue("Update B")
        client.enqueue("Update C")

        result = await client.flush_queue()

        assert result["ok"] is True
        assert result["flushed"] == 3
        assert client.queue_size == 0
        assert client.stats["messages_batched"] == 3

        # Verify combined message was sent
        call_args = mock_session.post.call_args
        sent_text = call_args.kwargs.get("json", call_args[1].get("json", {}))["text"]
        assert "Update A" in sent_text
        assert "Update B" in sent_text
        assert "Update C" in sent_text

    @pytest.mark.asyncio
    async def test_queue_max_size(self):
        client = TelegramClient(
            bot_token="123:FAKE",
            chat_id="456",
            max_queue_size=3,
        )
        for i in range(5):
            client.enqueue(f"msg {i}")
        # deque maxlen=3, so oldest are dropped
        assert client.queue_size == 3


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self, client):
        stats = client.stats
        assert stats["is_configured"] is True
        assert stats["messages_sent"] == 0
        assert stats["messages_throttled"] == 0
        assert stats["messages_failed"] == 0
        assert stats["retries_attempted"] == 0
        assert stats["messages_queued"] == 0
        assert stats["messages_batched"] == 0
        assert stats["queue_size"] == 0

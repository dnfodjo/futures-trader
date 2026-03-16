"""Telegram Bot API client — async notification delivery.

Uses aiohttp directly to call the Telegram Bot API. No extra dependency.

Features:
- Async message sending with retry on transient failures
- Message throttling (configurable min interval between sends)
- Priority bypass for critical alerts (kill switch, connection loss)
- HTML parse mode for rich formatting
- Message queue for burst handling
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Optional

import aiohttp
import structlog

logger = structlog.get_logger()

_TELEGRAM_API = "https://api.telegram.org"


class TelegramClient:
    """Async Telegram Bot API client for trading notifications.

    Usage:
        client = TelegramClient(bot_token="123:ABC", chat_id="456")
        await client.send("Hello from the trading system!")
        await client.send("<b>ALERT</b>: Kill switch triggered!", priority=True)

        # Queue non-priority messages for batching
        client.enqueue("Minor update 1")
        client.enqueue("Minor update 2")
        await client.flush_queue()  # Sends as single combined message

        await client.close()
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        throttle_sec: float = 10.0,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
        max_message_length: int = 4096,
        max_queue_size: int = 50,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._throttle_sec = throttle_sec
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec
        self._max_message_length = max_message_length
        self._max_queue_size = max_queue_size

        self._session: Optional[aiohttp.ClientSession] = None
        self._last_send_time: float = 0.0
        self._send_lock = asyncio.Lock()

        # Message queue for batching non-priority messages
        self._queue: deque[str] = deque(maxlen=max_queue_size)
        self._messages_queued: int = 0
        self._messages_batched: int = 0

        # Stats
        self._messages_sent: int = 0
        self._messages_throttled: int = 0
        self._messages_failed: int = 0
        self._retries_attempted: int = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def send(
        self,
        text: str,
        priority: bool = False,
        parse_mode: str = "HTML",
        disable_preview: bool = True,
    ) -> dict[str, Any]:
        """Send a message via Telegram Bot API.

        Args:
            text: Message text (supports HTML formatting).
            priority: If True, bypasses throttle (for kill switch alerts).
            parse_mode: Telegram parse mode ("HTML" or "Markdown").
            disable_preview: Disable link previews in message.

        Returns:
            Dict with send result: {"ok": True/False, ...}
        """
        if not self._bot_token or not self._chat_id:
            logger.warning("telegram.not_configured")
            return {"ok": False, "error": "not_configured"}

        # Truncate if too long
        if len(text) > self._max_message_length:
            text = text[: self._max_message_length - 20] + "\n\n... (truncated)"

        async with self._send_lock:
            # Throttle check (skip for priority messages)
            if not priority:
                now = time.monotonic()
                elapsed = now - self._last_send_time
                if elapsed < self._throttle_sec:
                    wait_time = self._throttle_sec - elapsed
                    self._messages_throttled += 1
                    logger.debug(
                        "telegram.throttled",
                        wait_sec=round(wait_time, 1),
                    )
                    await asyncio.sleep(wait_time)

            # Send with retry
            result = await self._send_with_retry(text, parse_mode, disable_preview)

            if result.get("ok"):
                self._messages_sent += 1
                self._last_send_time = time.monotonic()
            else:
                self._messages_failed += 1

            return result

    async def _send_with_retry(
        self,
        text: str,
        parse_mode: str,
        disable_preview: bool,
    ) -> dict[str, Any]:
        """Send with exponential backoff retry on transient failures."""
        url = f"{_TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1 + self._max_retries):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if resp.status == 200 and data.get("ok"):
                        return {"ok": True, "message_id": data["result"]["message_id"]}

                    # Rate limited by Telegram
                    if resp.status == 429:
                        retry_after = data.get("parameters", {}).get(
                            "retry_after", 5
                        )
                        logger.warning(
                            "telegram.rate_limited",
                            retry_after=retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        self._retries_attempted += 1
                        continue

                    # Other API error — don't retry
                    logger.error(
                        "telegram.api_error",
                        status=resp.status,
                        error=data.get("description", "unknown"),
                    )
                    return {
                        "ok": False,
                        "error": data.get("description", "unknown"),
                        "status": resp.status,
                    }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_delay_sec * (2**attempt)
                    self._retries_attempted += 1
                    logger.warning(
                        "telegram.retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            "telegram.send_failed",
            error=str(last_error),
            retries=self._max_retries,
        )
        return {"ok": False, "error": str(last_error)}

    def enqueue(self, text: str) -> None:
        """Add a non-priority message to the queue for batched sending.

        Queued messages are combined and sent as a single message
        when flush_queue() is called. Useful for minor updates that
        don't need immediate delivery.

        Args:
            text: Message text to queue.
        """
        self._queue.append(text)
        self._messages_queued += 1

    async def flush_queue(self) -> dict[str, Any]:
        """Flush all queued messages as a single combined message.

        Returns:
            Send result, or {"ok": True, "flushed": 0} if queue was empty.
        """
        if not self._queue:
            return {"ok": True, "flushed": 0}

        messages = list(self._queue)
        self._queue.clear()
        count = len(messages)

        combined = "\n\n\u2500\u2500\u2500\n\n".join(messages)
        self._messages_batched += count

        result = await self.send(combined)
        result["flushed"] = count
        return result

    @property
    def queue_size(self) -> int:
        """Number of messages waiting in the queue."""
        return len(self._queue)

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @property
    def is_configured(self) -> bool:
        """Whether bot token and chat ID are set."""
        return bool(self._bot_token) and bool(self._chat_id)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "is_configured": self.is_configured,
            "messages_sent": self._messages_sent,
            "messages_throttled": self._messages_throttled,
            "messages_failed": self._messages_failed,
            "retries_attempted": self._retries_attempted,
            "messages_queued": self._messages_queued,
            "messages_batched": self._messages_batched,
            "queue_size": len(self._queue),
        }

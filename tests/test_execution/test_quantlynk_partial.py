"""Tests for QuantLynk partial_close method.

TDD: Written before implementation to define expected behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import QuantLynkConfig
from src.execution.quantlynk_client import QuantLynkClient


class TestQuantLynkPartialClose:
    """Tests for the partial_close method on QuantLynkClient."""

    def _make_client(self, **overrides) -> QuantLynkClient:
        defaults = {
            "webhook_url": "https://quantlynk.io/webhook/test-123",
            "user_id": "test-user",
            "alert_id": "test-alert",
            "timeout_sec": 5.0,
            "max_retries": 1,
            "enabled": True,
        }
        defaults.update(overrides)
        config = QuantLynkConfig(**defaults)
        return QuantLynkClient(config)

    def _mock_response(self, status: int = 200, text: str = "OK"):
        resp = AsyncMock()
        resp.status = status
        resp.text = AsyncMock(return_value=text)
        return resp

    def _wire_mock_session(self, client, status=200, text="OK"):
        mock_resp = self._mock_response(status, text)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session
        return mock_session

    @pytest.mark.asyncio
    async def test_partial_close_long_sends_sell(self):
        """Partial close on a long position should send action='sell'."""
        client = self._make_client()
        mock_session = self._wire_mock_session(client)

        result = await client.partial_close(side="long", quantity=1, price=115.0)

        assert result["status"] == "sent"
        assert result["action"] == "sell"
        assert result["quantity"] == 1

        payload = mock_session.post.call_args.kwargs.get("json") or \
                  mock_session.post.call_args[1].get("json")
        assert payload["action"] == "sell"
        assert payload["quantity"] == "1"

    @pytest.mark.asyncio
    async def test_partial_close_short_sends_buy(self):
        """Partial close on a short position should send action='buy'."""
        client = self._make_client()
        mock_session = self._wire_mock_session(client)

        result = await client.partial_close(side="short", quantity=1, price=85.0)

        assert result["status"] == "sent"
        assert result["action"] == "buy"

        payload = mock_session.post.call_args.kwargs.get("json") or \
                  mock_session.post.call_args[1].get("json")
        assert payload["action"] == "buy"
        assert payload["quantity"] == "1"

    @pytest.mark.asyncio
    async def test_partial_close_respects_quantity(self):
        """partial_close should send the specified quantity, not flatten all."""
        client = self._make_client()
        mock_session = self._wire_mock_session(client)

        await client.partial_close(side="long", quantity=2, price=120.0)

        payload = mock_session.post.call_args.kwargs.get("json") or \
                  mock_session.post.call_args[1].get("json")
        assert payload["quantity"] == "2"
        assert payload["action"] == "sell"  # NOT flatten

    @pytest.mark.asyncio
    async def test_partial_close_disabled_returns_disabled(self):
        """When client is disabled, partial_close returns disabled status."""
        client = self._make_client(enabled=False)
        result = await client.partial_close(side="long", quantity=1, price=115.0)
        assert result["status"] == "disabled"

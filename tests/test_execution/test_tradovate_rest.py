"""Tests for Tradovate REST client.

Uses mocks for HTTP calls — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    InsufficientMarginError,
    OrderModifyFailedError,
    OrderRejectedError,
    TradovateConnectionError,
)
from src.execution.rate_limiter import RateLimiter
from src.execution.tradovate_auth import TradovateAuth
from src.execution.tradovate_rest import TradovateREST


@pytest.fixture
def mock_auth():
    auth = MagicMock(spec=TradovateAuth)
    auth.access_token = "test-token"
    auth.base_url = "https://demo.tradovateapi.com/v1"
    auth.account_id = 100
    auth.account_spec = "DEMO12345"
    return auth


@pytest.fixture
def limiter():
    return RateLimiter(budget=100, window_seconds=60)


@pytest.fixture
def rest(mock_auth, limiter):
    return TradovateREST(mock_auth, limiter)


def _mock_session(response_data: dict | list, status: int = 200):
    """Create a mock aiohttp session that returns given data."""
    mock_session = MagicMock()

    mock_ctx = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.status = status
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)

    mock_session.post = MagicMock(return_value=mock_ctx)
    mock_session.get = MagicMock(return_value=mock_ctx)
    mock_session.closed = False

    return mock_session


class TestPlaceMarketOrder:
    async def test_successful_market_order(self, rest: TradovateREST):
        response = {
            "orderId": 12345,
            "ordStatus": "Working",
            "action": "Buy",
            "symbol": "MNQM6",
            "orderQty": 3,
        }
        rest._session = _mock_session(response)

        result = await rest.place_market_order("MNQM6", "Buy", 3)
        assert result["orderId"] == 12345
        assert result["ordStatus"] == "Working"

    async def test_market_order_rejected(self, rest: TradovateREST):
        response = {"errorText": "Account is restricted"}
        rest._session = _mock_session(response)

        with pytest.raises(OrderRejectedError, match="restricted"):
            await rest.place_market_order("MNQM6", "Buy", 3)

    async def test_market_order_insufficient_margin(self, rest: TradovateREST):
        response = {"errorText": "Insufficient margin"}
        rest._session = _mock_session(response)

        with pytest.raises(InsufficientMarginError):
            await rest.place_market_order("MNQM6", "Buy", 3)


class TestPlaceBracketOrder:
    async def test_bracket_with_stop_only(self, rest: TradovateREST):
        response = {"orderId": 12345, "ordStatus": "Working"}
        rest._session = _mock_session(response)

        result = await rest.place_bracket_order(
            "MNQM6", "Buy", 3, stop_price=19835.0
        )
        assert result["orderId"] == 12345

        # Verify the payload included bracket1 but not bracket2
        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert "bracket1" in payload
        assert payload["bracket1"]["stopPrice"] == 19835.0
        assert "bracket2" not in payload

    async def test_bracket_with_stop_and_target(self, rest: TradovateREST):
        response = {"orderId": 12345}
        rest._session = _mock_session(response)

        result = await rest.place_bracket_order(
            "MNQM6", "Buy", 3, stop_price=19835.0, take_profit_price=19880.0
        )

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert "bracket2" in payload
        assert payload["bracket2"]["price"] == 19880.0

    async def test_bracket_sell_reverses_exit_action(self, rest: TradovateREST):
        response = {"orderId": 12345}
        rest._session = _mock_session(response)

        await rest.place_bracket_order("MNQM6", "Sell", 2, stop_price=19920.0)

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["bracket1"]["action"] == "Buy"  # Exit for short = Buy

    async def test_bracket_buy_sets_sell_exit(self, rest: TradovateREST):
        response = {"orderId": 12345}
        rest._session = _mock_session(response)

        await rest.place_bracket_order("MNQM6", "Buy", 2, stop_price=19835.0)

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["bracket1"]["action"] == "Sell"  # Exit for long = Sell


class TestPlaceStopOrder:
    async def test_stop_order_includes_gtc(self, rest: TradovateREST):
        response = {"orderId": 12345}
        rest._session = _mock_session(response)

        await rest.place_stop_order("MNQM6", "Sell", 3, 19835.0)

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["timeInForce"] == "GTC"
        assert payload["stopPrice"] == 19835.0


class TestModifyOrder:
    async def test_modify_order(self, rest: TradovateREST):
        response = {"orderId": 999, "stopPrice": 19840.0}
        rest._session = _mock_session(response)

        result = await rest.modify_order(999, quantity=3, stop_price=19840.0)
        assert result["stopPrice"] == 19840.0

    async def test_modify_order_includes_gtc(self, rest: TradovateREST):
        response = {"orderId": 999}
        rest._session = _mock_session(response)

        await rest.modify_order(999, quantity=3, stop_price=19840.0)

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["timeInForce"] == "GTC"
        assert payload["isAutomated"] is True

    async def test_modify_order_error_raises(self, rest: TradovateREST):
        response = {"errorText": "Order not found"}
        rest._session = _mock_session(response)

        with pytest.raises(OrderModifyFailedError, match="Order not found"):
            await rest.modify_order(999, quantity=3, stop_price=19840.0)


class TestCancelOrder:
    async def test_cancel_order_sends_correct_payload(self, rest: TradovateREST):
        response = {"orderId": 999, "ordStatus": "Cancelled"}
        rest._session = _mock_session(response)

        result = await rest.cancel_order(999)
        assert result["ordStatus"] == "Cancelled"

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["orderId"] == 999
        assert payload["isAutomated"] is True


class TestPositionQueries:
    async def test_get_positions(self, rest: TradovateREST):
        positions = [
            {"id": 1, "netPos": 3, "netPrice": 19850.0},
            {"id": 2, "netPos": -2, "netPrice": 19900.0},
        ]
        rest._session = _mock_session(positions)

        result = await rest.get_positions()
        assert len(result) == 2
        assert result[0]["netPos"] == 3

    async def test_get_positions_empty(self, rest: TradovateREST):
        rest._session = _mock_session([])
        result = await rest.get_positions()
        assert result == []

    async def test_get_orders(self, rest: TradovateREST):
        orders = [{"orderId": 123, "ordStatus": "Working"}]
        rest._session = _mock_session(orders)

        result = await rest.get_orders()
        assert len(result) == 1

    async def test_get_cash_balance(self, rest: TradovateREST):
        balance = {"amount": 50000.0, "realizedPnl": 250.0}
        rest._session = _mock_session(balance)

        result = await rest.get_cash_balance()
        assert result["amount"] == 50000.0


class TestLiquidatePosition:
    async def test_liquidate_uses_emergency(self, mock_auth):
        """Liquidation should bypass rate limiter."""
        # Use a limiter with high burst limit for this test
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=200)
        rest = TradovateREST(mock_auth, limiter)

        response = {"orderId": 999}
        rest._session = _mock_session(response)

        # Exhaust rate limiter
        for _ in range(100):
            await limiter.acquire()

        # Liquidate should still work (emergency bypass)
        result = await rest.liquidate_position()
        assert result["orderId"] == 999

    async def test_liquidate_includes_automated_flag(self, rest: TradovateREST):
        response = {}
        rest._session = _mock_session(response)

        await rest.liquidate_position()

        call_args = rest._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["isAutomated"] is True
        assert payload["accountId"] == 100


class TestIsAutomatedFlag:
    """Every order endpoint MUST include isAutomated: True (CME Rule 536-B)."""

    async def test_market_order_automated(self, rest: TradovateREST):
        rest._session = _mock_session({"orderId": 1})
        await rest.place_market_order("MNQM6", "Buy", 1)
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True

    async def test_bracket_order_automated(self, rest: TradovateREST):
        rest._session = _mock_session({"orderId": 1})
        await rest.place_bracket_order("MNQM6", "Buy", 1, stop_price=19835.0)
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True
        assert payload["bracket1"]["isAutomated"] is True

    async def test_stop_order_automated(self, rest: TradovateREST):
        rest._session = _mock_session({"orderId": 1})
        await rest.place_stop_order("MNQM6", "Sell", 1, 19835.0)
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True

    async def test_modify_order_automated(self, rest: TradovateREST):
        rest._session = _mock_session({"orderId": 1})
        await rest.modify_order(1, quantity=1, stop_price=19840.0)
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True

    async def test_cancel_order_automated(self, rest: TradovateREST):
        rest._session = _mock_session({"orderId": 1, "ordStatus": "Cancelled"})
        await rest.cancel_order(1)
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True

    async def test_liquidate_automated(self, rest: TradovateREST):
        rest._session = _mock_session({})
        await rest.liquidate_position()
        payload = rest._session.post.call_args[1]["json"]
        assert payload["isAutomated"] is True


class TestPostNon200:
    """_post should raise on non-200 status codes, not swallow them."""

    async def test_post_400_raises(self, rest: TradovateREST):
        rest._session = _mock_session({"errorText": "Bad request"}, status=400)

        with pytest.raises(TradovateConnectionError, match="400"):
            await rest.place_market_order("MNQM6", "Buy", 1)

    async def test_post_500_raises(self, rest: TradovateREST):
        rest._session = _mock_session({"error": "Internal server error"}, status=500)

        with pytest.raises(TradovateConnectionError, match="500"):
            await rest.place_market_order("MNQM6", "Buy", 1)

    async def test_post_401_raises(self, rest: TradovateREST):
        rest._session = _mock_session({"errorText": "Unauthorized"}, status=401)

        with pytest.raises(TradovateConnectionError, match="401"):
            await rest.place_market_order("MNQM6", "Buy", 1)


class TestModifyOrderWithVerify:
    """Tests for the silent-failure-aware modify path."""

    async def test_successful_modify_with_verify(self, rest: TradovateREST):
        """Should pass when modification is confirmed by GET."""
        # Mock POST to modify
        post_response = {"orderId": 999, "stopPrice": 19840.0}
        # Mock GET to verify — returns the expected stop
        get_response = {"orderId": 999, "stopPrice": 19840.0}

        mock_session = MagicMock()

        # POST returns modify result
        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=post_response)
        mock_post_resp.status = 200
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        # GET returns verified order
        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(return_value=get_response)
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        rest._session = mock_session

        result = await rest.modify_order_with_verify(
            999, quantity=3, stop_price=19840.0
        )
        assert result["stopPrice"] == 19840.0

    async def test_modify_with_verify_detects_silent_failure(self, rest: TradovateREST):
        """Should raise after retry when stop doesn't change."""
        post_response = {"orderId": 999}
        # GET always returns wrong stop price — silent failure
        get_response = {"orderId": 999, "stopPrice": 19830.0}

        mock_session = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=post_response)
        mock_post_resp.status = 200
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(return_value=get_response)
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        rest._session = mock_session

        with pytest.raises(OrderModifyFailedError, match="failed after retry"):
            await rest.modify_order_with_verify(
                999, quantity=3, stop_price=19840.0
            )

        # Should have called POST twice (initial + retry)
        assert mock_session.post.call_count == 2

    async def test_modify_with_verify_retries_once_on_silent_failure(self, rest: TradovateREST):
        """Should retry once and succeed if second attempt works."""
        post_response = {"orderId": 999}

        mock_session = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=post_response)
        mock_post_resp.status = 200
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        # GET returns wrong price first, then correct price
        get_responses = iter([
            {"orderId": 999, "stopPrice": 19830.0},  # First verify — wrong
            {"orderId": 999, "stopPrice": 19840.0},  # Second verify — correct
        ])

        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(side_effect=lambda: next(get_responses))
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        rest._session = mock_session

        result = await rest.modify_order_with_verify(
            999, quantity=3, stop_price=19840.0
        )
        # Should have retried and succeeded
        assert mock_session.post.call_count == 2


class TestRateLimiterIntegration:
    async def test_requests_consume_budget(self, rest: TradovateREST, limiter: RateLimiter):
        rest._session = _mock_session({"orderId": 1})

        await rest.place_market_order("MNQM6", "Buy", 1)
        await rest.place_market_order("MNQM6", "Buy", 1)

        assert limiter.stats["total_requests"] == 2

    async def test_close(self, rest: TradovateREST):
        rest._session = _mock_session({})
        rest._session.close = AsyncMock()
        await rest.close()
        rest._session.close.assert_called_once()

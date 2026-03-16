"""Tests for Tradovate authentication.

Uses aiohttp test utilities and mocks — no real API calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import TradovateConfig
from src.core.exceptions import TradovateConnectionError
from src.execution.tradovate_auth import TradovateAuth


@pytest.fixture
def demo_config():
    return TradovateConfig(
        username="test_user",
        password="test_pass",
        app_id="TestApp",
        app_version="1.0",
        cid=8,
        sec="test_secret",
        device_id="test-device-001",
        use_demo=True,
        account_name="",  # explicit empty — don't leak from .env
    )


@pytest.fixture
def auth(demo_config):
    return TradovateAuth(demo_config)


class TestAuthProperties:
    def test_initial_state(self, auth: TradovateAuth):
        assert auth.is_authenticated is False
        assert auth.user_id is None
        assert auth.account_id is None
        assert auth.account_spec is None

    def test_access_token_before_auth_raises(self, auth: TradovateAuth):
        with pytest.raises(TradovateConnectionError, match="Not authenticated"):
            _ = auth.access_token

    def test_base_url_demo(self, auth: TradovateAuth):
        assert "demo" in auth.base_url

    def test_base_url_live(self):
        cfg = TradovateConfig(use_demo=False)
        a = TradovateAuth(cfg)
        assert "live" in a.base_url


class TestAuthenticate:
    async def test_successful_auth(self, auth: TradovateAuth):
        """Successful auth should set tokens and user info."""
        mock_auth_response = {
            "accessToken": "test-token-123",
            "mdAccessToken": "test-md-token-456",
            "userId": 42,
            "expirationTime": "2026-03-14T12:00:00Z",
        }
        mock_accounts = [{"id": 100, "name": "DEMO12345"}]

        mock_session = MagicMock()
        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(return_value=mock_accounts)
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        auth._session = mock_session

        # Patch _start_refresh_loop to avoid background task
        with patch.object(auth, "_start_refresh_loop"):
            await auth.authenticate()

        assert auth.is_authenticated is True
        assert auth.access_token == "test-token-123"
        assert auth.md_access_token == "test-md-token-456"
        assert auth.user_id == 42
        assert auth.account_id == 100
        assert auth.account_spec == "DEMO12345"

    async def test_auth_with_error_text(self, auth: TradovateAuth):
        """Should raise on errorText in response."""
        mock_response = {
            "errorText": "Invalid credentials",
        }

        mock_session = MagicMock()
        mock_ctx = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session.closed = False

        auth._session = mock_session

        with pytest.raises(TradovateConnectionError, match="Invalid credentials"):
            await auth.authenticate()

    async def test_auth_with_penalty_ticket(self, auth: TradovateAuth):
        """Should raise on penalty ticket."""
        mock_response = {
            "p-ticket": "rate-limit-violation-123",
            "errorText": "",
        }

        mock_session = MagicMock()
        mock_ctx = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session.closed = False

        auth._session = mock_session

        with pytest.raises(TradovateConnectionError, match="penalty ticket"):
            await auth.authenticate()

    async def test_auth_no_access_token(self, auth: TradovateAuth):
        """Should raise when response has no accessToken."""
        mock_response = {"userId": 42}

        mock_session = MagicMock()
        mock_ctx = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session.closed = False

        auth._session = mock_session

        with pytest.raises(TradovateConnectionError, match="No access token"):
            await auth.authenticate()


class TestTokenExpiry:
    def test_token_expires_at_none_before_auth(self, auth: TradovateAuth):
        """Should be None when not authenticated."""
        assert auth.token_expires_at is None

    def test_token_seconds_remaining_zero_before_auth(self, auth: TradovateAuth):
        """Should be 0 when not authenticated."""
        assert auth.token_seconds_remaining == 0.0

    async def test_token_expires_at_set_after_auth(self, auth: TradovateAuth):
        """Should reflect token lifetime after authentication."""
        mock_auth_response = {
            "accessToken": "test-token-123",
            "userId": 42,
        }
        mock_accounts = [{"id": 100, "name": "DEMO12345"}]

        mock_session = MagicMock()
        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(return_value=mock_accounts)
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        auth._session = mock_session

        with patch.object(auth, "_start_refresh_loop"):
            await auth.authenticate()

        assert auth.token_expires_at is not None
        # Token lifetime is 90 min by default; seconds remaining should be close to 5400
        assert auth.token_seconds_remaining > 5300
        assert auth.token_seconds_remaining <= 5400


class TestAuthCallbacks:
    async def test_on_token_refresh_callback(self, auth: TradovateAuth):
        """Callback should receive new token after auth."""
        received_tokens = []

        async def token_callback(token: str):
            received_tokens.append(token)

        auth.on_token_refresh(token_callback)

        mock_auth_response = {
            "accessToken": "test-token-xyz",
            "userId": 42,
        }
        mock_accounts = [{"id": 100, "name": "DEMO12345"}]

        mock_session = MagicMock()
        mock_post_ctx = AsyncMock()
        mock_post_resp = AsyncMock()
        mock_post_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_get_ctx = AsyncMock()
        mock_get_resp = AsyncMock()
        mock_get_resp.json = AsyncMock(return_value=mock_accounts)
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)
        mock_session.closed = False

        auth._session = mock_session

        with patch.object(auth, "_start_refresh_loop"):
            await auth.authenticate()

        assert len(received_tokens) == 1
        assert received_tokens[0] == "test-token-xyz"

    async def test_multiple_callbacks(self, auth: TradovateAuth):
        """Multiple callbacks should all be called."""
        received_a = []
        received_b = []

        async def callback_a(token: str):
            received_a.append(token)

        async def callback_b(token: str):
            received_b.append(token)

        auth.on_token_refresh(callback_a)
        auth.on_token_refresh(callback_b)

        # Simulate internal notify
        await auth._notify_token_refresh("new-token")
        assert len(received_a) == 1
        assert len(received_b) == 1

    async def test_callback_error_doesnt_crash(self, auth: TradovateAuth):
        """A failing callback should not block other callbacks."""
        received = []

        async def bad_callback(token: str):
            raise ValueError("callback error")

        async def good_callback(token: str):
            received.append(token)

        auth.on_token_refresh(bad_callback)
        auth.on_token_refresh(good_callback)

        await auth._notify_token_refresh("token")
        assert len(received) == 1  # Good callback still ran


class TestClose:
    async def test_close_idempotent(self, auth: TradovateAuth):
        """Closing when not connected should be safe."""
        await auth.close()
        assert auth.is_authenticated is False

    async def test_close_cancels_refresh(self, auth: TradovateAuth):
        """Close should cancel the refresh task."""
        cancelled = False

        async def fake_refresh():
            nonlocal cancelled
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled = True
                raise

        task = asyncio.create_task(fake_refresh())
        auth._refresh_task = task

        await auth.close()
        assert task.cancelled() or cancelled


# ── Test: Multi-Account Selection ─────────────────────────────────────────


class TestAccountSelection:
    """Verify account selection when multiple accounts exist."""

    def _mock_auth_with_accounts(
        self, accounts: list[dict], account_name: str = ""
    ) -> TradovateAuth:
        """Create a TradovateAuth with mocked account list response."""
        config = TradovateConfig(
            username="test", password="test", cid=1, sec="s",
            account_name=account_name,
        )
        auth = TradovateAuth(config)
        auth._access_token = "fake"
        auth._authenticated = True

        # Build a proper async context manager mock for session.get()
        mock_resp = MagicMock()
        mock_resp.json = AsyncMock(return_value=accounts)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=ctx)
        mock_session.closed = False
        auth._session = mock_session

        return auth

    @pytest.mark.asyncio
    async def test_selects_first_account_when_no_name_set(self):
        """When TV_ACCOUNT_NAME is empty, use first account."""
        auth = self._mock_auth_with_accounts(
            accounts=[
                {"id": 100, "name": "DEMO-001", "active": True},
                {"id": 200, "name": "APEX-50K-123", "active": True},
            ],
            account_name="",
        )

        await auth._fetch_account_info()

        assert auth.account_id == 100
        assert auth.account_spec == "DEMO-001"

    @pytest.mark.asyncio
    async def test_selects_named_account(self):
        """When TV_ACCOUNT_NAME is set, select that specific account."""
        auth = self._mock_auth_with_accounts(
            accounts=[
                {"id": 100, "name": "DEMO-001", "active": True},
                {"id": 200, "name": "APEX-50K-123", "active": True},
                {"id": 300, "name": "APEX-50K-456", "active": True},
            ],
            account_name="APEX-50K-123",
        )

        await auth._fetch_account_info()

        assert auth.account_id == 200
        assert auth.account_spec == "APEX-50K-123"

    @pytest.mark.asyncio
    async def test_named_account_case_insensitive(self):
        """Account name matching is case-insensitive."""
        auth = self._mock_auth_with_accounts(
            accounts=[
                {"id": 200, "name": "APEX-50K-123", "active": True},
            ],
            account_name="apex-50k-123",  # lowercase
        )

        await auth._fetch_account_info()

        assert auth.account_id == 200

    @pytest.mark.asyncio
    async def test_named_account_not_found_raises(self):
        """If TV_ACCOUNT_NAME doesn't match any account, raise an error."""
        auth = self._mock_auth_with_accounts(
            accounts=[
                {"id": 100, "name": "DEMO-001", "active": True},
                {"id": 200, "name": "APEX-50K-123", "active": True},
            ],
            account_name="NONEXISTENT-999",
        )

        with pytest.raises(TradovateConnectionError, match="NONEXISTENT-999"):
            await auth._fetch_account_info()

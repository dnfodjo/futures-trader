"""Tradovate OAuth 2.0 authentication and token management.

Handles initial authentication, token refresh (every 85 min),
and re-authentication on failure. Thread-safe token access
via asyncio.Lock.

Token lifecycle:
  1. POST /auth/accesstokenrequest → accessToken (90 min lifetime)
  2. At 85 min → GET /auth/renewaccesstoken → new token
  3. On refresh failure → full re-auth
  4. On re-auth failure → raise (caller triggers kill switch)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Callable, Coroutine, Any

import aiohttp
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import TradovateConfig
from src.core.exceptions import TradovateConnectionError

logger = structlog.get_logger()

# Refresh buffer: refresh 5 min before expiry (85 min into 90 min lifetime)
_REFRESH_BUFFER_SEC = 5 * 60

# Type for auth state change callbacks
AuthCallback = Callable[[str], Coroutine[Any, Any, None]]


class TradovateAuth:
    """Manages Tradovate access tokens with auto-refresh.

    Usage:
        auth = TradovateAuth(config)
        auth.on_token_refresh(my_callback)   # notified on token change
        await auth.authenticate()
        token = auth.access_token            # always current
        expires = auth.token_expires_at      # check expiry
        # Later:
        await auth.close()
    """

    def __init__(self, config: TradovateConfig) -> None:
        self._config = config
        self._access_token: str | None = None
        self._md_access_token: str | None = None
        self._user_id: int | None = None
        self._account_id: int | None = None
        self._account_spec: str | None = None
        self._token_obtained_at: datetime | None = None
        self._session: aiohttp.ClientSession | None = None
        self._refresh_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._authenticated = False
        self._token_refresh_callbacks: list[AuthCallback] = []

    # ── Auth State Change Callbacks ──────────────────────────────────────────

    def on_token_refresh(self, callback: AuthCallback) -> None:
        """Register a callback for token refresh events.

        The callback receives the new access token string.
        Used by WebSocket client to update its token on refresh.
        """
        self._token_refresh_callbacks.append(callback)

    async def _notify_token_refresh(self, new_token: str) -> None:
        """Notify all registered callbacks of a token change."""
        for callback in self._token_refresh_callbacks:
            try:
                await callback(new_token)
            except Exception:
                logger.exception("tradovate_auth.token_callback_error")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def access_token(self) -> str:
        """Current access token. Raises if not authenticated."""
        if not self._access_token:
            raise TradovateConnectionError("Not authenticated. Call authenticate() first.")
        return self._access_token

    @property
    def md_access_token(self) -> str | None:
        """Market data access token (if available)."""
        return self._md_access_token

    @property
    def user_id(self) -> int | None:
        return self._user_id

    @property
    def account_id(self) -> int | None:
        return self._account_id

    @property
    def account_spec(self) -> str | None:
        return self._account_spec

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated and self._access_token is not None

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def token_expires_at(self) -> datetime | None:
        """When the current token expires (UTC). None if not authenticated."""
        if self._token_obtained_at is None:
            return None
        lifetime = timedelta(minutes=self._config.token_lifetime_min)
        return self._token_obtained_at + lifetime

    @property
    def token_seconds_remaining(self) -> float:
        """Seconds until token expires. 0 if not authenticated."""
        if self.token_expires_at is None:
            return 0.0
        remaining = (self.token_expires_at - datetime.now(tz=UTC)).total_seconds()
        return max(0.0, remaining)

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    def _auth_headers(self) -> dict[str, str]:
        """Headers with current Bearer token."""
        return {"Authorization": f"Bearer {self._access_token}"}

    async def authenticate(self) -> None:
        """Perform initial authentication with Tradovate.

        Retries up to 3 times on transient network errors. API-level
        errors (bad credentials, penalty tickets) are NOT retried.

        Raises:
            TradovateConnectionError: If authentication fails after retries.
        """
        try:
            await self._authenticate_with_retry()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # After tenacity exhausts retries, wrap as our exception
            raise TradovateConnectionError(f"HTTP error during auth: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _authenticate_with_retry(self) -> None:
        """Inner auth method — tenacity retries network errors here.

        TradovateConnectionError (bad creds, no token) is NOT in the retry
        list, so those fail immediately.
        aiohttp.ClientError and TimeoutError ARE retried up to 3 times.

        Penalty tickets (p-ticket) are handled inline: wait p-time seconds,
        then re-send with the ticket attached. If p-captcha is True, we still
        attempt the ticket retry — the API often accepts it without a real
        captcha solution for non-browser clients.
        """
        session = self._get_session()
        url = f"{self._config.base_url}/auth/accesstokenrequest"

        payload = {
            "name": self._config.username,
            "password": self._config.password,
            "appId": self._config.app_id,
            "appVersion": self._config.app_version,
            "cid": self._config.cid,
            "sec": self._config.sec,
            "deviceId": self._config.device_id,
        }

        logger.info(
            "tradovate_auth.authenticating",
            base_url=self._config.base_url,
            username=self._config.username,
        )

        data = await self._post_auth(session, url, payload)

        # Handle penalty ticket — wait and retry with the ticket
        if data.get("p-ticket"):
            wait_sec = int(data.get("p-time", 15))
            ticket = data["p-ticket"]
            has_captcha = data.get("p-captcha", False)
            logger.warning(
                "tradovate_auth.penalty_ticket",
                wait_sec=wait_sec,
                has_captcha=has_captcha,
            )
            await asyncio.sleep(wait_sec)

            # Retry with the penalty ticket attached
            payload["p-ticket"] = ticket
            data = await self._post_auth(session, url, payload)

            # If we get ANOTHER penalty ticket, fail
            if data.get("p-ticket"):
                raise TradovateConnectionError(
                    "Tradovate penalty ticket not resolved after retry. "
                    "Wait a few minutes and try again, or log out of "
                    "other Tradovate sessions (web, desktop)."
                )

        # Check for error — NOT retried (bad credentials)
        if data.get("errorText"):
            raise TradovateConnectionError(
                f"Tradovate auth failed: {data['errorText']}"
            )

        if "accessToken" not in data:
            raise TradovateConnectionError(
                f"No access token in auth response: {data}"
            )

        async with self._lock:
            self._access_token = data["accessToken"]
            self._md_access_token = data.get("mdAccessToken")
            self._user_id = data.get("userId")
            self._token_obtained_at = datetime.now(tz=UTC)
            self._authenticated = True

        logger.info(
            "tradovate_auth.authenticated",
            user_id=self._user_id,
            has_md_token=self._md_access_token is not None,
        )

        # Notify callbacks of the new token
        await self._notify_token_refresh(self._access_token)

        # Fetch account info
        await self._fetch_account_info()

        # Start refresh loop
        self._start_refresh_loop()

    async def _post_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
    ) -> dict:
        """POST to auth endpoint and return parsed JSON response."""
        async with session.post(url, json=payload) as resp:
            return await resp.json()

    async def _fetch_account_info(self) -> None:
        """Fetch account ID and spec after authentication.

        If TV_ACCOUNT_NAME is set, selects that specific account.
        Otherwise, uses the first account found. Logs all available
        accounts so the user can see what's available.
        """
        session = self._get_session()
        url = f"{self._config.base_url}/account/list"

        try:
            async with session.get(url, headers=self._auth_headers()) as resp:
                accounts = await resp.json()

                if not accounts:
                    logger.warning("tradovate_auth.no_accounts")
                    return

                # Log all available accounts
                logger.info(
                    "tradovate_auth.accounts_available",
                    count=len(accounts),
                    accounts=[
                        {"id": a.get("id"), "name": a.get("name"), "active": a.get("active")}
                        for a in accounts
                    ],
                )

                # Select account: by name if specified, otherwise first
                target_name = self._config.account_name.strip()
                account = None

                if target_name:
                    # Find matching account (case-insensitive)
                    for a in accounts:
                        if a.get("name", "").lower() == target_name.lower():
                            account = a
                            break

                    if account is None:
                        available = [a.get("name", "?") for a in accounts]
                        raise TradovateConnectionError(
                            f"Account '{target_name}' not found. "
                            f"Available accounts: {available}. "
                            f"Set TV_ACCOUNT_NAME to one of these, or leave empty for first account."
                        )
                else:
                    # Default: use first account
                    account = accounts[0]

                self._account_id = account.get("id")
                self._account_spec = account.get("name")

                logger.info(
                    "tradovate_auth.account_selected",
                    account_id=self._account_id,
                    account_spec=self._account_spec,
                    selection_method="by_name" if target_name else "first_available",
                )
        except TradovateConnectionError:
            raise  # re-raise our own error
        except Exception:
            logger.exception("tradovate_auth.account_fetch_failed")

    def _start_refresh_loop(self) -> None:
        """Start the background token refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

        refresh_interval = (self._config.token_lifetime_min * 60) - _REFRESH_BUFFER_SEC
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(refresh_interval)
        )

    async def _refresh_loop(self, interval: float) -> None:
        """Periodically refresh the access token."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self._refresh_token()
            except Exception:
                logger.exception("tradovate_auth.refresh_loop_error")
                # Try full re-auth
                try:
                    await self.authenticate()
                except Exception:
                    logger.critical("tradovate_auth.reauth_failed")
                    self._authenticated = False
                    raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _refresh_token(self) -> None:
        """Refresh the access token using the current token."""
        session = self._get_session()
        url = f"{self._config.base_url}/auth/renewaccesstoken"

        logger.info("tradovate_auth.refreshing_token")

        async with session.get(url, headers=self._auth_headers()) as resp:
            if resp.status != 200:
                raise TradovateConnectionError(
                    f"Token refresh failed with status {resp.status}"
                )

            data = await resp.json()

            if data.get("errorText"):
                raise TradovateConnectionError(
                    f"Token refresh error: {data['errorText']}"
                )

            new_token = None
            async with self._lock:
                if "accessToken" in data:
                    self._access_token = data["accessToken"]
                    new_token = self._access_token
                if "mdAccessToken" in data:
                    self._md_access_token = data["mdAccessToken"]
                self._token_obtained_at = datetime.now(tz=UTC)

            logger.info("tradovate_auth.token_refreshed")

            # Notify callbacks (e.g., WS client) of the new token
            if new_token:
                await self._notify_token_refresh(new_token)

    async def close(self) -> None:
        """Shut down the auth manager."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

        self._authenticated = False
        logger.info("tradovate_auth.closed")

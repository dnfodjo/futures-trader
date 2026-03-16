"""Multi-instrument poller for cross-market context.

Polls secondary instruments via Yahoo Finance for context that the LLM
uses in its reasoning. These don't need tick-level precision — approximate
values refreshed every 15-30 seconds are sufficient.

Instruments:
- ES (S&P 500 E-mini): primary cross-market reference
- TICK (NYSE TICK index): breadth indicator
- VIX (CBOE Volatility Index): fear gauge
- 10Y (10-Year Treasury Yield): rate context
- DXY (US Dollar Index): currency context

ES data comes from Databento (same feed as MNQ), so it's handled by
the DatabentoClient. This module only polls the non-futures instruments
that aren't available from Databento.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog

from src.core.types import CrossMarketContext

logger = structlog.get_logger()

# Yahoo Finance quote endpoint (no API key required)
_YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Symbol mapping: our name -> Yahoo Finance ticker
_SYMBOL_MAP: dict[str, str] = {
    "tick": "^TICK",     # NYSE TICK index
    "vix": "^VIX",       # CBOE Volatility Index
    "ten_year": "^TNX",  # 10-Year Treasury Yield
    "dxy": "DX-Y.NYB",  # US Dollar Index
}


async def _fetch_yahoo_quote(
    symbol: str,
    session: aiohttp.ClientSession,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch a single quote from Yahoo Finance.

    Returns dict with price, change, change_pct, or None on failure.
    """
    url = _YAHOO_QUOTE_URL.format(symbol=symbol)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; futures-trader/1.0)",
    }

    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                logger.debug("yahoo.fetch_failed", symbol=symbol, status=resp.status)
                return None

            data = await resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0.0)
            prev_close = meta.get("chartPreviousClose", 0.0) or meta.get(
                "previousClose", 0.0
            )

            change = price - prev_close if prev_close else 0.0
            change_pct = (change / prev_close * 100.0) if prev_close else 0.0

            return {
                "price": price,
                "prev_close": prev_close,
                "change": round(change, 4),
                "change_pct": round(change_pct, 3),
            }

    except asyncio.TimeoutError:
        logger.debug("yahoo.timeout", symbol=symbol)
        return None
    except Exception as e:
        logger.debug("yahoo.error", symbol=symbol, error=str(e))
        return None


class MultiInstrumentPoller:
    """Polls secondary instruments for cross-market context.

    Runs as an async loop, refreshing TICK, VIX, 10Y, and DXY
    at a configurable interval. ES price is set externally via
    update_es() (from the Databento feed).

    Usage:
        poller = MultiInstrumentPoller(poll_interval_sec=15)
        await poller.start()      # starts background polling loop
        ctx = poller.snapshot()    # get current CrossMarketContext
        await poller.stop()
    """

    def __init__(
        self,
        poll_interval_sec: float = 15.0,
        timeout_sec: float = 10.0,
    ) -> None:
        self._poll_interval = poll_interval_sec
        self._timeout = timeout_sec

        # Current values
        self._es_price: float = 0.0
        self._es_prev_close: float = 0.0
        self._tick_index: int = 0
        self._vix: float = 0.0
        self._vix_prev_close: float = 0.0
        self._ten_year_yield: float = 0.0
        self._dxy: float = 0.0

        # State
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_poll_time: datetime | None = None
        self._poll_count = 0
        self._error_count = 0

        # Lock for concurrent access
        self._lock = asyncio.Lock()

    # ── ES Update (from Databento feed) ───────────────────────────────────

    async def update_es(self, price: float, prev_close: float = 0.0) -> None:
        """Update ES price from the Databento feed (not polled)."""
        async with self._lock:
            self._es_price = price
            if prev_close > 0:
                self._es_prev_close = prev_close

    # ── Polling Loop ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return

        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("multi_instrument.started", interval_sec=self._poll_interval)

    async def stop(self) -> None:
        """Stop the polling loop and clean up."""
        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        logger.info("multi_instrument.stopped")

    async def _poll_loop(self) -> None:
        """Background loop that polls all instruments."""
        while self._running:
            try:
                await self._poll_all()
                self._poll_count += 1
                self._last_poll_time = datetime.now(tz=UTC)
            except asyncio.CancelledError:
                break
            except Exception:
                self._error_count += 1
                logger.exception("multi_instrument.poll_error")

            # Wait for next poll cycle
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

    async def _poll_all(self) -> None:
        """Poll all secondary instruments concurrently."""
        if self._session is None:
            return

        # Fetch all in parallel
        results = await asyncio.gather(
            _fetch_yahoo_quote(_SYMBOL_MAP["tick"], self._session, self._timeout),
            _fetch_yahoo_quote(_SYMBOL_MAP["vix"], self._session, self._timeout),
            _fetch_yahoo_quote(_SYMBOL_MAP["ten_year"], self._session, self._timeout),
            _fetch_yahoo_quote(_SYMBOL_MAP["dxy"], self._session, self._timeout),
            return_exceptions=True,
        )

        async with self._lock:
            # TICK
            tick_result = results[0]
            if isinstance(tick_result, dict) and tick_result:
                self._tick_index = int(tick_result["price"])

            # VIX
            vix_result = results[1]
            if isinstance(vix_result, dict) and vix_result:
                self._vix = vix_result["price"]
                self._vix_prev_close = vix_result.get("prev_close", 0.0)

            # 10Y
            ten_year_result = results[2]
            if isinstance(ten_year_result, dict) and ten_year_result:
                self._ten_year_yield = ten_year_result["price"]

            # DXY
            dxy_result = results[3]
            if isinstance(dxy_result, dict) and dxy_result:
                self._dxy = dxy_result["price"]

        logger.debug(
            "multi_instrument.polled",
            tick=self._tick_index,
            vix=self._vix,
            ten_year=self._ten_year_yield,
            dxy=self._dxy,
        )

    # ── Manual Polling ────────────────────────────────────────────────────

    async def poll_once(self) -> None:
        """Execute a single poll cycle (useful for testing or on-demand refresh).

        If no background polling loop is running, creates a temporary session
        that is cleaned up after the poll. If start() was called, reuses that session.
        """
        own_session = self._session is None
        if own_session:
            self._session = aiohttp.ClientSession()

        try:
            await self._poll_all()
            self._poll_count += 1
            self._last_poll_time = datetime.now(tz=UTC)
        finally:
            if own_session and self._session is not None and not self._running:
                await self._session.close()
                self._session = None

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> CrossMarketContext:
        """Return current cross-market context as a domain object."""
        es_change_pct = 0.0
        if self._es_prev_close > 0:
            es_change_pct = round(
                (self._es_price - self._es_prev_close) / self._es_prev_close * 100, 3
            )

        vix_change_pct = 0.0
        if self._vix_prev_close > 0:
            vix_change_pct = round(
                (self._vix - self._vix_prev_close) / self._vix_prev_close * 100, 3
            )

        return CrossMarketContext(
            es_price=self._es_price,
            es_change_pct=es_change_pct,
            tick_index=self._tick_index,
            vix=self._vix,
            vix_change_pct=vix_change_pct,
            ten_year_yield=self._ten_year_yield,
            dxy=self._dxy,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_poll_time(self) -> datetime | None:
        return self._last_poll_time

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "last_poll": self._last_poll_time.isoformat() if self._last_poll_time else None,
            "es_price": self._es_price,
            "tick_index": self._tick_index,
            "vix": self._vix,
            "ten_year_yield": self._ten_year_yield,
            "dxy": self._dxy,
        }

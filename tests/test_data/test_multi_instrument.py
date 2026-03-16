"""Tests for the multi-instrument poller.

Tests ES update, Yahoo Finance response parsing, snapshot generation,
and polling loop management without real network calls.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.core.types import CrossMarketContext
from src.data.multi_instrument import MultiInstrumentPoller, _fetch_yahoo_quote


@pytest.fixture
def poller():
    return MultiInstrumentPoller(poll_interval_sec=60)


class TestInitialization:
    def test_initial_state(self, poller: MultiInstrumentPoller):
        assert poller.is_running is False
        assert poller.last_poll_time is None
        assert poller._es_price == 0.0

    def test_initial_stats(self, poller: MultiInstrumentPoller):
        stats = poller.stats
        assert stats["running"] is False
        assert stats["poll_count"] == 0
        assert stats["error_count"] == 0
        assert stats["vix"] == 0.0


class TestESUpdate:
    async def test_update_es(self, poller: MultiInstrumentPoller):
        await poller.update_es(5420.0, prev_close=5400.0)
        assert poller._es_price == 5420.0
        assert poller._es_prev_close == 5400.0

    async def test_update_es_no_prev_close(self, poller: MultiInstrumentPoller):
        """Should not overwrite prev_close if not provided."""
        poller._es_prev_close = 5400.0
        await poller.update_es(5420.0)
        assert poller._es_prev_close == 5400.0


class TestSnapshot:
    def test_snapshot_initial(self, poller: MultiInstrumentPoller):
        snap = poller.snapshot()
        assert isinstance(snap, CrossMarketContext)
        assert snap.es_price == 0.0
        assert snap.tick_index == 0
        assert snap.vix == 0.0

    async def test_snapshot_with_es(self, poller: MultiInstrumentPoller):
        await poller.update_es(5420.0, prev_close=5400.0)
        snap = poller.snapshot()
        assert snap.es_price == 5420.0
        # Change pct = (5420 - 5400) / 5400 * 100 = 0.370%
        assert snap.es_change_pct == pytest.approx(0.370, abs=0.01)

    async def test_snapshot_with_manual_values(self, poller: MultiInstrumentPoller):
        """Manually set values should appear in snapshot."""
        poller._tick_index = 480
        poller._vix = 18.2
        poller._vix_prev_close = 18.5
        poller._ten_year_yield = 4.25
        poller._dxy = 104.5

        snap = poller.snapshot()
        assert snap.tick_index == 480
        assert snap.vix == 18.2
        assert snap.vix_change_pct == pytest.approx(-1.622, abs=0.01)
        assert snap.ten_year_yield == 4.25
        assert snap.dxy == 104.5

    def test_snapshot_zero_prev_close_no_divide_by_zero(self, poller: MultiInstrumentPoller):
        """Should handle zero prev_close without division by zero."""
        poller._es_price = 5420.0
        poller._es_prev_close = 0.0
        snap = poller.snapshot()
        assert snap.es_change_pct == 0.0


class TestYahooFetch:
    async def test_fetch_success(self):
        """Should parse Yahoo Finance chart response."""
        mock_response = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "regularMarketPrice": 18.5,
                            "chartPreviousClose": 19.0,
                        }
                    }
                ]
            }
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        result = await _fetch_yahoo_quote("^VIX", mock_session)
        assert result is not None
        assert result["price"] == 18.5
        assert result["prev_close"] == 19.0
        assert result["change"] == pytest.approx(-0.5, abs=0.01)

    async def test_fetch_non_200(self):
        """Should return None on non-200 status."""
        mock_resp = AsyncMock()
        mock_resp.status = 404

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        result = await _fetch_yahoo_quote("^VIX", mock_session)
        assert result is None

    async def test_fetch_empty_result(self):
        """Should return None if chart result is empty."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"chart": {"result": []}})

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        result = await _fetch_yahoo_quote("^VIX", mock_session)
        assert result is None

    async def test_fetch_timeout(self):
        """Should return None on timeout."""
        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        result = await _fetch_yahoo_quote("^VIX", mock_session)
        assert result is None

    async def test_fetch_network_error(self):
        """Should return None on network errors."""
        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        result = await _fetch_yahoo_quote("^VIX", mock_session)
        assert result is None


class TestPollAll:
    async def test_poll_all_updates_values(self, poller: MultiInstrumentPoller):
        """Polling should update internal values from responses."""
        tick_response = {"price": 480, "prev_close": 0, "change": 0, "change_pct": 0}
        vix_response = {"price": 18.5, "prev_close": 19.0, "change": -0.5, "change_pct": -2.6}
        ten_year_response = {"price": 4.25, "prev_close": 4.20, "change": 0.05, "change_pct": 1.2}
        dxy_response = {"price": 104.5, "prev_close": 104.0, "change": 0.5, "change_pct": 0.5}

        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            side_effect=[tick_response, vix_response, ten_year_response, dxy_response],
        ):
            poller._session = MagicMock()
            await poller._poll_all()

        assert poller._tick_index == 480
        assert poller._vix == 18.5
        assert poller._ten_year_yield == 4.25
        assert poller._dxy == 104.5

    async def test_poll_all_handles_failures(self, poller: MultiInstrumentPoller):
        """Partial failures should not crash polling."""
        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            side_effect=[None, {"price": 18.5, "prev_close": 19.0, "change": -0.5, "change_pct": -2.6}, None, None],
        ):
            poller._session = MagicMock()
            await poller._poll_all()

        # Only VIX should be updated
        assert poller._tick_index == 0  # unchanged
        assert poller._vix == 18.5

    async def test_poll_all_handles_exceptions(self, poller: MultiInstrumentPoller):
        """Exceptions in individual fetches should not crash."""
        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            side_effect=[Exception("boom"), None, None, None],
        ):
            poller._session = MagicMock()
            # Should not raise
            await poller._poll_all()


class TestPollOnce:
    async def test_poll_once(self, poller: MultiInstrumentPoller):
        """Manual poll should work and update stats."""
        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            return_value=None,
        ), patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()
            MockSession.return_value = mock_session
            await poller.poll_once()

        assert poller._poll_count == 1
        assert poller.last_poll_time is not None

    async def test_poll_once_creates_and_cleans_session(self, poller: MultiInstrumentPoller):
        """Should create a temporary session and clean it up when not running."""
        assert poller._session is None
        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            return_value=None,
        ), patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            MockSession.return_value = mock_session
            await poller.poll_once()
        # Session cleaned up since poller is not running
        assert poller._session is None
        mock_session.close.assert_awaited_once()


class TestStartStop:
    async def test_start_stop(self, poller: MultiInstrumentPoller):
        """Start/stop should manage the polling loop."""
        with patch.object(poller, "_poll_all", new_callable=AsyncMock):
            await poller.start()
            assert poller.is_running is True

            await asyncio.sleep(0.05)

            await poller.stop()
            assert poller.is_running is False
            assert poller._session is None

    async def test_stop_idempotent(self, poller: MultiInstrumentPoller):
        """Stopping when not running should be safe."""
        await poller.stop()
        assert poller.is_running is False

    async def test_start_idempotent(self, poller: MultiInstrumentPoller):
        """Starting when already running should be a no-op."""
        with patch.object(poller, "_poll_all", new_callable=AsyncMock):
            await poller.start()
            task1 = poller._task
            await poller.start()  # should not create second task
            assert poller._task is task1
            await poller.stop()


class TestStats:
    async def test_stats_after_polling(self, poller: MultiInstrumentPoller):
        poller._poll_count = 5
        poller._error_count = 1
        poller._vix = 18.5
        poller._tick_index = 480

        stats = poller.stats
        assert stats["poll_count"] == 5
        assert stats["error_count"] == 1
        assert stats["vix"] == 18.5
        assert stats["tick_index"] == 480


class TestPollOnceSessionCleanup:
    async def test_poll_once_cleans_up_session(self, poller: MultiInstrumentPoller):
        """poll_once should clean up session when not running in background."""
        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            return_value=None,
        ), patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            MockSession.return_value = mock_session
            await poller.poll_once()

        # Session should be cleaned up since poller is not running
        assert poller._session is None
        mock_session.close.assert_awaited_once()

    async def test_poll_once_keeps_session_when_running(self, poller: MultiInstrumentPoller):
        """poll_once should keep session if background loop is running."""
        poller._running = True
        poller._session = AsyncMock()  # pre-existing session

        with patch(
            "src.data.multi_instrument._fetch_yahoo_quote",
            return_value=None,
        ):
            await poller.poll_once()

        # Session should still be there since poller is running
        assert poller._session is not None

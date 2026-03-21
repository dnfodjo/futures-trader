"""Tests for the economic calendar.

Tests FOMC date detection (fallback), ForexFactory response parsing,
local file loading, event querying, and blackout window detection.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import aiohttp
import pytest

from src.core.types import EconomicEvent
from src.data.economic_calendar import (
    EconomicCalendar,
    _classify_ff_impact,
    _get_fomc_events,
    _parse_forexfactory_response,
    _parse_ff_time,
    fetch_from_forexfactory,
    load_from_file,
)

ET = ZoneInfo("US/Eastern")


class TestFOMCDates:
    def test_fomc_day(self):
        """Should return FOMC events on known FOMC dates."""
        events = _get_fomc_events("2025-01-29")
        assert len(events) == 2
        assert events[0].name == "FOMC Interest Rate Decision"
        assert events[0].impact == "high"
        assert events[1].name == "FOMC Press Conference"

    def test_non_fomc_day(self):
        """Should return empty list on non-FOMC dates."""
        events = _get_fomc_events("2025-01-30")
        assert events == []

    def test_fomc_2026(self):
        events = _get_fomc_events("2026-03-18")
        assert len(events) == 2

    def test_fomc_time(self):
        """FOMC decision should be at 2:00 PM ET."""
        events = _get_fomc_events("2025-01-29")
        assert events[0].time.hour == 14
        assert events[0].time.minute == 0

    def test_fomc_press_conference_time(self):
        """Press conference should be at 2:30 PM ET."""
        events = _get_fomc_events("2025-01-29")
        assert events[1].time.hour == 14
        assert events[1].time.minute == 30


class TestForexFactoryParsing:
    def test_parse_us_event(self):
        data = [
            {
                "country": "USD",
                "title": "CPI m/m",
                "impact": "High",
                "date": "2025-03-14T08:30:00-04:00",
                "previous": "0.3%",
                "forecast": "0.4%",
            }
        ]
        events = _parse_forexfactory_response(data)
        assert len(events) == 1
        assert events[0].name == "CPI m/m"
        assert events[0].impact == "high"

    def test_filter_non_us(self):
        """Should only include US events."""
        data = [
            {
                "country": "GBP",
                "title": "UK GDP",
                "impact": "High",
                "date": "2025-03-14T09:00:00-04:00",
            },
            {
                "country": "USD",
                "title": "Retail Sales",
                "impact": "High",
                "date": "2025-03-14T08:30:00-04:00",
            },
        ]
        events = _parse_forexfactory_response(data)
        assert len(events) == 1
        assert events[0].name == "Retail Sales"

    def test_empty_response(self):
        events = _parse_forexfactory_response([])
        assert events == []

    def test_parse_prior_and_forecast(self):
        data = [
            {
                "country": "USD",
                "title": "Initial Jobless Claims",
                "impact": "Medium",
                "date": "2025-03-14T08:30:00-04:00",
                "previous": "217K",
                "forecast": "220K",
            }
        ]
        events = _parse_forexfactory_response(data)
        assert events[0].prior == "217K"
        assert events[0].forecast == "220K"


class TestImpactClassification:
    def test_always_high_cpi(self):
        assert _classify_ff_impact("Medium", "CPI m/m") == "high"

    def test_always_high_fomc(self):
        assert _classify_ff_impact("Low", "FOMC Interest Rate Decision") == "high"

    def test_always_high_nfp(self):
        assert _classify_ff_impact("Medium", "Nonfarm Payrolls") == "high"

    def test_always_high_gdp(self):
        assert _classify_ff_impact("Low", "GDP q/q") == "high"

    def test_always_high_fomc_statement(self):
        assert _classify_ff_impact("High", "FOMC Statement") == "high"

    def test_always_high_core_cpi(self):
        assert _classify_ff_impact("Medium", "Core CPI m/m") == "high"

    def test_medium_impact(self):
        assert _classify_ff_impact("Medium", "Building Permits") == "medium"

    def test_low_impact(self):
        assert _classify_ff_impact("Low", "Redbook") == "low"

    def test_non_economic(self):
        assert _classify_ff_impact("Non-Economic", "Bank Holiday") == "low"


class TestTimeParser:
    def test_parse_iso_with_offset(self):
        dt = _parse_ff_time("2025-03-14T08:30:00-04:00")
        assert dt is not None
        assert dt.hour == 8
        assert dt.minute == 30

    def test_parse_iso_with_utc_offset(self):
        """Should convert UTC to ET."""
        dt = _parse_ff_time("2025-03-14T12:30:00+00:00")
        assert dt is not None
        # 12:30 UTC = 8:30 EDT
        assert dt.hour == 8
        assert dt.minute == 30

    def test_parse_empty_string(self):
        dt = _parse_ff_time("")
        assert dt is None

    def test_parse_none(self):
        dt = _parse_ff_time(None)
        assert dt is None


class TestLoadFromFile:
    def test_load_valid_file(self):
        data = [
            {
                "date": "2025-03-14",
                "time": "08:30",
                "name": "CPI m/m",
                "impact": "high",
                "prior": "0.3%",
                "forecast": "0.4%",
            },
            {
                "date": "2025-03-14",
                "time": "10:00",
                "name": "Consumer Sentiment",
                "impact": "medium",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            events = load_from_file(f.name)

        assert len(events) == 2
        assert events[0].name == "CPI m/m"
        assert events[0].impact == "high"
        assert events[0].prior == "0.3%"

    def test_load_nonexistent_file(self):
        events = load_from_file("/nonexistent/path.json")
        assert events == []

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json{{{")
            f.flush()
            events = load_from_file(f.name)

        assert events == []


class TestEconomicCalendar:
    @pytest.fixture
    def calendar(self):
        return EconomicCalendar()

    async def test_load_fomc_day_fallback(self, calendar: EconomicCalendar):
        """When ForexFactory is unavailable, FOMC fallback should be used."""
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await calendar.load_today("2025-01-29")
        assert any("FOMC" in e.name for e in calendar.events)

    async def test_load_non_event_day(self, calendar: EconomicCalendar):
        """Non-event day with no ForexFactory data should have 0 events."""
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await calendar.load_today("2025-01-30")
        assert len(calendar.events) == 0

    async def test_loaded_date(self, calendar: EconomicCalendar):
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await calendar.load_today("2025-03-14")
        assert calendar.loaded_date == "2025-03-14"

    async def test_high_impact_today(self, calendar: EconomicCalendar):
        """ForexFactory FOMC events should be classified as high impact."""
        mock_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 30, tzinfo=ET),
                name="FOMC Press Conference",
                impact="high",
            ),
        ]
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=mock_events,
        ):
            await calendar.load_today("2025-01-29")
        high = calendar.high_impact_today()
        assert len(high) == 2

    async def test_upcoming_events(self, calendar: EconomicCalendar):
        """Upcoming events should filter by time window."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=10),
                name="Event Soon",
                impact="high",
            ),
            EconomicEvent(
                time=now + timedelta(hours=5),
                name="Event Later",
                impact="medium",
            ),
        ]

        upcoming = calendar.upcoming_events(within_minutes=30, t=now)
        assert len(upcoming) == 1
        assert upcoming[0].name == "Event Soon"

    async def test_past_events(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now - timedelta(minutes=5),
                name="Recent Event",
                impact="high",
            ),
            EconomicEvent(
                time=now - timedelta(hours=2),
                name="Old Event",
                impact="medium",
            ),
        ]

        past = calendar.past_events(within_minutes=15, t=now)
        assert len(past) == 1
        assert past[0].name == "Recent Event"

    async def test_has_high_impact_upcoming(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=10),
                name="CPI m/m",
                impact="high",
            ),
        ]
        assert calendar.has_high_impact_upcoming(within_minutes=30, t=now)

    async def test_has_no_high_impact_upcoming(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=10),
                name="Redbook",
                impact="low",
            ),
        ]
        assert not calendar.has_high_impact_upcoming(within_minutes=30, t=now)

    async def test_next_event(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now - timedelta(hours=1),
                name="Past Event",
                impact="medium",
            ),
            EconomicEvent(
                time=now + timedelta(hours=1),
                name="Future Event",
                impact="high",
            ),
        ]
        nxt = calendar.next_event(t=now)
        assert nxt is not None
        assert nxt.name == "Future Event"

    async def test_next_event_none(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now - timedelta(hours=1),
                name="Past Event",
                impact="medium",
            ),
        ]
        assert calendar.next_event(t=now) is None

    async def test_next_high_impact_event(self, calendar: EconomicCalendar):
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=10),
                name="Redbook",
                impact="low",
            ),
            EconomicEvent(
                time=now + timedelta(minutes=30),
                name="CPI",
                impact="high",
            ),
        ]
        nxt = calendar.next_high_impact_event(t=now)
        assert nxt is not None
        assert nxt.name == "CPI"

    async def test_stats(self, calendar: EconomicCalendar):
        mock_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 30, tzinfo=ET),
                name="FOMC Press Conference",
                impact="high",
            ),
        ]
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=mock_events,
        ):
            await calendar.load_today("2025-01-29")
        stats = calendar.stats
        assert stats["loaded_date"] == "2025-01-29"
        assert stats["total_events"] == 2
        assert stats["high_impact"] == 2

    async def test_load_with_local_file(self):
        """Should merge local file events with ForexFactory data."""
        data = [
            {
                "date": "2025-01-29",
                "time": "08:30",
                "name": "GDP q/q",
                "impact": "high",
            }
        ]
        mock_ff_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cal = EconomicCalendar(local_calendar_path=f.name)
            with patch(
                "src.data.economic_calendar.fetch_from_forexfactory",
                new_callable=AsyncMock,
                return_value=mock_ff_events,
            ):
                await cal.load_today("2025-01-29")

        # Should have ForexFactory (1 event) + GDP from local (1 event) = 2
        assert len(cal.events) == 2

    async def test_load_deduplicates(self):
        """Should not duplicate events from both ForexFactory and file."""
        data = [
            {
                "date": "2025-01-29",
                "time": "14:00",
                "name": "Federal Funds Rate",
                "impact": "high",
            }
        ]
        mock_ff_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cal = EconomicCalendar(local_calendar_path=f.name)
            with patch(
                "src.data.economic_calendar.fetch_from_forexfactory",
                new_callable=AsyncMock,
                return_value=mock_ff_events,
            ):
                await cal.load_today("2025-01-29")

        fed_decisions = [e for e in cal.events if "Federal Funds" in e.name]
        assert len(fed_decisions) == 1

    async def test_events_sorted_by_time(self, calendar: EconomicCalendar):
        """Events should be sorted chronologically."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(hours=2),
                name="Later",
                impact="low",
            ),
            EconomicEvent(
                time=now + timedelta(minutes=10),
                name="Soon",
                impact="high",
            ),
        ]
        calendar._events.sort(key=lambda e: e.time)
        assert calendar._events[0].name == "Soon"

    async def test_is_in_blackout_before_event(self, calendar: EconomicCalendar):
        """Should return True within pre_minutes before a high-impact event."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=3),
                name="CPI",
                impact="high",
            ),
        ]
        assert calendar.is_in_blackout(t=now, pre_minutes=5, post_minutes=10)

    async def test_is_in_blackout_after_event(self, calendar: EconomicCalendar):
        """Should return True within post_minutes after a high-impact event."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now - timedelta(minutes=5),
                name="NFP",
                impact="high",
            ),
        ]
        assert calendar.is_in_blackout(t=now, pre_minutes=5, post_minutes=10)

    async def test_not_in_blackout(self, calendar: EconomicCalendar):
        """Should return False when outside blackout windows."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(hours=2),
                name="CPI",
                impact="high",
            ),
        ]
        assert not calendar.is_in_blackout(t=now)

    async def test_not_in_blackout_low_impact(self, calendar: EconomicCalendar):
        """Low-impact events should not trigger blackout."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(minutes=3),
                name="Redbook",
                impact="low",
            ),
        ]
        assert not calendar.is_in_blackout(t=now)

    async def test_blackout_until(self, calendar: EconomicCalendar):
        """Should return when the current blackout ends."""
        now = datetime.now(ET)
        event_time = now + timedelta(minutes=3)
        calendar._events = [
            EconomicEvent(
                time=event_time,
                name="CPI",
                impact="high",
            ),
        ]
        until = calendar.blackout_until(t=now, pre_minutes=5, post_minutes=10)
        assert until is not None
        assert until == event_time + timedelta(minutes=10)

    async def test_blackout_until_not_in_blackout(self, calendar: EconomicCalendar):
        """Should return None when not in blackout."""
        now = datetime.now(ET)
        calendar._events = [
            EconomicEvent(
                time=now + timedelta(hours=2),
                name="CPI",
                impact="high",
            ),
        ]
        assert calendar.blackout_until(t=now) is None

    async def test_forexfactory_deduplication(self):
        """ForexFactory events should be deduplicated with local file events."""
        mock_ff_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
            EconomicEvent(
                time=datetime(2025, 1, 29, 12, 30, tzinfo=ET),
                name="GDP q/q",
                impact="high",
            ),
        ]

        cal = EconomicCalendar()
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=mock_ff_events,
        ):
            await cal.load_today("2025-01-29")

        # Both events from ForexFactory should be included (no static FOMC since FF loaded)
        assert len(cal.events) == 2

    async def test_fomc_fallback_when_ff_fails(self):
        """Should use static FOMC dates when ForexFactory fails."""
        cal = EconomicCalendar()
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await cal.load_today("2025-01-29")

        # Should have FOMC events from static fallback
        assert len(cal.events) == 2
        assert any("FOMC" in e.name for e in cal.events)

    async def test_no_fomc_fallback_when_ff_succeeds(self):
        """Should NOT use static FOMC fallback when ForexFactory loads successfully."""
        mock_ff_events = [
            EconomicEvent(
                time=datetime(2025, 1, 29, 14, 0, tzinfo=ET),
                name="Federal Funds Rate",
                impact="high",
            ),
        ]

        cal = EconomicCalendar()
        with patch(
            "src.data.economic_calendar.fetch_from_forexfactory",
            new_callable=AsyncMock,
            return_value=mock_ff_events,
        ):
            await cal.load_today("2025-01-29")

        # Should only have ForexFactory events, no duplicates from static FOMC
        fomc_static = [e for e in cal.events if "FOMC Interest Rate Decision" in e.name]
        assert len(fomc_static) == 0  # Static FOMC not used since FF loaded


class TestFetchFromForexFactory:
    async def test_fetch_success(self):
        """Should fetch and parse events from ForexFactory."""
        response_data = [
            {
                "country": "USD",
                "title": "CPI m/m",
                "impact": "High",
                "date": "2025-03-14T08:30:00-04:00",
            },
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        events = await fetch_from_forexfactory(session=mock_session)
        assert len(events) == 1
        assert events[0].name == "CPI m/m"

    async def test_fetch_non_200(self):
        """Should return empty list on non-200 status."""
        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        events = await fetch_from_forexfactory(session=mock_session)
        assert events == []

    async def test_fetch_timeout(self):
        """Should return empty list on timeout."""
        import asyncio

        mock_session = MagicMock()
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        events = await fetch_from_forexfactory(session=mock_session)
        assert events == []

    async def test_fetch_creates_own_session(self):
        """Should create and close its own session if none provided."""
        response_data = []

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)

        with patch("aiohttp.ClientSession") as MockSession:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_get_ctx)
            mock_session.close = AsyncMock()
            MockSession.return_value = mock_session

            events = await fetch_from_forexfactory()
            assert events == []
            mock_session.close.assert_awaited_once()

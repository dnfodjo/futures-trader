"""Economic calendar — loads events and identifies blackout windows.

Sources:
1. ForexFactory free JSON feed (no API key required) — primary source
2. Local JSON fallback if ForexFactory is unavailable
3. Static FOMC dates as last-resort fallback (updated through 2026)

Events are loaded once at startup (pre-market) and cached for the day.
The clock module uses these events to determine blackout windows.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import structlog

from src.core.types import EconomicEvent

logger = structlog.get_logger()

ET = ZoneInfo("US/Eastern")

# ── Static FOMC dates (last-resort fallback) ──────────────────────────────
# Only used if ForexFactory is unavailable. ForexFactory already includes
# FOMC events with correct times and impact levels.

_FOMC_DATES: dict[int, list[str]] = {
    2025: [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    ],
    2026: [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ],
}


def _get_fomc_events(target_date: str) -> list[EconomicEvent]:
    """Get FOMC events for a given date (YYYY-MM-DD). Fallback only."""
    year = int(target_date[:4])
    fomc_dates = _FOMC_DATES.get(year, [])

    events = []
    if target_date in fomc_dates:
        # FOMC decision at 2:00 PM ET
        dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
            hour=14, minute=0, tzinfo=ET
        )
        events.append(
            EconomicEvent(
                time=dt,
                name="FOMC Interest Rate Decision",
                impact="high",
            )
        )
        # Press conference at 2:30 PM ET
        events.append(
            EconomicEvent(
                time=dt.replace(minute=30),
                name="FOMC Press Conference",
                impact="high",
            )
        )
    return events


# ── ForexFactory Integration ─────────────────────────────────────────────


async def fetch_from_forexfactory(
    session: aiohttp.ClientSession | None = None,
) -> list[EconomicEvent]:
    """Fetch economic calendar events from ForexFactory free JSON feed.

    No API key required. Returns all events for the current week.
    Source: https://nfs.faireconomy.media/ff_calendar_thisweek.json
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning("forexfactory.fetch_failed", status=resp.status)
                return []
            data = await resp.json(content_type=None)
    except Exception as e:
        logger.warning("forexfactory.fetch_error", error=str(e))
        return []
    finally:
        if own_session:
            await session.close()

    return _parse_forexfactory_response(data)


def _parse_forexfactory_response(data: list[dict[str, Any]]) -> list[EconomicEvent]:
    """Parse ForexFactory JSON feed into EconomicEvent list."""
    events = []

    for item in data:
        # Only US events matter for MNQ trading
        country = item.get("country", "")
        if country != "USD":
            continue

        name = item.get("title", "Unknown Event")
        impact_raw = item.get("impact", "Low")

        # Map ForexFactory impact levels
        impact = _classify_ff_impact(impact_raw, name)

        # Parse time — ForexFactory uses ISO 8601 with timezone offset
        # e.g., "2026-03-18T14:00:00-04:00"
        date_str = item.get("date", "")
        event_time = _parse_ff_time(date_str)

        if event_time is None:
            continue

        events.append(
            EconomicEvent(
                time=event_time,
                name=name,
                impact=impact,
                prior=item.get("previous") or None,
                forecast=item.get("forecast") or None,
            )
        )

    return events


def _classify_ff_impact(impact_str: str, name: str) -> str:
    """Classify event impact level.

    Some events are always high-impact regardless of ForexFactory's classification.
    """
    # Always high-impact events (override ForexFactory's classification)
    always_high = {
        "nonfarm payrolls",
        "cpi",
        "consumer price index",
        "core cpi",
        "fomc",
        "federal funds rate",
        "fed interest rate",
        "unemployment claims",
        "initial jobless claims",
        "gdp",
        "pce",
        "core pce",
        "ppi",
        "core ppi",
        "ism manufacturing",
        "ism services",
        "retail sales",
        "fomc statement",
        "fomc press conference",
        "fomc economic projections",
    }

    name_lower = name.lower()
    for keyword in always_high:
        if keyword in name_lower:
            return "high"

    # Map ForexFactory impact levels
    impact_map = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "non-economic": "low",
        "holiday": "low",
    }
    return impact_map.get(impact_str.lower(), "low")


def _parse_ff_time(date_str: str) -> datetime | None:
    """Parse ForexFactory ISO 8601 datetime to timezone-aware datetime."""
    if not date_str:
        return None

    try:
        # ISO 8601 with offset: "2026-03-18T14:00:00-04:00"
        dt = datetime.fromisoformat(date_str)
        # Convert to ET for consistency
        return dt.astimezone(ET)
    except (ValueError, TypeError):
        pass

    return None


# ── Local Fallback ────────────────────────────────────────────────────────


def load_from_file(filepath: str | Path) -> list[EconomicEvent]:
    """Load economic events from a local JSON file.

    Expected format:
    [
        {
            "date": "2025-03-14",
            "time": "08:30",
            "name": "CPI m/m",
            "impact": "high",
            "prior": "0.3%",
            "forecast": "0.4%"
        },
        ...
    ]
    """
    path = Path(filepath)
    if not path.exists():
        logger.debug("economic_calendar.file_not_found", path=str(path))
        return []

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("economic_calendar.file_parse_error", error=str(e))
        return []

    events = []
    for item in raw:
        date_str = item.get("date", "")
        time_str = item.get("time", "08:30")
        name = item.get("name", "")
        impact = item.get("impact", "low")

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            dt = dt.replace(tzinfo=ET)
        except ValueError:
            continue

        events.append(
            EconomicEvent(
                time=dt,
                name=name,
                impact=impact,
                prior=item.get("prior"),
                forecast=item.get("forecast"),
            )
        )

    return events


# ── Main Calendar Class ──────────────────────────────────────────────────


class EconomicCalendar:
    """Economic calendar manager.

    Loads events from ForexFactory (free, no API key), caches for the day,
    and provides query methods for checking upcoming events and blackout windows.

    Usage:
        calendar = EconomicCalendar()
        await calendar.load_today()
        events = calendar.upcoming_events(within_minutes=30)
    """

    def __init__(
        self,
        local_calendar_path: str | Path | None = None,
    ) -> None:
        self._local_path = local_calendar_path
        self._events: list[EconomicEvent] = []
        self._loaded_date: str = ""
        self._ff_events_cache: list[EconomicEvent] = []

    async def load_today(self, target_date: str | None = None) -> None:
        """Load economic events for today (or a specific date).

        Priority:
        1. ForexFactory free feed (primary — has FOMC, CPI, NFP, everything)
        2. Local JSON file (if configured)
        3. Static FOMC dates (last-resort fallback if ForexFactory is down)

        Args:
            target_date: "YYYY-MM-DD" format. Defaults to today (ET).
        """
        if target_date is None:
            target_date = datetime.now(ET).strftime("%Y-%m-%d")

        events: list[EconomicEvent] = []
        existing_names: set[str] = set()
        ff_loaded = False

        # 1. ForexFactory (primary source — free, no API key)
        try:
            ff_all = await fetch_from_forexfactory()
            # Filter to target date only
            ff_today = [
                e for e in ff_all
                if e.time.astimezone(ET).strftime("%Y-%m-%d") == target_date
            ]
            for e in ff_today:
                events.append(e)
                existing_names.add(e.name.lower())
            self._ff_events_cache = ff_all  # Cache full week for next-day lookups
            ff_loaded = len(ff_today) > 0 or len(ff_all) > 0
            logger.info(
                "economic_calendar.forexfactory_loaded",
                total_week=len(ff_all),
                today=len(ff_today),
                date=target_date,
            )
        except Exception:
            logger.exception("economic_calendar.forexfactory_error")

        # 2. Local file fallback
        if self._local_path:
            local_events = load_from_file(self._local_path)
            local_for_today = [
                e for e in local_events if e.time.strftime("%Y-%m-%d") == target_date
            ]
            for e in local_for_today:
                if e.name.lower() not in existing_names:
                    events.append(e)
                    existing_names.add(e.name.lower())

        # 3. Static FOMC fallback (only if ForexFactory didn't load)
        if not ff_loaded:
            fomc = _get_fomc_events(target_date)
            for e in fomc:
                if e.name.lower() not in existing_names:
                    events.append(e)
                    existing_names.add(e.name.lower())
            if fomc:
                logger.info(
                    "economic_calendar.fomc_fallback_used",
                    count=len(fomc),
                    msg="ForexFactory unavailable — using static FOMC dates",
                )

        # Sort by time
        events.sort(key=lambda e: e.time)

        self._events = events
        self._loaded_date = target_date

        logger.info(
            "economic_calendar.loaded",
            date=target_date,
            total_events=len(events),
            high_impact=sum(1 for e in events if e.impact == "high"),
        )

    @property
    def events(self) -> list[EconomicEvent]:
        """All loaded events for the day."""
        return self._events

    @property
    def loaded_date(self) -> str:
        """The date for which events were loaded."""
        return self._loaded_date

    def upcoming_events(
        self,
        within_minutes: int = 60,
        t: datetime | None = None,
    ) -> list[EconomicEvent]:
        """Get events happening within the next N minutes.

        Args:
            within_minutes: Look-ahead window in minutes.
            t: Reference time (defaults to now ET).
        """
        if t is None:
            t = datetime.now(ET)

        window_end = t + timedelta(minutes=within_minutes)

        return [
            e for e in self._events
            if t <= e.time <= window_end
        ]

    def past_events(
        self,
        within_minutes: int = 15,
        t: datetime | None = None,
    ) -> list[EconomicEvent]:
        """Get events that happened within the last N minutes.

        Useful for checking if we're in a post-news blackout.
        """
        if t is None:
            t = datetime.now(ET)

        window_start = t - timedelta(minutes=within_minutes)

        return [
            e for e in self._events
            if window_start <= e.time <= t
        ]

    def high_impact_today(self) -> list[EconomicEvent]:
        """Get all high-impact events for today."""
        return [e for e in self._events if e.impact == "high"]

    def has_high_impact_upcoming(
        self,
        within_minutes: int = 30,
        t: datetime | None = None,
    ) -> bool:
        """Check if any high-impact events are coming up soon."""
        upcoming = self.upcoming_events(within_minutes, t)
        return any(e.impact == "high" for e in upcoming)

    def is_in_blackout(
        self,
        t: datetime | None = None,
        pre_minutes: int = 5,
        post_minutes: int = 10,
    ) -> bool:
        """Check if we're in a news blackout window around a high-impact event.

        Blackout = within `pre_minutes` before OR `post_minutes` after
        any high-impact event.

        Args:
            t: Reference time (defaults to now ET).
            pre_minutes: Minutes before event to start blackout.
            post_minutes: Minutes after event to end blackout.
        """
        if t is None:
            t = datetime.now(ET)

        for event in self._events:
            if event.impact != "high":
                continue
            pre_start = event.time - timedelta(minutes=pre_minutes)
            post_end = event.time + timedelta(minutes=post_minutes)
            if pre_start <= t <= post_end:
                return True
        return False

    def blackout_until(
        self,
        t: datetime | None = None,
        pre_minutes: int = 5,
        post_minutes: int = 10,
    ) -> datetime | None:
        """Return when the current blackout ends, or None if not in blackout.

        Useful for knowing how long to wait before resuming trading.
        """
        if t is None:
            t = datetime.now(ET)

        for event in self._events:
            if event.impact != "high":
                continue
            pre_start = event.time - timedelta(minutes=pre_minutes)
            post_end = event.time + timedelta(minutes=post_minutes)
            if pre_start <= t <= post_end:
                return post_end
        return None

    def next_event(self, t: datetime | None = None) -> EconomicEvent | None:
        """Get the next upcoming event (any impact level)."""
        if t is None:
            t = datetime.now(ET)

        for event in self._events:
            if event.time > t:
                return event
        return None

    def next_high_impact_event(self, t: datetime | None = None) -> EconomicEvent | None:
        """Get the next upcoming high-impact event."""
        if t is None:
            t = datetime.now(ET)

        for event in self._events:
            if event.time > t and event.impact == "high":
                return event
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "loaded_date": self._loaded_date,
            "total_events": len(self._events),
            "high_impact": sum(1 for e in self._events if e.impact == "high"),
            "medium_impact": sum(1 for e in self._events if e.impact == "medium"),
            "low_impact": sum(1 for e in self._events if e.impact == "low"),
        }

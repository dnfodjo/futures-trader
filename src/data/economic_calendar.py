"""Economic calendar — loads events and identifies blackout windows.

Sources:
1. Static FOMC dates (hardcoded — always known far in advance)
2. Finnhub free API for daily economic releases (CPI, NFP, etc.)
3. Local JSON fallback if Finnhub is unavailable

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

# ── Static FOMC dates ─────────────────────────────────────────────────────
# These are ALWAYS high impact. Fed decisions at 2:00 PM ET, press conf at 2:30.
# Update annually from the Federal Reserve website.

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
    """Get FOMC events for a given date (YYYY-MM-DD)."""
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


# ── Finnhub Integration ──────────────────────────────────────────────────


async def fetch_from_finnhub(
    api_key: str,
    target_date: str,
    session: aiohttp.ClientSession | None = None,
) -> list[EconomicEvent]:
    """Fetch economic calendar events from Finnhub for a given date.

    Finnhub free tier: https://finnhub.io/api/v1/calendar/economic
    Rate limit: 60 calls/minute on free tier.

    Args:
        api_key: Finnhub API key (free tier works)
        target_date: "YYYY-MM-DD" format
        session: Optional aiohttp session to reuse
    """
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": target_date,
        "to": target_date,
        "token": api_key,
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning("finnhub.fetch_failed", status=resp.status)
                return []
            data = await resp.json()
    except Exception as e:
        logger.warning("finnhub.fetch_error", error=str(e))
        return []
    finally:
        if own_session:
            await session.close()

    return _parse_finnhub_response(data)


def _parse_finnhub_response(data: dict[str, Any]) -> list[EconomicEvent]:
    """Parse Finnhub economic calendar response into EconomicEvent list."""
    events = []
    raw_events = data.get("economicCalendar", [])

    for item in raw_events:
        # Only US events matter for MNQ trading
        country = item.get("country", "")
        if country != "US":
            continue

        name = item.get("event", "Unknown Event")
        impact_str = item.get("impact", "low")

        # Parse Finnhub impact levels
        impact = _classify_finnhub_impact(impact_str, name)

        # Parse time — Finnhub uses "YYYY-MM-DD HH:MM:SS" in UTC
        time_str = item.get("time", "")
        event_time = _parse_finnhub_time(time_str, item.get("date", ""))

        if event_time is None:
            continue

        events.append(
            EconomicEvent(
                time=event_time,
                name=name,
                impact=impact,
                prior=str(item.get("prev", "")) if item.get("prev") is not None else None,
                forecast=str(item.get("estimate", "")) if item.get("estimate") is not None else None,
            )
        )

    return events


def _classify_finnhub_impact(impact_str: str, name: str) -> str:
    """Classify event impact level.

    Some events are always high-impact regardless of Finnhub's classification.
    """
    # Always high-impact events
    always_high = {
        "nonfarm payrolls",
        "cpi",
        "consumer price index",
        "fomc",
        "fed interest rate",
        "initial jobless claims",
        "gdp",
        "pce",
        "ppi",
        "ism manufacturing",
        "ism services",
        "retail sales",
    }

    name_lower = name.lower()
    for keyword in always_high:
        if keyword in name_lower:
            return "high"

    # Map Finnhub's impact levels
    impact_map = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "3": "high",  # Finnhub numeric scale
        "2": "medium",
        "1": "low",
    }
    return impact_map.get(str(impact_str).lower(), "low")


def _parse_finnhub_time(time_str: str, date_str: str) -> datetime | None:
    """Parse Finnhub time format to timezone-aware datetime."""
    if time_str:
        try:
            # Format: "HH:MM:SS" — combine with date
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=ZoneInfo("UTC"))
        except ValueError:
            pass

    if date_str:
        try:
            # Fallback: just the date, assume 8:30 AM ET (common release time)
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=8, minute=30, tzinfo=ET
            )
            return dt
        except ValueError:
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

    Loads events from multiple sources, caches for the day, and provides
    query methods for the system to check upcoming events and blackout windows.

    Usage:
        calendar = EconomicCalendar(finnhub_api_key="xxx")
        await calendar.load_today()
        events = calendar.upcoming_events(within_minutes=30)
    """

    def __init__(
        self,
        finnhub_api_key: str = "",
        local_calendar_path: str | Path | None = None,
    ) -> None:
        self._finnhub_key = finnhub_api_key
        self._local_path = local_calendar_path
        self._events: list[EconomicEvent] = []
        self._loaded_date: str = ""

    async def load_today(self, target_date: str | None = None) -> None:
        """Load economic events for today (or a specific date).

        Fetches from Finnhub first, falls back to local file, and always
        includes static FOMC dates.

        Args:
            target_date: "YYYY-MM-DD" format. Defaults to today (ET).
        """
        if target_date is None:
            target_date = datetime.now(ET).strftime("%Y-%m-%d")

        events: list[EconomicEvent] = []

        # 1. Static FOMC dates (always included)
        fomc = _get_fomc_events(target_date)
        events.extend(fomc)
        if fomc:
            logger.info("economic_calendar.fomc_day", count=len(fomc))

        # Track existing event names for deduplication across all sources
        existing_names = {e.name.lower() for e in events}

        # 2. Finnhub API
        if self._finnhub_key:
            try:
                finnhub_events = await fetch_from_finnhub(self._finnhub_key, target_date)
                for e in finnhub_events:
                    if e.name.lower() not in existing_names:
                        events.append(e)
                        existing_names.add(e.name.lower())
                logger.info(
                    "economic_calendar.finnhub_loaded",
                    count=len(finnhub_events),
                    date=target_date,
                )
            except Exception:
                logger.exception("economic_calendar.finnhub_error")

        # 3. Local file fallback
        if self._local_path:
            local_events = load_from_file(self._local_path)
            # Filter to target date
            local_for_today = [
                e for e in local_events if e.time.strftime("%Y-%m-%d") == target_date
            ]
            # Deduplicate by name
            for e in local_for_today:
                if e.name.lower() not in existing_names:
                    events.append(e)
                    existing_names.add(e.name.lower())

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

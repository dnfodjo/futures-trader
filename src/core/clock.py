"""Market hours, session phase detection, and timezone logic.

All times are handled in US/Eastern. The clock is the single source of
truth for "can we trade right now?"
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.core.types import EconomicEvent, SessionPhase

ET = ZoneInfo("US/Eastern")

# ── US Market Holidays ────────────────────────────────────────────────────────
# CME Globex is CLOSED on these days. Dates must be updated annually.
# Source: CME Group holiday calendar.

_MARKET_HOLIDAYS: dict[int, set[date]] = {
    2025: {
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
    },
    2026: {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Jr. Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
    },
}

# Half-day early close dates (trading ends at 1:00 PM ET on these days)
# Typically day before/after major holidays.
_EARLY_CLOSE_DATES: dict[int, set[date]] = {
    2025: {
        date(2025, 7, 3),   # Day before July 4th
        date(2025, 11, 28), # Day after Thanksgiving
        date(2025, 12, 24), # Christmas Eve
    },
    2026: {
        date(2026, 7, 2),   # Day before July 4th (observed)
        date(2026, 11, 27), # Day after Thanksgiving
        date(2026, 12, 24), # Christmas Eve
    },
}


def _get_holidays(year: int) -> set[date]:
    """Get known holidays for a year. Returns empty set for unknown years."""
    return _MARKET_HOLIDAYS.get(year, set())


def _get_early_close_dates(year: int) -> set[date]:
    """Get known early close dates for a year."""
    return _EARLY_CLOSE_DATES.get(year, set())


def now_et() -> datetime:
    """Current time in US/Eastern."""
    return datetime.now(ET)


def to_et(dt: datetime) -> datetime:
    """Convert a datetime to US/Eastern."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
    return dt.astimezone(ET)


def get_session_phase(t: datetime | None = None) -> SessionPhase:
    """Determine the current session phase based on time of day (ET)."""
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    current = t.time()

    if current < time(9, 30):
        return SessionPhase.PRE_MARKET
    if current < time(10, 0):
        return SessionPhase.OPEN_DRIVE
    if current < time(12, 0):
        return SessionPhase.MORNING
    if current < time(14, 0):
        return SessionPhase.MIDDAY
    if current < time(15, 30):
        return SessionPhase.AFTERNOON
    if current < time(16, 0):
        return SessionPhase.CLOSE
    return SessionPhase.AFTER_HOURS


def is_trading_hours(start: str = "09:35", end: str = "15:50", t: datetime | None = None) -> bool:
    """Check if current time is within allowed trading hours.

    On early close dates, trading ends at 12:50 regardless of the `end` parameter.
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    current = t.time()
    start_time = time(*[int(x) for x in start.split(":")])
    end_time = time(*[int(x) for x in end.split(":")])

    # On early close days, cap the trading window at 12:50 ET
    if is_early_close(t):
        early_end = time(12, 50)
        if end_time > early_end:
            end_time = early_end

    return start_time <= current <= end_time


def is_past_hard_flatten(flatten_time: str = "15:55", t: datetime | None = None) -> bool:
    """Check if we're past the hard flatten time.

    On early close dates, hard flatten is at 12:55 PM ET.
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    current = t.time()

    # On early close days, override flatten time
    if is_early_close(t):
        flat_time = time(12, 55)
    else:
        flat_time = time(*[int(x) for x in flatten_time.split(":")])

    return current >= flat_time


def is_market_day(t: datetime | None = None) -> bool:
    """Check if today is a trading day (weekday and not a known holiday)."""
    if t is None:
        t = now_et()

    # Weekend check
    if t.weekday() >= 5:
        return False

    # Holiday check
    holidays = _get_holidays(t.year)
    if t.date() in holidays:
        return False

    return True


def is_early_close(t: datetime | None = None) -> bool:
    """Check if today is a half-day early close (trading ends ~1:00 PM ET)."""
    if t is None:
        t = now_et()

    early_dates = _get_early_close_dates(t.year)
    return t.date() in early_dates


def is_in_news_blackout(
    events: list[EconomicEvent],
    before_min: int = 5,
    after_min: int = 10,
    t: datetime | None = None,
) -> bool:
    """Check if we're in a news blackout window around a high/medium impact event.

    High impact events use the full blackout window (default: 5 min before, 10 min after).
    Medium impact events use a reduced window (2 min before, 5 min after).
    Low impact events are ignored.
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    for event in events:
        if event.impact == "low":
            continue

        event_time = to_et(event.time)

        if event.impact == "medium":
            # Reduced window for medium impact
            blackout_start = event_time - timedelta(minutes=max(1, before_min // 2))
            blackout_end = event_time + timedelta(minutes=max(2, after_min // 2))
        else:
            # Full window for high impact
            blackout_start = event_time - timedelta(minutes=before_min)
            blackout_end = event_time + timedelta(minutes=after_min)

        if blackout_start <= t <= blackout_end:
            return True

    return False


def seconds_until(target_time_str: str, t: datetime | None = None) -> float:
    """Seconds remaining until a target time today (ET)."""
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    parts = [int(x) for x in target_time_str.split(":")]
    target = t.replace(hour=parts[0], minute=parts[1], second=0, microsecond=0)

    if target <= t:
        return 0.0
    return (target - t).total_seconds()


def format_time_in_session(t: datetime | None = None) -> str:
    """Human-readable description of where we are in the session."""
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    market_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
    if t < market_open:
        return "pre-market"

    minutes_in = int((t - market_open).total_seconds() / 60)
    return f"{t.strftime('%I:%M %p')} ET, {minutes_in} min into session"


def get_effective_close_time(t: datetime | None = None) -> str:
    """Get the effective close time for today.

    Returns "13:00" on early close days, "16:00" on normal days.
    """
    if t is None:
        t = now_et()
    if is_early_close(t):
        return "13:00"
    return "16:00"


def get_effective_flatten_time(t: datetime | None = None) -> str:
    """Get the effective hard flatten time for today.

    Returns "12:55" on early close days, "15:55" on normal days.
    """
    if t is None:
        t = now_et()
    if is_early_close(t):
        return "12:55"
    return "15:55"

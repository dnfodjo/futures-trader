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
    """Determine the current session phase based on time of day (ET).

    CME Globex MNQ sessions (all times ET):
        18:00 - 02:00  Asian session (Sunday open at 18:00)
        02:00 - 08:00  London session
        08:00 - 09:30  Pre-RTH (econ data at 8:30)
        09:30 - 10:00  Open drive
        10:00 - 12:00  Morning
        12:00 - 14:00  Midday
        14:00 - 15:30  Afternoon
        15:30 - 16:00  Close
        16:00 - 16:55  Post-RTH
        17:00 - 18:00  Daily halt (CME maintenance)
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    current = t.time()

    # RTH phases (most common path — checked first)
    if time(9, 30) <= current < time(10, 0):
        return SessionPhase.OPEN_DRIVE
    if time(10, 0) <= current < time(12, 0):
        return SessionPhase.MORNING
    if time(12, 0) <= current < time(14, 0):
        return SessionPhase.MIDDAY
    if time(14, 0) <= current < time(15, 30):
        return SessionPhase.AFTERNOON
    if time(15, 30) <= current < time(16, 0):
        return SessionPhase.CLOSE

    # Post-RTH and daily halt
    if time(16, 0) <= current < time(17, 0):
        return SessionPhase.POST_RTH
    if time(17, 0) <= current < time(18, 0):
        return SessionPhase.DAILY_HALT

    # Overnight sessions (wraps past midnight)
    if current >= time(18, 0):
        return SessionPhase.ASIAN
    if current < time(2, 0):
        return SessionPhase.ASIAN
    if time(2, 0) <= current < time(8, 0):
        return SessionPhase.LONDON
    if time(8, 0) <= current < time(9, 30):
        return SessionPhase.PRE_RTH

    # Should never reach here, but defensive
    return SessionPhase.DAILY_HALT


def is_rth(phase: SessionPhase) -> bool:
    """Check if a session phase is Regular Trading Hours."""
    return phase in _RTH_PHASES


def is_eth(phase: SessionPhase) -> bool:
    """Check if a session phase is Extended Trading Hours (overnight/pre/post)."""
    return phase in _ETH_PHASES


# Phase sets for quick membership checks
_RTH_PHASES = frozenset({
    SessionPhase.OPEN_DRIVE,
    SessionPhase.MORNING,
    SessionPhase.MIDDAY,
    SessionPhase.AFTERNOON,
    SessionPhase.CLOSE,
})

_ETH_PHASES = frozenset({
    SessionPhase.ASIAN,
    SessionPhase.LONDON,
    SessionPhase.PRE_RTH,
    SessionPhase.POST_RTH,
})


def is_trading_hours(start: str = "18:05", end: str = "16:50", t: datetime | None = None) -> bool:
    """Check if current time is within allowed trading hours.

    Supports cross-midnight windows (e.g. start=18:05, end=16:50 wraps overnight).
    On early close dates, trading ends at 12:50 regardless of the `end` parameter.
    Always excludes the daily halt (17:00-18:00 ET).
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
        # Daily halt is still blocked
        if time(17, 0) <= current < time(18, 0):
            return False
        # Cross-midnight: session started prev evening, ends at 12:50 on early day
        if start_time > early_end:
            return current >= start_time or current <= early_end
        return start_time <= current <= early_end

    # Always block during daily halt (17:00-18:00 ET)
    if time(17, 0) <= current < time(18, 0):
        return False

    # Cross-midnight window: start > end means the window wraps past midnight
    # e.g. start=18:05, end=16:50 → active from 18:05 to midnight AND midnight to 16:50
    if start_time > end_time:
        return current >= start_time or current <= end_time

    # Normal non-wrapping window
    return start_time <= current <= end_time


def is_past_hard_flatten(flatten_time: str = "16:55", t: datetime | None = None) -> bool:
    """Check if we must flatten before the daily halt.

    The CME daily halt is 17:00-18:00 ET. We flatten at 16:55 to avoid
    being caught in the halt with an open position.

    On early close dates, hard flatten is at 12:55 PM ET.

    This function returns True ONLY in the narrow pre-halt window
    (flatten_time to 17:00). After 18:00, a new session begins and
    we should NOT be flattening.
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    current = t.time()

    # On early close days, override flatten time
    if is_early_close(t):
        flat_time = time(12, 55)
        return flat_time <= current < time(13, 5)

    flat_time = time(*[int(x) for x in flatten_time.split(":")])

    # Only flatten in the narrow pre-halt window, not for the entire night
    return flat_time <= current < time(17, 0)


def is_market_day(t: datetime | None = None) -> bool:
    """Check if Globex is open at the given time.

    CME Globex runs Sunday 18:00 ET through Friday 17:00 ET.
    Closed all day Saturday and Sunday before 18:00.
    Also closed on market holidays (checked against next business day
    for overnight sessions).
    """
    if t is None:
        t = now_et()
    else:
        t = to_et(t)

    weekday = t.weekday()  # 0=Mon ... 6=Sun

    # Saturday: always closed
    if weekday == 6 and t.time() < time(18, 0):
        # Sunday before 18:00 → closed
        return False
    if weekday == 5:
        # Saturday: always closed (Globex closes Friday 17:00)
        return False

    # Friday after 17:00: Globex is closed for the weekend
    if weekday == 4 and t.time() >= time(17, 0):
        return False

    # Holiday check: for overnight sessions (18:00+), check the NEXT day's
    # holiday status since that's the trading session it belongs to
    check_date = t.date()
    if t.time() >= time(18, 0):
        check_date = (t + timedelta(days=1)).date()

    holidays = _get_holidays(check_date.year)
    if check_date in holidays:
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

    phase = get_session_phase(t)
    return f"{t.strftime('%I:%M %p')} ET ({phase.value} session)"


def get_effective_close_time(t: datetime | None = None) -> str:
    """Get the effective session close time for today.

    Returns "13:00" on early close days, "17:00" on normal days
    (CME daily halt begins at 17:00 ET).
    """
    if t is None:
        t = now_et()
    if is_early_close(t):
        return "13:00"
    return "17:00"


def get_effective_flatten_time(t: datetime | None = None) -> str:
    """Get the effective hard flatten time for today.

    Returns "12:55" on early close days, "16:55" on normal days
    (flatten 5 min before the daily halt).
    """
    if t is None:
        t = now_et()
    if is_early_close(t):
        return "12:55"
    return "16:55"


def is_daily_halt(t: datetime | None = None) -> bool:
    """Check if we're in the CME daily maintenance halt (17:00-18:00 ET)."""
    if t is None:
        t = now_et()
    else:
        t = to_et(t)
    return time(17, 0) <= t.time() < time(18, 0)

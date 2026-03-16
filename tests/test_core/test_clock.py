"""Tests for the market clock module."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.core.clock import (
    format_time_in_session,
    get_effective_close_time,
    get_effective_flatten_time,
    get_session_phase,
    is_daily_halt,
    is_early_close,
    is_in_news_blackout,
    is_market_day,
    is_past_hard_flatten,
    is_trading_hours,
    seconds_until,
)
from src.core.types import EconomicEvent, SessionPhase

ET = ZoneInfo("US/Eastern")


def _et(hour: int, minute: int = 0, weekday: int = 0) -> datetime:
    """Create a datetime in ET for a specific time on a Monday (weekday=0).

    2026-03-16 is a Monday. weekday offsets from there.
    """
    day = 16 + weekday
    return datetime(2026, 3, day, hour, minute, tzinfo=ET)


# ── Session Phase Detection (24h Globex) ────────────────────────────────────


class TestGetSessionPhase:
    def test_asian_evening(self):
        assert get_session_phase(_et(18, 30)) == SessionPhase.ASIAN
        assert get_session_phase(_et(23, 0)) == SessionPhase.ASIAN

    def test_asian_past_midnight(self):
        assert get_session_phase(_et(0, 30)) == SessionPhase.ASIAN
        assert get_session_phase(_et(1, 59)) == SessionPhase.ASIAN

    def test_london(self):
        assert get_session_phase(_et(2, 0)) == SessionPhase.LONDON
        assert get_session_phase(_et(5, 0)) == SessionPhase.LONDON
        assert get_session_phase(_et(7, 59)) == SessionPhase.LONDON

    def test_pre_rth(self):
        assert get_session_phase(_et(8, 0)) == SessionPhase.PRE_RTH
        assert get_session_phase(_et(8, 30)) == SessionPhase.PRE_RTH
        assert get_session_phase(_et(9, 29)) == SessionPhase.PRE_RTH

    def test_open_drive(self):
        assert get_session_phase(_et(9, 30)) == SessionPhase.OPEN_DRIVE
        assert get_session_phase(_et(9, 45)) == SessionPhase.OPEN_DRIVE

    def test_morning(self):
        assert get_session_phase(_et(10, 0)) == SessionPhase.MORNING
        assert get_session_phase(_et(11, 30)) == SessionPhase.MORNING

    def test_midday(self):
        assert get_session_phase(_et(12, 0)) == SessionPhase.MIDDAY
        assert get_session_phase(_et(13, 30)) == SessionPhase.MIDDAY

    def test_afternoon(self):
        assert get_session_phase(_et(14, 0)) == SessionPhase.AFTERNOON
        assert get_session_phase(_et(15, 0)) == SessionPhase.AFTERNOON

    def test_close(self):
        assert get_session_phase(_et(15, 30)) == SessionPhase.CLOSE
        assert get_session_phase(_et(15, 55)) == SessionPhase.CLOSE

    def test_post_rth(self):
        assert get_session_phase(_et(16, 0)) == SessionPhase.POST_RTH
        assert get_session_phase(_et(16, 50)) == SessionPhase.POST_RTH

    def test_daily_halt(self):
        assert get_session_phase(_et(17, 0)) == SessionPhase.DAILY_HALT
        assert get_session_phase(_et(17, 30)) == SessionPhase.DAILY_HALT
        assert get_session_phase(_et(17, 59)) == SessionPhase.DAILY_HALT

    def test_boundaries_are_correct(self):
        """Test exact boundary times between phases."""
        assert get_session_phase(_et(18, 0)) == SessionPhase.ASIAN  # Asian starts
        assert get_session_phase(_et(2, 0)) == SessionPhase.LONDON  # London starts
        assert get_session_phase(_et(8, 0)) == SessionPhase.PRE_RTH  # Pre-RTH starts
        assert get_session_phase(_et(9, 30)) == SessionPhase.OPEN_DRIVE
        assert get_session_phase(_et(10, 0)) == SessionPhase.MORNING
        assert get_session_phase(_et(12, 0)) == SessionPhase.MIDDAY
        assert get_session_phase(_et(14, 0)) == SessionPhase.AFTERNOON
        assert get_session_phase(_et(15, 30)) == SessionPhase.CLOSE
        assert get_session_phase(_et(16, 0)) == SessionPhase.POST_RTH
        assert get_session_phase(_et(17, 0)) == SessionPhase.DAILY_HALT


# ── Trading Hours (cross-midnight) ──────────────────────────────────────────


class TestIsTradingHours:
    """Default window: 18:05 to 16:50 (wraps overnight)."""

    def test_asian_session_active(self):
        assert is_trading_hours(t=_et(18, 30)) is True
        assert is_trading_hours(t=_et(23, 0)) is True

    def test_past_midnight_active(self):
        assert is_trading_hours(t=_et(1, 0)) is True

    def test_london_active(self):
        assert is_trading_hours(t=_et(3, 0)) is True

    def test_rth_active(self):
        assert is_trading_hours(t=_et(10, 0)) is True
        assert is_trading_hours(t=_et(15, 0)) is True

    def test_at_open(self):
        assert is_trading_hours(t=_et(18, 5)) is True

    def test_before_open(self):
        # 18:04 is before the 18:05 start
        assert is_trading_hours(t=_et(18, 4)) is False

    def test_at_close(self):
        assert is_trading_hours(t=_et(16, 50)) is True

    def test_after_close(self):
        assert is_trading_hours(t=_et(16, 51)) is False

    def test_daily_halt_blocked(self):
        """Daily halt (17:00-18:00) is always blocked."""
        assert is_trading_hours(t=_et(17, 0)) is False
        assert is_trading_hours(t=_et(17, 30)) is False
        assert is_trading_hours(t=_et(17, 59)) is False


# ── Hard Flatten ────────────────────────────────────────────────────────────


class TestIsPastHardFlatten:
    def test_before_flatten(self):
        assert is_past_hard_flatten(t=_et(16, 54)) is False

    def test_at_flatten(self):
        assert is_past_hard_flatten(t=_et(16, 55)) is True

    def test_during_pre_halt(self):
        assert is_past_hard_flatten(t=_et(16, 58)) is True

    def test_not_past_halt(self):
        """After the halt resumes (18:00+), we should NOT be flattening."""
        assert is_past_hard_flatten(t=_et(18, 30)) is False

    def test_overnight_not_flattening(self):
        """At 2 AM in London session, flatten should be False."""
        assert is_past_hard_flatten(t=_et(2, 0)) is False

    def test_morning_not_flattening(self):
        assert is_past_hard_flatten(t=_et(10, 0)) is False


# ── Daily Halt ──────────────────────────────────────────────────────────────


class TestIsDailyHalt:
    def test_in_halt(self):
        assert is_daily_halt(_et(17, 0)) is True
        assert is_daily_halt(_et(17, 30)) is True

    def test_not_in_halt(self):
        assert is_daily_halt(_et(16, 59)) is False
        assert is_daily_halt(_et(18, 0)) is False
        assert is_daily_halt(_et(10, 0)) is False


# ── Market Day (Globex schedule) ────────────────────────────────────────────


class TestIsMarketDay:
    def test_monday(self):
        assert is_market_day(_et(10, 0, weekday=0)) is True

    def test_friday_rth(self):
        assert is_market_day(_et(10, 0, weekday=4)) is True

    def test_friday_after_close(self):
        """Friday 17:00+ → Globex closed for weekend."""
        assert is_market_day(_et(17, 30, weekday=4)) is False

    def test_saturday(self):
        assert is_market_day(_et(10, 0, weekday=5)) is False

    def test_sunday_before_open(self):
        """Sunday 10 AM → Globex not open yet."""
        assert is_market_day(_et(10, 0, weekday=6)) is False

    def test_sunday_after_open(self):
        """Sunday 18:30 → Globex just opened."""
        assert is_market_day(_et(18, 30, weekday=6)) is True

    def test_holiday_christmas_2026(self):
        """Christmas 2026 is a Friday — market closed."""
        xmas = datetime(2026, 12, 25, 10, 0, tzinfo=ET)
        assert is_market_day(xmas) is False

    def test_holiday_thanksgiving_2026(self):
        """Thanksgiving 2026 is Nov 26 — market closed."""
        thanksgiving = datetime(2026, 11, 26, 10, 0, tzinfo=ET)
        assert is_market_day(thanksgiving) is False

    def test_holiday_mlk_2026(self):
        """MLK Day 2026 is Jan 19."""
        mlk = datetime(2026, 1, 19, 10, 0, tzinfo=ET)
        assert is_market_day(mlk) is False

    def test_normal_weekday_not_holiday(self):
        """A regular Wednesday in March is a market day."""
        wed = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert is_market_day(wed) is True

    def test_unknown_year_has_no_holidays(self):
        """For years without a holiday list, only weekday check applies."""
        future = datetime(2030, 12, 25, 10, 0, tzinfo=ET)  # Wed Dec 25 2030
        assert is_market_day(future) is True  # It's a Wednesday

    def test_overnight_before_holiday(self):
        """Night before a holiday: 18:00+ should check next day (holiday)."""
        # Night before MLK Day 2026 (Jan 19 is Mon, so Jan 18 is Sun 18:00+)
        # Actually let's use a weekday example: night before Christmas (Dec 24 evening)
        # Dec 25 is the holiday, so Dec 24 at 18:30 should check Dec 25
        xmas_eve_night = datetime(2026, 12, 24, 18, 30, tzinfo=ET)
        assert is_market_day(xmas_eve_night) is False


# ── News Blackout ───────────────────────────────────────────────────────────


class TestIsInNewsBlackout:
    def test_before_blackout(self):
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="FOMC",
            impact="high",
        )
        # 12 minutes before — outside blackout
        t = datetime(2026, 3, 16, 13, 48, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is False

    def test_in_blackout_before(self):
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="FOMC",
            impact="high",
        )
        # 3 minutes before — inside blackout
        t = datetime(2026, 3, 16, 13, 57, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is True

    def test_in_blackout_after(self):
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="FOMC",
            impact="high",
        )
        # 5 minutes after — inside blackout
        t = datetime(2026, 3, 16, 14, 5, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is True

    def test_after_blackout(self):
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="FOMC",
            impact="high",
        )
        # 11 minutes after — outside blackout
        t = datetime(2026, 3, 16, 14, 11, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is False

    def test_low_impact_ignored(self):
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="Some Low Impact",
            impact="low",
        )
        t = datetime(2026, 3, 16, 13, 57, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is False

    def test_medium_impact_has_reduced_window(self):
        """Medium impact uses half the blackout window."""
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="ADP Employment",
            impact="medium",
        )
        # 2 minutes before — inside medium blackout (default before=5, half=2)
        t = datetime(2026, 3, 16, 13, 58, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is True

        # 4 minutes before — outside medium blackout but inside high blackout
        t = datetime(2026, 3, 16, 13, 56, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is False

    def test_medium_impact_after_reduced_window(self):
        """Medium impact after window is shorter (5 min, not 10)."""
        event = EconomicEvent(
            time=datetime(2026, 3, 16, 14, 0, tzinfo=ET),
            name="ADP Employment",
            impact="medium",
        )
        # 4 minutes after — inside medium blackout (half of 10 = 5)
        t = datetime(2026, 3, 16, 14, 4, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is True

        # 6 minutes after — outside medium blackout
        t = datetime(2026, 3, 16, 14, 6, tzinfo=ET)
        assert is_in_news_blackout([event], t=t) is False

    def test_no_events(self):
        assert is_in_news_blackout([], t=_et(12, 0)) is False


# ── Utility Functions ───────────────────────────────────────────────────────


class TestSecondsUntil:
    def test_future_time(self):
        t = _et(15, 0)
        secs = seconds_until("15:55", t=t)
        assert secs == 55 * 60  # 55 minutes

    def test_past_time(self):
        t = _et(16, 0)
        secs = seconds_until("15:55", t=t)
        assert secs == 0.0


class TestFormatTimeInSession:
    def test_overnight(self):
        result = format_time_in_session(_et(3, 0))
        assert "london" in result

    def test_rth(self):
        result = format_time_in_session(_et(10, 30))
        assert "morning" in result

    def test_pre_rth(self):
        result = format_time_in_session(_et(9, 0))
        assert "pre_rth" in result


# ── Early Close ─────────────────────────────────────────────────────────────


class TestEarlyClose:
    def test_christmas_eve_2026_is_early_close(self):
        xmas_eve = datetime(2026, 12, 24, 10, 0, tzinfo=ET)
        assert is_early_close(xmas_eve) is True

    def test_normal_day_is_not_early_close(self):
        normal = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert is_early_close(normal) is False

    def test_day_after_thanksgiving_2026_is_early_close(self):
        black_friday = datetime(2026, 11, 27, 10, 0, tzinfo=ET)
        assert is_early_close(black_friday) is True

    def test_trading_hours_capped_on_early_close(self):
        """On early close days, trading should end at 12:50."""
        xmas_eve = datetime(2026, 12, 24, 12, 45, tzinfo=ET)
        assert is_trading_hours(t=xmas_eve) is True

        xmas_eve_late = datetime(2026, 12, 24, 13, 0, tzinfo=ET)
        assert is_trading_hours(t=xmas_eve_late) is False

    def test_hard_flatten_on_early_close(self):
        """On early close, hard flatten should be at 12:55."""
        xmas_eve = datetime(2026, 12, 24, 12, 54, tzinfo=ET)
        assert is_past_hard_flatten(t=xmas_eve) is False

        xmas_eve_flat = datetime(2026, 12, 24, 12, 55, tzinfo=ET)
        assert is_past_hard_flatten(t=xmas_eve_flat) is True


# ── Effective Times ─────────────────────────────────────────────────────────


class TestEffectiveTimes:
    def test_effective_close_normal_day(self):
        normal = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert get_effective_close_time(normal) == "17:00"

    def test_effective_close_early_day(self):
        early = datetime(2026, 12, 24, 10, 0, tzinfo=ET)
        assert get_effective_close_time(early) == "13:00"

    def test_effective_flatten_normal_day(self):
        normal = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert get_effective_flatten_time(normal) == "16:55"

    def test_effective_flatten_early_day(self):
        early = datetime(2026, 12, 24, 10, 0, tzinfo=ET)
        assert get_effective_flatten_time(early) == "12:55"

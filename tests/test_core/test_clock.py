"""Tests for the market clock module."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.core.clock import (
    format_time_in_session,
    get_effective_close_time,
    get_effective_flatten_time,
    get_session_phase,
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
    """Create a datetime in ET for a specific time on a Monday (weekday=0)."""
    # 2026-03-16 is a Monday
    day = 16 + weekday
    return datetime(2026, 3, day, hour, minute, tzinfo=ET)


class TestGetSessionPhase:
    def test_pre_market(self):
        assert get_session_phase(_et(8, 30)) == SessionPhase.PRE_MARKET

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

    def test_after_hours(self):
        assert get_session_phase(_et(16, 0)) == SessionPhase.AFTER_HOURS
        assert get_session_phase(_et(17, 30)) == SessionPhase.AFTER_HOURS


class TestIsTradingHours:
    def test_before_open(self):
        assert is_trading_hours(t=_et(9, 30)) is False

    def test_at_open(self):
        assert is_trading_hours(t=_et(9, 35)) is True

    def test_midday(self):
        assert is_trading_hours(t=_et(12, 0)) is True

    def test_at_close(self):
        assert is_trading_hours(t=_et(15, 50)) is True

    def test_after_close(self):
        assert is_trading_hours(t=_et(15, 51)) is False


class TestIsPastHardFlatten:
    def test_before(self):
        assert is_past_hard_flatten(t=_et(15, 54)) is False

    def test_at_flatten(self):
        assert is_past_hard_flatten(t=_et(15, 55)) is True

    def test_after(self):
        assert is_past_hard_flatten(t=_et(16, 0)) is True


class TestIsMarketDay:
    def test_monday(self):
        assert is_market_day(_et(10, 0, weekday=0)) is True

    def test_friday(self):
        assert is_market_day(_et(10, 0, weekday=4)) is True

    def test_saturday(self):
        assert is_market_day(_et(10, 0, weekday=5)) is False

    def test_sunday(self):
        assert is_market_day(_et(10, 0, weekday=6)) is False

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
        # No holiday list for 2030, so weekday check only
        assert is_market_day(future) is True  # It's a Wednesday


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
    def test_pre_market(self):
        result = format_time_in_session(_et(9, 0))
        assert result == "pre-market"

    def test_during_session(self):
        result = format_time_in_session(_et(10, 30))
        assert "60 min into session" in result


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
        """On early close days, trading should end at 12:50, not 15:50."""
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


class TestEffectiveTimes:
    def test_effective_close_normal_day(self):
        normal = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert get_effective_close_time(normal) == "16:00"

    def test_effective_close_early_day(self):
        early = datetime(2026, 12, 24, 10, 0, tzinfo=ET)
        assert get_effective_close_time(early) == "13:00"

    def test_effective_flatten_normal_day(self):
        normal = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert get_effective_flatten_time(normal) == "15:55"

    def test_effective_flatten_early_day(self):
        early = datetime(2026, 12, 24, 10, 0, tzinfo=ET)
        assert get_effective_flatten_time(early) == "12:55"

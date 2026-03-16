"""Tests for the trade logger — SQLite trade journal."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.types import Regime, SessionPhase, Side, TradeRecord
from src.learning.trade_logger import TradeLogger


def _trade(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),
        timestamp_exit=datetime(2026, 3, 14, 15, 0, 0, tzinfo=UTC),
        side=Side.LONG,
        entry_quantity=2,
        exit_quantity=2,
        entry_price=19850.0,
        exit_price=19870.0,
        stop_price=19830.0,
        pnl=80.0,
        commissions=1.72,
        hold_time_sec=1800,
        max_favorable_excursion=100.0,
        max_adverse_excursion=-20.0,
        reasoning_entry="Breakout above VWAP with strong delta",
        reasoning_exit="Target reached",
        regime_at_entry=Regime.TRENDING_UP,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ── Initialization ────────────────────────────────────────────────────────


class TestInitialization:
    def test_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            logger = TradeLogger(db_path=str(db_path))
            assert db_path.exists()
            logger.close()

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "deep" / "test.db"
            logger = TradeLogger(db_path=str(db_path))
            assert db_path.exists()
            logger.close()

    def test_initializes_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            logger = TradeLogger(db_path=str(db_path))

            # Should be able to query empty tables
            trades = logger.get_recent_trades()
            assert trades == []
            logger.close()


# ── Log Trade ─────────────────────────────────────────────────────────────


class TestLogTrade:
    def test_log_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            trade = _trade()
            logger.log_trade(trade)

            trades = logger.get_recent_trades()
            assert len(trades) == 1
            assert trades[0].id == trade.id
            assert trades[0].entry_price == 19850.0
            assert trades[0].pnl == 80.0
            logger.close()

    def test_log_preserves_all_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            trade = _trade(
                reasoning_entry="Strong breakout",
                reasoning_exit="Hit target",
                debate_result={"bull": "Trend is strong", "bear": "Overextended"},
                adds=1,
                scale_outs=2,
            )
            logger.log_trade(trade)

            result = logger.get_recent_trades()[0]
            assert result.reasoning_entry == "Strong breakout"
            assert result.reasoning_exit == "Hit target"
            assert result.debate_result == {"bull": "Trend is strong", "bear": "Overextended"}
            assert result.adds == 1
            assert result.scale_outs == 2
            logger.close()

    def test_log_trade_with_none_pnl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            trade = _trade(pnl=None, exit_price=None, timestamp_exit=None)
            logger.log_trade(trade)

            result = logger.get_recent_trades()[0]
            assert result.pnl is None
            assert result.exit_price is None
            logger.close()

    def test_log_multiple_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            for i in range(5):
                logger.log_trade(_trade(pnl=float(i * 10)))

            trades = logger.get_recent_trades()
            assert len(trades) == 5
            logger.close()

    def test_upsert_on_same_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            trade = _trade(pnl=50.0)
            logger.log_trade(trade)

            # Update same trade with new pnl
            trade.pnl = 100.0
            logger.log_trade(trade)

            trades = logger.get_recent_trades()
            assert len(trades) == 1
            assert trades[0].pnl == 100.0
            logger.close()


# ── Query by Date ─────────────────────────────────────────────────────────


class TestQueryByDate:
    def test_get_trades_by_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            logger.log_trade(
                _trade(timestamp_entry=datetime(2026, 3, 14, 10, 0, tzinfo=UTC))
            )
            logger.log_trade(
                _trade(timestamp_entry=datetime(2026, 3, 15, 10, 0, tzinfo=UTC))
            )

            trades_14 = logger.get_trades_by_date("2026-03-14")
            trades_15 = logger.get_trades_by_date("2026-03-15")

            assert len(trades_14) == 1
            assert len(trades_15) == 1
            logger.close()

    def test_get_trades_by_date_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            for day in [14, 15, 16]:
                logger.log_trade(
                    _trade(
                        timestamp_entry=datetime(2026, 3, day, 10, 0, tzinfo=UTC)
                    )
                )

            trades = logger.get_trades_by_date_range("2026-03-14", "2026-03-15")
            assert len(trades) == 2
            logger.close()

    def test_get_trades_empty_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            trades = logger.get_trades_by_date("2026-01-01")
            assert trades == []
            logger.close()


# ── Query by Regime / Side ────────────────────────────────────────────────


class TestQueryByAttribute:
    def test_get_trades_by_regime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            logger.log_trade(_trade(regime_at_entry=Regime.TRENDING_UP))
            logger.log_trade(_trade(regime_at_entry=Regime.CHOPPY))
            logger.log_trade(_trade(regime_at_entry=Regime.TRENDING_UP))

            trending = logger.get_trades_by_regime(Regime.TRENDING_UP)
            choppy = logger.get_trades_by_regime(Regime.CHOPPY)

            assert len(trending) == 2
            assert len(choppy) == 1
            logger.close()

    def test_get_trades_by_side(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            logger.log_trade(_trade(side=Side.LONG))
            logger.log_trade(_trade(side=Side.SHORT))
            logger.log_trade(_trade(side=Side.LONG))

            longs = logger.get_trades_by_side(Side.LONG)
            shorts = logger.get_trades_by_side(Side.SHORT)

            assert len(longs) == 2
            assert len(shorts) == 1
            logger.close()

    def test_get_trades_by_session_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            logger.log_trade(_trade(session_phase_at_entry=SessionPhase.MORNING))
            logger.log_trade(_trade(session_phase_at_entry=SessionPhase.AFTERNOON))
            logger.log_trade(_trade(session_phase_at_entry=SessionPhase.MORNING))

            morning = logger.get_trades_by_session_phase(SessionPhase.MORNING)
            afternoon = logger.get_trades_by_session_phase(SessionPhase.AFTERNOON)

            assert len(morning) == 2
            assert len(afternoon) == 1
            logger.close()


# ── Daily Summary ─────────────────────────────────────────────────────────


class TestDailySummary:
    def test_log_and_retrieve_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            logger.log_daily_summary(
                date="2026-03-14",
                total_trades=5,
                winners=3,
                losers=2,
                gross_pnl=250.0,
                net_pnl=240.0,
                commissions=10.0,
                max_drawdown=80.0,
                postmortem="Good day overall.",
                grade="B",
            )

            summary = logger.get_daily_summary("2026-03-14")
            assert summary is not None
            assert summary["total_trades"] == 5
            assert summary["winners"] == 3
            assert summary["net_pnl"] == 240.0
            assert summary["grade"] == "B"
            logger.close()

    def test_get_recent_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            for day in [14, 15, 16]:
                logger.log_daily_summary(
                    date=f"2026-03-{day:02d}",
                    total_trades=day,
                    winners=day - 1,
                    losers=1,
                    gross_pnl=100.0 * day,
                    net_pnl=90.0 * day,
                    commissions=10.0,
                )

            summaries = logger.get_recent_summaries(limit=2)
            assert len(summaries) == 2
            # Most recent first
            assert summaries[0]["date"] == "2026-03-16"
            logger.close()

    def test_missing_summary_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            assert logger.get_daily_summary("2026-01-01") is None
            logger.close()


# ── Consecutive Red Days ──────────────────────────────────────────────────


class TestConsecutiveRedDays:
    def test_no_summaries_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            assert logger.get_consecutive_red_days() == 0
            logger.close()

    def test_three_consecutive_red_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            for day, pnl in [(12, -50), (13, -80), (14, -30)]:
                logger.log_daily_summary(
                    date=f"2026-03-{day:02d}",
                    total_trades=3,
                    winners=1,
                    losers=2,
                    gross_pnl=pnl,
                    net_pnl=pnl,
                    commissions=5.0,
                )

            assert logger.get_consecutive_red_days() == 3
            logger.close()

    def test_green_day_breaks_streak(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            # Day 12: green, Day 13: red, Day 14: red
            for day, pnl in [(12, 100), (13, -80), (14, -30)]:
                logger.log_daily_summary(
                    date=f"2026-03-{day:02d}",
                    total_trades=3,
                    winners=2,
                    losers=1,
                    gross_pnl=pnl,
                    net_pnl=pnl,
                    commissions=5.0,
                )

            # Only 2 red days (13 and 14), day 12 is green
            assert logger.get_consecutive_red_days() == 2
            logger.close()


# ── Trade Count ───────────────────────────────────────────────────────────


class TestTradeCount:
    def test_total_trade_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            for _ in range(7):
                logger.log_trade(_trade())

            assert logger.get_total_trade_count() == 7
            logger.close()

    def test_empty_db_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(db_path=f"{tmpdir}/test.db")
            assert logger.get_total_trade_count() == 0
            logger.close()

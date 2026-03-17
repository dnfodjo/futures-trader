"""Tests for the SessionController — P&L tracking and profit preservation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import Side, TradeRecord
from src.agents.session_controller import SessionController


def _make_trade(pnl: float, commission: float = 0.86) -> TradeRecord:
    """Create a test TradeRecord with specified P&L."""
    return TradeRecord(
        timestamp_entry=datetime.now(tz=UTC),
        side=Side.LONG,
        entry_quantity=1,
        entry_price=19850.0,
        exit_price=19850.0 + pnl / 2.0,  # $2/pt for MNQ
        stop_price=19830.0,
        pnl=pnl,
        commissions=commission,
    )


# ── Test: Initialization ────────────────────────────────────────────────────


class TestInit:
    def test_defaults(self) -> None:
        ctrl = SessionController()
        assert ctrl.daily_pnl == 0.0
        assert ctrl.total_trades == 0
        assert not ctrl.should_stop_trading

    def test_custom_config(self) -> None:
        ctrl = SessionController(
            max_daily_loss=500.0,
            profit_tier1_pnl=300.0,
            profit_tier1_max_size=4,
        )
        assert ctrl._max_daily_loss == 500.0
        assert ctrl._tier1_pnl == 300.0
        assert ctrl._tier1_max == 4


# ── Test: Session Lifecycle ──────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_start_session(self) -> None:
        ctrl = SessionController()
        ctrl.start_session("2025-03-14")
        assert ctrl.session_date == "2025-03-14"
        assert ctrl.daily_pnl == 0.0

    def test_start_session_auto_date(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        assert ctrl.session_date != ""

    def test_start_session_resets_stats(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        assert ctrl.total_trades == 1

        ctrl.start_session("2025-03-15")
        assert ctrl.total_trades == 0
        assert ctrl.daily_pnl == 0.0
        assert ctrl.winners == 0


# ── Test: Trade Recording ────────────────────────────────────────────────────


class TestTradeRecording:
    def test_winning_trade(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        assert ctrl.winners == 1
        assert ctrl.losers == 0
        assert ctrl.gross_pnl == 100.0
        assert ctrl.commissions == 0.86

    def test_losing_trade(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(-50.0))
        assert ctrl.winners == 0
        assert ctrl.losers == 1
        assert ctrl.consecutive_losers == 1

    def test_scratch_trade(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(0.0))
        assert ctrl.scratches == 1
        assert ctrl.winners == 0
        assert ctrl.losers == 0

    def test_daily_pnl_is_net(self) -> None:
        """Daily P&L = gross - commissions."""
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0, commission=0.86))
        # gross=100, commission=0.86, net=99.14
        assert ctrl.daily_pnl == pytest.approx(99.14, abs=0.01)

    def test_multiple_trades(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        ctrl.record_trade(_make_trade(50.0))
        ctrl.record_trade(_make_trade(-30.0))
        assert ctrl.total_trades == 3
        assert ctrl.winners == 2
        assert ctrl.losers == 1
        assert ctrl.gross_pnl == 120.0

    def test_consecutive_losers_tracking(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(-10.0))
        ctrl.record_trade(_make_trade(-20.0))
        assert ctrl.consecutive_losers == 2
        assert ctrl.max_consecutive_losers == 2

        # Win resets consecutive losers
        ctrl.record_trade(_make_trade(50.0))
        assert ctrl.consecutive_losers == 0
        assert ctrl.max_consecutive_losers == 2  # keeps the max


# ── Test: Win Rate ───────────────────────────────────────────────────────────


class TestWinRate:
    def test_no_trades(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        assert ctrl.win_rate == 0.0

    def test_all_winners(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        ctrl.record_trade(_make_trade(50.0))
        assert ctrl.win_rate == 100.0

    def test_mixed(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        ctrl.record_trade(_make_trade(-50.0))
        assert ctrl.win_rate == 50.0

    def test_scratches_excluded(self) -> None:
        """Scratches don't count in win rate calculation."""
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        ctrl.record_trade(_make_trade(0.0))  # scratch
        # Only 1 decided trade (the winner), win rate = 100%
        assert ctrl.win_rate == 100.0


# ── Test: Max Drawdown ───────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_no_drawdown(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0))
        ctrl.record_trade(_make_trade(50.0))
        assert ctrl.max_drawdown == 0.0

    def test_drawdown_from_peak(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(200.0, commission=0.0))
        ctrl.record_trade(_make_trade(-80.0, commission=0.0))
        # Peak was 200, current = 120, drawdown = 80
        assert ctrl.max_drawdown == pytest.approx(80.0, abs=0.01)

    def test_drawdown_tracks_worst(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(300.0, commission=0.0))
        ctrl.record_trade(_make_trade(-100.0, commission=0.0))
        # Drawdown = 100
        ctrl.record_trade(_make_trade(50.0, commission=0.0))
        # Peak still 300, current = 250, drawdown now 50 from peak
        # But max drawdown stays at 100
        assert ctrl.max_drawdown == pytest.approx(100.0, abs=0.01)


# ── Test: Profit Preservation ────────────────────────────────────────────────


class TestProfitPreservation:
    def test_no_preservation_by_default(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        assert ctrl.effective_max_contracts == 10
        assert ctrl.profit_preservation_tier == 0
        assert not ctrl.profit_preservation_active

    def test_tier1_at_200(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(250.0, commission=0.0))
        assert ctrl.effective_max_contracts == 6
        assert ctrl.profit_preservation_tier == 1
        assert ctrl.profit_preservation_active

    def test_tier2_at_400(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(450.0, commission=0.0))
        assert ctrl.effective_max_contracts == 4
        assert ctrl.profit_preservation_tier == 2

    def test_tier_not_active_below_threshold(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(150.0, commission=0.0))
        assert ctrl.effective_max_contracts == 10
        assert ctrl.profit_preservation_tier == 0

    def test_custom_tier_values(self) -> None:
        ctrl = SessionController(
            profit_tier1_pnl=100.0,
            profit_tier1_max_size=4,
            profit_tier2_pnl=300.0,
            profit_tier2_max_size=1,
        )
        ctrl.start_session()
        ctrl.record_trade(_make_trade(150.0, commission=0.0))
        assert ctrl.effective_max_contracts == 4

        ctrl.record_trade(_make_trade(200.0, commission=0.0))
        assert ctrl.effective_max_contracts == 1


# ── Test: Daily Loss Limit ───────────────────────────────────────────────────


class TestDailyLossLimit:
    def test_stop_at_daily_loss(self) -> None:
        ctrl = SessionController(max_daily_loss=400.0)
        ctrl.start_session()
        ctrl.record_trade(_make_trade(-410.0, commission=0.0))
        assert ctrl.should_stop_trading
        assert "Daily loss limit" in ctrl.stop_reason

    def test_not_stopped_within_limit(self) -> None:
        ctrl = SessionController(max_daily_loss=400.0)
        ctrl.start_session()
        ctrl.record_trade(_make_trade(-200.0, commission=0.0))
        assert not ctrl.should_stop_trading

    def test_force_stop(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.force_stop("Manual intervention")
        assert ctrl.should_stop_trading
        assert ctrl.stop_reason == "Manual intervention"


# ── Test: PnL Per Trade ──────────────────────────────────────────────────────


class TestPnlPerTrade:
    def test_no_trades(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        assert ctrl.pnl_per_trade == 0.0

    def test_positive_average(self) -> None:
        ctrl = SessionController()
        ctrl.start_session()
        ctrl.record_trade(_make_trade(100.0, commission=0.0))
        ctrl.record_trade(_make_trade(50.0, commission=0.0))
        assert ctrl.pnl_per_trade == 75.0


# ── Test: Stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_structure(self) -> None:
        ctrl = SessionController()
        ctrl.start_session("2025-03-14")
        stats = ctrl.stats
        assert stats["session_date"] == "2025-03-14"
        assert "daily_pnl" in stats
        assert "win_rate" in stats
        assert "effective_max_contracts" in stats
        assert "is_stopped" in stats

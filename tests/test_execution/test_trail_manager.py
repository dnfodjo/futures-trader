"""Tests for the TrailManager — client-side trailing stop logic."""

from __future__ import annotations

import pytest

from src.core.types import PositionState, Side
from src.execution.trail_manager import TrailManager


def _position(
    side: Side = Side.LONG,
    avg_entry: float = 19850.0,
    stop_price: float = 19840.0,
    quantity: int = 2,
) -> PositionState:
    return PositionState(
        side=side,
        quantity=quantity,
        avg_entry=avg_entry,
        stop_price=stop_price,
    )


# ── Activation ───────────────────────────────────────────────────────────────


class TestActivation:
    def test_activate_long(self):
        trail = TrailManager(trail_distance=8.0, activation_profit_pts=4.0)
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19855.0)

        assert trail.is_active is True
        assert trail.peak_price == 19855.0
        assert trail.last_sent_stop == 19840.0

    def test_activate_short(self):
        trail = TrailManager(trail_distance=8.0, activation_profit_pts=4.0)
        pos = _position(side=Side.SHORT, avg_entry=19850.0, stop_price=19860.0)
        trail.activate(pos, 19845.0)

        assert trail.is_active is True

    def test_deactivate(self):
        trail = TrailManager()
        pos = _position()
        trail.activate(pos, 19855.0)
        trail.deactivate()

        assert trail.is_active is False
        assert trail.peak_price == 0.0

    def test_not_active_initially(self):
        trail = TrailManager()
        assert trail.is_active is False
        assert trail.update(19860.0) is None


# ── Long Position Trailing ───────────────────────────────────────────────────


class TestLongTrailing:
    def test_no_trail_below_activation_threshold(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19852.0)

        # Only 2 pts profit, need 4 to activate
        result = trail.update(19852.0)
        assert result is None

    def test_trail_after_activation(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # Price goes up enough to activate and produce trail stop > last sent
        # Entry=19850, current=19858 → 8pts profit ≥ 4 activation
        # Peak=19858, trail=8 → ideal_stop=19850 > last_sent(19840) → yes
        # Diff = 19850 - 19840 = 10 ≥ batch(3) → send
        result = trail.update(19858.0)
        assert result == 19850.0

    def test_trail_batch_skips_small_moves(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # Big move: triggers first trail
        trail.update(19858.0)  # sends stop at 19850

        # Small move: only 1 pt, below batch threshold of 3
        result = trail.update(19859.0)
        assert result is None

    def test_trail_sends_after_batch_threshold(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # First trail: peak=19858, stop=19850
        trail.update(19858.0)

        # Move 3+ more: peak=19861, stop=19853 → diff=3 ≥ batch=3
        result = trail.update(19861.0)
        assert result == 19853.0

    def test_trail_never_moves_stop_down(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # Move up to set peak
        trail.update(19862.0)  # peak=19862, stop=19854

        # Price drops — peak stays 19862, ideal_stop still 19854 (unchanged)
        result = trail.update(19855.0)
        assert result is None  # no new stop to send


# ── Short Position Trailing ──────────────────────────────────────────────────


class TestShortTrailing:
    def test_trail_short_position(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.SHORT, avg_entry=19850.0, stop_price=19860.0)
        trail.activate(pos, 19850.0)

        # Price drops (favorable for short): entry=19850, current=19842 → 8pts profit
        # Peak becomes 19842 (for shorts, peak tracks lowest)
        # Ideal stop = 19842 + 8 = 19850
        # Last sent stop is 19860, new 19850 < 19860 → this is a tighter stop → send
        # Diff = |19850 - 19860| = 10 ≥ batch(3) → send
        result = trail.update(19842.0)
        assert result == 19850.0

    def test_short_trail_never_widens(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.SHORT, avg_entry=19850.0, stop_price=19860.0)
        trail.activate(pos, 19850.0)

        # Set initial trail stop
        trail.update(19842.0)  # stop at 19850

        # Price goes up (adverse for short) — peak stays at 19842
        # ideal_stop = 19842 + 8 = 19850, same as current → no change
        result = trail.update(19848.0)
        assert result is None


# ── Tightening ───────────────────────────────────────────────────────────────


class TestTightening:
    def test_tighten_at_milestone(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
            tighten_at_pts=15.0,
            tighten_distance=5.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # Move 15+ points: entry=19850, price=19866 → 16pts profit
        # Tighten: trail_distance becomes 5.0
        # Peak=19866, stop=19866-5=19861
        # Diff = |19861 - 19840| = 21 ≥ batch(3) → send
        result = trail.update(19866.0)
        assert result == 19861.0  # tighter than 19858 (8pt trail)


# ── Stats ────────────────────────────────────────────────────────────────────


class TestTrailStats:
    def test_initial_stats(self):
        trail = TrailManager(trail_distance=8.0, batch_points=3.0)
        stats = trail.stats
        assert stats["is_active"] is False
        assert stats["updates_sent"] == 0
        assert stats["updates_skipped"] == 0
        assert stats["trail_distance"] == 8.0

    def test_stats_after_trailing(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        trail.update(19858.0)  # sent
        trail.update(19859.0)  # skipped (below batch)
        trail.update(19862.0)  # sent (3pt move)

        stats = trail.stats
        assert stats["is_active"] is True
        assert stats["updates_sent"] == 2
        assert stats["updates_skipped"] == 1


# ── Modify Verification ─────────────────────────────────────────────────────


class TestConfirmModify:
    def test_confirm_success(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        new_stop = trail.update(19858.0)
        assert new_stop == 19850.0

        trail.confirm_modify(success=True)
        assert trail.stats["last_confirmed_stop"] == 19850.0
        assert trail.stats["modify_failures"] == 0

    def test_confirm_failure_rolls_back(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        new_stop = trail.update(19858.0)  # returns 19850
        assert new_stop == 19850.0

        trail.confirm_modify(success=False)

        # last_sent_stop rolled back to last_confirmed (19840)
        assert trail.last_sent_stop == 19840.0
        assert trail.stats["modify_failures"] == 1

    def test_rollback_allows_re_attempt(self):
        trail = TrailManager(
            trail_distance=8.0,
            batch_points=3.0,
            activation_profit_pts=4.0,
        )
        pos = _position(side=Side.LONG, avg_entry=19850.0, stop_price=19840.0)
        trail.activate(pos, 19850.0)

        # First attempt: returns 19850, but modify fails
        new_stop = trail.update(19858.0)
        assert new_stop == 19850.0
        trail.confirm_modify(success=False)

        # Second attempt at same price: should re-send because rollback
        new_stop = trail.update(19858.0)
        assert new_stop == 19850.0  # re-sent after rollback

        trail.confirm_modify(success=True)
        assert trail.stats["last_confirmed_stop"] == 19850.0

    def test_confirm_noop_without_pending(self):
        trail = TrailManager()
        trail.confirm_modify(success=True)  # should not raise
        assert trail.stats["modify_failures"] == 0

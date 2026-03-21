"""Tests for dynamic stop loss computation (OB zone edge + ATR buffer).

Step 3 of profitability overhaul: replace static 40pt stops with dynamic
stops based on Order Block zone boundaries + ATR buffer.
"""

import pytest

from src.core.types import SessionPhase
from src.execution.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Long side — OB zone
# ---------------------------------------------------------------------------


class TestDynamicStopLong:
    def test_long_with_ob_zone_clamped_to_min(self):
        """Long entry at 21450, OB zone 21445-21452, ATR=4.5.

        raw = 21450 - 21445 + (4.5 * 0.5) = 5 + 2.25 = 7.25
        Clamped to min 8.0.
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21445.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
        )
        assert sl == 8.0

    def test_long_wider_ob_zone(self):
        """Long entry at 21450, OB zone 21435-21452, ATR=4.5.

        raw = 21450 - 21435 + 2.25 = 17.25
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21435.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
        )
        assert sl == 17.25


# ---------------------------------------------------------------------------
# Short side — OB zone
# ---------------------------------------------------------------------------


class TestDynamicStopShort:
    def test_short_with_ob_zone(self):
        """Short entry at 21450, OB zone 21448-21460, ATR=4.5.

        raw = 21460 - 21450 + 2.25 = 12.25
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="short",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21448.0, "ob_zone_high": 21460.0, "ob_side": "bear"},
            atr=4.5,
            phase=SessionPhase.MORNING,
        )
        assert sl == 12.25

    def test_short_narrow_ob_zone(self):
        """Short entry at 21450, OB zone 21448-21453, ATR=3.0.

        raw = 21453 - 21450 + 1.5 = 4.5  -> clamped to 8.0
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="short",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21448.0, "ob_zone_high": 21453.0, "ob_side": "bear"},
            atr=3.0,
            phase=SessionPhase.MORNING,
        )
        assert sl == 8.0


# ---------------------------------------------------------------------------
# Clamping — min and max
# ---------------------------------------------------------------------------


class TestDynamicStopClamp:
    def test_clamped_to_min(self):
        """Very narrow OB zone -> clamped to 8 pts minimum."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21448.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=2.0,
            phase=SessionPhase.MORNING,
        )
        # raw = 21450 - 21448 + 1.0 = 3.0 -> clamped to 8.0
        assert sl == 8.0

    def test_clamped_to_session_max(self):
        """Very wide OB zone -> clamped to session max (40 for MORNING)."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21400.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=10.0,
            phase=SessionPhase.MORNING,
        )
        # raw = 50 + 5 = 55 -> clamped to 40
        assert sl == 40.0


# ---------------------------------------------------------------------------
# Fallbacks — no OB zone
# ---------------------------------------------------------------------------


class TestDynamicStopFallback:
    def test_no_ob_uses_atr_fallback(self):
        """No OB zone -> 3 * ATR."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone=None,
            atr=5.0,
            phase=SessionPhase.MORNING,
        )
        assert sl == 15.0  # 3 * 5.0

    def test_no_ob_atr_fallback_clamped_min(self):
        """ATR very small -> fallback clamped to 8."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone=None,
            atr=2.0,
            phase=SessionPhase.MORNING,
        )
        assert sl == 8.0  # 3 * 2 = 6 -> clamped to 8

    def test_no_ob_no_atr_uses_session_default(self):
        """No OB, no ATR -> session max."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone=None,
            atr=0.0,
            phase=SessionPhase.MORNING,
        )
        assert sl == 40.0  # session default for MORNING

    def test_no_ob_atr_fallback_clamped_max(self):
        """ATR very large without OB -> clamped to session max."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="short",
            entry_price=21450.0,
            ob_zone=None,
            atr=20.0,
            phase=SessionPhase.MORNING,
        )
        # 3 * 20 = 60 -> clamped to 40
        assert sl == 40.0


# ---------------------------------------------------------------------------
# Session variation
# ---------------------------------------------------------------------------


class TestDynamicStopSessionVariation:
    def test_asian_session_max_30(self):
        """Asian session max is 30, not 40."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21410.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=5.0,
            phase=SessionPhase.ASIAN,
        )
        # raw = 40 + 2.5 = 42.5 -> clamped to 30
        assert sl == 30.0

    def test_london_session_max_35(self):
        """London session max is 35."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21410.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=5.0,
            phase=SessionPhase.LONDON,
        )
        # raw = 40 + 2.5 = 42.5 -> clamped to 35
        assert sl == 35.0

    def test_open_drive_session_max_40(self):
        """Open drive session max is 40."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="short",
            entry_price=21450.0,
            ob_zone=None,
            atr=5.0,
            phase=SessionPhase.OPEN_DRIVE,
        )
        # 3 * 5 = 15
        assert sl == 15.0


# ---------------------------------------------------------------------------
# widen_stops interaction
# ---------------------------------------------------------------------------


class TestWidenStopsOnDynamic:
    def test_widen_stops_multiplies_dynamic(self):
        """widen_stops should multiply the dynamic stop by 1.25.

        This is applied in the orchestrator, not in compute_dynamic_stop.
        Verify the base value and the multiplied value.
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21435.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
        )
        widened = sl * 1.25
        assert sl == 17.25
        assert widened == 21.5625  # orchestrator applies the 1.25x


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDynamicStopEdgeCases:
    def test_case_insensitive_side(self):
        """Side should be case-insensitive."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="LONG",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21435.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
        )
        assert sl == 17.25

    def test_rounding(self):
        """Result should be rounded to 2 decimal places."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21440.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=3.33,
            phase=SessionPhase.MORNING,
        )
        # raw = 10 + 1.665 = 11.665 -> round(11.665, 2)
        # Python banker's rounding: 11.665 rounds to 11.66
        assert sl == 11.66


# ---------------------------------------------------------------------------
# HTF structure zone (Priority 2 — used when no OB zone)
# ---------------------------------------------------------------------------


class TestDynamicStopStructureZone:
    def test_long_structure_support_zone(self):
        """Long bouncing off 4h support at 21430-21435, entry 21440, ATR=4.5.

        raw = 21440 - 21430 + (4.5 * 0.75) = 10 + 3.375 = 13.375 -> 13.38
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21440.0,
            ob_zone=None,
            atr=4.5,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21430.0, "zone_high": 21435.0, "timeframe": "4h", "level_type": "support"},
        )
        assert sl == 13.38

    def test_short_structure_resistance_zone(self):
        """Short bouncing off Daily resistance at 21460-21470, entry 21458, ATR=5.0.

        raw = 21470 - 21458 + (5.0 * 0.75) = 12 + 3.75 = 15.75
        """
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="short",
            entry_price=21458.0,
            ob_zone=None,
            atr=5.0,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21460.0, "zone_high": 21470.0, "timeframe": "D", "level_type": "resistance"},
        )
        assert sl == 15.75

    def test_structure_zone_clamped_min(self):
        """Very narrow structure zone -> clamped to 8pt minimum."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21435.0,
            ob_zone=None,
            atr=2.0,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21433.0, "zone_high": 21435.0, "timeframe": "1h", "level_type": "support"},
        )
        # raw = 2 + 1.5 = 3.5 -> clamped to 8
        assert sl == 8.0

    def test_structure_zone_clamped_max(self):
        """Very wide structure zone -> clamped to session max."""
        rm = RiskManager()
        sl = rm.compute_dynamic_stop(
            side="long",
            entry_price=21440.0,
            ob_zone=None,
            atr=5.0,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21390.0, "zone_high": 21395.0, "timeframe": "W", "level_type": "support"},
        )
        # raw = 50 + 3.75 = 53.75 -> clamped to 40
        assert sl == 40.0

    def test_ob_takes_priority_over_structure(self):
        """When both OB and structure zone exist, OB wins (tighter stop)."""
        rm = RiskManager()
        sl_with_both = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21440.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21420.0, "zone_high": 21425.0, "timeframe": "D", "level_type": "support"},
        )
        sl_ob_only = rm.compute_dynamic_stop(
            side="long",
            entry_price=21450.0,
            ob_zone={"ob_zone_low": 21440.0, "ob_zone_high": 21452.0, "ob_side": "bull"},
            atr=4.5,
            phase=SessionPhase.MORNING,
            structure_zone=None,
        )
        assert sl_with_both == sl_ob_only  # OB takes priority, structure ignored

    def test_structure_used_when_no_ob(self):
        """Structure zone used as fallback when no OB zone."""
        rm = RiskManager()
        sl_structure = rm.compute_dynamic_stop(
            side="long",
            entry_price=21440.0,
            ob_zone=None,
            atr=4.5,
            phase=SessionPhase.MORNING,
            structure_zone={"zone_low": 21430.0, "zone_high": 21435.0, "timeframe": "4h", "level_type": "support"},
        )
        sl_atr_fallback = rm.compute_dynamic_stop(
            side="long",
            entry_price=21440.0,
            ob_zone=None,
            atr=4.5,
            phase=SessionPhase.MORNING,
            structure_zone=None,
        )
        # Structure: 10 + 3.375 = 13.38; ATR fallback: 13.5
        # Both similar in this case, but structure is zone-anchored
        assert sl_structure == 13.38
        assert sl_atr_fallback == 13.5
        assert sl_structure != sl_atr_fallback  # different computation paths

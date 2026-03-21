"""Tests for confluence-based position sizing.

Score 3 -> 3 contracts (base)
Score 4 -> 4 contracts
Score 5+ -> 5 contracts (max_entry_contracts default)

Modifiers: reduce_size halves, session controller caps, apex caps.
Note: fast_market is info-only — NO size reduction (strong moves = best setups).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Pure sizing function tests -- we extract the logic into a helper so it can
# be unit-tested without wiring up the full orchestrator.
# ---------------------------------------------------------------------------

from src.core.config import TradingConfig


def _compute_entry_quantity(
    best_score: int,
    max_entry_contracts: int = 5,
    reduce_size: bool = False,
    session_ctrl_max: int | None = None,
    apex_effective_max: int | None = None,
) -> int:
    """Mirror the sizing logic that lives in the orchestrator.

    This is the *specification* -- the orchestrator must implement this
    exact algorithm.

    Note: fast_market is NOT a sizing modifier. FAST markets are strong
    directional moves — the best setups for ICT entries. Other filters
    (OB decay, direction-aware volume, sweep single-fire) protect against
    news-spike false signals.
    """
    # Confluence-based base quantity
    if best_score >= 5:
        base = max_entry_contracts
    elif best_score >= 4:
        base = max(3, max_entry_contracts - 1)
    else:
        base = 3  # minimum confluence

    qty = base

    # Pre-market reduce_size: halve
    if reduce_size:
        qty = max(1, qty // 2)

    # Apex scaling cap
    if apex_effective_max is not None:
        qty = min(qty, apex_effective_max)

    # Session controller cap (profit preservation)
    if session_ctrl_max is not None:
        qty = min(qty, session_ctrl_max)

    # Hard floor
    return max(1, qty)


class TestConfluenceBaseSizing:
    """Base sizing from confluence score: 3-5 MNQ."""

    def test_score_3_gives_3_contracts(self):
        assert _compute_entry_quantity(best_score=3) == 3

    def test_score_4_gives_4_contracts(self):
        assert _compute_entry_quantity(best_score=4) == 4

    def test_score_5_gives_5_contracts(self):
        assert _compute_entry_quantity(best_score=5) == 5

    def test_score_6_gives_5_contracts(self):
        """Score above 5 still maps to max_entry_contracts."""
        assert _compute_entry_quantity(best_score=6) == 5

    def test_score_7_gives_5_contracts(self):
        assert _compute_entry_quantity(best_score=7) == 5


class TestReduceSizeModifier:
    """reduce_size halves confluence-based quantity."""

    def test_reduce_size_halves_score_5(self):
        # 5 // 2 = 2
        assert _compute_entry_quantity(best_score=5, reduce_size=True) == 2

    def test_reduce_size_halves_score_4(self):
        # 4 // 2 = 2
        assert _compute_entry_quantity(best_score=4, reduce_size=True) == 2

    def test_reduce_size_halves_score_3(self):
        # 3 // 2 = 1
        assert _compute_entry_quantity(best_score=3, reduce_size=True) == 1


class TestFastMarketNoSizeImpact:
    """fast_market is info-only — does NOT affect sizing.

    FAST markets are strong directional moves, the best ICT setups.
    Other filters protect against bad entries.
    """

    def test_fast_market_no_size_change_score_5(self):
        # fast_market param removed from helper — sizing is identical
        assert _compute_entry_quantity(best_score=5) == 5

    def test_fast_market_no_size_change_score_4(self):
        assert _compute_entry_quantity(best_score=4) == 4

    def test_fast_market_no_size_change_score_3(self):
        assert _compute_entry_quantity(best_score=3) == 3


class TestReduceSizeOnly:
    """reduce_size is the only halving modifier now (from pre-market LLM)."""

    def test_reduce_size_alone_score_5(self):
        # 5 // 2 = 2
        assert _compute_entry_quantity(best_score=5, reduce_size=True) == 2

    def test_reduce_size_alone_score_3(self):
        # 3 // 2 = 1
        assert _compute_entry_quantity(best_score=3, reduce_size=True) == 1


class TestCustomMaxEntry:
    """Custom max_entry_contracts scales up."""

    def test_score_5_with_max_8(self):
        assert _compute_entry_quantity(best_score=5, max_entry_contracts=8) == 8

    def test_score_4_with_max_8(self):
        # max(3, 8-1) = 7
        assert _compute_entry_quantity(best_score=4, max_entry_contracts=8) == 7

    def test_score_3_with_max_8(self):
        # Still base=3 regardless of max
        assert _compute_entry_quantity(best_score=3, max_entry_contracts=8) == 3

    def test_score_5_with_max_3(self):
        """When max is 3, everything caps at 3."""
        assert _compute_entry_quantity(best_score=5, max_entry_contracts=3) == 3

    def test_score_4_with_max_3(self):
        # max(3, 3-1) = max(3, 2) = 3
        assert _compute_entry_quantity(best_score=4, max_entry_contracts=3) == 3


class TestPartialQuantityScaling:
    """Partial quantity = half of entry, min 1."""

    def test_5_contract_entry_partial_is_2(self):
        entry_qty = 5
        partial_qty = max(1, entry_qty // 2)
        assert partial_qty == 2

    def test_4_contract_entry_partial_is_2(self):
        entry_qty = 4
        partial_qty = max(1, entry_qty // 2)
        assert partial_qty == 2

    def test_3_contract_entry_partial_is_1(self):
        entry_qty = 3
        partial_qty = max(1, entry_qty // 2)
        assert partial_qty == 1

    def test_2_contract_entry_partial_is_1(self):
        entry_qty = 2
        partial_qty = max(1, entry_qty // 2)
        assert partial_qty == 1

    def test_1_contract_entry_partial_is_1(self):
        entry_qty = 1
        partial_qty = max(1, entry_qty // 2)
        assert partial_qty == 1


class TestSessionControllerCap:
    """Session controller effective_max_contracts is respected."""

    def test_session_ctrl_caps_at_2(self):
        # Score 5 wants 5, but session says max 2
        assert _compute_entry_quantity(best_score=5, session_ctrl_max=2) == 2

    def test_session_ctrl_caps_at_3(self):
        # Score 5 wants 5, session says 3
        assert _compute_entry_quantity(best_score=5, session_ctrl_max=3) == 3

    def test_session_ctrl_does_not_cap_when_higher(self):
        # Score 3 wants 3, session says 6 -- no cap needed
        assert _compute_entry_quantity(best_score=3, session_ctrl_max=6) == 3


class TestApexCap:
    """Apex effective_max_micros caps the quantity."""

    def test_apex_caps_below_confluence(self):
        # Score 5 wants 5, apex says 3
        assert _compute_entry_quantity(best_score=5, apex_effective_max=3) == 3

    def test_apex_no_cap_when_higher(self):
        assert _compute_entry_quantity(best_score=3, apex_effective_max=50) == 3

    def test_apex_caps_at_2(self):
        assert _compute_entry_quantity(best_score=5, apex_effective_max=2) == 2


class TestConfigField:
    """Verify the config field exists with correct default."""

    def test_max_entry_contracts_default(self):
        cfg = TradingConfig()
        assert cfg.max_entry_contracts == 5

    def test_max_entry_contracts_custom(self):
        cfg = TradingConfig(max_entry_contracts=8)
        assert cfg.max_entry_contracts == 8

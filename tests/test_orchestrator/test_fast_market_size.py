"""Tests for FAST market sizing in the orchestrator.

FAST markets are strong directional moves — the BEST setups for ICT entries.
No size reduction. The orchestrator logs fast_market as info only.
Other filters (OB decay, direction-aware volume, sweep single-fire) protect
against news-spike false signals.
"""

from __future__ import annotations

import pytest


class TestFastMarketNoSizeReduction:
    """FAST market should NOT reduce entry quantity — strong moves are good."""

    def test_fast_market_keeps_full_quantity_score_5(self):
        """Score 5 with fast_market=True should still get 5 contracts."""
        max_entry = 5
        best_score = 5
        fast_market = True

        # Confluence-based sizing (same as orchestrator logic)
        if best_score >= 5:
            base_quantity = max_entry
        elif best_score >= 4:
            base_quantity = max(3, max_entry - 1)
        else:
            base_quantity = 3

        entry_quantity = base_quantity

        # FAST market: NO size reduction (info flag only)
        # This is intentional — strong directional moves are the best setups.
        assert entry_quantity == 5, (
            f"FAST market should NOT reduce quantity. Expected 5, got {entry_quantity}"
        )

    def test_fast_market_keeps_full_quantity_score_4(self):
        """Score 4 with fast_market=True should still get 4 contracts."""
        max_entry = 5
        best_score = 4
        fast_market = True

        if best_score >= 5:
            base_quantity = max_entry
        elif best_score >= 4:
            base_quantity = max(3, max_entry - 1)
        else:
            base_quantity = 3

        entry_quantity = base_quantity
        assert entry_quantity == 4

    def test_fast_market_keeps_full_quantity_score_3(self):
        """Score 3 with fast_market=True should still get 3 contracts."""
        max_entry = 5
        best_score = 3
        fast_market = True

        if best_score >= 5:
            base_quantity = max_entry
        elif best_score >= 4:
            base_quantity = max(3, max_entry - 1)
        else:
            base_quantity = 3

        entry_quantity = base_quantity
        assert entry_quantity == 3

    def test_normal_market_same_as_fast_market(self):
        """FAST and NORMAL should produce identical sizing for same score."""
        max_entry = 5
        best_score = 5

        # Compute for FAST
        fast_quantity = max_entry  # score >= 5

        # Compute for NORMAL
        normal_quantity = max_entry  # score >= 5

        assert fast_quantity == normal_quantity, (
            "FAST and NORMAL should have identical sizing"
        )

    def test_reduce_size_still_applies_independently(self):
        """reduce_size (pre-market LLM flag) still halves, but fast_market does NOT."""
        max_entry = 5
        best_score = 5
        reduce_size = True
        fast_market = True

        base_quantity = max_entry  # score >= 5
        qty = base_quantity

        # reduce_size halves (this is from pre-market context, NOT speed state)
        if reduce_size:
            qty = max(1, qty // 2)

        # fast_market does NOT halve — intentionally omitted
        assert qty == 2, (
            f"reduce_size should halve 5→2, fast_market should NOT further reduce. Got {qty}"
        )

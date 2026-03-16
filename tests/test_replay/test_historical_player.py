"""Tests for the historical player — session replay through LLM."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import (
    ActionType,
    LLMAction,
    MarketState,
    Regime,
    SessionPhase,
    Side,
)
from src.replay.data_recorder import DataRecorder
from src.replay.historical_player import HistoricalPlayer, ReplayResult


def _state(**overrides) -> MarketState:
    defaults = dict(
        timestamp=datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        bid=19849.75,
        ask=19850.25,
        session_phase=SessionPhase.MORNING,
        regime=Regime.TRENDING_UP,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _action(**overrides) -> LLMAction:
    defaults = dict(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=2,
        stop_distance=10.0,
        reasoning="Good setup",
        confidence=0.8,
        model_used="sonnet",
    )
    defaults.update(overrides)
    return LLMAction(**defaults)


def _create_recorded_session(tmpdir: str) -> Path:
    """Helper — creates a recorded session with states + decisions."""
    recorder = DataRecorder(base_dir=tmpdir)
    recorder.start_session("2026-03-14")

    # Record 3 states and 3 decisions
    for i, price in enumerate([19850.0, 19855.0, 19860.0]):
        state = _state(last_price=price)
        recorder.record_state(state)

        actions = [
            _action(action=ActionType.DO_NOTHING, confidence=0.3),
            _action(action=ActionType.ENTER, confidence=0.8),
            _action(action=ActionType.DO_NOTHING, confidence=0.4),
        ]
        recorder.record_decision(state, actions[i])

    recorder.flush()
    return Path(tmpdir) / "2026-03-14"


# ── ReplayResult ──────────────────────────────────────────────────────────


class TestReplayResult:
    def test_to_dict(self):
        r = ReplayResult(
            timestamp="2026-03-14T14:30:00",
            original_action="ENTER",
            replay_action="ENTER",
            original_confidence=0.8,
            replay_confidence=0.75,
            agreed=True,
            last_price=19850.0,
            replay_reasoning="Good setup",
            latency_ms=250,
        )
        d = r.to_dict()
        assert d["original_action"] == "ENTER"
        assert d["replay_action"] == "ENTER"
        assert d["agreed"] is True
        assert d["latency_ms"] == 250


# ── Replay Session ────────────────────────────────────────────────────────


class TestReplaySession:
    @pytest.mark.asyncio
    async def test_replay_returns_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_recorded_session(tmpdir)

            mock_reasoner = MagicMock()
            mock_reasoner.decide = AsyncMock(return_value=LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning="Replay: no setup",
                confidence=0.3,
            ))

            player = HistoricalPlayer(reasoner=mock_reasoner)
            results = await player.replay_session(session_dir)

            assert len(results) == 3
            assert mock_reasoner.decide.call_count == 3

    @pytest.mark.asyncio
    async def test_replay_max_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_recorded_session(tmpdir)

            mock_reasoner = MagicMock()
            mock_reasoner.decide = AsyncMock(return_value=LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning="Replay",
                confidence=0.3,
            ))

            player = HistoricalPlayer(reasoner=mock_reasoner)
            results = await player.replay_session(session_dir, max_decisions=2)

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_replay_agreement_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_recorded_session(tmpdir)

            # Replay always says DO_NOTHING
            mock_reasoner = MagicMock()
            mock_reasoner.decide = AsyncMock(return_value=LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning="Replay: waiting",
                confidence=0.3,
            ))

            player = HistoricalPlayer(reasoner=mock_reasoner)
            results = await player.replay_session(session_dir)

            # Original decisions: DO_NOTHING, ENTER, DO_NOTHING
            # Replay: all DO_NOTHING
            # So 2 agree (index 0 and 2), 1 disagrees (index 1)
            agreed_count = sum(1 for r in results if r.agreed)
            assert agreed_count == 2

    @pytest.mark.asyncio
    async def test_replay_missing_decisions_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty").mkdir()

            mock_reasoner = MagicMock()
            player = HistoricalPlayer(reasoner=mock_reasoner)
            results = await player.replay_session(Path(tmpdir) / "empty")

            assert results == []

    @pytest.mark.asyncio
    async def test_replay_llm_error_skips_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_recorded_session(tmpdir)

            mock_reasoner = MagicMock()
            call_count = 0

            async def _decide_with_error(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("LLM timeout")
                return LLMAction(
                    action=ActionType.DO_NOTHING,
                    reasoning="OK",
                    confidence=0.3,
                )

            mock_reasoner.decide = _decide_with_error

            player = HistoricalPlayer(reasoner=mock_reasoner)
            results = await player.replay_session(session_dir)

            # 3 decisions, 1 error → 2 results
            assert len(results) == 2


# ── Summarize ─────────────────────────────────────────────────────────────


class TestSummarize:
    def test_summarize_empty(self):
        summary = HistoricalPlayer.summarize([])
        assert summary["total"] == 0
        assert summary["agreement_rate"] == 0.0

    def test_summarize_results(self):
        results = [
            ReplayResult(
                timestamp="t1", original_action="ENTER", replay_action="ENTER",
                original_confidence=0.8, replay_confidence=0.7, agreed=True,
                last_price=19850.0, latency_ms=200,
            ),
            ReplayResult(
                timestamp="t2", original_action="DO_NOTHING", replay_action="ENTER",
                original_confidence=0.3, replay_confidence=0.8, agreed=False,
                last_price=19855.0, latency_ms=300,
            ),
        ]

        summary = HistoricalPlayer.summarize(results)
        assert summary["total"] == 2
        assert summary["agreed"] == 1
        assert summary["agreement_rate"] == 0.5
        assert summary["avg_latency_ms"] == 250.0
        assert summary["original_action_distribution"]["ENTER"] == 1
        assert summary["original_action_distribution"]["DO_NOTHING"] == 1

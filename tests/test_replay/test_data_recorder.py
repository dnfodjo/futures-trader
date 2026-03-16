"""Tests for the data recorder — MarketState + LLM decision persistence."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.types import (
    ActionType,
    LLMAction,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    Side,
)
from src.replay.data_recorder import DataRecorder


def _state(**overrides) -> MarketState:
    defaults = dict(
        timestamp=datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        bid=19849.75,
        ask=19850.25,
        spread=0.5,
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


# ── Session Lifecycle ────────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_start_session_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            path = recorder.start_session("2026-03-14")
            assert path.exists()
            assert path.name == "2026-03-14"

    def test_start_session_default_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            path = recorder.start_session()
            today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            assert path.name == today

    def test_stop_session_sets_not_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            assert recorder._is_recording is True
            recorder.stop_session()
            assert recorder._is_recording is False

    def test_stop_session_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.stop_session()  # no-op, should not raise


# ── Recording States ─────────────────────────────────────────────────────


class TestRecordState:
    def test_record_state_adds_to_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_state(_state())
            assert len(recorder._state_records) == 1

    def test_record_state_not_recording_is_noop(self):
        recorder = DataRecorder()
        recorder.record_state(_state())
        assert len(recorder._state_records) == 0

    def test_record_state_captures_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")

            state = _state(
                position=PositionState(
                    side=Side.LONG,
                    quantity=3,
                    avg_entry=19840.0,
                    unrealized_pnl=60.0,
                )
            )
            recorder.record_state(state)

            record = recorder._state_records[0]
            assert record["has_position"] is True
            assert record["position_side"] == "long"
            assert record["position_qty"] == 3
            assert record["position_pnl"] == 60.0

    def test_record_state_flat_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_state(_state())

            record = recorder._state_records[0]
            assert record["has_position"] is False
            assert record["position_side"] == ""
            assert record["position_qty"] == 0


# ── Recording Decisions ──────────────────────────────────────────────────


class TestRecordDecision:
    def test_record_decision_adds_to_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_decision(_state(), _action())
            assert len(recorder._decision_records) == 1

    def test_record_decision_captures_block_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_decision(
                _state(),
                _action(),
                was_blocked=True,
                block_reason="Max position exceeded",
                modified_quantity=2,
            )

            record = recorder._decision_records[0]
            assert record["was_blocked"] is True
            assert record["block_reason"] == "Max position exceeded"
            assert record["modified_quantity"] == 2

    def test_record_decision_captures_action_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_decision(_state(), _action(confidence=0.9))

            record = recorder._decision_records[0]
            assert record["action"] == "ENTER"
            assert record["side"] == "long"
            assert record["quantity"] == 2
            assert record["confidence"] == 0.9
            assert record["model_used"] == "sonnet"


# ── Flush to Parquet ─────────────────────────────────────────────────────


class TestFlush:
    def test_flush_states_creates_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_state(_state())
            recorder.flush()

            path = Path(tmpdir) / "2026-03-14" / "states.parquet"
            assert path.exists()

    def test_flush_decisions_creates_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_decision(_state(), _action())
            recorder.flush()

            path = Path(tmpdir) / "2026-03-14" / "decisions.parquet"
            assert path.exists()

    def test_flush_clears_buffers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_state(_state())
            recorder.record_decision(_state(), _action())

            assert len(recorder._state_records) == 1
            assert len(recorder._decision_records) == 1

            recorder.flush()

            assert len(recorder._state_records) == 0
            assert len(recorder._decision_records) == 0

    def test_flush_appends_to_existing(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")

            # First batch
            recorder.record_state(_state(last_price=19850.0))
            recorder.flush()

            # Second batch
            recorder.record_state(_state(last_price=19855.0))
            recorder.flush()

            # Load and verify both rows
            path = Path(tmpdir) / "2026-03-14" / "states.parquet"
            df = pd.read_parquet(path)
            assert len(df) == 2
            assert df.iloc[0]["last_price"] == 19850.0
            assert df.iloc[1]["last_price"] == 19855.0

    def test_flush_empty_buffer_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.flush()  # nothing to flush

            path = Path(tmpdir) / "2026-03-14" / "states.parquet"
            assert not path.exists()


# ── Load Session ─────────────────────────────────────────────────────────


class TestLoadSession:
    def test_load_session_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")

            recorder.record_state(_state(last_price=19850.0))
            recorder.record_decision(_state(), _action())
            recorder.flush()

            data = DataRecorder.load_session(Path(tmpdir) / "2026-03-14")
            assert data["states"] is not None
            assert data["decisions"] is not None
            assert len(data["states"]) == 1
            assert len(data["decisions"]) == 1

    def test_load_session_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty").mkdir()
            data = DataRecorder.load_session(Path(tmpdir) / "empty")
            assert data["states"] is None
            assert data["decisions"] is None


# ── List Sessions ────────────────────────────────────────────────────────


class TestListSessions:
    def test_list_sessions_returns_dates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)

            recorder.start_session("2026-03-12")
            recorder.record_state(_state())
            recorder.flush()

            recorder.start_session("2026-03-14")
            recorder.record_state(_state())
            recorder.flush()

            sessions = DataRecorder.list_sessions(tmpdir)
            assert sessions == ["2026-03-12", "2026-03-14"]

    def test_list_sessions_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = DataRecorder.list_sessions(tmpdir)
            assert sessions == []

    def test_list_sessions_nonexistent_dir(self):
        sessions = DataRecorder.list_sessions("/nonexistent/path")
        assert sessions == []


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_initial(self):
        recorder = DataRecorder()
        stats = recorder.stats
        assert stats["is_recording"] is False
        assert stats["buffered_states"] == 0

    def test_stats_during_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(base_dir=tmpdir)
            recorder.start_session("2026-03-14")
            recorder.record_state(_state())
            recorder.record_decision(_state(), _action())

            stats = recorder.stats
            assert stats["is_recording"] is True
            assert stats["session_date"] == "2026-03-14"
            assert stats["buffered_states"] == 1
            assert stats["buffered_decisions"] == 1


# ── Time-Based Flush ──────────────────────────────────────────────────────


class TestTimeBasedFlush:
    def test_time_flush_triggers_when_elapsed(self):
        """Verify time-based flush fires after flush_interval_sec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 0 second interval so it always triggers
            recorder = DataRecorder(
                base_dir=tmpdir,
                flush_interval_records=999,  # won't trigger count-based
                flush_interval_sec=0.0,
            )
            recorder.start_session("2026-03-14")

            # Record a state — should trigger time flush
            recorder.record_state(_state())

            # Buffer should have been flushed to disk
            states_path = Path(tmpdir) / "2026-03-14" / "states.parquet"
            assert states_path.exists()

    def test_time_flush_does_not_trigger_when_recent(self):
        """Verify time-based flush does NOT fire when within interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                base_dir=tmpdir,
                flush_interval_records=999,  # won't trigger count-based
                flush_interval_sec=3600.0,  # 1 hour — won't trigger
            )
            recorder.start_session("2026-03-14")

            recorder.record_state(_state())

            # Buffer should still be in memory
            states_path = Path(tmpdir) / "2026-03-14" / "states.parquet"
            assert not states_path.exists()
            assert len(recorder._state_records) == 1

    def test_time_flush_on_decisions(self):
        """Verify time-based flush works for decisions too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                base_dir=tmpdir,
                flush_interval_records=999,
                flush_interval_sec=0.0,
            )
            recorder.start_session("2026-03-14")

            recorder.record_decision(_state(), _action())

            decisions_path = Path(tmpdir) / "2026-03-14" / "decisions.parquet"
            assert decisions_path.exists()

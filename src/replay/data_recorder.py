"""Data recorder — persists market states, LLM decisions, and events to disk.

During live trading, the recorder captures:
- MarketState snapshots (every decision cycle)
- LLM decisions (input state + output action)
- Trade executions (fills, modifications)

Data is stored as compressed Parquet files, one per session:
    data/recordings/YYYY-MM-DD/states.parquet
    data/recordings/YYYY-MM-DD/decisions.parquet

This enables:
- Historical replay through the LLM
- Decision scoring and postmortem analysis
- Monte Carlo simulation from observed stats
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from src.core.types import LLMAction, MarketState

logger = structlog.get_logger()


class DataRecorder:
    """Records live trading data for replay and analysis.

    Usage:
        recorder = DataRecorder(base_dir="data/recordings")
        recorder.start_session("2026-03-14")

        # During each decision cycle:
        recorder.record_state(market_state)
        recorder.record_decision(state, action, latency_ms=250)

        # At end of day:
        recorder.flush()
    """

    def __init__(
        self,
        base_dir: str = "data/recordings",
        flush_interval_records: int = 100,
        flush_interval_sec: float = 60.0,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._session_date: str = ""
        self._session_dir: Optional[Path] = None

        # In-memory buffers (flushed to disk periodically)
        self._state_records: list[dict[str, Any]] = []
        self._decision_records: list[dict[str, Any]] = []
        self._flush_interval: int = flush_interval_records  # flush every N records
        self._flush_interval_sec: float = flush_interval_sec  # flush every N seconds
        self._last_flush_time: float = 0.0
        self._is_recording: bool = False

    # ── Session Lifecycle ────────────────────────────────────────────────

    def start_session(self, date: str = "") -> Path:
        """Start a new recording session.

        Args:
            date: Session date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Path to the session directory.
        """
        if not date:
            date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

        self._session_date = date
        self._session_dir = self._base_dir / date
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._state_records.clear()
        self._decision_records.clear()
        self._last_flush_time = time.monotonic()
        self._is_recording = True

        logger.info(
            "recorder.session_started",
            date=date,
            dir=str(self._session_dir),
        )
        return self._session_dir

    def stop_session(self) -> None:
        """Stop recording and flush remaining data."""
        if not self._is_recording:
            return

        self.flush()
        self._is_recording = False

        logger.info(
            "recorder.session_stopped",
            date=self._session_date,
            states=len(self._state_records),
            decisions=len(self._decision_records),
        )

    # ── Recording ────────────────────────────────────────────────────────

    def record_state(self, state: MarketState) -> None:
        """Record a MarketState snapshot."""
        if not self._is_recording:
            return

        record = {
            "timestamp": state.timestamp.isoformat(),
            "symbol": state.symbol,
            "last_price": state.last_price,
            "bid": state.bid,
            "ask": state.ask,
            "spread": state.spread,
            "session_phase": state.session_phase.value,
            "regime": state.regime.value,
            "regime_confidence": state.regime_confidence,
            # Key levels
            "vwap": state.levels.vwap if state.levels else 0.0,
            "session_high": state.levels.session_high if state.levels else 0.0,
            "session_low": state.levels.session_low if state.levels else 0.0,
            "poc": state.levels.poc if state.levels else 0.0,
            # Order flow
            "cumulative_delta": state.flow.cumulative_delta if state.flow else 0.0,
            "delta_1min": state.flow.delta_1min if state.flow else 0.0,
            "rvol": state.flow.rvol if state.flow else 0.0,
            "tape_speed": state.flow.tape_speed if state.flow else 0.0,
            # Cross-market
            "es_price": state.cross_market.es_price if state.cross_market else 0.0,
            "vix": state.cross_market.vix if state.cross_market else 0.0,
            "tick_index": state.cross_market.tick_index if state.cross_market else 0,
            # Position
            "has_position": state.position is not None,
            "position_side": state.position.side.value if state.position else "",
            "position_qty": state.position.quantity if state.position else 0,
            "position_pnl": state.position.unrealized_pnl if state.position else 0.0,
            # Session
            "daily_pnl": state.daily_pnl,
            "daily_trades": state.daily_trades,
            "in_blackout": state.in_blackout,
        }
        self._state_records.append(record)

        if len(self._state_records) % self._flush_interval == 0:
            self._flush_states()
        elif self._should_time_flush():
            self.flush()

    def record_decision(
        self,
        state: MarketState,
        action: LLMAction,
        latency_ms: int = 0,
        was_blocked: bool = False,
        block_reason: str = "",
        modified_quantity: Optional[int] = None,
    ) -> None:
        """Record an LLM decision with the state that produced it."""
        if not self._is_recording:
            return

        record = {
            "timestamp": state.timestamp.isoformat(),
            "last_price": state.last_price,
            "regime": state.regime.value,
            "session_phase": state.session_phase.value,
            "daily_pnl": state.daily_pnl,
            "has_position": state.position is not None,
            "position_side": state.position.side.value if state.position else "",
            "position_qty": state.position.quantity if state.position else 0,
            "position_pnl": state.position.unrealized_pnl if state.position else 0.0,
            # Decision
            "action": action.action.value,
            "side": action.side.value if action.side else "",
            "quantity": action.quantity or 0,
            "stop_distance": action.stop_distance or 0.0,
            "new_stop_price": action.new_stop_price or 0.0,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "model_used": action.model_used,
            "latency_ms": latency_ms or action.latency_ms,
            # Guardrail result
            "was_blocked": was_blocked,
            "block_reason": block_reason,
            "modified_quantity": modified_quantity,
        }
        self._decision_records.append(record)

        if len(self._decision_records) % self._flush_interval == 0:
            self._flush_decisions()
        elif self._should_time_flush():
            self.flush()

    def _should_time_flush(self) -> bool:
        """Check if enough time has elapsed to trigger a time-based flush."""
        now = time.monotonic()
        if now - self._last_flush_time >= self._flush_interval_sec:
            return True
        return False

    # ── Flush ────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all buffered records to disk."""
        self._flush_states()
        self._flush_decisions()
        self._last_flush_time = time.monotonic()

    def _flush_states(self) -> None:
        """Write state records to Parquet."""
        if not self._state_records or not self._session_dir:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(self._state_records)
            path = self._session_dir / "states.parquet"

            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)

            df.to_parquet(path, index=False, compression="snappy")
            count = len(self._state_records)
            self._state_records.clear()
            logger.debug("recorder.states_flushed", count=count)
        except Exception:
            logger.exception("recorder.flush_states_failed")

    def _flush_decisions(self) -> None:
        """Write decision records to Parquet."""
        if not self._decision_records or not self._session_dir:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(self._decision_records)
            path = self._session_dir / "decisions.parquet"

            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)

            df.to_parquet(path, index=False, compression="snappy")
            count = len(self._decision_records)
            self._decision_records.clear()
            logger.debug("recorder.decisions_flushed", count=count)
        except Exception:
            logger.exception("recorder.flush_decisions_failed")

    # ── Loading ──────────────────────────────────────────────────────────

    @staticmethod
    def load_session(session_dir: str | Path) -> dict[str, Any]:
        """Load a recorded session from disk.

        Returns:
            Dict with 'states' and 'decisions' DataFrames (or empty dicts).
        """
        import pandas as pd

        session_dir = Path(session_dir)
        result: dict[str, Any] = {"states": None, "decisions": None}

        states_path = session_dir / "states.parquet"
        if states_path.exists():
            result["states"] = pd.read_parquet(states_path)

        decisions_path = session_dir / "decisions.parquet"
        if decisions_path.exists():
            result["decisions"] = pd.read_parquet(decisions_path)

        return result

    @staticmethod
    def list_sessions(base_dir: str = "data/recordings") -> list[str]:
        """List all recorded session dates."""
        base = Path(base_dir)
        if not base.exists():
            return []

        sessions = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "states.parquet").exists()
        )
        return sessions

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "is_recording": self._is_recording,
            "session_date": self._session_date,
            "buffered_states": len(self._state_records),
            "buffered_decisions": len(self._decision_records),
        }

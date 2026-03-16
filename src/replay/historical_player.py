"""Historical player — replays recorded sessions through the reasoning engine.

Given a recorded session (states.parquet + decisions.parquet), the player:
1. Feeds each MarketState through the LLM reasoning engine (temperature=0)
2. Compares the replay decisions to what was actually done
3. Tracks "what if" P&L for the replay decisions

This allows:
- A/B testing prompt changes against real market data
- Measuring decision consistency (same input → same output?)
- Estimating P&L impact of alternative decisions
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from src.core.types import (
    ActionType,
    KeyLevels,
    LLMAction,
    MarketState,
    OrderFlowData,
    CrossMarketContext,
    PositionState,
    Regime,
    SessionPhase,
    Side,
)

logger = structlog.get_logger()


class ReplayResult:
    """Result of replaying a single MarketState through the reasoning engine."""

    def __init__(
        self,
        timestamp: str,
        original_action: str,
        replay_action: str,
        original_confidence: float,
        replay_confidence: float,
        agreed: bool,
        last_price: float,
        replay_reasoning: str = "",
        latency_ms: int = 0,
    ) -> None:
        self.timestamp = timestamp
        self.original_action = original_action
        self.replay_action = replay_action
        self.original_confidence = original_confidence
        self.replay_confidence = replay_confidence
        self.agreed = agreed
        self.last_price = last_price
        self.replay_reasoning = replay_reasoning
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "original_action": self.original_action,
            "replay_action": self.replay_action,
            "original_confidence": self.original_confidence,
            "replay_confidence": self.replay_confidence,
            "agreed": self.agreed,
            "last_price": self.last_price,
            "replay_reasoning": self.replay_reasoning,
            "latency_ms": self.latency_ms,
        }


class HistoricalPlayer:
    """Replays recorded sessions through the LLM reasoning engine.

    Usage:
        player = HistoricalPlayer(reasoner=reasoner)
        results = await player.replay_session("data/recordings/2026-03-14")
        summary = player.summarize(results)
    """

    def __init__(
        self,
        reasoner: Any,  # Reasoner, but typed as Any to avoid circular import issues
        game_plan: str = "",
    ) -> None:
        self._reasoner = reasoner
        self._game_plan = game_plan

    async def replay_session(
        self,
        session_dir: str | Path,
        max_decisions: int = 0,
    ) -> list[ReplayResult]:
        """Replay a recorded session, comparing LLM decisions.

        Args:
            session_dir: Path to the session directory.
            max_decisions: Max decisions to replay (0 = all).

        Returns:
            List of ReplayResult comparing original vs replay decisions.
        """
        import pandas as pd

        session_dir = Path(session_dir)
        decisions_path = session_dir / "decisions.parquet"
        states_path = session_dir / "states.parquet"

        if not decisions_path.exists():
            logger.warning("replay.no_decisions_file", dir=str(session_dir))
            return []

        decisions_df = pd.read_parquet(decisions_path)
        states_df = pd.read_parquet(states_path) if states_path.exists() else None

        results: list[ReplayResult] = []
        total = len(decisions_df)
        if max_decisions > 0:
            total = min(total, max_decisions)

        logger.info("replay.starting", total_decisions=total, dir=str(session_dir))

        for i in range(total):
            row = decisions_df.iloc[i]

            # Reconstruct MarketState from recorded data
            state = self._reconstruct_state(row, states_df)

            # Get replay decision from LLM
            t0 = time.monotonic()
            try:
                replay_action = await self._reasoner.decide(
                    state=state,
                    game_plan=self._game_plan,
                )
            except Exception as e:
                logger.error("replay.decision_failed", error=str(e), index=i)
                continue
            latency_ms = int((time.monotonic() - t0) * 1000)

            original_action = row["action"]
            agreed = replay_action.action.value == original_action

            result = ReplayResult(
                timestamp=str(row["timestamp"]),
                original_action=original_action,
                replay_action=replay_action.action.value,
                original_confidence=float(row.get("confidence", 0.0)),
                replay_confidence=replay_action.confidence,
                agreed=agreed,
                last_price=float(row["last_price"]),
                replay_reasoning=replay_action.reasoning,
                latency_ms=latency_ms,
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                agreement_rate = sum(1 for r in results if r.agreed) / len(results)
                logger.info(
                    "replay.progress",
                    done=i + 1,
                    total=total,
                    agreement_rate=round(agreement_rate, 2),
                )

        logger.info("replay.complete", total=len(results))
        return results

    def _reconstruct_state(self, decision_row, states_df) -> MarketState:
        """Reconstruct a MarketState from recorded decision data.

        Uses the decision row's snapshot fields. If a full states DataFrame
        is available, finds the closest matching state by timestamp.
        """
        timestamp_str = str(decision_row["timestamp"])

        # Build position if present
        position = None
        if decision_row.get("has_position", False):
            side_str = decision_row.get("position_side", "")
            position = PositionState(
                side=Side(side_str) if side_str else Side.LONG,
                quantity=int(decision_row.get("position_qty", 0)),
                avg_entry=float(decision_row.get("last_price", 0)),
                unrealized_pnl=float(decision_row.get("position_pnl", 0)),
            )

        # Build MarketState
        state = MarketState(
            timestamp=datetime.fromisoformat(timestamp_str)
            if "T" in timestamp_str
            else datetime.now(tz=UTC),
            symbol=str(decision_row.get("symbol", "MNQM6")),
            last_price=float(decision_row["last_price"]),
            regime=Regime(decision_row.get("regime", "choppy")),
            session_phase=SessionPhase(
                decision_row.get("session_phase", "morning")
            ),
            position=position,
            daily_pnl=float(decision_row.get("daily_pnl", 0)),
            daily_trades=int(decision_row.get("daily_trades", 0)),
            in_blackout=bool(decision_row.get("in_blackout", False)),
        )

        # Enrich from states DataFrame if available
        if states_df is not None and len(states_df) > 0:
            # Find closest state by timestamp
            closest_idx = self._find_closest_state(states_df, timestamp_str)
            if closest_idx is not None:
                srow = states_df.iloc[closest_idx]
                state.bid = float(srow.get("bid", 0))
                state.ask = float(srow.get("ask", 0))
                state.spread = float(srow.get("spread", 0))
                state.levels = KeyLevels(
                    vwap=float(srow.get("vwap", 0)),
                    session_high=float(srow.get("session_high", 0)),
                    session_low=float(srow.get("session_low", 0)),
                    poc=float(srow.get("poc", 0)),
                )
                state.flow = OrderFlowData(
                    cumulative_delta=float(srow.get("cumulative_delta", 0)),
                    delta_1min=float(srow.get("delta_1min", 0)),
                    rvol=float(srow.get("rvol", 0)),
                    tape_speed=float(srow.get("tape_speed", 0)),
                )
                state.cross_market = CrossMarketContext(
                    es_price=float(srow.get("es_price", 0)),
                    vix=float(srow.get("vix", 0)),
                    tick_index=int(srow.get("tick_index", 0)),
                )

        return state

    def _find_closest_state(self, states_df, timestamp_str: str) -> int | None:
        """Find the index of the state closest to the given timestamp."""
        try:
            timestamps = states_df["timestamp"].tolist()
            # Simple approach: find exact or nearest
            for i, ts in enumerate(timestamps):
                if str(ts) == timestamp_str:
                    return i
            # If no exact match, return the last state before the decision
            return len(timestamps) - 1
        except Exception:
            return None

    @staticmethod
    def summarize(results: list[ReplayResult]) -> dict[str, Any]:
        """Compute summary statistics from replay results.

        Returns:
            Dict with agreement rate, action distribution, avg latency, etc.
        """
        if not results:
            return {"total": 0, "agreement_rate": 0.0}

        total = len(results)
        agreed = sum(1 for r in results if r.agreed)

        # Action distribution
        original_actions: dict[str, int] = {}
        replay_actions: dict[str, int] = {}
        for r in results:
            original_actions[r.original_action] = (
                original_actions.get(r.original_action, 0) + 1
            )
            replay_actions[r.replay_action] = (
                replay_actions.get(r.replay_action, 0) + 1
            )

        avg_latency = sum(r.latency_ms for r in results) / total

        return {
            "total": total,
            "agreed": agreed,
            "agreement_rate": round(agreed / total, 4),
            "original_action_distribution": original_actions,
            "replay_action_distribution": replay_actions,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_original_confidence": round(
                sum(r.original_confidence for r in results) / total, 4
            ),
            "avg_replay_confidence": round(
                sum(r.replay_confidence for r in results) / total, 4
            ),
        }

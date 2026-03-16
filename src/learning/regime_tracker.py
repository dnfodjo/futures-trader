"""Regime tracker — tracks accuracy of market regime classifications.

The LLM classifies market conditions into regimes (trending_up, trending_down,
choppy, breakout, news_driven, low_volume). This module tracks:

1. How often each regime is classified
2. What the actual outcome was (profitable trades vs not)
3. Accuracy of the classification over time
4. Regime transitions and their patterns

If a regime classification is consistently associated with bad outcomes,
it signals the need for prompt tuning (e.g., "choppy" regime needs
more conservative settings, or "trending_up" is being called too eagerly).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from src.core.types import Regime

logger = structlog.get_logger()

_CREATE_REGIME_TABLE = """
CREATE TABLE IF NOT EXISTS regime_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime TEXT NOT NULL,
    price_at_classification REAL NOT NULL,
    price_after_5min REAL,
    price_after_15min REAL,
    was_correct INTEGER,
    trade_pnl REAL,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
)
"""

_CREATE_REGIME_INDEX = """
CREATE INDEX IF NOT EXISTS idx_regime_timestamp
    ON regime_observations(timestamp)
"""


class RegimeTracker:
    """Tracks regime classification accuracy over time.

    Usage:
        tracker = RegimeTracker(db_path="data/journal.db")
        tracker.record_classification(Regime.TRENDING_UP, price=19850.0)
        tracker.update_outcome(observation_id, price_after_5min=19860.0)

        accuracy = tracker.get_accuracy_by_regime()
        # {"trending_up": 0.72, "choppy": 0.55, ...}
    """

    def __init__(self, db_path: str = "data/journal.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_REGIME_TABLE)
        self._conn.execute(_CREATE_REGIME_INDEX)
        self._conn.commit()

    # ── Recording ─────────────────────────────────────────────────────────

    def record_classification(
        self,
        regime: Regime,
        price: float,
        timestamp: datetime | None = None,
    ) -> int:
        """Record a regime classification event.

        Args:
            regime: The classified regime.
            price: Current price at classification time.
            timestamp: When the classification happened.

        Returns:
            The observation ID for later outcome update.
        """
        if self._conn is None:
            return -1

        ts = timestamp or datetime.now(tz=UTC)

        cursor = self._conn.execute(
            """
            INSERT INTO regime_observations (timestamp, regime, price_at_classification, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (ts.isoformat(), regime.value, price, datetime.now(tz=UTC).isoformat()),
        )
        self._conn.commit()

        obs_id = cursor.lastrowid or -1
        logger.debug(
            "regime_tracker.classified",
            regime=regime.value,
            price=price,
            id=obs_id,
        )
        return obs_id

    def update_outcome(
        self,
        observation_id: int,
        price_after_5min: float | None = None,
        price_after_15min: float | None = None,
        trade_pnl: float | None = None,
        was_correct: bool | None = None,
        notes: str = "",
    ) -> None:
        """Update an observation with the actual outcome.

        Args:
            observation_id: ID from record_classification.
            price_after_5min: Price 5 minutes after classification.
            price_after_15min: Price 15 minutes after classification.
            trade_pnl: P&L of any trade taken during this regime.
            was_correct: Manual override for correctness.
            notes: Additional notes.
        """
        if self._conn is None:
            return

        updates: list[str] = []
        values: list[Any] = []

        if price_after_5min is not None:
            updates.append("price_after_5min = ?")
            values.append(price_after_5min)

        if price_after_15min is not None:
            updates.append("price_after_15min = ?")
            values.append(price_after_15min)

        if trade_pnl is not None:
            updates.append("trade_pnl = ?")
            values.append(trade_pnl)

        if was_correct is not None:
            updates.append("was_correct = ?")
            values.append(1 if was_correct else 0)

        if notes:
            updates.append("notes = ?")
            values.append(notes)

        if not updates:
            return

        values.append(observation_id)
        sql = f"UPDATE regime_observations SET {', '.join(updates)} WHERE id = ?"
        self._conn.execute(sql, values)
        self._conn.commit()

    def auto_evaluate(
        self,
        observation_id: int,
        regime: Regime,
        price_at_classification: float,
        price_after: float,
    ) -> bool:
        """Auto-evaluate whether a regime classification was correct.

        Heuristic:
        - trending_up: price went up (>2 points)
        - trending_down: price went down (>2 points)
        - choppy: price didn't move much (abs < 5 points)
        - breakout: price moved significantly (abs > 10 points)
        - low_volume: neutral (always considered "correct")
        - news_driven: neutral (always considered "correct")

        Args:
            observation_id: ID from record_classification.
            regime: The classified regime.
            price_at_classification: Price when classified.
            price_after: Price after some time.

        Returns:
            Whether the classification was correct.
        """
        delta = price_after - price_at_classification
        correct = False

        if regime == Regime.TRENDING_UP:
            correct = delta > 2.0
        elif regime == Regime.TRENDING_DOWN:
            correct = delta < -2.0
        elif regime == Regime.CHOPPY:
            correct = abs(delta) < 5.0
        elif regime == Regime.BREAKOUT:
            correct = abs(delta) > 10.0
        elif regime in (Regime.LOW_VOLUME, Regime.NEWS_DRIVEN):
            correct = True  # can't easily auto-evaluate these

        self.update_outcome(observation_id, was_correct=correct)
        return correct

    # ── Queries ────────────────────────────────────────────────────────────

    def get_accuracy_by_regime(
        self, min_observations: int = 10
    ) -> dict[str, float]:
        """Get classification accuracy for each regime.

        Args:
            min_observations: Minimum observations needed to report accuracy.

        Returns:
            Dict mapping regime name → accuracy (0-1).
        """
        if self._conn is None:
            return {}

        rows = self._conn.execute(
            """
            SELECT regime,
                   COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM regime_observations
            WHERE was_correct IS NOT NULL
            GROUP BY regime
            """
        ).fetchall()

        result: dict[str, float] = {}
        for row in rows:
            if row["total"] >= min_observations:
                result[row["regime"]] = row["correct"] / row["total"]

        return result

    def get_regime_distribution(self) -> dict[str, int]:
        """Get count of each regime classification.

        Returns:
            Dict mapping regime name → count.
        """
        if self._conn is None:
            return {}

        rows = self._conn.execute(
            "SELECT regime, COUNT(*) as cnt FROM regime_observations GROUP BY regime"
        ).fetchall()

        return {row["regime"]: row["cnt"] for row in rows}

    def get_regime_pnl(self) -> dict[str, dict[str, Any]]:
        """Get aggregate P&L stats by regime.

        Returns:
            Dict mapping regime → {"total_pnl": float, "avg_pnl": float, "count": int}.
        """
        if self._conn is None:
            return {}

        rows = self._conn.execute(
            """
            SELECT regime,
                   COUNT(*) as cnt,
                   COALESCE(SUM(trade_pnl), 0) as total_pnl,
                   COALESCE(AVG(trade_pnl), 0) as avg_pnl
            FROM regime_observations
            WHERE trade_pnl IS NOT NULL
            GROUP BY regime
            """
        ).fetchall()

        return {
            row["regime"]: {
                "total_pnl": round(row["total_pnl"], 2),
                "avg_pnl": round(row["avg_pnl"], 2),
                "count": row["cnt"],
            }
            for row in rows
        }

    def get_recent_observations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent regime observations.

        Args:
            limit: Max observations to return.

        Returns:
            List of observation dicts.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM regime_observations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_total_observations(self) -> int:
        """Get total number of regime observations."""
        if self._conn is None:
            return 0

        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM regime_observations"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_worst_regimes(self, min_trades: int = 5) -> list[dict[str, Any]]:
        """Get regimes with worst P&L performance.

        Returns regimes sorted by avg P&L (worst first) — useful for
        identifying regime classifications that consistently lead to losses.

        Args:
            min_trades: Minimum trades in regime to include.

        Returns:
            List of dicts with regime, avg_pnl, total_pnl, count.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            """
            SELECT regime,
                   COUNT(*) as cnt,
                   COALESCE(SUM(trade_pnl), 0) as total_pnl,
                   COALESCE(AVG(trade_pnl), 0) as avg_pnl
            FROM regime_observations
            WHERE trade_pnl IS NOT NULL
            GROUP BY regime
            HAVING cnt >= ?
            ORDER BY avg_pnl ASC
            """,
            (min_trades,),
        ).fetchall()

        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

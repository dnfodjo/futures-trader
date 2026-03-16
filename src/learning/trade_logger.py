"""Trade logger — full trade journal persisted to SQLite.

Every trade gets a complete record: entry/exit reasoning, market state
snapshots, regime, P&L, hold time, MFE/MAE. This is the system's memory
of every decision it ever made.

Supports:
- Writing new trade records
- Querying by date range, regime, side, outcome
- Aggregating stats for Kelly/Monte Carlo
- Loading recent trades for LLM context (postmortem, pre-market)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from src.core.types import Regime, SessionPhase, Side, TradeRecord

logger = structlog.get_logger()

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    timestamp_entry TEXT NOT NULL,
    timestamp_exit TEXT,
    side TEXT NOT NULL,
    entry_quantity INTEGER NOT NULL,
    exit_quantity INTEGER DEFAULT 0,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_price REAL NOT NULL,
    pnl REAL,
    commissions REAL DEFAULT 0.0,
    hold_time_sec INTEGER,
    max_favorable_excursion REAL DEFAULT 0.0,
    max_adverse_excursion REAL DEFAULT 0.0,
    reasoning_entry TEXT DEFAULT '',
    reasoning_exit TEXT,
    regime_at_entry TEXT DEFAULT 'choppy',
    session_phase_at_entry TEXT DEFAULT 'morning',
    debate_result TEXT,
    adds INTEGER DEFAULT 0,
    scale_outs INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

_CREATE_DAILY_SUMMARIES_TABLE = """
CREATE TABLE IF NOT EXISTS daily_summaries (
    date TEXT PRIMARY KEY,
    total_trades INTEGER DEFAULT 0,
    winners INTEGER DEFAULT 0,
    losers INTEGER DEFAULT 0,
    gross_pnl REAL DEFAULT 0.0,
    net_pnl REAL DEFAULT 0.0,
    commissions REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    postmortem TEXT DEFAULT '',
    grade TEXT DEFAULT '',
    created_at TEXT NOT NULL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_entry)
"""


class TradeLogger:
    """SQLite-backed trade journal for full lifecycle recording.

    Usage:
        logger = TradeLogger(db_path="data/journal.db")
        logger.log_trade(trade_record)
        trades = logger.get_trades_by_date("2026-03-14")
        recent = logger.get_recent_trades(limit=20)
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
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_CREATE_TRADES_TABLE)
        self._conn.execute(_CREATE_DAILY_SUMMARIES_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()
        logger.info("trade_logger.initialized", db=str(self._db_path))

    # ── Write ─────────────────────────────────────────────────────────────

    def log_trade(self, trade: TradeRecord) -> None:
        """Log a completed trade to the journal.

        Args:
            trade: TradeRecord with entry/exit data.
        """
        if self._conn is None:
            return

        debate_json = json.dumps(trade.debate_result) if trade.debate_result else None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO trades (
                id, timestamp_entry, timestamp_exit, side,
                entry_quantity, exit_quantity, entry_price, exit_price,
                stop_price, pnl, commissions, hold_time_sec,
                max_favorable_excursion, max_adverse_excursion,
                reasoning_entry, reasoning_exit, regime_at_entry,
                session_phase_at_entry,
                debate_result, adds, scale_outs, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.id,
                trade.timestamp_entry.isoformat(),
                trade.timestamp_exit.isoformat() if trade.timestamp_exit else None,
                trade.side.value,
                trade.entry_quantity,
                trade.exit_quantity,
                trade.entry_price,
                trade.exit_price,
                trade.stop_price,
                trade.pnl,
                trade.commissions,
                trade.hold_time_sec,
                trade.max_favorable_excursion,
                trade.max_adverse_excursion,
                trade.reasoning_entry,
                trade.reasoning_exit,
                trade.regime_at_entry.value,
                trade.session_phase_at_entry.value,
                debate_json,
                trade.adds,
                trade.scale_outs,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
        self._conn.commit()
        logger.debug(
            "trade_logger.logged",
            trade_id=trade.id,
            pnl=trade.pnl,
            side=trade.side.value,
        )

    def log_daily_summary(
        self,
        date: str,
        total_trades: int,
        winners: int,
        losers: int,
        gross_pnl: float,
        net_pnl: float,
        commissions: float,
        max_drawdown: float = 0.0,
        postmortem: str = "",
        grade: str = "",
    ) -> None:
        """Log an end-of-day summary.

        Args:
            date: Date string YYYY-MM-DD.
            total_trades: Total trades taken.
            winners: Number of winning trades.
            losers: Number of losing trades.
            gross_pnl: P&L before commissions.
            net_pnl: P&L after commissions.
            commissions: Total commissions paid.
            max_drawdown: Worst drawdown during the day.
            postmortem: LLM postmortem analysis text.
            grade: Session grade (A-F).
        """
        if self._conn is None:
            return

        self._conn.execute(
            """
            INSERT OR REPLACE INTO daily_summaries (
                date, total_trades, winners, losers,
                gross_pnl, net_pnl, commissions, max_drawdown,
                postmortem, grade, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                total_trades,
                winners,
                losers,
                gross_pnl,
                net_pnl,
                commissions,
                max_drawdown,
                postmortem,
                grade,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
        self._conn.commit()
        logger.info(
            "trade_logger.daily_summary",
            date=date,
            trades=total_trades,
            net_pnl=net_pnl,
            grade=grade,
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def get_trades_by_date(self, date: str) -> list[TradeRecord]:
        """Get all trades for a specific date.

        Args:
            date: Date string YYYY-MM-DD.

        Returns:
            List of TradeRecord for that date.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM trades WHERE timestamp_entry LIKE ? ORDER BY timestamp_entry",
            (f"{date}%",),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_trades_by_date_range(
        self, start_date: str, end_date: str
    ) -> list[TradeRecord]:
        """Get trades in a date range (inclusive).

        Args:
            start_date: Start date YYYY-MM-DD.
            end_date: End date YYYY-MM-DD.

        Returns:
            List of TradeRecord in range.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            """
            SELECT * FROM trades
            WHERE timestamp_entry >= ? AND timestamp_entry < ?
            ORDER BY timestamp_entry
            """,
            (f"{start_date}T00:00:00", f"{end_date}T23:59:59"),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_recent_trades(self, limit: int = 20) -> list[TradeRecord]:
        """Get the most recent trades.

        Args:
            limit: Maximum number of trades to return.

        Returns:
            List of TradeRecord, most recent first.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY timestamp_entry DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_trades_by_regime(self, regime: Regime) -> list[TradeRecord]:
        """Get all trades with a specific regime at entry.

        Args:
            regime: Regime enum value.

        Returns:
            List of TradeRecord with that regime.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM trades WHERE regime_at_entry = ? ORDER BY timestamp_entry",
            (regime.value,),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_trades_by_session_phase(self, phase: SessionPhase) -> list[TradeRecord]:
        """Get all trades with a specific session phase at entry.

        Args:
            phase: SessionPhase enum value.

        Returns:
            List of TradeRecord with that session phase.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM trades WHERE session_phase_at_entry = ? ORDER BY timestamp_entry",
            (phase.value,),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_trades_by_side(self, side: Side) -> list[TradeRecord]:
        """Get all trades for a specific side (long/short).

        Args:
            side: Side enum value.

        Returns:
            List of TradeRecord for that side.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM trades WHERE side = ? ORDER BY timestamp_entry",
            (side.value,),
        ).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def get_daily_summary(self, date: str) -> Optional[dict[str, Any]]:
        """Get the daily summary for a specific date.

        Args:
            date: Date string YYYY-MM-DD.

        Returns:
            Dict with summary data or None.
        """
        if self._conn is None:
            return None

        row = self._conn.execute(
            "SELECT * FROM daily_summaries WHERE date = ?",
            (date,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_recent_summaries(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent daily summaries.

        Args:
            limit: Maximum number of summaries.

        Returns:
            List of summary dicts, most recent first.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM daily_summaries ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_consecutive_red_days(self) -> int:
        """Count consecutive red (losing) days looking backwards from today.

        Returns:
            Number of consecutive days with negative net_pnl.
        """
        if self._conn is None:
            return 0

        rows = self._conn.execute(
            "SELECT net_pnl FROM daily_summaries ORDER BY date DESC LIMIT 10"
        ).fetchall()

        count = 0
        for row in rows:
            if row["net_pnl"] < 0:
                count += 1
            else:
                break

        return count

    def get_total_trade_count(self) -> int:
        """Get total number of logged trades."""
        if self._conn is None:
            return 0

        row = self._conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
        return row["cnt"] if row else 0

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_weekly_pnl(self) -> float:
        """Get net P&L for the current week (Mon-Sun)."""
        if self._conn is None:
            return 0.0

        # Get current week's Monday
        today = datetime.now(tz=UTC)
        monday = today.date()
        while monday.weekday() != 0:  # 0 = Monday
            from datetime import timedelta

            monday -= timedelta(days=1)

        rows = self._conn.execute(
            "SELECT COALESCE(SUM(net_pnl), 0) as total FROM daily_summaries WHERE date >= ?",
            (monday.isoformat(),),
        ).fetchone()

        return float(rows["total"]) if rows else 0.0

    def get_monthly_pnl(self) -> float:
        """Get net P&L for the current month."""
        if self._conn is None:
            return 0.0

        today = datetime.now(tz=UTC)
        month_start = today.strftime("%Y-%m-01")

        rows = self._conn.execute(
            "SELECT COALESCE(SUM(net_pnl), 0) as total FROM daily_summaries WHERE date >= ?",
            (month_start,),
        ).fetchone()

        return float(rows["total"]) if rows else 0.0

    def get_daily_pnl_history(self, limit: int = 60) -> list[tuple[str, float]]:
        """Get daily P&L history as (date, net_pnl) tuples.

        Returns results ordered chronologically (oldest first),
        suitable for loading into CircuitBreakers.

        Args:
            limit: Maximum number of days to return.

        Returns:
            List of (date_string, net_pnl) tuples.
        """
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT date, net_pnl FROM daily_summaries "
            "ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()

        # Reverse to chronological order (oldest first)
        return [(row["date"], float(row["net_pnl"])) for row in reversed(rows)]

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        """Convert a database row to a TradeRecord."""
        debate = None
        if row["debate_result"]:
            try:
                debate = json.loads(row["debate_result"])
            except (json.JSONDecodeError, TypeError):
                debate = None

        return TradeRecord(
            id=row["id"],
            timestamp_entry=datetime.fromisoformat(row["timestamp_entry"]),
            timestamp_exit=(
                datetime.fromisoformat(row["timestamp_exit"])
                if row["timestamp_exit"]
                else None
            ),
            side=Side(row["side"]),
            entry_quantity=row["entry_quantity"],
            exit_quantity=row["exit_quantity"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            stop_price=row["stop_price"],
            pnl=row["pnl"],
            commissions=row["commissions"],
            hold_time_sec=row["hold_time_sec"],
            max_favorable_excursion=row["max_favorable_excursion"],
            max_adverse_excursion=row["max_adverse_excursion"],
            reasoning_entry=row["reasoning_entry"],
            reasoning_exit=row["reasoning_exit"],
            regime_at_entry=Regime(row["regime_at_entry"]),
            session_phase_at_entry=SessionPhase(
                row["session_phase_at_entry"]
            ) if row["session_phase_at_entry"] else SessionPhase.MORNING,
            debate_result=debate,
            adds=row["adds"],
            scale_outs=row["scale_outs"],
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

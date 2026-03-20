"""Session-aware risk management for MNQ futures trading.

Replaces the old guardrail pipeline with a unified risk manager that
enforces hard gates before LLM calls and mechanical exit conditions
on every state update.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

from src.core.types import SessionPhase

logger = structlog.get_logger()

_RISK_STATE_PATH = Path("data/risk_state.json")


# ---------------------------------------------------------------------------
# Session risk parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionRiskParams:
    """Risk parameters for a specific trading session."""

    sl_points: float
    tp_points: float             # hard take-profit ceiling in points
    min_confluence: int          # out of 6
    min_confidence: float        # 0.0 - 1.0
    extra_entropy_check: bool = False   # True for NY midday
    max_entropy: float = 0.6            # only enforced when extra_entropy_check=True
    entries_allowed: bool = True        # False for NY last 30 min
    max_trades_per_session: int = 3    # max entries per session phase


# Canonical session parameter table
_SESSION_PARAMS: dict[SessionPhase, SessionRiskParams] = {
    # Overnight — TP 100 pts (tighter ranges)
    SessionPhase.ASIAN: SessionRiskParams(
        sl_points=30, tp_points=100, min_confluence=4, min_confidence=0.80,
        max_trades_per_session=1,
    ),
    SessionPhase.LONDON: SessionRiskParams(
        sl_points=35, tp_points=125, min_confluence=3, min_confidence=0.75,
        max_trades_per_session=2,
    ),
    SessionPhase.PRE_RTH: SessionRiskParams(
        sl_points=35, tp_points=125, min_confluence=3, min_confidence=0.75,
        max_trades_per_session=2,
    ),
    # RTH — TP 150 pts (most volatility)
    SessionPhase.OPEN_DRIVE: SessionRiskParams(
        sl_points=40, tp_points=150, min_confluence=3, min_confidence=0.70,
        max_trades_per_session=3,
    ),
    SessionPhase.MORNING: SessionRiskParams(
        sl_points=40, tp_points=150, min_confluence=3, min_confidence=0.70,
        max_trades_per_session=3,
    ),
    SessionPhase.MIDDAY: SessionRiskParams(
        sl_points=40, tp_points=150, min_confluence=4, min_confidence=0.75,
        extra_entropy_check=True, max_entropy=0.6,
        max_trades_per_session=2,
    ),
    SessionPhase.AFTERNOON: SessionRiskParams(
        sl_points=40, tp_points=150, min_confluence=3, min_confidence=0.70,
        max_trades_per_session=3,
    ),
    SessionPhase.CLOSE: SessionRiskParams(
        sl_points=40, tp_points=150, min_confluence=3, min_confidence=0.70,
        entries_allowed=False,
    ),
    # Post-RTH
    SessionPhase.POST_RTH: SessionRiskParams(
        sl_points=35, tp_points=100, min_confluence=4, min_confidence=0.80,
        entries_allowed=False,
    ),
    SessionPhase.DAILY_HALT: SessionRiskParams(
        sl_points=0, tp_points=0, min_confluence=6, min_confidence=1.0,
        entries_allowed=False,
    ),
    # Legacy aliases
    SessionPhase.PRE_MARKET: SessionRiskParams(
        sl_points=35, tp_points=125, min_confluence=3, min_confidence=0.75,
    ),
    SessionPhase.AFTER_HOURS: SessionRiskParams(
        sl_points=35, tp_points=100, min_confluence=4, min_confidence=0.80,
        entries_allowed=False,
    ),
}


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Centralised risk management for MNQ futures trading.

    Responsibilities:
      - Enforce hard gates before every LLM decision call
      - Enforce mechanical exit conditions on every state update
      - Track cooldown after losing trades
      - Provide session-specific risk parameters
    """

    def __init__(
        self,
        daily_loss_limit: float = 400.0,
        cooldown_bars_5m: int = 3,          # 3 bars x 5 min = 15 minutes
        daily_target: float = 500.0,        # advisory threshold for LLM
    ) -> None:
        self._daily_loss_limit = daily_loss_limit
        self._cooldown_bars = cooldown_bars_5m
        self._cooldown_duration = timedelta(minutes=cooldown_bars_5m * 5)
        self._daily_target = daily_target
        self._last_loss_time: Optional[datetime] = None
        self._shutdown = False
        # Per-session trade counting for anti-revenge-trading
        self._session_trade_counts: dict[str, int] = {}  # phase.value -> count
        self._current_session_phase: Optional[SessionPhase] = None

        logger.info(
            "risk_manager.init",
            daily_loss_limit=daily_loss_limit,
            cooldown_minutes=cooldown_bars_5m * 5,
            daily_target=daily_target,
        )

    # ------------------------------------------------------------------
    # Session parameters
    # ------------------------------------------------------------------

    def get_session_params(self, phase: SessionPhase) -> SessionRiskParams:
        """Return risk parameters for the current session phase."""
        params = _SESSION_PARAMS.get(phase)
        if params is None:
            logger.warning(
                "risk_manager.unknown_phase",
                phase=phase,
                fallback="morning_defaults",
            )
            return _SESSION_PARAMS[SessionPhase.MORNING]
        return params

    def get_sl_points(self, phase: SessionPhase) -> float:
        """Get stop-loss distance in points for the current session."""
        return self.get_session_params(phase).sl_points

    def get_tp_points(self, phase: SessionPhase) -> float:
        """Get take-profit distance in points for the current session."""
        return self.get_session_params(phase).tp_points

    # ------------------------------------------------------------------
    # Hard gates - checked BEFORE the LLM call
    # ------------------------------------------------------------------

    def check_entry_allowed(
        self,
        phase: SessionPhase,
        daily_pnl: float,
        has_position: bool,
        confluence_score: int,
        confidence: float,
        entropy: float,
        speed_state: str,
        trend_30m_agrees: bool,
        volume_sufficient: bool,
    ) -> tuple[bool, str]:
        """Check whether a new entry is allowed.

        Returns ``(allowed, reason)``.  If *allowed* is ``False`` the
        caller should force the action to FLAT and include *reason* in
        the decision log.
        """
        params = self.get_session_params(phase)

        # 1. System shutdown (daily loss already triggered)
        if self._shutdown:
            return False, "SHUTDOWN: daily loss limit already triggered"

        # 2. Daily P&L hard stop
        if daily_pnl <= -self._daily_loss_limit:
            self._shutdown = True
            logger.critical(
                "risk_manager.daily_loss_limit",
                daily_pnl=daily_pnl,
                limit=self._daily_loss_limit,
            )
            return False, f"DAILY_LOSS_LIMIT: P&L ${daily_pnl:.2f} <= -${self._daily_loss_limit:.2f}"

        # 3. Session doesn't allow new entries (e.g. last 30 min, post-RTH)
        if not params.entries_allowed:
            return False, f"SESSION_CLOSED: {phase.value} does not allow new entries"

        # 4. Already in a position (no stacking)
        if has_position:
            return False, "POSITION_EXISTS: must exit current position first"

        # 5. Cooldown active after a loss
        if self._is_cooldown_active():
            remaining = self._cooldown_remaining()
            return False, f"COOLDOWN: {remaining.total_seconds():.0f}s remaining after last loss"

        # 5b. Max trades per session phase
        phase_key = phase.value
        session_trades = self._session_trade_counts.get(phase_key, 0)
        if session_trades >= params.max_trades_per_session:
            return False, (
                f"SESSION_TRADE_LIMIT: {session_trades} trades in {phase_key} "
                f"(max {params.max_trades_per_session})"
            )

        # 6. Entropy too high (market too random)
        if entropy > 0.85:
            return False, f"HIGH_ENTROPY: {entropy:.3f} > 0.85 threshold"

        # 7. Speed state too slow (no momentum)
        if speed_state.upper() == "SLOW":
            return False, f"SLOW_SPEED: speed_state={speed_state}"

        # 8. 30-minute trend disagrees with signal direction
        # Only hard-block in NORMAL/SLOW markets.  In FAST markets (news
        # events, momentum moves) the 30m EMAs lag too much — a 200pt pump
        # can happen while EMA9 is still below EMA50.  The confluence score
        # threshold is the real protection: without trend points you need
        # stronger setup from other factors.
        if not trend_30m_agrees and speed_state.upper() != "FAST":
            return False, "TREND_DISAGREE: 30m trend against signal direction"

        # 9. Insufficient volume
        if not volume_sufficient:
            return False, "LOW_VOLUME: 5-bar avg volume below 20-bar avg"

        # 10. Confluence below session minimum
        if confluence_score < params.min_confluence:
            return False, (
                f"LOW_CONFLUENCE: {confluence_score}/{6} < "
                f"min {params.min_confluence}/{6} for {phase.value}"
            )

        # 11. Confidence below session minimum
        # Skip when confidence=0.0 — means LLM hasn't been called yet.
        # The real confidence check happens after the LLM returns.
        if confidence > 0.0 and confidence < params.min_confidence:
            return False, (
                f"LOW_CONFIDENCE: {confidence:.2f} < "
                f"min {params.min_confidence:.2f} for {phase.value}"
            )

        # 12. NY midday extra entropy check
        if params.extra_entropy_check and entropy > params.max_entropy:
            return False, (
                f"MIDDAY_ENTROPY: {entropy:.3f} > {params.max_entropy} "
                f"(extra check for {phase.value})"
            )

        logger.debug(
            "risk_manager.entry_allowed",
            phase=phase.value,
            confluence=confluence_score,
            confidence=confidence,
            entropy=entropy,
        )
        return True, "PASS"

    # ------------------------------------------------------------------
    # Mechanical exit conditions - checked on every state update
    # ------------------------------------------------------------------

    def check_exit_needed(
        self,
        phase: SessionPhase,
        daily_pnl: float,
        entropy: float,
        position_pnl: float,
        delta_against_minutes: int,
        absorption_against: bool,
    ) -> tuple[bool, str]:
        """Check whether a mechanical exit is required (independent of LLM).

        Returns ``(should_exit, reason)``.  When *should_exit* is
        ``True`` the orchestrator must flatten immediately.
        """
        reasons: list[str] = []

        # 1. Daily P&L hard stop - flatten everything
        if daily_pnl <= -self._daily_loss_limit:
            self._shutdown = True
            logger.critical(
                "risk_manager.force_exit.daily_loss",
                daily_pnl=daily_pnl,
            )
            return True, f"DAILY_LOSS_LIMIT: P&L ${daily_pnl:.2f} breached -${self._daily_loss_limit:.2f}"

        # 2. Session boundary approaching (last 5 min of CLOSE phase, i.e. 3:55-4:00 PM ET)
        if phase == SessionPhase.CLOSE:
            now_et = datetime.now(ZoneInfo("US/Eastern"))
            minutes_left = (16 * 60) - (now_et.hour * 60 + now_et.minute)  # minutes until 4:00 PM
            if minutes_left <= 5:
                reasons.append(f"SESSION_BOUNDARY: {minutes_left}min to RTH close, must exit")

        # 3. Entropy spiking - conditions too choppy to hold
        if entropy > 0.85:
            reasons.append(f"HIGH_ENTROPY: {entropy:.3f} > 0.85, conditions too random")

        # If we accumulated any advisory reasons, return the first critical one
        # (non-daily-loss reasons are strong suggestions but the orchestrator
        # can choose to let the LLM weigh in)
        if reasons:
            combined = "; ".join(reasons)
            logger.warning("risk_manager.exit_suggested", reasons=combined)
            return True, combined

        return False, "HOLD"

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def record_entry(self, phase: SessionPhase) -> None:
        """Record that an entry was made in the given session phase."""
        phase_key = phase.value
        self._session_trade_counts[phase_key] = self._session_trade_counts.get(phase_key, 0) + 1
        logger.info(
            "risk_manager.entry_recorded",
            phase=phase_key,
            count=self._session_trade_counts[phase_key],
        )
        self._persist_state()

    def record_loss(self, timestamp: datetime) -> None:
        """Record a losing trade timestamp for cooldown tracking."""
        self._last_loss_time = timestamp
        logger.info(
            "risk_manager.loss_recorded",
            timestamp=timestamp.isoformat(),
            cooldown_until=(timestamp + self._cooldown_duration).isoformat(),
        )
        self._persist_state()

    def _is_cooldown_active(self) -> bool:
        if self._last_loss_time is None:
            return False
        now = datetime.now(UTC)
        return now < self._last_loss_time + self._cooldown_duration

    def _cooldown_remaining(self) -> timedelta:
        if self._last_loss_time is None:
            return timedelta(0)
        end = self._last_loss_time + self._cooldown_duration
        remaining = end - datetime.now(UTC)
        return max(remaining, timedelta(0))

    # ------------------------------------------------------------------
    # Advisory helpers
    # ------------------------------------------------------------------

    def is_daily_target_met(self, daily_pnl: float) -> bool:
        """Signal that the daily target has been reached.

        Returns ``True`` when ``daily_pnl >= daily_target``.  This does
        NOT force a stop -- the LLM decides whether to continue trading
        or shut down for the day.
        """
        met = daily_pnl >= self._daily_target
        if met:
            logger.info(
                "risk_manager.daily_target_met",
                daily_pnl=daily_pnl,
                target=self._daily_target,
            )
        return met

    @property
    def is_shutdown(self) -> bool:
        """Whether the risk manager has triggered a daily shutdown."""
        return self._shutdown

    def reset_daily(self) -> None:
        """Reset daily state (call at session open / 6 PM ET)."""
        self._shutdown = False
        self._last_loss_time = None
        self._session_trade_counts.clear()
        self._persist_state()
        logger.info("risk_manager.daily_reset")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Save cooldown and trade count state to disk for crash recovery."""
        try:
            _RISK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "session_date": datetime.now(ZoneInfo("US/Eastern")).date().isoformat(),
                "last_loss_time": self._last_loss_time.isoformat() if self._last_loss_time else None,
                "session_trade_counts": self._session_trade_counts,
                "shutdown": self._shutdown,
            }
            _RISK_STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.warning("risk_manager.persist_failed", exc_info=True)

    def load_persisted_state(self) -> None:
        """Load state from disk on startup."""
        try:
            if not _RISK_STATE_PATH.exists():
                return
            data = json.loads(_RISK_STATE_PATH.read_text())

            # Date validation: don't restore session counts or shutdown from a different day
            today_et = datetime.now(ZoneInfo("US/Eastern")).date().isoformat()
            persisted_date = data.get("session_date", "")
            is_same_day = (persisted_date == today_et)

            if not is_same_day:
                logger.info(
                    "risk_manager.stale_state_discarded",
                    persisted_date=persisted_date,
                    today=today_et,
                )

            # Cooldown: always restore if still active (loss could be minutes ago)
            if data.get("last_loss_time"):
                dt = datetime.fromisoformat(data["last_loss_time"])
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                self._last_loss_time = dt
                if not self._is_cooldown_active():
                    self._last_loss_time = None
                else:
                    logger.info(
                        "risk_manager.cooldown_restored",
                        remaining=self._cooldown_remaining().total_seconds(),
                    )

            # Session trade counts and shutdown: only restore on same trading day
            if is_same_day:
                if data.get("session_trade_counts"):
                    self._session_trade_counts = data["session_trade_counts"]
                if data.get("shutdown"):
                    self._shutdown = True
                    logger.warning("risk_manager.shutdown_restored")
        except Exception:
            logger.warning("risk_manager.load_failed", exc_info=True)

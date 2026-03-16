"""Kill switch — emergency flatten on critical conditions.

The kill switch is the last line of defense. When triggered, it:
1. Cancels ALL working orders
2. Liquidates the entire position via REST
3. Signals the system to shut down for the day

Trigger conditions:
- Flash crash: price moves 50+ points in <60 seconds
- Connection lost: no data for 30+ seconds while in position
- LLM failure: 3+ consecutive API failures while in position
- Daily loss limit: P&L hits -$400 (or configured limit)
- Manual trigger: operator sends kill command

The kill switch uses emergency=True on REST calls to bypass
rate limiting — safety always takes priority.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.exceptions import (
    ConnectionTimeoutError,
    FlashCrashDetected,
    KillSwitchTriggered,
    LLMFailureThreshold,
)
from src.core.types import Event, EventType, PositionState

logger = structlog.get_logger()


class KillSwitch:
    """Emergency position flatten and system shutdown.

    Usage:
        ks = KillSwitch(event_bus=bus, flatten_fn=flatten)
        ks.check_flash_crash(price, timestamp)
        ks.check_connection(last_data_time, in_position)
        ks.check_llm_failures(consecutive_failures, in_position)
        ks.check_daily_loss(daily_pnl)
    """

    def __init__(
        self,
        event_bus: EventBus,
        flatten_fn: Any = None,
        flash_crash_points: float = 50.0,
        flash_crash_seconds: float = 60.0,
        connection_timeout_sec: float = 30.0,
        llm_failure_threshold: int = 3,
        daily_loss_limit: float = 400.0,
    ) -> None:
        self._bus = event_bus
        self._flatten_fn = flatten_fn

        # Thresholds
        self._flash_crash_points = flash_crash_points
        self._flash_crash_seconds = flash_crash_seconds
        self._connection_timeout_sec = connection_timeout_sec
        self._llm_failure_threshold = llm_failure_threshold
        self._daily_loss_limit = daily_loss_limit

        # Price tracking for flash crash detection
        self._price_window: list[tuple[float, float]] = []  # (price, timestamp)
        self._window_max_sec: float = flash_crash_seconds
        self._flash_crash_warmup_sec: float = 15.0  # Need 15s of data before checking
        self._flash_crash_min_samples: int = 3       # Need at least 3 price points
        self._flash_crash_start_time: Optional[float] = None  # monotonic time of first price

        # State
        self._is_triggered: bool = False
        self._trigger_reason: str = ""
        self._trigger_time: Optional[datetime] = None
        self._trigger_count: int = 0

    # ── Flash Crash Detection ──────────────────────────────────────────────────

    def reset_price_window(self) -> None:
        """Clear the flash crash price window.

        Call this when transitioning to live trading to prevent false
        positives from overnight/pre-market price gaps.
        """
        self._price_window.clear()
        self._flash_crash_start_time = None
        logger.info("kill_switch.price_window_reset")

    def check_flash_crash(self, price: float, timestamp: float) -> bool:
        """Check if a flash crash has occurred.

        Records the price and checks if price has moved >= flash_crash_points
        within the flash_crash_seconds window.

        Requires a warmup period (15s and 3+ samples) to avoid false
        positives from session open price gaps.

        Args:
            price: Current market price.
            timestamp: Monotonic timestamp (time.monotonic() or similar).

        Returns:
            True if flash crash detected and kill switch triggered.
        """
        if self._is_triggered:
            return True

        # Track when we first started receiving prices
        if self._flash_crash_start_time is None:
            self._flash_crash_start_time = timestamp

        # Add to window
        self._price_window.append((price, timestamp))

        # Prune old entries
        cutoff = timestamp - self._window_max_sec
        self._price_window = [
            (p, t) for p, t in self._price_window if t >= cutoff
        ]

        # Warmup guard: need enough time and samples for reliable detection
        elapsed = timestamp - self._flash_crash_start_time
        if elapsed < self._flash_crash_warmup_sec:
            return False
        if len(self._price_window) < self._flash_crash_min_samples:
            return False

        # Check max price movement in window
        prices = [p for p, _ in self._price_window]
        max_move = max(prices) - min(prices)

        # Sanity ceiling: >500 NQ points in 60s is a data artifact, not a
        # real flash crash.  Real NQ crashes are 50-200 points.
        if max_move > 500:
            logger.warning(
                "kill_switch.price_spike_ignored",
                max_move=max_move,
                window_size=len(self._price_window),
                msg="Ignoring likely data artifact (>500pt move)",
            )
            # Prune to only the latest price to reset baseline
            self._price_window = [self._price_window[-1]]
            return False

        if max_move >= self._flash_crash_points:
            self._trigger(
                f"Flash crash detected: {max_move:.1f} point move in "
                f"<{self._flash_crash_seconds:.0f}s"
            )
            return True

        return False

    # ── Connection Monitoring ──────────────────────────────────────────────────

    def check_connection(
        self,
        last_data_time: datetime,
        in_position: bool,
    ) -> bool:
        """Check if data connection has been lost too long while in position.

        Args:
            last_data_time: When we last received market data.
            in_position: Whether we currently hold a position.

        Returns:
            True if connection timeout triggered kill switch.
        """
        if self._is_triggered:
            return True

        if not in_position:
            return False

        elapsed = (datetime.now(tz=UTC) - last_data_time).total_seconds()

        if elapsed >= self._connection_timeout_sec:
            self._trigger(
                f"Connection lost for {elapsed:.0f}s while in position "
                f"(threshold: {self._connection_timeout_sec:.0f}s)"
            )
            return True

        return False

    # ── LLM Failure Monitoring ─────────────────────────────────────────────────

    def check_llm_failures(
        self,
        consecutive_failures: int,
        in_position: bool,
    ) -> bool:
        """Check if too many consecutive LLM failures while in position.

        Args:
            consecutive_failures: Number of consecutive LLM API failures.
            in_position: Whether we currently hold a position.

        Returns:
            True if LLM failure threshold triggered kill switch.
        """
        if self._is_triggered:
            return True

        if not in_position:
            return False

        if consecutive_failures >= self._llm_failure_threshold:
            self._trigger(
                f"LLM API failed {consecutive_failures} consecutive times "
                f"while in position (threshold: {self._llm_failure_threshold})"
            )
            return True

        return False

    # ── Daily Loss Check ───────────────────────────────────────────────────────

    def check_daily_loss(self, daily_pnl: float) -> bool:
        """Check if daily loss limit has been exceeded.

        Args:
            daily_pnl: Current daily net P&L (negative = loss).

        Returns:
            True if daily loss limit triggered kill switch.
        """
        if self._is_triggered:
            return True

        if daily_pnl <= -self._daily_loss_limit:
            self._trigger(
                f"Daily loss limit hit: ${daily_pnl:.2f} "
                f"(limit: -${self._daily_loss_limit:.2f})"
            )
            return True

        return False

    # ── Manual Trigger ─────────────────────────────────────────────────────────

    def trigger_manual(self, reason: str = "Manual kill switch") -> None:
        """Manually trigger the kill switch."""
        self._trigger(reason)

    # ── Core Trigger ───────────────────────────────────────────────────────────

    def _trigger(self, reason: str) -> None:
        """Execute the kill switch."""
        self._is_triggered = True
        self._trigger_reason = reason
        self._trigger_time = datetime.now(tz=UTC)
        self._trigger_count += 1

        logger.critical(
            "kill_switch.TRIGGERED",
            reason=reason,
            trigger_count=self._trigger_count,
        )

        self._bus.publish_nowait(Event(
            type=EventType.KILL_SWITCH_ACTIVATED,
            data={"reason": reason, "trigger_count": self._trigger_count},
        ))

    async def execute_flatten(self, timeout_sec: float = 10.0) -> dict[str, Any]:
        """Execute emergency flatten via the provided flatten function.

        Args:
            timeout_sec: Maximum time to wait for flatten to complete.
                         Defaults to 10 seconds — if the flatten hangs,
                         we report timeout so the operator can intervene.

        Returns:
            Result from the flatten operation.
        """
        if self._flatten_fn is None:
            logger.error("kill_switch.no_flatten_fn")
            return {"status": "error", "reason": "no_flatten_function_configured"}

        try:
            if asyncio.iscoroutinefunction(self._flatten_fn):
                result = await asyncio.wait_for(
                    self._flatten_fn(), timeout=timeout_sec
                )
            else:
                result = self._flatten_fn()

            logger.info("kill_switch.flatten_executed", result=str(result)[:100])
            return {"status": "flattened", "result": result}

        except asyncio.TimeoutError:
            logger.critical(
                "kill_switch.flatten_timeout",
                timeout_sec=timeout_sec,
            )
            return {
                "status": "timeout",
                "error": f"Flatten did not complete within {timeout_sec}s",
            }

        except Exception as e:
            logger.error("kill_switch.flatten_failed", error=str(e))
            return {"status": "flatten_failed", "error": str(e)}

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the kill switch for a new session.

        Only call this at the start of a new trading day.
        """
        self._is_triggered = False
        self._trigger_reason = ""
        self._trigger_time = None
        self._price_window.clear()

        logger.info("kill_switch.reset")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_triggered(self) -> bool:
        return self._is_triggered

    @property
    def trigger_reason(self) -> str:
        return self._trigger_reason

    @property
    def trigger_time(self) -> Optional[datetime]:
        return self._trigger_time

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "is_triggered": self._is_triggered,
            "trigger_reason": self._trigger_reason,
            "trigger_time": (
                self._trigger_time.isoformat() if self._trigger_time else None
            ),
            "trigger_count": self._trigger_count,
            "price_window_size": len(self._price_window),
            "thresholds": {
                "flash_crash_points": self._flash_crash_points,
                "flash_crash_seconds": self._flash_crash_seconds,
                "connection_timeout_sec": self._connection_timeout_sec,
                "llm_failure_threshold": self._llm_failure_threshold,
                "daily_loss_limit": self._daily_loss_limit,
            },
        }

"""Token bucket rate limiter for Tradovate API.

Tradovate allows 5,000 actions per rolling 60-minute window (IP-based).
We budget 4,500/hr for normal operations and reserve 500 for emergencies
(kill switch, position queries during incidents).

Emergency calls bypass the limiter entirely.

Burst protection: a per-second sub-limit prevents spiking the full
hourly budget in a short burst (default: 25 requests/second).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog

from src.core.exceptions import RateLimitExceeded

logger = structlog.get_logger()

# Default budget per hour
_DEFAULT_BUDGET = 4500
_EMERGENCY_RESERVE = 500
_WINDOW_SECONDS = 3600  # 60-minute rolling window
_DEFAULT_BURST_LIMIT = 25  # Max requests per second


class RateLimiter:
    """Sliding window rate limiter with emergency bypass and burst protection.

    Usage:
        limiter = RateLimiter()
        await limiter.acquire()            # blocks if budget exhausted
        await limiter.acquire(emergency=True)  # always passes
    """

    def __init__(
        self,
        budget: int = _DEFAULT_BUDGET,
        window_seconds: int = _WINDOW_SECONDS,
        burst_limit: int = _DEFAULT_BURST_LIMIT,
    ) -> None:
        self._budget = budget
        self._window = window_seconds
        self._burst_limit = burst_limit
        self._timestamps: deque[float] = deque()
        self._burst_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._total_emergency = 0
        self._total_throttled = 0
        self._total_burst_limited = 0

    def _prune_old(self) -> None:
        """Remove timestamps older than the rolling window."""
        cutoff = time.monotonic() - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def _prune_burst(self) -> None:
        """Remove burst timestamps older than 1 second."""
        cutoff = time.monotonic() - 1.0
        while self._burst_timestamps and self._burst_timestamps[0] < cutoff:
            self._burst_timestamps.popleft()

    @property
    def remaining(self) -> int:
        """Number of requests remaining in the current window."""
        self._prune_old()
        return max(0, self._budget - len(self._timestamps))

    @property
    def usage_pct(self) -> float:
        """Current usage as a percentage of budget (0-100)."""
        self._prune_old()
        if self._budget == 0:
            return 100.0
        return (len(self._timestamps) / self._budget) * 100.0

    async def acquire(self, emergency: bool = False) -> None:
        """Acquire a rate limit token.

        Args:
            emergency: If True, bypasses the limiter (for kill switch, etc.)

        Raises:
            RateLimitExceeded: If budget is exhausted and not an emergency.
        """
        if emergency:
            self._total_emergency += 1
            async with self._lock:
                self._timestamps.append(time.monotonic())
                self._total_requests += 1
            logger.debug("rate_limiter.emergency_bypass", remaining=self.remaining)
            return

        async with self._lock:
            self._prune_old()
            self._prune_burst()

            # Check burst limit (per-second sub-limit)
            if self._burst_limit > 0 and len(self._burst_timestamps) >= self._burst_limit:
                self._total_burst_limited += 1
                logger.warning(
                    "rate_limiter.burst_limited",
                    burst_count=len(self._burst_timestamps),
                    burst_limit=self._burst_limit,
                )
                raise RateLimitExceeded(
                    f"Burst limit exceeded ({self._burst_limit}/sec). "
                    f"Slow down request rate."
                )

            if len(self._timestamps) >= self._budget:
                self._total_throttled += 1

                # Calculate wait time until oldest request falls out of window
                if self._timestamps:
                    wait = self._timestamps[0] + self._window - time.monotonic()
                else:
                    wait = self._window  # Budget is zero — full window wait

                logger.warning(
                    "rate_limiter.budget_exhausted",
                    used=len(self._timestamps),
                    budget=self._budget,
                    wait_sec=round(wait, 1),
                )
                raise RateLimitExceeded(
                    f"Rate limit budget exhausted ({self._budget}/hr). "
                    f"Wait {wait:.0f}s or use emergency=True."
                )

            now = time.monotonic()
            self._timestamps.append(now)
            self._burst_timestamps.append(now)
            self._total_requests += 1

        # Log warnings at usage thresholds
        pct = self.usage_pct
        if pct >= 90:
            logger.warning("rate_limiter.near_limit", usage_pct=round(pct, 1))
        elif pct >= 75:
            logger.info("rate_limiter.high_usage", usage_pct=round(pct, 1))

    async def acquire_or_wait(self, timeout: float = 60.0, emergency: bool = False) -> None:
        """Acquire a token, waiting if necessary (up to timeout).

        Unlike `acquire()`, this will wait for capacity instead of raising.

        Args:
            timeout: Maximum seconds to wait for capacity.
            emergency: If True, bypasses the limiter.

        Raises:
            RateLimitExceeded: If timeout expires without acquiring a token.
        """
        if emergency:
            await self.acquire(emergency=True)
            return

        start = time.monotonic()
        while True:
            try:
                await self.acquire()
                return
            except RateLimitExceeded:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise RateLimitExceeded(
                        f"Rate limit timeout after {timeout}s. "
                        f"Budget: {self._budget}/hr"
                    )
                # Wait a bit and retry
                await asyncio.sleep(min(1.0, timeout - elapsed))

    @property
    def stats(self) -> dict:
        """Current rate limiter statistics."""
        self._prune_old()
        return {
            "remaining": self.remaining,
            "used_in_window": len(self._timestamps),
            "budget": self._budget,
            "usage_pct": round(self.usage_pct, 1),
            "total_requests": self._total_requests,
            "total_emergency": self._total_emergency,
            "total_throttled": self._total_throttled,
            "total_burst_limited": self._total_burst_limited,
        }

    def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        self._timestamps.clear()
        self._burst_timestamps.clear()
        self._total_requests = 0
        self._total_emergency = 0
        self._total_throttled = 0
        self._total_burst_limited = 0

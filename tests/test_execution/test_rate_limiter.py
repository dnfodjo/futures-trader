"""Tests for the token bucket rate limiter."""

import asyncio
import time

import pytest

from src.core.exceptions import RateLimitExceeded
from src.execution.rate_limiter import RateLimiter


@pytest.fixture
def limiter():
    """Small-budget limiter for fast tests."""
    return RateLimiter(budget=10, window_seconds=60)


class TestAcquire:
    async def test_acquire_within_budget(self, limiter: RateLimiter):
        """Should succeed when under budget."""
        await limiter.acquire()
        assert limiter.remaining == 9

    async def test_acquire_multiple(self, limiter: RateLimiter):
        """Acquiring multiple tokens decrements remaining."""
        for _ in range(5):
            await limiter.acquire()
        assert limiter.remaining == 5

    async def test_acquire_exhausted_raises(self, limiter: RateLimiter):
        """Should raise RateLimitExceeded when budget is exhausted."""
        for _ in range(10):
            await limiter.acquire()

        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()

    async def test_emergency_bypasses_limit(self, limiter: RateLimiter):
        """Emergency calls always succeed, even when over budget."""
        # Exhaust the budget
        for _ in range(10):
            await limiter.acquire()

        # Emergency should still work
        await limiter.acquire(emergency=True)
        # Remaining goes negative relative to budget
        stats = limiter.stats
        assert stats["total_emergency"] == 1

    async def test_emergency_tracks_separately(self, limiter: RateLimiter):
        await limiter.acquire(emergency=True)
        await limiter.acquire(emergency=True)
        assert limiter.stats["total_emergency"] == 2


class TestSlidingWindow:
    async def test_old_requests_expire(self):
        """Requests older than the window should be pruned."""
        # Use a very short window for testing
        limiter = RateLimiter(budget=3, window_seconds=1)

        # Exhaust budget
        for _ in range(3):
            await limiter.acquire()
        assert limiter.remaining == 0

        # Wait for the window to expire
        await asyncio.sleep(1.1)

        # Should be able to acquire again
        assert limiter.remaining == 3
        await limiter.acquire()
        assert limiter.remaining == 2


class TestAcquireOrWait:
    async def test_acquire_or_wait_immediate(self, limiter: RateLimiter):
        """Should acquire immediately when capacity is available."""
        await limiter.acquire_or_wait()
        assert limiter.remaining == 9

    async def test_acquire_or_wait_timeout(self, limiter: RateLimiter):
        """Should raise after timeout when budget is exhausted."""
        # Exhaust budget
        for _ in range(10):
            await limiter.acquire()

        # Should timeout quickly
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire_or_wait(timeout=0.5)

    async def test_acquire_or_wait_emergency_bypasses(self, limiter: RateLimiter):
        """Emergency should bypass even in acquire_or_wait."""
        for _ in range(10):
            await limiter.acquire()
        await limiter.acquire_or_wait(emergency=True)

    async def test_acquire_or_wait_succeeds_after_expiry(self):
        """Should succeed after old requests expire during wait."""
        limiter = RateLimiter(budget=2, window_seconds=1)

        await limiter.acquire()
        await limiter.acquire()
        assert limiter.remaining == 0

        # This should wait for ~1 second until old requests expire, then succeed
        start = time.monotonic()
        await limiter.acquire_or_wait(timeout=2.0)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.8  # Should have waited for expiry


class TestProperties:
    async def test_remaining_initial(self, limiter: RateLimiter):
        assert limiter.remaining == 10

    async def test_usage_pct_empty(self, limiter: RateLimiter):
        assert limiter.usage_pct == 0.0

    async def test_usage_pct_half(self, limiter: RateLimiter):
        for _ in range(5):
            await limiter.acquire()
        assert limiter.usage_pct == 50.0

    async def test_usage_pct_full(self, limiter: RateLimiter):
        for _ in range(10):
            await limiter.acquire()
        assert limiter.usage_pct == 100.0


class TestStats:
    async def test_stats_initial(self, limiter: RateLimiter):
        stats = limiter.stats
        assert stats["remaining"] == 10
        assert stats["budget"] == 10
        assert stats["total_requests"] == 0
        assert stats["total_emergency"] == 0
        assert stats["total_throttled"] == 0
        assert stats["total_burst_limited"] == 0

    async def test_stats_after_usage(self, limiter: RateLimiter):
        for _ in range(5):
            await limiter.acquire()
        await limiter.acquire(emergency=True)

        stats = limiter.stats
        assert stats["total_requests"] == 6
        assert stats["total_emergency"] == 1
        assert stats["used_in_window"] == 6

    async def test_stats_after_throttle(self, limiter: RateLimiter):
        for _ in range(10):
            await limiter.acquire()

        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()

        assert limiter.stats["total_throttled"] == 1


class TestReset:
    async def test_reset_clears_everything(self, limiter: RateLimiter):
        for _ in range(5):
            await limiter.acquire()

        limiter.reset()
        assert limiter.remaining == 10
        stats = limiter.stats
        assert stats["total_requests"] == 0
        assert stats["used_in_window"] == 0


class TestBurstProtection:
    async def test_burst_limit_enforced(self):
        """Should reject requests that exceed burst limit (per-second)."""
        # Budget of 100/hr but burst limit of 5/sec
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=5)

        # First 5 should succeed
        for _ in range(5):
            await limiter.acquire()

        # 6th should be burst-limited
        with pytest.raises(RateLimitExceeded, match="Burst limit"):
            await limiter.acquire()

    async def test_burst_limit_resets_after_1_second(self):
        """Burst counter should clear after 1 second."""
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=3)

        # Exhaust burst limit
        for _ in range(3):
            await limiter.acquire()

        with pytest.raises(RateLimitExceeded, match="Burst"):
            await limiter.acquire()

        # Wait for burst window to expire
        await asyncio.sleep(1.1)

        # Should work again
        await limiter.acquire()
        assert limiter.stats["total_requests"] == 4

    async def test_burst_limited_tracked_in_stats(self):
        """Burst-limited requests should be counted in stats."""
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=2)

        await limiter.acquire()
        await limiter.acquire()

        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()

        assert limiter.stats["total_burst_limited"] == 1

    async def test_emergency_bypasses_burst_limit(self):
        """Emergency calls should bypass burst protection."""
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=2)

        await limiter.acquire()
        await limiter.acquire()

        # Burst-limited for normal
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()

        # Emergency should still work
        await limiter.acquire(emergency=True)
        assert limiter.stats["total_emergency"] == 1

    async def test_burst_limit_zero_disables(self):
        """burst_limit=0 should disable burst protection."""
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=0)

        # Should be able to fire rapidly
        for _ in range(50):
            await limiter.acquire()
        assert limiter.stats["total_requests"] == 50

    async def test_reset_clears_burst(self):
        """Reset should clear burst timestamps too."""
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=3)

        for _ in range(3):
            await limiter.acquire()

        limiter.reset()

        # Should work again without waiting
        await limiter.acquire()
        assert limiter.stats["total_burst_limited"] == 0


class TestEdgeCases:
    async def test_zero_budget(self):
        """Zero budget should immediately exhaust."""
        limiter = RateLimiter(budget=0)
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()

    async def test_concurrent_acquires(self):
        """Multiple concurrent acquires should be thread-safe."""
        # High burst limit to avoid burst protection during concurrency test
        limiter = RateLimiter(budget=100, window_seconds=60, burst_limit=200)

        async def acquire_n(n: int):
            for _ in range(n):
                await limiter.acquire()

        # Run 10 concurrent tasks each acquiring 10 tokens
        await asyncio.gather(*[acquire_n(10) for _ in range(10)])
        assert limiter.remaining == 0
        assert limiter.stats["total_requests"] == 100

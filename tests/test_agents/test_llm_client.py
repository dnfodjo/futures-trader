"""Tests for the LLM client wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.llm_client import (
    LLMCallFailed,
    LLMClient,
    LLMCostCapExceeded,
    LLMResponse,
    _estimate_cost,
    _get_model_tier,
)


# ── Test: Cost Estimation ────────────────────────────────────────────────────


class TestCostEstimation:
    def test_haiku_input_only(self) -> None:
        """Haiku cost for input tokens."""
        cost = _estimate_cost("haiku", input_tokens=1000, output_tokens=0)
        assert cost == pytest.approx(0.0008, abs=0.0001)

    def test_sonnet_input_output(self) -> None:
        """Sonnet cost for input + output tokens."""
        cost = _estimate_cost("sonnet", input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_cache_read_reduces_cost(self) -> None:
        """Cache read tokens are cheaper than regular input."""
        regular = _estimate_cost("sonnet", input_tokens=1000, output_tokens=0)
        cached = _estimate_cost(
            "sonnet",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1000,
        )
        assert cached < regular

    def test_unknown_tier_defaults_to_sonnet(self) -> None:
        """Unknown model tier uses sonnet pricing."""
        cost = _estimate_cost("unknown_tier", input_tokens=1000, output_tokens=0)
        sonnet_cost = _estimate_cost("sonnet", input_tokens=1000, output_tokens=0)
        assert cost == sonnet_cost


class TestModelTier:
    def test_haiku_detection(self) -> None:
        assert _get_model_tier("claude-haiku-4-5-20251001") == "haiku"

    def test_sonnet_detection(self) -> None:
        assert _get_model_tier("claude-sonnet-4-6-20260314") == "sonnet"

    def test_unknown_defaults_to_sonnet(self) -> None:
        assert _get_model_tier("claude-opus-5") == "sonnet"


# ── Test: LLMResponse ────────────────────────────────────────────────────────


class TestLLMResponse:
    def test_basic_response(self) -> None:
        resp = LLMResponse(text="Hello", input_tokens=100, output_tokens=50)
        assert resp.text == "Hello"
        assert not resp.has_tool_calls

    def test_tool_calls(self) -> None:
        resp = LLMResponse(
            tool_calls=[
                {"id": "1", "name": "test_tool", "input": {"key": "value"}}
            ]
        )
        assert resp.has_tool_calls
        assert resp.first_tool_input == {"key": "value"}
        assert resp.get_tool_input("test_tool") == {"key": "value"}
        assert resp.get_tool_input("nonexistent") is None

    def test_empty_tool_calls(self) -> None:
        resp = LLMResponse()
        assert not resp.has_tool_calls
        assert resp.first_tool_input is None


# ── Test: LLMClient Initialization ──────────────────────────────────────────


class TestLLMClientInit:
    def test_defaults(self) -> None:
        client = LLMClient(api_key="test-key")
        assert client.daily_cost == 0.0
        assert client.total_cost == 0.0
        assert client.call_count == 0
        assert client.consecutive_failures == 0

    def test_custom_config(self) -> None:
        client = LLMClient(
            api_key="test-key",
            daily_cost_cap=5.0,
            daily_cost_alert=2.0,
            max_retries=5,
        )
        assert client._daily_cost_cap == 5.0
        assert client._daily_cost_alert == 2.0
        assert client._max_retries == 5


# ── Test: Model Resolution ──────────────────────────────────────────────────


class TestModelResolution:
    def test_haiku_shorthand(self) -> None:
        client = LLMClient(api_key="test", haiku_model="claude-haiku-test")
        assert client._resolve_model("haiku") == "claude-haiku-test"

    def test_sonnet_shorthand(self) -> None:
        client = LLMClient(api_key="test", sonnet_model="claude-sonnet-test")
        assert client._resolve_model("sonnet") == "claude-sonnet-test"

    def test_full_model_id(self) -> None:
        client = LLMClient(api_key="test")
        assert client._resolve_model("claude-opus-5") == "claude-opus-5"


# ── Test: API Call Behavior ──────────────────────────────────────────────────


class TestAPICall:
    def _make_mock_response(
        self,
        text: str = "test response",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> MagicMock:
        """Create a mock Anthropic Message response."""
        text_block = MagicMock()
        text_block.text = text
        text_block.__class__ = type("TextBlock", (), {})
        # Need to patch isinstance checks
        from anthropic.types import TextBlock as RealTextBlock

        mock = MagicMock()
        mock.content = [MagicMock(text=text)]
        mock.content[0].__class__ = RealTextBlock
        mock.usage = MagicMock()
        mock.usage.input_tokens = input_tokens
        mock.usage.output_tokens = output_tokens
        mock.usage.cache_creation_input_tokens = 0
        mock.usage.cache_read_input_tokens = 0
        mock.model = "claude-sonnet-test"
        mock.stop_reason = "end_turn"
        return mock

    @pytest.mark.asyncio
    async def test_successful_call(self) -> None:
        """Successful API call tracks cost and count."""
        client = LLMClient(api_key="test")

        mock_response = self._make_mock_response()
        client._client.messages.create = AsyncMock(return_value=mock_response)

        response = await client.call(
            system="test system",
            messages=[{"role": "user", "content": "test"}],
            model="sonnet",
        )

        assert client.call_count == 1
        assert client.daily_cost > 0
        assert client.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_cost_cap_blocks_call(self) -> None:
        """Cost cap prevents further calls."""
        client = LLMClient(api_key="test", daily_cost_cap=0.001)
        client._daily_cost = 0.002  # Already exceeded

        with pytest.raises(LLMCostCapExceeded):
            await client.call(
                system="test",
                messages=[{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Client retries on API errors."""
        client = LLMClient(api_key="test", max_retries=3, timeout_sec=5)

        client._client.messages.create = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(LLMCallFailed):
            await client.call(
                system="test",
                messages=[{"role": "user", "content": "test"}],
            )

        assert client._error_count == 1
        assert client.consecutive_failures == 1
        # Should have attempted 3 times
        assert client._client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_counts_as_failure(self) -> None:
        """Timeout results in retry."""
        client = LLMClient(api_key="test", max_retries=2, timeout_sec=0.01)

        async def slow_call(**kwargs):
            await asyncio.sleep(10)

        client._client.messages.create = AsyncMock(side_effect=slow_call)

        with pytest.raises(LLMCallFailed):
            await client.call(
                system="test",
                messages=[{"role": "user", "content": "test"}],
            )

        assert client.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_consecutive_failures_reset_on_success(self) -> None:
        """Consecutive failures counter resets after success."""
        client = LLMClient(api_key="test", max_retries=1)

        # First call fails
        client._client.messages.create = AsyncMock(
            side_effect=Exception("fail")
        )
        with pytest.raises(LLMCallFailed):
            await client.call(
                system="test",
                messages=[{"role": "user", "content": "test"}],
            )
        assert client.consecutive_failures == 1

        # Second call succeeds
        mock_response = self._make_mock_response()
        client._client.messages.create = AsyncMock(return_value=mock_response)
        await client.call(
            system="test",
            messages=[{"role": "user", "content": "test"}],
        )
        assert client.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_reset_daily_cost(self) -> None:
        """Daily cost can be reset."""
        client = LLMClient(api_key="test")
        client._daily_cost = 5.0
        client._cost_alert_sent = True

        client.reset_daily_cost()
        assert client.daily_cost == 0.0
        assert client._cost_alert_sent is False


# ── Test: Stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self) -> None:
        client = LLMClient(api_key="test")
        stats = client.stats
        assert stats["call_count"] == 0
        assert stats["error_count"] == 0
        assert stats["daily_cost"] == 0.0
        assert stats["total_cost"] == 0.0

    def test_stats_after_cost(self) -> None:
        client = LLMClient(api_key="test", daily_cost_cap=10.0)
        client._daily_cost = 2.5
        client._call_count = 50
        stats = client.stats
        assert stats["daily_cost"] == 2.5
        assert stats["call_count"] == 50
        assert stats["cost_cap"] == 10.0

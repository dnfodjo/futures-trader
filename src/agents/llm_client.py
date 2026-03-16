"""Anthropic SDK wrapper with prompt caching and cost tracking.

Provides a clean interface for calling Claude models with:
- Prompt caching via cache_control for system prompts
- Model selection (Haiku for routine, Sonnet for decisions)
- Structured output via tool/function calling
- Cost tracking per call and daily totals
- Retry logic with exponential backoff
- Latency measurement
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog
from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock, ToolUseBlock

logger = structlog.get_logger()

# ── Approximate pricing per million tokens (as of 2025) ──────────────────────
# These are estimates — actual pricing may vary.
_PRICING: dict[str, dict[str, float]] = {
    "haiku": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
}


def _estimate_cost(
    model_tier: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost for a single API call."""
    pricing = _PRICING.get(model_tier, _PRICING["sonnet"])
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_creation_tokens / 1_000_000) * pricing["cache_write"]
        + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
    )
    return round(cost, 6)


def _get_model_tier(model_id: str) -> str:
    """Determine pricing tier from model ID."""
    model_lower = model_id.lower()
    if "haiku" in model_lower:
        return "haiku"
    return "sonnet"


class LLMClient:
    """Async wrapper around the Anthropic SDK with caching and cost tracking.

    Usage:
        client = LLMClient(api_key="sk-...", sonnet_model="claude-sonnet-4-6-20260314")
        response = await client.call(
            system="You are an elite trader...",
            messages=[{"role": "user", "content": "..."}],
            model="sonnet",
        )
        print(response.text)
        print(client.daily_cost)
    """

    def __init__(
        self,
        api_key: str,
        haiku_model: str = "claude-haiku-4-5-20251001",
        sonnet_model: str = "claude-sonnet-4-6-20260314",
        max_retries: int = 3,
        timeout_sec: int = 30,
        daily_cost_cap: float = 10.0,
        daily_cost_alert: float = 5.0,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._haiku_model = haiku_model
        self._sonnet_model = sonnet_model
        self._max_retries = max_retries
        self._timeout_sec = timeout_sec
        self._daily_cost_cap = daily_cost_cap
        self._daily_cost_alert = daily_cost_alert

        # Cost tracking
        self._daily_cost: float = 0.0
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._error_count: int = 0
        self._consecutive_failures: int = 0
        self._cost_alert_sent: bool = False

        # Token usage
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0

    # ── Core Call Method ─────────────────────────────────────────────────────

    async def call(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        model: str = "sonnet",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Make an API call to Claude with retry logic and cost tracking.

        Args:
            system: System prompt string or list of content blocks with cache_control.
            messages: Conversation messages in Anthropic format.
            model: "haiku" or "sonnet" — maps to configured model IDs.
            tools: Optional tool definitions for structured output.
            tool_choice: Optional tool choice directive (e.g., {"type": "tool", "name": "..."}).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with text, tool calls, usage stats, and cost.

        Raises:
            LLMCostCapExceeded: If daily cost cap would be exceeded.
            LLMCallFailed: If all retries are exhausted.
        """
        # Check cost cap
        if self._daily_cost >= self._daily_cost_cap:
            raise LLMCostCapExceeded(
                f"Daily cost cap ${self._daily_cost_cap:.2f} reached "
                f"(current: ${self._daily_cost:.4f})"
            )

        model_id = self._resolve_model(model)
        model_tier = _get_model_tier(model_id)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": model_id,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Retry loop
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                start = time.monotonic()
                response = await asyncio.wait_for(
                    self._client.messages.create(**kwargs),
                    timeout=self._timeout_sec,
                )
                latency_ms = int((time.monotonic() - start) * 1000)

                # Parse response
                result = self._parse_response(response, model_tier, latency_ms)

                # Track success
                self._call_count += 1
                self._consecutive_failures = 0
                self._daily_cost += result.cost
                self._total_cost += result.cost
                self._total_input_tokens += result.input_tokens
                self._total_output_tokens += result.output_tokens
                self._total_cache_read_tokens += result.cache_read_tokens
                self._total_cache_creation_tokens += result.cache_creation_tokens

                # Cost alert
                if (
                    self._daily_cost >= self._daily_cost_alert
                    and not self._cost_alert_sent
                ):
                    logger.warning(
                        "llm_client.cost_alert",
                        daily_cost=self._daily_cost,
                        threshold=self._daily_cost_alert,
                    )
                    self._cost_alert_sent = True

                logger.debug(
                    "llm_client.call_success",
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost=result.cost,
                )

                return result

            except asyncio.TimeoutError:
                last_error = LLMCallFailed(f"Timeout after {self._timeout_sec}s")
                logger.warning(
                    "llm_client.timeout",
                    attempt=attempt + 1,
                    model=model,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "llm_client.call_error",
                    attempt=attempt + 1,
                    model=model,
                    error=str(e),
                )

            # Exponential backoff between retries
            if attempt < self._max_retries - 1:
                wait = min(2**attempt, 8)
                await asyncio.sleep(wait)

        # All retries exhausted
        self._error_count += 1
        self._consecutive_failures += 1
        raise LLMCallFailed(
            f"All {self._max_retries} retries exhausted. Last error: {last_error}"
        )

    # ── Convenience Methods ──────────────────────────────────────────────────

    async def call_with_tool(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_name: str,
        tool_schema: dict[str, Any],
        model: str = "sonnet",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Call with a single tool and force its use.

        Shortcut for the common pattern of forcing structured output via a tool.
        """
        tools = [
            {
                "name": tool_name,
                "description": f"Provide your {tool_name} response",
                "input_schema": tool_schema,
            }
        ]
        tool_choice = {"type": "tool", "name": tool_name}

        return await self.call(
            system=system,
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ── Response Parsing ─────────────────────────────────────────────────────

    def _parse_response(
        self,
        response: Message,
        model_tier: str,
        latency_ms: int,
    ) -> LLMResponse:
        """Parse an Anthropic API response into our LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        # Token usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        cost = _estimate_cost(
            model_tier, input_tokens, output_tokens, cache_creation, cache_read
        )

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else "",
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            cost=cost,
            latency_ms=latency_ms,
            model=response.model,
            stop_reason=response.stop_reason or "",
        )

    def _resolve_model(self, model: str) -> str:
        """Map model shorthand to full model ID."""
        if model == "haiku":
            return self._haiku_model
        if model == "sonnet":
            return self._sonnet_model
        # Allow passing full model ID directly
        return model

    # ── Cost & Usage ─────────────────────────────────────────────────────────

    @property
    def daily_cost(self) -> float:
        return self._daily_cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def reset_daily_cost(self) -> None:
        """Reset daily cost tracking. Called at session start."""
        self._daily_cost = 0.0
        self._cost_alert_sent = False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "consecutive_failures": self._consecutive_failures,
            "daily_cost": round(self._daily_cost, 4),
            "total_cost": round(self._total_cost, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "cache_read_tokens": self._total_cache_read_tokens,
            "cache_creation_tokens": self._total_cache_creation_tokens,
            "cost_cap": self._daily_cost_cap,
            "cost_alert_threshold": self._daily_cost_alert,
        }


# ── Response Model ───────────────────────────────────────────────────────────


class LLMResponse:
    """Parsed response from an LLM call."""

    def __init__(
        self,
        text: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: int = 0,
        model: str = "",
        stop_reason: str = "",
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cost = cost
        self.latency_ms = latency_ms
        self.model = model
        self.stop_reason = stop_reason

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_tool_input(self, name: str) -> dict[str, Any] | None:
        """Get the input dict for a specific tool call by name."""
        for tc in self.tool_calls:
            if tc["name"] == name:
                return tc["input"]
        return None

    @property
    def first_tool_input(self) -> dict[str, Any] | None:
        """Get the input dict of the first tool call, if any."""
        if self.tool_calls:
            return self.tool_calls[0]["input"]
        return None


# ── Exceptions ───────────────────────────────────────────────────────────────


class LLMCostCapExceeded(Exception):
    """Raised when daily LLM cost cap would be exceeded."""

    pass


class LLMCallFailed(Exception):
    """Raised when all retry attempts are exhausted."""

    pass

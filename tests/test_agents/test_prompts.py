"""Tests for prompt templates."""

from __future__ import annotations

from src.agents.prompts.system_prompt import (
    SYSTEM_PROMPT,
    TRADING_DECISION_TOOL,
    build_system_blocks,
)


class TestSystemPrompt:
    def test_system_prompt_exists(self) -> None:
        assert len(SYSTEM_PROMPT) > 100
        assert "VALIDATION" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT.lower()

    def test_trading_decision_tool_schema(self) -> None:
        assert TRADING_DECISION_TOOL["name"] == "trading_decision"
        schema = TRADING_DECISION_TOOL["input_schema"]
        props = schema["properties"]
        assert "action" in props
        assert "reasoning" in props
        assert "confidence" in props
        assert "primary_timeframe" in props
        assert "confluence_factors" in props
        assert "order_flow_assessment" in props
        assert "risk_flags" in props

    def test_action_enum_values(self) -> None:
        actions = TRADING_DECISION_TOOL["input_schema"]["properties"]["action"]["enum"]
        assert "LONG" in actions
        assert "SHORT" in actions
        assert "EXIT" in actions
        assert "FLAT" in actions
        assert "STOP_TRADING" in actions
        assert len(actions) == 5


class TestBuildSystemBlocks:
    def test_basic_blocks(self) -> None:
        blocks = build_system_blocks()
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_with_game_plan(self) -> None:
        blocks = build_system_blocks(game_plan="Go long above VWAP")
        assert len(blocks) == 2
        assert "Game Plan" in blocks[1]["text"]
        assert "Go long above VWAP" in blocks[1]["text"]

    def test_with_extra_context(self) -> None:
        blocks = build_system_blocks(extra_context="Debate result: long bias")
        assert len(blocks) == 2
        assert "Additional Context" in blocks[1]["text"]

    def test_with_both(self) -> None:
        blocks = build_system_blocks(
            game_plan="Bullish day",
            extra_context="Strong delta",
        )
        assert len(blocks) == 3
        assert "cache_control" in blocks[0]  # only first block cached
        assert "cache_control" not in blocks[1]
        assert "cache_control" not in blocks[2]

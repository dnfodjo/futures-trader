"""Tests for PreMarketContext and PreMarketContextGenerator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.intelligence.pre_market_context import (
    PreMarketContext,
    PreMarketContextGenerator,
)


# ── PreMarketContext dataclass tests ─────────────────────────────────────────


class TestPreMarketContext:
    def test_default_returns_normal_risk(self):
        ctx = PreMarketContext.default()
        assert ctx.risk_level == "normal"
        assert ctx.events == []
        assert ctx.no_trade_windows == []
        assert ctx.reduce_size is False
        assert ctx.widen_stops is False
        assert ctx.min_confluence_override is None
        assert ctx.notes == ""

    def test_min_confluence_override_capped_at_5(self):
        ctx = PreMarketContext(min_confluence_override=10)
        assert ctx.min_confluence_override == 5

    def test_min_confluence_override_normal_value_unchanged(self):
        ctx = PreMarketContext(min_confluence_override=3)
        assert ctx.min_confluence_override == 3

    def test_min_confluence_override_none_stays_none(self):
        ctx = PreMarketContext(min_confluence_override=None)
        assert ctx.min_confluence_override is None

    def test_invalid_risk_level_normalized(self):
        ctx = PreMarketContext(risk_level="extreme")
        assert ctx.risk_level == "normal"

    def test_valid_risk_levels_preserved(self):
        for level in ("low", "normal", "high"):
            ctx = PreMarketContext(risk_level=level)
            assert ctx.risk_level == level

    def test_no_trade_windows_stored(self):
        windows = [("09:25", "09:45"), ("14:00", "14:30")]
        ctx = PreMarketContext(no_trade_windows=windows)
        assert ctx.no_trade_windows == windows

    def test_events_list(self):
        ctx = PreMarketContext(events=["FOMC", "CPI"])
        assert ctx.events == ["FOMC", "CPI"]


# ── PreMarketContextGenerator tests ──────────────────────────────────────────


class TestPreMarketContextGenerator:
    def test_init_defaults(self):
        gen = PreMarketContextGenerator()
        assert gen._llm is None
        assert gen._calendar_path == Path("config/economic_events.json")

    def test_init_custom_calendar_path(self):
        gen = PreMarketContextGenerator(calendar_path="/tmp/cal.json")
        assert gen._calendar_path == Path("/tmp/cal.json")

    @pytest.mark.asyncio
    async def test_generate_no_llm_returns_default(self):
        """Without LLM client, should return safe defaults."""
        gen = PreMarketContextGenerator(llm_client=None)
        ctx = await gen.generate()
        assert ctx.risk_level == "normal"
        assert ctx.events == []

    @pytest.mark.asyncio
    async def test_generate_llm_success(self, tmp_path):
        """With working LLM, should parse response into PreMarketContext."""
        # Create a calendar file
        cal_path = tmp_path / "calendar.json"
        cal_path.write_text(json.dumps({}))

        llm = AsyncMock()
        llm.call = AsyncMock()
        response_mock = MagicMock()
        response_mock.text = json.dumps({
            "events": ["FOMC"],
            "risk_level": "high",
            "no_trade_windows": [["13:45", "14:30"]],
            "reduce_size": True,
            "widen_stops": False,
            "min_confluence_override": 4,
            "notes": "FOMC day",
        })
        llm.call.return_value = response_mock

        gen = PreMarketContextGenerator(
            llm_client=llm,
            calendar_path=str(cal_path),
        )
        ctx = await gen.generate()

        assert ctx.events == ["FOMC"]
        assert ctx.risk_level == "high"
        assert ctx.no_trade_windows == [["13:45", "14:30"]]
        assert ctx.reduce_size is True
        assert ctx.min_confluence_override == 4

    @pytest.mark.asyncio
    async def test_generate_llm_failure_returns_default(self, tmp_path):
        """LLM failure should gracefully degrade to defaults."""
        cal_path = tmp_path / "calendar.json"
        cal_path.write_text(json.dumps({}))

        llm = AsyncMock()
        llm.call = AsyncMock(side_effect=Exception("LLM down"))

        gen = PreMarketContextGenerator(
            llm_client=llm,
            calendar_path=str(cal_path),
        )
        ctx = await gen.generate()

        assert ctx.risk_level == "normal"
        assert ctx.events == []

    @pytest.mark.asyncio
    async def test_generate_calendar_missing_no_crash(self, tmp_path):
        """Missing calendar file should not crash."""
        gen = PreMarketContextGenerator(
            llm_client=None,
            calendar_path=str(tmp_path / "nonexistent.json"),
        )
        ctx = await gen.generate()
        assert ctx.risk_level == "normal"

    def test_load_calendar_warns_stale(self, tmp_path):
        """Calendar older than 7 days should trigger warning."""
        cal_path = tmp_path / "calendar.json"
        cal_path.write_text(json.dumps({"_description": "test"}))

        # Make file appear 10 days old
        import os
        old_time = time.time() - (10 * 86400)
        os.utime(cal_path, (old_time, old_time))

        gen = PreMarketContextGenerator(calendar_path=str(cal_path))
        # Should not crash, just log warning
        result = gen._load_calendar()
        assert "_description" in result

    def test_build_prompt_contains_date(self):
        gen = PreMarketContextGenerator()
        prompt = gen._build_prompt({"2026-03-20": [{"time": "08:30", "event": "CPI"}]})
        assert "pre-market analyst" in prompt
        assert "JSON" in prompt

    def test_build_prompt_no_events(self):
        gen = PreMarketContextGenerator()
        prompt = gen._build_prompt({})
        assert "None scheduled" in prompt

    def test_load_calendar_validates_date_keys(self, tmp_path):
        """Calendar with bad date keys should log warning and skip them."""
        cal_path = tmp_path / "calendar.json"
        cal_path.write_text(json.dumps({
            "2026-03-20": [{"time": "08:30", "event": "CPI"}],
            "bad-key": [{"time": "10:00", "event": "Invalid"}],
            "_description": "metadata is fine",
            "20260320": [{"time": "09:00", "event": "Also invalid"}],
        }))
        gen = PreMarketContextGenerator(calendar_path=str(cal_path))
        result = gen._load_calendar()
        # Valid date key should survive
        assert "2026-03-20" in result
        # Metadata key should survive
        assert "_description" in result
        # Bad keys should be removed
        assert "bad-key" not in result
        assert "20260320" not in result

    def test_reduce_size_from_llm(self):
        """LLM can set reduce_size for quad witching / options expiry."""
        ctx = PreMarketContext(reduce_size=True)
        assert ctx.reduce_size is True
        ctx2 = PreMarketContext.default()
        assert ctx2.reduce_size is False

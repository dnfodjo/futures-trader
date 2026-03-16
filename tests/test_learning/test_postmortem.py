"""Tests for the nightly postmortem analyzer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import Regime, Side, TradeRecord
from src.learning.postmortem import (
    PostmortemAnalyzer,
    PostmortemResult,
    combine_recent_lessons,
)


def _trade(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),
        timestamp_exit=datetime(2026, 3, 14, 15, 0, 0, tzinfo=UTC),
        side=Side.LONG,
        entry_quantity=2,
        exit_quantity=2,
        entry_price=19850.0,
        exit_price=19870.0,
        stop_price=19830.0,
        pnl=80.0,
        commissions=1.72,
        hold_time_sec=1800,
        max_favorable_excursion=100.0,
        max_adverse_excursion=-20.0,
        reasoning_entry="Breakout above VWAP",
        reasoning_exit="Target reached",
        regime_at_entry=Regime.TRENDING_UP,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ── PostmortemResult ──────────────────────────────────────────────────────


class TestPostmortemResult:
    def test_to_dict(self):
        result = PostmortemResult(
            grade="B",
            what_worked=["Good entries"],
            what_didnt_work=["Late exits"],
            improvements=["Tighten stops"],
            market_observations=["Trending day"],
            key_lesson="Be patient",
            tomorrow_focus="Wait for pullbacks",
        )
        d = result.to_dict()
        assert d["grade"] == "B"
        assert len(d["what_worked"]) == 1
        assert d["key_lesson"] == "Be patient"

    def test_to_summary_text(self):
        result = PostmortemResult(
            grade="A",
            what_worked=["Great discipline"],
            what_didnt_work=[],
            improvements=["Keep it up"],
            key_lesson="Consistency is key",
            tomorrow_focus="Same approach",
        )
        text = result.to_summary_text()
        assert "Grade: A" in text
        assert "Great discipline" in text
        assert "Consistency is key" in text

    def test_defaults(self):
        result = PostmortemResult()
        assert result.grade == "C"
        assert result.what_worked == []
        assert result.key_lesson == ""


# ── Empty Trades ──────────────────────────────────────────────────────────


class TestEmptyTrades:
    @pytest.mark.asyncio
    async def test_no_trades_returns_na_grade(self):
        analyzer = PostmortemAnalyzer()
        result = await analyzer.analyze(trades=[], daily_stats={})
        assert result.grade == "N/A"
        assert "No trades" in result.key_lesson


# ── Basic Postmortem (No LLM) ────────────────────────────────────────────


class TestBasicPostmortem:
    @pytest.mark.asyncio
    async def test_basic_postmortem_profitable(self):
        analyzer = PostmortemAnalyzer(llm_client=None)
        trades = [
            _trade(pnl=100.0),
            _trade(pnl=150.0),
            _trade(pnl=-30.0),
        ]
        result = await analyzer.analyze(
            trades=trades,
            daily_stats={"net_pnl": 220.0, "winners": 2, "losers": 1},
        )
        # Net P&L > 200 should get grade A
        assert result.grade == "A"
        assert len(result.what_worked) > 0

    @pytest.mark.asyncio
    async def test_basic_postmortem_small_profit(self):
        analyzer = PostmortemAnalyzer(llm_client=None)
        trades = [_trade(pnl=50.0)]
        result = await analyzer.analyze(
            trades=trades,
            daily_stats={"net_pnl": 50.0, "winners": 1, "losers": 0},
        )
        # 0 < net_pnl <= 200 = grade B
        assert result.grade == "B"

    @pytest.mark.asyncio
    async def test_basic_postmortem_small_loss(self):
        analyzer = PostmortemAnalyzer(llm_client=None)
        trades = [_trade(pnl=-50.0)]
        result = await analyzer.analyze(
            trades=trades,
            daily_stats={"net_pnl": -50.0, "winners": 0, "losers": 1},
        )
        # -100 < net_pnl <= 0 = grade C
        assert result.grade == "C"

    @pytest.mark.asyncio
    async def test_basic_postmortem_large_loss(self):
        analyzer = PostmortemAnalyzer(llm_client=None)
        trades = [_trade(pnl=-350.0)]
        result = await analyzer.analyze(
            trades=trades,
            daily_stats={"net_pnl": -350.0, "winners": 0, "losers": 1},
        )
        # net_pnl < -300 = grade F... wait, -350 > -300? No, -350 < -300
        assert result.grade == "F"

    @pytest.mark.asyncio
    async def test_basic_postmortem_medium_loss(self):
        analyzer = PostmortemAnalyzer(llm_client=None)
        trades = [_trade(pnl=-200.0)]
        result = await analyzer.analyze(
            trades=trades,
            daily_stats={"net_pnl": -200.0, "winners": 0, "losers": 1},
        )
        # -300 < net_pnl <= -100 = grade D
        assert result.grade == "D"


# ── LLM Postmortem ───────────────────────────────────────────────────────


class TestLLMPostmortem:
    @pytest.mark.asyncio
    async def test_llm_postmortem_parses_json(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '''{
            "grade": "B",
            "what_worked": ["Good timing on entries"],
            "what_didnt_work": ["Held losers too long"],
            "improvements": ["Cut losers faster"],
            "market_observations": ["Strong trending day"],
            "key_lesson": "Respect the trend",
            "tomorrow_focus": "Look for continuation"
        }'''
        mock_client.call = AsyncMock(return_value=mock_response)

        analyzer = PostmortemAnalyzer(llm_client=mock_client)
        result = await analyzer.analyze(
            trades=[_trade()],
            daily_stats={"net_pnl": 80.0, "winners": 1, "losers": 0},
        )

        assert result.grade == "B"
        assert "Good timing on entries" in result.what_worked
        assert "Respect the trend" == result.key_lesson
        mock_client.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_basic(self):
        mock_client = MagicMock()
        mock_client.call = AsyncMock(side_effect=RuntimeError("API down"))

        analyzer = PostmortemAnalyzer(llm_client=mock_client)
        result = await analyzer.analyze(
            trades=[_trade(pnl=50.0)],
            daily_stats={"net_pnl": 50.0},
        )

        # Should fall back to basic postmortem
        assert result.grade in ("A", "B", "C", "D", "F")
        assert "LLM postmortem unavailable" in result.improvements[0]

    @pytest.mark.asyncio
    async def test_llm_bad_json_falls_back(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON at all"
        mock_client.call = AsyncMock(return_value=mock_response)

        analyzer = PostmortemAnalyzer(llm_client=mock_client)
        result = await analyzer.analyze(
            trades=[_trade()],
            daily_stats={"net_pnl": 80.0},
        )

        # Should parse gracefully
        assert result.grade == "C"  # default on parse failure


# ── Prompt Building ──────────────────────────────────────────────────────


class TestPromptBuilding:
    def test_build_prompt_includes_trade_details(self):
        analyzer = PostmortemAnalyzer()
        trades = [_trade(entry_price=19850.0, exit_price=19870.0, pnl=80.0)]
        prompt = analyzer._build_prompt(
            trades=trades,
            daily_stats={"net_pnl": 80.0, "total_trades": 1, "winners": 1, "losers": 0},
        )
        assert "19850.00" in prompt
        assert "19870.00" in prompt
        assert "$80.00" in prompt

    def test_build_prompt_includes_regime_changes(self):
        analyzer = PostmortemAnalyzer()
        changes = [
            {"time": "10:30", "from": "choppy", "to": "trending_up"},
        ]
        prompt = analyzer._build_prompt(
            trades=[_trade()],
            daily_stats={},
            regime_changes=changes,
        )
        assert "choppy" in prompt
        assert "trending_up" in prompt


# ── Test: Reasoner Lesson Extraction ─────────────────────────────────────────


class TestReasonerLessons:
    """Tests for PostmortemResult.to_reasoner_lessons() — the feedback loop."""

    def test_lessons_include_key_lesson(self) -> None:
        pm = PostmortemResult(
            grade="B",
            key_lesson="VWAP pullbacks work best in morning session.",
            tomorrow_focus="Focus on morning VWAP setups.",
        )
        lessons = pm.to_reasoner_lessons()
        assert "VWAP pullbacks" in lessons
        assert "morning" in lessons.lower()

    def test_lessons_include_improvements(self) -> None:
        pm = PostmortemResult(
            grade="C",
            improvements=[
                "Cut losers faster — held 2 trades past thesis invalidation.",
                "Reduce size in midday chop.",
                "Third improvement that should be excluded",
            ],
        )
        lessons = pm.to_reasoner_lessons()
        assert "Cut losers faster" in lessons
        assert "Reduce size in midday" in lessons
        assert "Third improvement" not in lessons  # only top 2

    def test_lessons_include_mistakes(self) -> None:
        pm = PostmortemResult(
            grade="D",
            what_didnt_work=[
                "Entered 3 trades in midday chop — all losers.",
                "Widened stop on trade 2.",
            ],
        )
        lessons = pm.to_reasoner_lessons()
        assert "midday chop" in lessons
        # Only first mistake included
        assert "Widened stop" not in lessons

    def test_empty_result_gives_empty_lessons(self) -> None:
        pm = PostmortemResult(grade="N/A")
        lessons = pm.to_reasoner_lessons()
        assert lessons == ""

    def test_no_trades_gives_empty_lessons(self) -> None:
        pm = PostmortemResult(
            grade="N/A",
            key_lesson="No trades taken today.",
        )
        lessons = pm.to_reasoner_lessons()
        # "No trades taken today." should be excluded
        assert "No trades" not in lessons


class TestCombineRecentLessons:
    """Tests for combine_recent_lessons() — multi-day aggregation."""

    def test_single_day(self) -> None:
        pms = [
            PostmortemResult(
                grade="A",
                key_lesson="Morning VWAP pullbacks were excellent.",
                tomorrow_focus="Look for same setups.",
            ),
        ]
        result = combine_recent_lessons(pms)
        assert "Yesterday" in result
        assert "Grade A" in result
        assert "VWAP pullbacks" in result

    def test_multiple_days(self) -> None:
        pms = [
            PostmortemResult(
                grade="B",
                key_lesson="Good day, need faster exits.",
                tomorrow_focus="Tighten trailing stops.",
            ),
            PostmortemResult(
                grade="D",
                key_lesson="Overtraded in midday.",
                improvements=["Skip midday entirely."],
            ),
        ]
        result = combine_recent_lessons(pms)
        assert "Yesterday" in result
        assert "2 days ago" in result
        assert "faster exits" in result
        assert "midday" in result

    def test_respects_max_days(self) -> None:
        pms = [
            PostmortemResult(grade="A", key_lesson=f"Day {i}")
            for i in range(5)
        ]
        result = combine_recent_lessons(pms, max_days=2)
        assert "Day 0" in result
        assert "Day 1" in result
        assert "Day 2" not in result

    def test_empty_list(self) -> None:
        assert combine_recent_lessons([]) == ""

    def test_skips_empty_lessons(self) -> None:
        pms = [
            PostmortemResult(grade="N/A"),  # empty lessons
        ]
        result = combine_recent_lessons(pms)
        assert result == ""  # nothing useful to include

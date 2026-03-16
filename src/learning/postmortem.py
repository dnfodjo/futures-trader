"""Nightly postmortem — LLM analysis of the day's trading performance.

At end of day, the system feeds a Sonnet call with:
- All trades taken (entry/exit, reasoning, P&L)
- Market regime changes during the day
- Missed opportunities (signals without entries)
- Daily stats (win rate, drawdown, commissions)

Output: structured analysis with:
- What worked and why
- What didn't work and why
- Specific improvements for tomorrow
- Overall session grade (A-F)
- Key observations about market behavior

The postmortem is stored in the trade journal and referenced by
the pre-market analyst for the next trading day.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.types import TradeRecord

logger = structlog.get_logger()

_POSTMORTEM_SYSTEM_PROMPT = """You are an expert trading performance analyst. Analyze the day's \
trading results and provide actionable feedback.

Your analysis should be specific and constructive. Reference actual trades, prices, and \
decisions. Grade the session A-F based on:
- A: Excellent execution, good discipline, profitable or small loss with strong process
- B: Good execution with minor issues, profitable or controlled loss
- C: Average execution, some discipline issues, mediocre results
- D: Poor execution, discipline violations, avoidable losses
- F: Very poor execution, multiple discipline violations, large avoidable losses

Focus on PROCESS over outcomes. A disciplined loss is better than a lucky win.

Respond in JSON format:
{
    "grade": "A-F",
    "what_worked": ["specific observation 1", "..."],
    "what_didnt_work": ["specific observation 1", "..."],
    "improvements": ["actionable improvement 1", "..."],
    "market_observations": ["observation about today's market behavior", "..."],
    "key_lesson": "single most important takeaway from today",
    "tomorrow_focus": "what to focus on tomorrow based on today's results"
}"""


class PostmortemAnalyzer:
    """Generates nightly postmortem analysis using LLM.

    Usage:
        analyzer = PostmortemAnalyzer(llm_client=client)
        result = await analyzer.analyze(trades, daily_stats)
    """

    def __init__(
        self,
        llm_client: Any = None,  # LLMClient, typed as Any to avoid circular import
    ) -> None:
        self._llm_client = llm_client

    async def analyze(
        self,
        trades: list[TradeRecord],
        daily_stats: dict[str, Any],
        regime_changes: list[dict[str, Any]] | None = None,
    ) -> PostmortemResult:
        """Analyze the day's trading performance.

        Args:
            trades: All trades taken today.
            daily_stats: Dict with win_rate, net_pnl, etc.
            regime_changes: Optional list of regime transitions.

        Returns:
            PostmortemResult with analysis and grade.
        """
        if not trades:
            return PostmortemResult(
                grade="N/A",
                what_worked=[],
                what_didnt_work=[],
                improvements=[],
                market_observations=[],
                key_lesson="No trades taken today.",
                tomorrow_focus="Look for cleaner setups.",
                raw_text="No trades to analyze.",
            )

        # Build the prompt
        prompt = self._build_prompt(trades, daily_stats, regime_changes)

        # Call LLM if available
        if self._llm_client is None:
            return self._generate_basic_postmortem(trades, daily_stats)

        try:
            response = await self._llm_client.call(
                system=_POSTMORTEM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                model="sonnet",
                max_tokens=1500,
                temperature=0.3,
            )
            return self._parse_response(response.text)
        except Exception as e:
            logger.error("postmortem.llm_failed", error=str(e))
            return self._generate_basic_postmortem(trades, daily_stats)

    def _build_prompt(
        self,
        trades: list[TradeRecord],
        daily_stats: dict[str, Any],
        regime_changes: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the analysis prompt from today's data."""
        lines: list[str] = []

        lines.append("# Today's Trading Summary\n")
        lines.append(f"Date: {datetime.now(tz=UTC).strftime('%Y-%m-%d')}")
        lines.append(f"Total trades: {daily_stats.get('total_trades', len(trades))}")
        lines.append(f"Winners: {daily_stats.get('winners', 0)}")
        lines.append(f"Losers: {daily_stats.get('losers', 0)}")
        lines.append(f"Win rate: {daily_stats.get('win_rate', 0.0):.1%}")
        lines.append(f"Net P&L: ${daily_stats.get('net_pnl', 0.0):.2f}")
        lines.append(f"Gross P&L: ${daily_stats.get('gross_pnl', 0.0):.2f}")
        lines.append(f"Commissions: ${daily_stats.get('commissions', 0.0):.2f}")
        lines.append(f"Max drawdown: ${daily_stats.get('max_drawdown', 0.0):.2f}")
        lines.append("")

        # Trade details
        lines.append("## Trade Details\n")
        for i, t in enumerate(trades, 1):
            lines.append(f"### Trade {i}")
            lines.append(f"- Side: {t.side.value}")
            lines.append(f"- Entry: {t.entry_price:.2f} → Exit: {t.exit_price:.2f}" if t.exit_price else f"- Entry: {t.entry_price:.2f}")
            lines.append(f"- Quantity: {t.entry_quantity}")
            lines.append(f"- P&L: ${t.pnl:.2f}" if t.pnl is not None else "- P&L: pending")
            lines.append(f"- Hold time: {t.hold_time_sec}s" if t.hold_time_sec else "- Hold time: unknown")
            lines.append(f"- Regime: {t.regime_at_entry.value}")
            lines.append(f"- Entry reasoning: {t.reasoning_entry[:200]}")
            if t.reasoning_exit:
                lines.append(f"- Exit reasoning: {t.reasoning_exit[:200]}")
            lines.append(f"- MFE: ${t.max_favorable_excursion:.2f}, MAE: ${t.max_adverse_excursion:.2f}")
            lines.append("")

        # Regime changes
        if regime_changes:
            lines.append("## Regime Changes\n")
            for change in regime_changes:
                lines.append(
                    f"- {change.get('time', '?')}: {change.get('from', '?')} → {change.get('to', '?')}"
                )
            lines.append("")

        lines.append("Analyze this trading day and provide your assessment.")
        return "\n".join(lines)

    def _parse_response(self, text: str) -> PostmortemResult:
        """Parse LLM JSON response into PostmortemResult."""
        import json

        try:
            # Try to extract JSON from the response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(text[json_start:json_end])
                return PostmortemResult(
                    grade=data.get("grade", "C"),
                    what_worked=data.get("what_worked", []),
                    what_didnt_work=data.get("what_didnt_work", []),
                    improvements=data.get("improvements", []),
                    market_observations=data.get("market_observations", []),
                    key_lesson=data.get("key_lesson", ""),
                    tomorrow_focus=data.get("tomorrow_focus", ""),
                    raw_text=text,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("postmortem.parse_failed", error=str(e))

        return PostmortemResult(
            grade="C",
            what_worked=[],
            what_didnt_work=[],
            improvements=[],
            market_observations=[],
            key_lesson="Unable to parse LLM analysis.",
            tomorrow_focus="",
            raw_text=text,
        )

    def _generate_basic_postmortem(
        self,
        trades: list[TradeRecord],
        daily_stats: dict[str, Any],
    ) -> PostmortemResult:
        """Generate a basic postmortem without LLM (fallback)."""
        winners = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losers = [t for t in trades if t.pnl is not None and t.pnl < 0]
        net_pnl = daily_stats.get("net_pnl", sum(t.pnl or 0 for t in trades))

        what_worked: list[str] = []
        what_didnt_work: list[str] = []

        if winners:
            avg_win = sum(t.pnl for t in winners if t.pnl) / len(winners)
            what_worked.append(f"{len(winners)} winning trades, avg win ${avg_win:.2f}")

        if losers:
            avg_loss = sum(t.pnl for t in losers if t.pnl) / len(losers)
            what_didnt_work.append(f"{len(losers)} losing trades, avg loss ${avg_loss:.2f}")

        # Grade based on outcome and process
        if net_pnl > 200:
            grade = "A"
        elif net_pnl > 0:
            grade = "B"
        elif net_pnl > -100:
            grade = "C"
        elif net_pnl > -300:
            grade = "D"
        else:
            grade = "F"

        return PostmortemResult(
            grade=grade,
            what_worked=what_worked,
            what_didnt_work=what_didnt_work,
            improvements=["LLM postmortem unavailable — review trades manually."],
            market_observations=[],
            key_lesson=f"Net P&L: ${net_pnl:.2f} across {len(trades)} trades.",
            tomorrow_focus="Focus on process over outcomes.",
            raw_text="Basic postmortem (no LLM).",
        )


class PostmortemResult:
    """Result of a postmortem analysis."""

    def __init__(
        self,
        grade: str = "C",
        what_worked: list[str] | None = None,
        what_didnt_work: list[str] | None = None,
        improvements: list[str] | None = None,
        market_observations: list[str] | None = None,
        key_lesson: str = "",
        tomorrow_focus: str = "",
        raw_text: str = "",
    ) -> None:
        self.grade = grade
        self.what_worked = what_worked or []
        self.what_didnt_work = what_didnt_work or []
        self.improvements = improvements or []
        self.market_observations = market_observations or []
        self.key_lesson = key_lesson
        self.tomorrow_focus = tomorrow_focus
        self.raw_text = raw_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "what_worked": self.what_worked,
            "what_didnt_work": self.what_didnt_work,
            "improvements": self.improvements,
            "market_observations": self.market_observations,
            "key_lesson": self.key_lesson,
            "tomorrow_focus": self.tomorrow_focus,
        }

    def to_summary_text(self) -> str:
        """Generate a human-readable summary for Telegram or logs."""
        lines = [
            f"📊 Session Grade: {self.grade}",
            "",
        ]

        if self.what_worked:
            lines.append("✅ What Worked:")
            for item in self.what_worked:
                lines.append(f"  • {item}")
            lines.append("")

        if self.what_didnt_work:
            lines.append("❌ What Didn't Work:")
            for item in self.what_didnt_work:
                lines.append(f"  • {item}")
            lines.append("")

        if self.improvements:
            lines.append("🔧 Improvements:")
            for item in self.improvements:
                lines.append(f"  • {item}")
            lines.append("")

        if self.key_lesson:
            lines.append(f"💡 Key Lesson: {self.key_lesson}")

        if self.tomorrow_focus:
            lines.append(f"🎯 Tomorrow Focus: {self.tomorrow_focus}")

        return "\n".join(lines)

    def to_reasoner_lessons(self) -> str:
        """Extract compact actionable lessons for injection into the Reasoner.

        This is the feedback loop: postmortem insights → next day's LLM context.
        Kept deliberately short to fit in the system prompt without inflating costs.
        """
        parts: list[str] = []

        if self.key_lesson and self.key_lesson != "No trades taken today.":
            parts.append(f"Key lesson: {self.key_lesson}")

        if self.tomorrow_focus:
            parts.append(f"Focus today: {self.tomorrow_focus}")

        # Top 2 improvements (most actionable)
        for imp in self.improvements[:2]:
            parts.append(f"Improve: {imp}")

        # Top mistake to avoid
        for mistake in self.what_didnt_work[:1]:
            parts.append(f"Avoid: {mistake}")

        return "\n".join(parts) if parts else ""


def combine_recent_lessons(
    postmortems: list[PostmortemResult],
    max_days: int = 3,
) -> str:
    """Combine lessons from the last N postmortems into a single context block.

    Used by the orchestrator to load multi-day learning context into the Reasoner.
    Keeps only the most recent and actionable insights.

    Args:
        postmortems: List of PostmortemResult from recent days (newest first).
        max_days: Maximum number of days to include.

    Returns:
        Compact multi-line string suitable for Reasoner.set_postmortem_lessons().
    """
    if not postmortems:
        return ""

    lines: list[str] = []
    for i, pm in enumerate(postmortems[:max_days]):
        day_label = ["Yesterday", "2 days ago", "3 days ago"][i] if i < 3 else f"{i + 1} days ago"
        lessons = pm.to_reasoner_lessons()
        if lessons:
            lines.append(f"[{day_label} — Grade {pm.grade}]")
            lines.append(lessons)
            lines.append("")

    return "\n".join(lines).strip()

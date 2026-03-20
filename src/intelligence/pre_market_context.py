"""Pre-Market Context -- One-shot daily LLM analysis for trading parameter adjustment.

Runs ONCE at ~9:15 AM ET before market open. Outputs a simple JSON config that
mechanically adjusts trading parameters for the day (no-trade windows, size
reduction, confluence overrides).

If the LLM fails or is unavailable, safe defaults are used.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger()

ET = ZoneInfo("America/New_York")


@dataclass
class PreMarketContext:
    """Trading parameter adjustments for today's session."""

    events: list[str] = field(default_factory=list)
    risk_level: str = "normal"  # "low" / "normal" / "high"
    no_trade_windows: list[tuple[str, str]] = field(default_factory=list)
    reduce_size: bool = False
    widen_stops: bool = False
    min_confluence_override: int | None = None
    notes: str = ""

    def __post_init__(self):
        """Validate and cap override values."""
        if self.min_confluence_override is not None:
            self.min_confluence_override = min(self.min_confluence_override, 5)
        # Validate risk_level
        if self.risk_level not in ("low", "normal", "high"):
            self.risk_level = "normal"

    @classmethod
    def default(cls) -> PreMarketContext:
        """Safe defaults when LLM is unavailable."""
        return cls()


class PreMarketContextGenerator:
    """Generates pre-market context using LLM analysis of economic calendar."""

    def __init__(self, llm_client=None, calendar_path: str | None = None):
        self._llm = llm_client
        self._calendar_path = (
            Path(calendar_path) if calendar_path else Path("config/economic_events.json")
        )

    async def generate(self) -> PreMarketContext:
        """Run at ~9:15 AM ET. Returns PreMarketContext."""
        try:
            calendar = self._load_calendar()
            prompt = self._build_prompt(calendar)

            if self._llm is None:
                logger.warning("pre_market.no_llm_client", msg="Using defaults")
                return PreMarketContext.default()

            response = await self._llm.call(
                system="You are a pre-market economic calendar analyst. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                model="sonnet",
                max_tokens=512,
                temperature=0.1,
            )

            # Parse JSON from LLM response text
            raw_text = response.text.strip()
            # Handle markdown code blocks (including trailing newlines after ```)
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-z]*\s*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```\s*$", "", raw_text)
                raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            # Filter to valid fields only — LLM may add extra keys like "reasoning"
            valid_fields = {f.name for f in fields(PreMarketContext)}
            filtered = {k: v for k, v in parsed.items() if k in valid_fields}
            ctx = PreMarketContext(**filtered)

            logger.info(
                "pre_market.context_generated",
                events=ctx.events,
                risk_level=ctx.risk_level,
                no_trade_windows=ctx.no_trade_windows,
                reduce_size=ctx.reduce_size,
                min_confluence_override=ctx.min_confluence_override,
            )
            return ctx

        except Exception:
            logger.warning("pre_market.llm_failed", exc_info=True, msg="Using defaults")
            return PreMarketContext.default()

    def _load_calendar(self) -> dict:
        """Load economic events calendar. Warn if stale."""
        if not self._calendar_path.exists():
            logger.warning("pre_market.no_calendar", path=str(self._calendar_path))
            return {}

        # Staleness check: warn if > 7 days old
        mtime = self._calendar_path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        if age_days > 7:
            logger.warning(
                "pre_market.calendar_stale",
                age_days=round(age_days, 1),
                msg=f"Economic calendar is {age_days:.0f} days old -- update recommended",
            )

        with open(self._calendar_path) as f:
            data = json.load(f)

        # Validate date key format (YYYY-MM-DD) — skip metadata keys like "_description"
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        bad_keys = [
            k for k in data
            if not k.startswith("_") and not date_pattern.match(k)
        ]
        for key in bad_keys:
            logger.warning(
                "pre_market.calendar_bad_key",
                key=key,
                msg=f"Calendar key '{key}' is not YYYY-MM-DD format -- skipping",
            )
            del data[key]

        return data

    def _build_prompt(self, calendar: dict) -> str:
        """Build the LLM prompt for pre-market analysis."""
        today = datetime.now(ET).strftime("%Y-%m-%d")
        today_events = calendar.get(today, [])

        return f"""You are a pre-market analyst for an MNQ futures day trading system.
Today is {today}. Analyze today's economic calendar and return a JSON object.

Today's events: {json.dumps(today_events) if today_events else "None scheduled"}

Rules:
- FOMC announcements: add no_trade_window 15min before through 30min after
- CPI/PPI/NFP releases: add no_trade_window 5min before through 15min after
- Quad witching / options expiry: set reduce_size=true
- If multiple high-impact events: set risk_level="high", min_confluence_override=4
- If no events: return defaults (risk_level="normal", empty windows)
- All times in ET (Eastern Time)

Return ONLY valid JSON with this schema:
{{
  "events": ["list of today's events"],
  "risk_level": "low|normal|high",
  "no_trade_windows": [["HH:MM", "HH:MM"]],
  "reduce_size": false,
  "widen_stops": false,
  "min_confluence_override": null,
  "notes": ""
}}"""

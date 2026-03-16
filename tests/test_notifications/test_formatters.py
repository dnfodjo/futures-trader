"""Tests for notification formatters — pure function tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import (
    LLMAction,
    ActionType,
    PositionState,
    Regime,
    SessionSummary,
    Side,
    TradeRecord,
)
from src.notifications.formatters import (
    format_guardrail_block,
    format_heartbeat,
    format_risk_alert,
    format_session_summary,
    format_shutdown,
    format_startup,
    format_system_alert,
    format_trade_entry,
    format_trade_exit,
)


def _action(**overrides) -> LLMAction:
    defaults = dict(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=3,
        stop_distance=10.0,
        reasoning="Strong momentum above VWAP with high delta",
        confidence=0.85,
    )
    defaults.update(overrides)
    return LLMAction(**defaults)


def _trade(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 3, 14, 10, 30, tzinfo=UTC),
        timestamp_exit=datetime(2026, 3, 14, 10, 45, tzinfo=UTC),
        side=Side.LONG,
        entry_quantity=3,
        exit_quantity=3,
        entry_price=19850.0,
        exit_price=19860.0,
        stop_price=19840.0,
        pnl=140.0,
        commissions=5.16,
        hold_time_sec=900,
        max_favorable_excursion=180.0,
        max_adverse_excursion=-30.0,
        reasoning_entry="Strong momentum",
        reasoning_exit="Target reached",
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _summary(**overrides) -> SessionSummary:
    defaults = dict(
        date="2026-03-14",
        total_trades=5,
        winners=3,
        losers=2,
        gross_pnl=310.0,
        commissions=30.0,
        net_pnl=280.0,
        max_drawdown=95.0,
        max_position_size=3,
        session_grade="B+",
        trades=[
            _trade(pnl=120.0),
            _trade(pnl=100.0),
            _trade(pnl=90.0),
            _trade(pnl=-15.0),
            _trade(pnl=-25.0),
        ],
    )
    defaults.update(overrides)
    return SessionSummary(**defaults)


# ── Trade Entry ──────────────────────────────────────────────────────────────


class TestFormatTradeEntry:
    def test_basic_long_entry(self):
        text = format_trade_entry(
            action=_action(),
            fill_price=19850.0,
            position_qty=3,
            daily_pnl=0.0,
        )
        assert "LONG 3 MNQ" in text
        assert "19,850.00" in text
        assert "Stop: 19,840.00" in text
        assert "Conf: 85%" in text

    def test_short_entry(self):
        text = format_trade_entry(
            action=_action(side=Side.SHORT),
            fill_price=19850.0,
            position_qty=3,
        )
        assert "SHORT 3 MNQ" in text
        # Stop for short = entry + distance = 19860
        assert "Stop: 19,860.00" in text

    def test_entry_with_daily_pnl(self):
        text = format_trade_entry(
            action=_action(),
            fill_price=19850.0,
            position_qty=3,
            daily_pnl=280.0,
        )
        assert "+$280.00" in text

    def test_add_shows_position_size(self):
        text = format_trade_entry(
            action=_action(quantity=1),
            fill_price=19855.0,
            position_qty=4,  # 4 total after add
        )
        assert "Position now: 4 contracts" in text

    def test_long_reasoning(self):
        long_reason = "A" * 200
        text = format_trade_entry(
            action=_action(reasoning=long_reason),
            fill_price=19850.0,
            position_qty=3,
        )
        assert "..." in text  # Truncated
        assert len(text) < 500  # Reasonable length


# ── Trade Exit ───────────────────────────────────────────────────────────────


class TestFormatTradeExit:
    def test_winning_trade(self):
        text = format_trade_exit(
            trade=_trade(pnl=140.0),
            daily_pnl=280.0,
            winners=2,
            losers=0,
        )
        assert "+$140.00" in text
        assert "2W/0L" in text
        assert "+$280.00" in text
        assert "\u2705" in text  # Green check

    def test_losing_trade(self):
        text = format_trade_exit(
            trade=_trade(pnl=-50.0),
            daily_pnl=-50.0,
            winners=0,
            losers=1,
        )
        assert "-$50.00" in text
        assert "\u274c" in text  # Red X

    def test_hold_time_shown(self):
        text = format_trade_exit(
            trade=_trade(hold_time_sec=900),
            daily_pnl=0.0,
            winners=0,
            losers=0,
        )
        assert "15.0 min" in text

    def test_mfe_mae_shown(self):
        text = format_trade_exit(
            trade=_trade(max_favorable_excursion=180.0, max_adverse_excursion=-30.0),
            daily_pnl=0.0,
            winners=0,
            losers=0,
        )
        assert "MFE: +$180.00" in text
        assert "MAE: -$30.00" in text

    def test_exit_reasoning_shown(self):
        text = format_trade_exit(
            trade=_trade(reasoning_exit="Target reached"),
            daily_pnl=0.0,
            winners=0,
            losers=0,
        )
        assert "Target reached" in text


# ── Risk Alert ───────────────────────────────────────────────────────────────


class TestFormatRiskAlert:
    def test_daily_loss_alert(self):
        text = format_risk_alert(
            alert_type="Daily loss approaching limit",
            current_value=-320.0,
            limit_value=-400.0,
        )
        assert "RISK ALERT" in text
        assert "-$320.00" in text
        assert "-$400.00" in text

    def test_with_details(self):
        text = format_risk_alert(
            alert_type="Consecutive losers",
            current_value=3.0,
            limit_value=4.0,
            details="Consider reducing size",
        )
        assert "Consider reducing size" in text


# ── System Alert ─────────────────────────────────────────────────────────────


class TestFormatSystemAlert:
    def test_critical_alert(self):
        text = format_system_alert(
            alert_type="KILL SWITCH",
            message="Flash crash detected",
            severity="critical",
        )
        assert "\U0001f6a8" in text
        assert "KILL SWITCH" in text
        assert "Flash crash" in text

    def test_warning_alert(self):
        text = format_system_alert(
            alert_type="HIGH VOLATILITY",
            message="VIX above 30",
            severity="warning",
        )
        assert "\u26a0\ufe0f" in text

    def test_info_alert(self):
        text = format_system_alert(
            alert_type="CONNECTION RESTORED",
            message="Reconnected after 15s",
            severity="info",
        )
        assert "\u2139\ufe0f" in text


# ── Guardrail Block ──────────────────────────────────────────────────────────


class TestFormatGuardrailBlock:
    def test_block_message(self):
        text = format_guardrail_block(
            reason="risk_check: confidence 0.15 below minimum 0.30",
            action_type="ENTER",
        )
        assert "BLOCKED: ENTER" in text
        assert "confidence" in text


# ── Heartbeat ────────────────────────────────────────────────────────────────


class TestFormatHeartbeat:
    def test_flat_position(self):
        text = format_heartbeat(
            daily_pnl=145.0,
            position=None,
            system_uptime_min=150.0,
            trades_today=3,
            winners=2,
            losers=1,
        )
        assert "System alive" in text
        assert "+$145.00" in text
        assert "flat" in text
        assert "2.5 hrs" in text
        assert "2W/1L" in text

    def test_in_position(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
            unrealized_pnl=60.0,
        )
        text = format_heartbeat(
            daily_pnl=200.0,
            position=pos,
            system_uptime_min=60.0,
            trades_today=1,
            winners=1,
            losers=0,
        )
        assert "LONG 3" in text
        assert "19,850.00" in text
        assert "Unrealized: +$60.00" in text


# ── Session Summary ──────────────────────────────────────────────────────────


class TestFormatSessionSummary:
    def test_green_day(self):
        text = format_session_summary(_summary())
        assert "END OF DAY" in text
        assert "2026-03-14" in text
        assert "+$280.00" in text
        assert "3W/2L" in text
        assert "60.0%" in text
        assert "B+" in text
        assert "\U0001f7e2" in text  # Green circle

    def test_red_day(self):
        text = format_session_summary(
            _summary(net_pnl=-100.0, gross_pnl=-80.0, winners=1, losers=4)
        )
        assert "\U0001f534" in text  # Red circle
        assert "-$100.00" in text

    def test_with_postmortem(self):
        text = format_session_summary(
            _summary(postmortem="Good execution on trend moves, struggled in chop")
        )
        assert "Good execution" in text

    def test_no_losers_infinite_pf(self):
        summary = _summary(
            losers=0,
            trades=[_trade(pnl=100.0), _trade(pnl=200.0)],
        )
        text = format_session_summary(summary)
        assert "\u221e" in text  # Infinity symbol


# ── Startup / Shutdown ───────────────────────────────────────────────────────


class TestStartupShutdown:
    def test_startup(self):
        text = format_startup(
            mode="demo",
            max_contracts=6,
            daily_loss_limit=400.0,
            symbol="MNQM6",
        )
        assert "System starting" in text
        assert "demo" in text
        assert "MNQM6" in text
        assert "6" in text
        assert "-$400.00" in text

    def test_shutdown(self):
        text = format_shutdown(
            reason="End of trading day",
            daily_pnl=280.0,
        )
        assert "System shutdown" in text
        assert "End of trading day" in text
        assert "+$280.00" in text


# ── HTML Escaping ────────────────────────────────────────────────────────────


class TestHTMLEscaping:
    def test_entry_reasoning_escaped(self):
        """LLM reasoning with HTML chars must be escaped."""
        text = format_trade_entry(
            action=_action(reasoning="Price > VWAP & delta < -100"),
            fill_price=19850.0,
            position_qty=3,
        )
        assert "&gt;" in text  # > escaped
        assert "&amp;" in text  # & escaped
        assert "&lt;" in text  # < escaped

    def test_exit_reasoning_escaped(self):
        text = format_trade_exit(
            trade=_trade(reasoning_exit="P&L < target & risk > reward"),
            daily_pnl=0.0,
            winners=0,
            losers=0,
        )
        assert "&amp;" in text
        assert "&lt;" in text
        assert "&gt;" in text

    def test_system_alert_message_escaped(self):
        text = format_system_alert(
            alert_type="TEST <script>",
            message="Error: x < 0 & y > 100",
            severity="warning",
        )
        assert "&lt;script&gt;" in text
        assert "&amp;" in text

    def test_guardrail_reason_escaped(self):
        text = format_guardrail_block(
            reason="stop distance < minimum & confidence > 0",
            action_type="ENTER",
        )
        assert "&lt;" in text
        assert "&amp;" in text

    def test_risk_alert_details_escaped(self):
        text = format_risk_alert(
            alert_type="Test",
            current_value=-300.0,
            limit_value=-400.0,
            details="Loss > limit & approaching <danger>",
        )
        assert "&gt;" in text
        assert "&lt;" in text

    def test_session_postmortem_escaped(self):
        text = format_session_summary(
            _summary(postmortem="Win rate < 50% & profit factor > 1.5")
        )
        assert "&lt;" in text
        assert "&amp;" in text

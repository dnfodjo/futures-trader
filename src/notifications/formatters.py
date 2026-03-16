"""Message formatters — convert trading events into Telegram-friendly text.

All formatters output HTML-formatted strings for Telegram's HTML parse mode.
Each formatter is a pure function (no side effects, easy to test).

Message types:
- Trade entry / exit
- Risk alerts
- System alerts (kill switch, connection loss)
- Heartbeat (periodic health check)
- End-of-day summary
- Guardrail blocks
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any, Optional

from src.core.types import (
    LLMAction,
    PositionState,
    SessionSummary,
    Side,
    TradeRecord,
)


def _esc(text: str) -> str:
    """Escape HTML special characters in user-generated text.

    Telegram's HTML parser will reject or silently drop messages
    containing unescaped <, >, or & characters in free-text fields.
    """
    return html.escape(text, quote=False)


def _sign(value: float) -> str:
    """Format a dollar value with +/- sign."""
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _price(value: float) -> str:
    """Format a price with comma separation."""
    return f"{value:,.2f}"


def format_trade_entry(
    action: LLMAction,
    fill_price: float,
    position_qty: int,
    daily_pnl: float = 0.0,
) -> str:
    """Format a trade entry notification.

    Example:
        LONG 3 MNQ @ 19,850.00 | Stop: 19,840.00 | Daily P&L: +$280.00
        Reasoning: Strong momentum above VWAP...
    """
    side = action.side.value.upper() if action.side else "UNKNOWN"
    qty = action.quantity or 1

    stop_text = ""
    if action.stop_distance is not None:
        if action.side == Side.LONG:
            stop_price = fill_price - action.stop_distance
        else:
            stop_price = fill_price + action.stop_distance
        stop_text = f" | Stop: {_price(stop_price)}"

    reasoning_raw = action.reasoning[:120] + "..." if len(action.reasoning) > 120 else action.reasoning
    reasoning_short = _esc(reasoning_raw)

    lines = [
        f"<b>{side} {qty} MNQ @ {_price(fill_price)}</b>{stop_text}",
        f"Daily P&L: {_sign(daily_pnl)} | Conf: {action.confidence:.0%}",
        f"<i>{reasoning_short}</i>",
    ]

    if position_qty > qty:
        lines.insert(1, f"Position now: {position_qty} contracts")

    return "\n".join(lines)


def format_trade_exit(
    trade: TradeRecord,
    daily_pnl: float,
    winners: int,
    losers: int,
) -> str:
    """Format a trade exit notification.

    Example:
        CLOSED +$140.00 | 3 MNQ @ 19,860.00 -> 19,870.00
        Daily P&L: +$280.00 | Trades: 2W/0L
        Hold: 12.5 min | MFE: +$180 | MAE: -$30
    """
    pnl = trade.pnl or 0.0
    emoji = "\u2705" if pnl >= 0 else "\u274c"

    exit_price = trade.exit_price or 0.0
    hold_min = (trade.hold_time_sec or 0) / 60.0

    lines = [
        f"{emoji} <b>CLOSED {_sign(pnl)}</b> | {trade.entry_quantity} MNQ "
        f"@ {_price(trade.entry_price)} \u2192 {_price(exit_price)}",
        f"Daily P&L: {_sign(daily_pnl)} | Trades: {winners}W/{losers}L",
        f"Hold: {hold_min:.1f} min | MFE: {_sign(trade.max_favorable_excursion)} | MAE: {_sign(trade.max_adverse_excursion)}",
    ]

    if trade.reasoning_exit:
        reason_raw = trade.reasoning_exit[:100] + "..." if len(trade.reasoning_exit) > 100 else trade.reasoning_exit
        lines.append(f"<i>{_esc(reason_raw)}</i>")

    return "\n".join(lines)


def format_risk_alert(
    alert_type: str,
    current_value: float,
    limit_value: float,
    details: str = "",
) -> str:
    """Format a risk alert notification.

    Example:
        \u26a0\ufe0f RISK ALERT: Daily loss approaching limit
        Current: -$320.00 / Limit: -$400.00
    """
    lines = [
        f"\u26a0\ufe0f <b>RISK ALERT: {alert_type}</b>",
        f"Current: {_sign(current_value)} / Limit: {_sign(limit_value)}",
    ]
    if details:
        lines.append(f"<i>{_esc(details)}</i>")
    return "\n".join(lines)


def format_system_alert(
    alert_type: str,
    message: str,
    severity: str = "warning",
) -> str:
    """Format a system alert notification.

    Example:
        \U0001f6a8 KILL SWITCH: Flash crash detected — flattened all positions
    """
    icons = {
        "critical": "\U0001f6a8",
        "warning": "\u26a0\ufe0f",
        "info": "\u2139\ufe0f",
    }
    icon = icons.get(severity, "\u26a0\ufe0f")
    return f"{icon} <b>{_esc(alert_type)}</b>\n{_esc(message)}"


def format_guardrail_block(
    reason: str,
    action_type: str,
) -> str:
    """Format a guardrail block notification.

    Example:
        \U0001f6d1 BLOCKED: ENTER — confidence 0.15 below minimum 0.30
    """
    return f"\U0001f6d1 <b>BLOCKED: {_esc(action_type)}</b>\n{_esc(reason)}"


def format_heartbeat(
    daily_pnl: float,
    position: Optional[PositionState],
    system_uptime_min: float,
    trades_today: int,
    winners: int,
    losers: int,
) -> str:
    """Format a periodic heartbeat notification.

    Example:
        \U0001f49a System alive | P&L: +$145.00 | Position: flat
        Uptime: 2.5 hrs | Trades: 3 (2W/1L)
    """
    if position is not None:
        side = position.side.value.upper()
        pos_text = f"{side} {position.quantity} @ {_price(position.avg_entry)}"
        pnl_text = f"Unrealized: {_sign(position.unrealized_pnl)}"
    else:
        pos_text = "flat"
        pnl_text = ""

    uptime_hrs = system_uptime_min / 60.0

    lines = [
        f"\U0001f49a <b>System alive</b> | P&L: {_sign(daily_pnl)} | Position: {pos_text}",
    ]
    if pnl_text:
        lines.append(pnl_text)

    lines.append(
        f"Uptime: {uptime_hrs:.1f} hrs | Trades: {trades_today} ({winners}W/{losers}L)"
    )

    return "\n".join(lines)


def format_session_summary(summary: SessionSummary) -> str:
    """Format an end-of-day session summary.

    Example:
        \U0001f4ca END OF DAY SUMMARY — 2026-03-14
        Net P&L: +$280.00 (Gross: +$310.00, Comm: -$30.00)
        Trades: 5 (3W/2L) | Win Rate: 60.0%
        Profit Factor: 2.10 | Avg Win: +$120 | Avg Loss: -$45
        Max Drawdown: -$95.00
        Grade: B+
    """
    day_emoji = "\U0001f7e2" if summary.is_green_day else "\U0001f534"

    lines = [
        f"{day_emoji} <b>END OF DAY SUMMARY \u2014 {summary.date}</b>",
        f"Net P&L: {_sign(summary.net_pnl)} (Gross: {_sign(summary.gross_pnl)}, Comm: {_sign(-summary.commissions)})",
        f"Trades: {summary.total_trades} ({summary.winners}W/{summary.losers}L) | Win Rate: {summary.win_rate:.1f}%",
    ]

    pf = summary.profit_factor
    pf_text = f"{pf:.2f}" if pf != float("inf") else "\u221e"
    lines.append(
        f"Profit Factor: {pf_text} | Avg Win: {_sign(summary.avg_winner)} | Avg Loss: {_sign(summary.avg_loser)}"
    )

    lines.append(f"Max Drawdown: {_sign(-summary.max_drawdown)}")

    if summary.session_grade:
        lines.append(f"Grade: <b>{_esc(summary.session_grade)}</b>")

    if summary.postmortem:
        pm_raw = summary.postmortem[:200] + "..." if len(summary.postmortem) > 200 else summary.postmortem
        lines.append(f"\n<i>{_esc(pm_raw)}</i>")

    return "\n".join(lines)


def format_startup(
    mode: str,
    max_contracts: int,
    daily_loss_limit: float,
    symbol: str,
) -> str:
    """Format a system startup notification."""
    return (
        f"\U0001f680 <b>System starting</b>\n"
        f"Mode: {mode} | Symbol: {symbol}\n"
        f"Max contracts: {max_contracts} | Daily loss limit: {_sign(-daily_loss_limit)}"
    )


def format_shutdown(reason: str, daily_pnl: float) -> str:
    """Format a system shutdown notification."""
    return (
        f"\U0001f6d1 <b>System shutdown</b>\n"
        f"Reason: {reason}\n"
        f"Final P&L: {_sign(daily_pnl)}"
    )

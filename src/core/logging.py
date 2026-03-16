"""Structured logging configuration using structlog.

All log output is JSON for machine parsing. Trading context (session date,
position state, daily P&L) can be bound to all log lines via contextvars.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging to go through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Suppress noisy third-party loggers
    for noisy in ["websockets", "aiohttp", "httpcore", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def bind_trading_context(
    session_date: str | None = None,
    daily_pnl: float | None = None,
    position_side: str | None = None,
    position_qty: int | None = None,
    system_state: str | None = None,
) -> None:
    """Bind trading context to all subsequent log lines in this async context.

    Call this whenever session state changes so every log line includes
    the current trading context automatically.
    """
    ctx: dict = {}
    if session_date is not None:
        ctx["session_date"] = session_date
    if daily_pnl is not None:
        ctx["daily_pnl"] = daily_pnl
    if position_side is not None:
        ctx["position"] = f"{position_side} {position_qty}"
    elif position_qty is not None and position_qty == 0:
        ctx["position"] = "flat"
    if system_state is not None:
        ctx["system_state"] = system_state

    structlog.contextvars.bind_contextvars(**ctx)


def clear_trading_context() -> None:
    """Clear all bound trading context."""
    structlog.contextvars.clear_contextvars()

"""Exception hierarchy for the trading system.

Every exception type maps to a specific failure mode with a clear
recovery path documented in the class docstring.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""


# ── Connection Errors ──────────────────────────────────────────────────────────


class ConnectionError(TradingSystemError):
    """Base for connection failures."""


class TradovateConnectionError(ConnectionError):
    """Tradovate WebSocket or REST connection failed.

    Recovery: exponential backoff reconnect. After 30s total → kill switch.
    """


class DatabentoConnectionError(ConnectionError):
    """Databento market data feed disconnected.

    Recovery: Databento SDK handles reconnect internally.
    After 30s with no ticks during RTH → kill switch.
    """


class LLMConnectionError(ConnectionError):
    """Anthropic API unreachable or timing out.

    Recovery: retry with backoff. After 3 consecutive failures → flatten + shutdown.
    """


# ── Order Errors ───────────────────────────────────────────────────────────────


class OrderError(TradingSystemError):
    """Base for order-related failures."""


class OrderRejectedError(OrderError):
    """Order was rejected by Tradovate/CME.

    Recovery: log rejection reason, alert via Telegram, do not retry automatically.
    """


class OrderModifyFailedError(OrderError):
    """modifyOrder returned 200 but didn't actually modify (Tradovate bug).

    Recovery: retry once with explicit timeInForce. If still fails, place new
    order and cancel old one. Alert via Telegram.
    """


class InsufficientMarginError(OrderError):
    """Not enough margin for the requested position.

    Recovery: log, alert, do not retry. May need to reduce position size.
    """


class BracketLegRejectedError(OrderError):
    """OSO bracket stop/target leg was rejected after entry filled.

    Recovery: immediately place stop as standalone order. This is critical —
    a filled entry without a stop is unprotected.
    """


# ── Guardrail Violations ──────────────────────────────────────────────────────


class GuardrailViolation(TradingSystemError):
    """LLM requested an action that violates hard limits.

    Recovery: action is blocked. Log the violation. Not an error per se —
    the guardrails are working as intended.
    """


class MaxPositionExceeded(GuardrailViolation):
    """Requested position would exceed max contracts."""


class DailyLossLimitHit(GuardrailViolation):
    """Daily P&L has hit the maximum loss threshold.

    Recovery: flatten all positions, shutdown for the day.
    """


class BlackoutPeriodViolation(GuardrailViolation):
    """Attempted to enter during a news blackout window."""


# ── Kill Switch Triggers ──────────────────────────────────────────────────────


class KillSwitchTriggered(TradingSystemError):
    """Emergency flatten activated. Always results in full position closure."""


class FlashCrashDetected(KillSwitchTriggered):
    """Price moved 50+ points in under 60 seconds."""


class ConnectionTimeoutError(KillSwitchTriggered):
    """Connection lost for more than 30 seconds while in a position."""


class LLMFailureThreshold(KillSwitchTriggered):
    """3 consecutive LLM API failures while in a position."""


# ── Rate Limiting ─────────────────────────────────────────────────────────────


class RateLimitExceeded(TradingSystemError):
    """API rate limit budget exhausted.

    Recovery: queue the request. Emergency calls (kill switch) bypass this.
    """

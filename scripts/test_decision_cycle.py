#!/usr/bin/env python3
"""Test a full LLM decision cycle with live market data.

Run on VPS: sudo -u trader .venv/bin/python scripts/test_decision_cycle.py
"""
import asyncio
import json
import time

from dotenv import load_dotenv
load_dotenv()

from src.core.config import load_config
from src.core.types import MarketState, KeyLevels, SessionPhase, Regime
from src.agents.llm_client import LLMClient
from src.agents.reasoner import Reasoner
from src.guardrails.guardrail_engine import GuardrailEngine
from src.core.events import EventBus
from src.data.databento_client import DatabentoClient


async def test_decision_cycle():
    config = load_config()

    # 1. Get live price from Databento
    print("=== 1. GETTING LIVE MARKET DATA ===")
    trades = []
    db = DatabentoClient(config.databento, config.trading)
    db.on_trade(lambda t: trades.append(t))
    await db.connect()

    start = time.time()
    while time.time() - start < 10 and len(trades) < 3:
        await asyncio.sleep(0.2)
    await db.close()

    if trades:
        last_price = trades[-1]["price"]
        print(f"Got {len(trades)} trades. Last price: {last_price}")
    else:
        last_price = 24500.0
        print(f"No trades (low volume). Using placeholder: {last_price}")

    # 2. Build a realistic MarketState
    print()
    print("=== 2. BUILDING MARKET STATE ===")
    state = MarketState(
        timestamp=time.time(),
        symbol="MNQM6",
        last_price=last_price,
        bid=last_price - 0.25,
        ask=last_price + 0.25,
        spread=0.50,
        session_phase=SessionPhase.OPEN_DRIVE,
        regime=Regime.CHOPPY,
        regime_confidence=0.65,
        levels=KeyLevels(
            prior_day_high=25219.0,
            prior_day_low=24270.0,
            prior_day_close=24650.0,
            overnight_high=24920.5,
            overnight_low=24270.0,
            session_high=last_price + 15,
            session_low=last_price - 30,
            vwap=last_price - 5,
            poc=last_price - 2,
        ),
        cumulative_delta=150.0,
        delta_5min=-20.0,
        tape_speed=2.5,
        large_lot_imbalance=0.1,
        rvol=0.0,
        daily_pnl=0.0,
        position=None,
        game_plan="Testing pre-market game plan. Focus on key levels around PDH 25219 and PDL 24270.",
    )

    state_dict = state.to_llm_dict()
    print(f"MarketState JSON: {len(json.dumps(state_dict))} chars")
    print(f"Keys: {list(state_dict.keys())}")

    # 3. Run LLM decision
    print()
    print("=== 3. LLM DECISION (Reasoner.decide) ===")
    llm = LLMClient(
        api_key=config.anthropic.api_key,
        haiku_model=config.anthropic.haiku_model,
        sonnet_model=config.anthropic.sonnet_model,
        max_retries=2,
        timeout_sec=30,
    )
    reasoner = Reasoner(llm_client=llm)

    t0 = time.time()
    action = await reasoner.decide(
        state=state,
        game_plan="Testing. Focus on key levels.",
    )
    elapsed = int((time.time() - t0) * 1000)

    print(f"Action: {action.action.value}")
    print(f"Side: {action.side}")
    print(f"Quantity: {action.quantity}")
    print(f"Stop distance: {action.stop_distance}")
    print(f"Confidence: {action.confidence}")
    print(f"Reasoning: {action.reasoning}")
    print(f"Model: {action.model_used}")
    print(f"Latency: {elapsed}ms")

    # 4. Run guardrails
    print()
    print("=== 4. GUARDRAIL VALIDATION ===")
    bus = EventBus()
    guardrails = GuardrailEngine(event_bus=bus)
    result = guardrails.check(
        action=action,
        state=state,
        position=None,
        daily_pnl=0.0,
    )
    print(f"Allowed: {result.allowed}")
    print(f"Reason: {result.reason}")
    if result.modified_quantity is not None:
        print(f"Modified quantity: {result.modified_quantity}")

    # 5. LLM cost tracking
    print()
    print("=== 5. COST TRACKING ===")
    cost = llm.total_cost
    print(f"Total cost so far: {cost:.4f} USD")
    print(f"Call count: {llm.call_count}")

    print()
    print("=== FULL DECISION CYCLE TEST PASSED ===")


if __name__ == "__main__":
    asyncio.run(test_decision_cycle())

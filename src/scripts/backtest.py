"""Backtest the trading system on historical Databento data.

Usage:
    python -m src.scripts.backtest --start 2026-03-10 --end 2026-03-14
    python -m src.scripts.backtest --start 2026-03-10 --end 2026-03-14 --output results/

Downloads historical data from Databento, feeds it through the full
pipeline (TickProcessor -> StateEngine -> Reasoner), and scores
the LLM's decisions. Uses temperature=0 for reproducibility.

This is NOT a simulation of actual trades — it replays the data through
the LLM reasoning engine and records what it WOULD have done, then
scores those decisions against what actually happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


async def run_backtest(
    api_key: str,
    anthropic_key: str,
    start: str,
    end: str,
    output_dir: str = "results",
    symbol: str = "MNQ.FUT",
) -> dict[str, Any]:
    """Run a backtest over a date range.

    1. Fetch historical trades from Databento
    2. Feed through TickProcessor to build bars/delta/flow
    3. Build MarketState snapshots at regular intervals
    4. Send each snapshot to the Reasoner (temperature=0)
    5. Score the decisions using DecisionScorer
    """
    from src.agents.llm_client import LLMClient
    from src.agents.reasoner import Reasoner
    from src.data.databento_client import DatabentoClient
    from src.data.tick_processor import TickProcessor
    from src.data.state_engine import StateEngine
    from src.data.economic_calendar import EconomicCalendar
    from src.data.multi_instrument import MultiInstrumentPoller
    from src.core.events import EventBus
    from src.replay.decision_scorer import DecisionScorer

    print(f"Backtest: {symbol} from {start} to {end}")

    # 1. Fetch historical data
    print("Fetching historical trades from Databento...")
    trades = await DatabentoClient.fetch_historical(
        api_key=api_key,
        symbol=symbol,
        start=start,
        end=end,
        schema="trades",
    )
    print(f"  Loaded {len(trades):,} trades")

    if not trades:
        print("No trades found for the specified date range.")
        return {"error": "no_data"}

    # 2. Build components
    event_bus = EventBus()
    tick_processor = TickProcessor()
    multi_instrument = MultiInstrumentPoller()
    calendar = EconomicCalendar()

    state_engine = StateEngine(
        tick_processor=tick_processor,
        multi_instrument=multi_instrument,
        calendar=calendar,
        event_bus=event_bus,
    )

    llm_client = LLMClient(
        api_key=anthropic_key,
        daily_cost_cap=5.0,
    )
    reasoner = Reasoner(llm_client=llm_client)

    # 3. Process trades and collect states
    print("Processing trades through pipeline...")
    states: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    snapshot_interval = 30  # seconds between snapshots
    last_snapshot_time: datetime | None = None

    for i, trade in enumerate(trades):
        # Feed trade to tick processor
        await tick_processor.process_trade(trade)

        ts = trade["timestamp"]

        # Take a snapshot every N seconds
        if last_snapshot_time is None or (ts - last_snapshot_time).total_seconds() >= snapshot_interval:
            last_snapshot_time = ts

            # Compute state
            state = await state_engine.compute_state()
            if state is None:
                continue

            state_dict = state.to_llm_dict()
            states.append({
                "timestamp": ts.isoformat(),
                "state": state_dict,
                "price": trade["price"],
            })

            # 4. Get LLM decision (every 5th snapshot to save cost)
            if len(states) % 5 == 0 and len(states) > 0:
                try:
                    action = await reasoner.decide(state)
                    decisions.append({
                        "timestamp": ts.isoformat(),
                        "action": action.action.value,
                        "side": action.side.value if action.side else None,
                        "confidence": action.confidence,
                        "reasoning": action.reasoning,
                        "price": trade["price"],
                    })
                    print(f"  [{ts.strftime('%H:%M:%S')}] {action.action.value} "
                          f"{'@ ' + str(trade['price']) if action.action.value != 'DO_NOTHING' else ''}"
                          f" (conf={action.confidence:.0%})")
                except Exception as e:
                    print(f"  [{ts.strftime('%H:%M:%S')}] LLM error: {e}")

        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(trades):,} trades...")

    # 5. Score decisions
    print(f"\nBacktest complete. {len(decisions)} LLM decisions made.")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_trades": len(trades),
        "snapshots": len(states),
        "decisions": decisions,
        "llm_cost": llm_client.daily_cost,
    }

    output_file = Path(output_dir) / f"backtest_{start}_{end}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_file}")
    print(f"LLM cost: ${llm_client.daily_cost:.2f}")

    # Print decision summary
    from collections import Counter
    action_counts = Counter(d["action"] for d in decisions)
    print(f"\nDecision distribution:")
    for action, count in action_counts.most_common():
        print(f"  {action}: {count} ({count/len(decisions)*100:.0f}%)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--symbol", default="MNQ.FUT", help="Symbol")
    args = parser.parse_args()

    api_key = os.environ.get("DB_API_KEY", "") or os.environ.get("DATABENTO_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("Error: DB_API_KEY not set")
        return
    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return

    asyncio.run(run_backtest(
        api_key=api_key,
        anthropic_key=anthropic_key,
        start=args.start,
        end=args.end,
        output_dir=args.output,
        symbol=args.symbol,
    ))


if __name__ == "__main__":
    main()

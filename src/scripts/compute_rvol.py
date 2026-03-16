"""Compute RVOL baseline from Databento historical data.

Usage:
    python -m src.scripts.compute_rvol
    python -m src.scripts.compute_rvol --days 20 --output data/rvol_baseline.json

Downloads the last N trading days of MNQ trade data from Databento
and computes average volume per 5-minute bucket. Saves the result
as a JSON file that StateEngine.load_rvol_baseline() can consume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


async def compute(
    api_key: str,
    lookback_days: int = 20,
    output_path: str = "data/rvol_baseline.json",
    symbol: str = "MNQ.FUT",
) -> None:
    """Compute and save RVOL baseline."""
    from src.data.databento_client import DatabentoClient

    print(f"Computing RVOL baseline for {symbol}...")
    print(f"  Lookback: {lookback_days} trading days")
    print(f"  Output: {output_path}")

    baseline = await DatabentoClient.compute_rvol_baseline(
        api_key=api_key,
        symbol=symbol,
        lookback_days=lookback_days,
    )

    # Convert to int for the RVOLBaseline schema
    int_baseline = {k: int(v) for k, v in baseline.items()}

    output = {"volume_by_time": int_baseline}

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved RVOL baseline with {len(int_baseline)} time buckets to {output_path}")

    # Print a sample
    print("\nSample buckets:")
    for i, (time_str, vol) in enumerate(sorted(int_baseline.items())):
        if i < 5 or i >= len(int_baseline) - 3:
            print(f"  {time_str}: {vol:,} avg contracts")
        elif i == 5:
            print("  ...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RVOL baseline")
    parser.add_argument("--days", type=int, default=20, help="Lookback days")
    parser.add_argument("--output", default="data/rvol_baseline.json", help="Output path")
    parser.add_argument("--symbol", default="MNQ.FUT", help="Symbol")
    args = parser.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("Error: DATABENTO_API_KEY environment variable not set")
        print("Set it with: export DATABENTO_API_KEY=your_key_here")
        return

    asyncio.run(compute(
        api_key=api_key,
        lookback_days=args.days,
        output_path=args.output,
        symbol=args.symbol,
    ))


if __name__ == "__main__":
    main()

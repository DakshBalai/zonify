"""
scripts/analyze_ticker.py
Command-line entry point for running the full pipeline -- structure,
POI, multi-timeframe alignment, and backtester -- against REAL NSE data
for one ticker.

The data_loader.py SANDBOX NOTE describes a network restriction that
was specific to this project's original dev sandbox; this machine has
been confirmed to reach Yahoo Finance directly, so this script is safe
to run as-is.

Usage:
    python scripts/analyze_ticker.py RELIANCE
    python scripts/analyze_ticker.py TCS --timeframes daily 4h 15min
    python scripts/analyze_ticker.py INFY --reward-r 1.5 --max-bars 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_timeframe import TIMEFRAME_ORDER, analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from backtester import format_stats, run_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SMC screener pipeline against real NSE data.")
    parser.add_argument("ticker", help="NSE ticker, e.g. RELIANCE or RELIANCE.NS")
    parser.add_argument("--timeframes", nargs="+", default=["weekly", "daily", "4h"], choices=TIMEFRAME_ORDER)
    parser.add_argument("--reward-r", type=float, default=2.0, help="Backtest target as a multiple of structural risk")
    parser.add_argument("--max-bars", type=int, default=20, help="Bars to wait before scoring a trade a timeout")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} -> {args.timeframes} ...")
    data = fetch_multi_timeframe_data(args.ticker, timeframes=args.timeframes)
    result = analyze_multi_timeframe(data)

    print()
    print(result["summary"])
    print()

    for tf in args.timeframes:
        df = data[tf]
        per_tf = result["per_timeframe"][tf]
        bt = run_backtest(df, per_tf["structure"], per_tf["poi"], reward_r=args.reward_r, max_bars=args.max_bars)
        print(f"--- {tf.upper()} ({len(df)} candles) ---")
        print(format_stats(bt["stats"]))
        print()


if __name__ == "__main__":
    main()

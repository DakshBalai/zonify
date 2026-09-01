"""
scripts/basket_backtest.py
Runs the full pipeline (structure + POI + backtester) across a basket
of NSE tickers and pools every trade together per timeframe, so signal
accuracy can be judged with real statistical footing instead of the
one or two tickers' worth of noise a single run gives you.

Also reports, per source_type, how many tickers individually had
positive expectancy -- a pooled number can look great while actually
being carried by one or two outlier tickers; this makes that visible
instead of hiding it.

Usage:
    python scripts/basket_backtest.py
    python scripts/basket_backtest.py --tickers RELIANCE TCS INFY
    python scripts/basket_backtest.py --timeframes daily 4h --reward-r 2.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_timeframe import TIMEFRAME_ORDER, analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from backtester import format_stats, run_backtest, summarize_trades  # noqa: E402

# 20 liquid, large-cap NSE names spanning banking, IT, energy, FMCG,
# auto, pharma, and materials -- a reasonable cross-sector basket, not
# a cherry-picked one.
DEFAULT_BASKET = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "HINDUNILVR",
    "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "BHARTIARTL",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC",
]


def collect_trades_for_ticker(ticker: str, timeframes: list, reward_r: float, max_bars: int):
    """Returns {timeframe: [Trade, ...]} for one ticker, or None on fetch failure."""
    try:
        data = fetch_multi_timeframe_data(ticker, timeframes=timeframes)
    except Exception as exc:
        print(f"  [skip] {ticker}: {exc}")
        return None

    result = analyze_multi_timeframe(data)
    trades_by_tf = {}
    for tf in timeframes:
        df = data[tf]
        per_tf = result["per_timeframe"][tf]
        bt = run_backtest(df, per_tf["structure"], per_tf["poi"], reward_r=reward_r, max_bars=max_bars)
        trades_by_tf[tf] = bt["trades"]
    return trades_by_tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the SMC pipeline across a basket of NSE tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_BASKET)
    parser.add_argument("--timeframes", nargs="+", default=["daily", "4h"], choices=TIMEFRAME_ORDER)
    parser.add_argument("--reward-r", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between tickers")
    args = parser.parse_args()

    pooled_trades = {tf: [] for tf in args.timeframes}
    per_ticker_stats = {tf: {} for tf in args.timeframes}
    n_ok, n_failed = 0, 0

    for i, ticker in enumerate(args.tickers):
        print(f"[{i + 1}/{len(args.tickers)}] {ticker} ...")
        trades_by_tf = collect_trades_for_ticker(ticker, args.timeframes, args.reward_r, args.max_bars)
        if trades_by_tf is None:
            n_failed += 1
            continue
        n_ok += 1
        for tf, trades in trades_by_tf.items():
            pooled_trades[tf] += trades
            per_ticker_stats[tf][ticker] = summarize_trades(trades)
        if i < len(args.tickers) - 1:
            time.sleep(args.delay)

    print(f"\n{n_ok} tickers fetched OK, {n_failed} skipped\n")

    for tf in args.timeframes:
        print(f"=== {tf.upper()} -- pooled across {n_ok} tickers ===")
        pooled_stats = summarize_trades(pooled_trades[tf])
        print(format_stats(pooled_stats))
        print()

        print(f"--- {tf.upper()} -- consistency across tickers (positive-expectancy count) ---")
        source_types = sorted(pooled_stats.keys())
        for source_type in source_types:
            per_ticker = [
                stats[source_type] for stats in per_ticker_stats[tf].values() if source_type in stats
            ]
            n_seen = len(per_ticker)
            n_positive = sum(1 for s in per_ticker if s.expectancy_r > 0)
            print(f"{source_type:24s} positive in {n_positive:2d}/{n_seen:2d} tickers")
        print()


if __name__ == "__main__":
    main()

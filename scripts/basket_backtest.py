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
from backtester import format_stats, run_extended_backtest, summarize_trades  # noqa: E402
from session_model import DailyRangeFormation, classify_daily_range_formation  # noqa: E402

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
        bt = run_extended_backtest(df, per_tf["structure"], per_tf["poi"], reward_r=reward_r, max_bars=max_bars)
        trades_by_tf[tf] = bt["trades"]
    return trades_by_tf


def collect_daily_range_formation_stats(ticker: str, timeframe: str) -> dict | None:
    """
    Descriptive check ONLY, not a backtest: classifies each day (on
    intraday data) as OLHC/OHLC/undetermined, and separately checks
    whether that day closed above or below its own open. This is NOT a
    same-day forward signal (see session_model.py's module docstring)
    -- it's here purely to check the concept is descriptively coherent,
    i.e. that an OLHC ("buy day") read actually correlates with the day
    closing bullish more often than an OHLC ("sell day") read does.
    """
    try:
        data = fetch_multi_timeframe_data(ticker, timeframes=[timeframe])
    except Exception:
        return None

    df = data[timeframe]
    formations = classify_daily_range_formation(df)
    if not formations:
        return None

    days = df.index.normalize()
    counts = {DailyRangeFormation.OLHC: [0, 0], DailyRangeFormation.OHLC: [0, 0]}  # [bullish_days, total_days]
    for day, formation in formations.items():
        if formation not in counts:
            continue
        day_rows = df[days == day]
        if day_rows.empty:
            continue
        bullish_day = day_rows["close"].iloc[-1] > day_rows["open"].iloc[0]
        counts[formation][1] += 1
        if bullish_day:
            counts[formation][0] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the SMC pipeline across a basket of NSE tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_BASKET)
    parser.add_argument("--timeframes", nargs="+", default=["daily", "4h"], choices=TIMEFRAME_ORDER)
    parser.add_argument("--reward-r", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between tickers")
    parser.add_argument(
        "--range-formation-timeframe", default="4h", choices=[*TIMEFRAME_ORDER, ""],
        help="Intraday timeframe to run the OLHC/OHLC descriptive check on; pass '' to skip it",
    )
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

    if args.range_formation_timeframe:
        tf = args.range_formation_timeframe
        print(f"=== Daily range formation (OLHC vs OHLC) -- descriptive check on {tf.upper()} ===")
        print("(NOT a same-day forward signal -- see session_model.py docstring. Checking only")
        print(" whether the read is descriptively coherent with how that day actually closed.)\n")
        totals = {DailyRangeFormation.OLHC: [0, 0], DailyRangeFormation.OHLC: [0, 0]}
        for ticker in args.tickers:
            counts = collect_daily_range_formation_stats(ticker, tf)
            if counts is None:
                continue
            for formation in totals:
                totals[formation][0] += counts[formation][0]
                totals[formation][1] += counts[formation][1]

        for formation in (DailyRangeFormation.OLHC, DailyRangeFormation.OHLC):
            bullish, total = totals[formation]
            rate = bullish / total if total else 0.0
            print(f"{formation.value.upper():6s} days: {bullish:4d}/{total:4d} closed bullish ({rate:.1%})")
        print()


if __name__ == "__main__":
    main()

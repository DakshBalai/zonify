"""
scripts/top_down_backtest.py
Backtests the top-down HTF-zone -> LTF MSS+FVG entry workflow
(src/top_down.py) against a basket of NSE tickers, comparing it
directly against the plain "FVG" baseline on the SAME lower timeframe
-- the question this answers is whether requiring an LTF Market
Structure Shift (a CHoCH) + FVG to occur INSIDE an HTF zone actually
improves on just taking every valid FVG on that timeframe.

Runs one representative LTF per HTF level (1h for daily, 15min for 4h,
5min for 1h) rather than every pair in top_down.HTF_TO_LTF, to keep the
number of yfinance calls reasonable -- the mechanism is identical for
the other LTF choice per HTF (30min, 5min-under-4h, 1min); extending to
those is a one-line change to PAIRS below. 1min in particular is capped
at ~7 days of history by yfinance itself, so it would contribute little
to a pooled backtest regardless.

Usage:
    python scripts/top_down_backtest.py
    python scripts/top_down_backtest.py --tickers RELIANCE TCS
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_timeframe import analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from top_down import collect_htf_zones, find_top_down_entries  # noqa: E402
from backtester import backtest_pois, format_stats, summarize_trades  # noqa: E402

from basket_backtest import DEFAULT_BASKET  # noqa: E402 -- reuse the same 20-ticker basket

PAIRS = [("daily", "1h"), ("4h", "15min"), ("1h", "5min")]


def process_ticker(ticker: str, pairs: list, reward_r: float, max_bars: int):
    """Returns {(htf, ltf): {"baseline": [...], "topdown": [...]}} or None on fetch failure."""
    needed_tfs = sorted({tf for pair in pairs for tf in pair})
    try:
        data = fetch_multi_timeframe_data(ticker, timeframes=needed_tfs)
    except Exception as exc:
        print(f"  [skip] {ticker}: {exc}")
        return None

    result = analyze_multi_timeframe(data)

    out = {}
    for htf, ltf in pairs:
        htf_df, htf_res = data[htf], result["per_timeframe"][htf]
        ltf_df, ltf_res = data[ltf], result["per_timeframe"][ltf]

        htf_zones = collect_htf_zones(htf_df, htf_res["structure"], htf_res["poi"], htf)

        valid_ltf_fvgs = [f for f in ltf_res["poi"]["fvgs"] if f.valid]
        baseline_trades = backtest_pois(ltf_df, valid_ltf_fvgs, "FVG (baseline)", reward_r=reward_r, max_bars=max_bars)

        topdown_fvgs = find_top_down_entries(ltf_df, ltf_res["structure"], valid_ltf_fvgs, htf_zones)
        topdown_trades = backtest_pois(ltf_df, topdown_fvgs, "TopDown MSS+FVG", reward_r=reward_r, max_bars=max_bars)

        out[(htf, ltf)] = {"baseline": baseline_trades, "topdown": topdown_trades}

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the top-down HTF-zone -> LTF MSS+FVG workflow.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_BASKET)
    parser.add_argument("--reward-r", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    pooled = {pair: {"baseline": [], "topdown": []} for pair in PAIRS}
    n_ok, n_failed = 0, 0

    for i, ticker in enumerate(args.tickers):
        print(f"[{i + 1}/{len(args.tickers)}] {ticker} ...")
        out = process_ticker(ticker, PAIRS, args.reward_r, args.max_bars)
        if out is None:
            n_failed += 1
            continue
        n_ok += 1
        for pair, trades in out.items():
            pooled[pair]["baseline"] += trades["baseline"]
            pooled[pair]["topdown"] += trades["topdown"]
        if i < len(args.tickers) - 1:
            time.sleep(args.delay)

    print(f"\n{n_ok} tickers fetched OK, {n_failed} skipped\n")

    for (htf, ltf), trades in pooled.items():
        print(f"=== {htf.upper()} zones -> {ltf.upper()} entries (pooled across {n_ok} tickers) ===")
        stats = summarize_trades(trades["baseline"] + trades["topdown"])
        print(format_stats(stats))
        print()


if __name__ == "__main__":
    main()

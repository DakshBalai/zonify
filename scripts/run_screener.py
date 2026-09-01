"""
scripts/run_screener.py
Runs the actual screener (src/screener.py) across a basket of NSE
tickers and prints the ranked STRONG/SETUP results -- a short, ranked
list of "these look good today, here's why," using only what
backtesting has actually validated: HTF bias alignment, a fresh Order
Block/Extreme OB, and the 4h->15min top-down MSS+FVG confluence for
the top tier.

Usage:
    python scripts/run_screener.py
    python scripts/run_screener.py --tickers RELIANCE TCS INFY
    python scripts/run_screener.py --with-fundamentals
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screener import screen_ticker  # noqa: E402
from basket_backtest import DEFAULT_BASKET  # noqa: E402 -- reuse the same 20-ticker basket


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SMC screener across a basket of NSE tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_BASKET)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--with-fundamentals", action="store_true",
        help="Also show %% from all-time high for each result (one extra fetch per matching ticker -- slower)",
    )
    args = parser.parse_args()

    results = []
    for i, ticker in enumerate(args.tickers):
        print(f"[{i + 1}/{len(args.tickers)}] {ticker} ...", file=sys.stderr)
        try:
            result = screen_ticker(ticker)
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}", file=sys.stderr)
            result = None
        if result is not None:
            results.append(result)
        if i < len(args.tickers) - 1:
            time.sleep(args.delay)

    results.sort(key=lambda r: (r.tier != "STRONG", r.ticker))

    print(f"\n{len(results)} tickers qualified (STRONG or SETUP) out of {len(args.tickers)} scanned\n")
    for r in results:
        print(
            f"[{r.tier:6s}] {r.ticker:12s} {r.direction.upper():8s} "
            f"{r.daily_zone.poi_type} zone {r.daily_zone.zone_low:.2f}-{r.daily_zone.zone_high:.2f}  "
            f"price={r.current_price:.2f}  live_LTF_entry={r.live_ltf_entry}"
        )
        if args.with_fundamentals:
            from fundamentals import fetch_ticker_profile
            try:
                profile = fetch_ticker_profile(r.ticker)
                if profile.all_time_high:
                    pct_from_ath = (r.current_price - profile.all_time_high) / profile.all_time_high * 100
                    print(f"           {pct_from_ath:+.1f}% from all-time high ({profile.all_time_high:.2f})")
            except Exception:
                pass


if __name__ == "__main__":
    main()

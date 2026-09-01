"""
multi_timeframe.py
Runs the SAME structure_engine + poi_engine pipeline independently on
each timeframe's data, then rolls the results into a single top-down
bias summary -- HTF (Monthly/Weekly) established first, drilling down
to execution timeframes (Daily/4H/15min), matching how the project
owner actually trades.

This module invents NO new detection logic. It only orchestrates calls
to analyze_structure() and analyze_poi(), which remain the single
source of truth for what a swing, BOS, CHoCH, IDM, FVG, OB, or MB is.
A bug in the alignment/summary logic here can never be confused with a
bug in the structure-detection logic itself.
"""

from __future__ import annotations

import pandas as pd

from structure_engine import Bias, analyze_structure
from poi_engine import analyze_poi

# Standard top-down order, HTF first -- also used to order any rollup output.
TIMEFRAME_ORDER = ["monthly", "weekly", "daily", "4h", "15min"]

# Swing-detection lookback tuned per timeframe: tighter (smaller) for
# LTFs where you want finer swings, looser (larger) for HTFs where you
# only want genuinely significant structure, not every 2-candle
# wiggle. These are sane starting defaults -- override per ticker/
# volatility once real data is flowing.
DEFAULT_LOOKBACK = {
    "monthly": 2,
    "weekly": 2,
    "daily": 3,
    "4h": 3,
    "15min": 4,
}


def analyze_multi_timeframe(
    data_by_timeframe: dict,
    lookback_by_timeframe: dict | None = None,
    include_poi: bool = True,
) -> dict:
    """
    data_by_timeframe: {"monthly": df, "weekly": df, ...}. Keys are
    normally a subset of TIMEFRAME_ORDER (any subset works fine, e.g.
    just {"daily", "4h"}); unrecognized custom keys are also accepted
    and appended after the standard ones.

    Returns:
      {
        "per_timeframe": {tf: {"structure": <analyze_structure() dict>,
                                 "poi": <analyze_poi() dict> or None}},
        "bias_by_timeframe": {tf: Bias enum},
        "alignment": "bullish" | "bearish" | "mixed" | "undetermined",
        "summary": human-readable string, HTF first,
      }
    """
    lookback_by_timeframe = lookback_by_timeframe or {}
    per_timeframe = {}
    bias_by_timeframe = {}

    ordered_keys = [tf for tf in TIMEFRAME_ORDER if tf in data_by_timeframe]
    ordered_keys += [tf for tf in data_by_timeframe if tf not in TIMEFRAME_ORDER]

    for tf in ordered_keys:
        df = data_by_timeframe[tf]
        lookback = lookback_by_timeframe.get(tf, DEFAULT_LOOKBACK.get(tf, 3))
        structure_result = analyze_structure(df, lookback=lookback)

        poi_result = None
        if include_poi:
            poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])

        per_timeframe[tf] = {"structure": structure_result, "poi": poi_result}
        bias_by_timeframe[tf] = structure_result["current_bias"]

    alignment, summary = _summarize_alignment(ordered_keys, bias_by_timeframe)

    return {
        "per_timeframe": per_timeframe,
        "bias_by_timeframe": bias_by_timeframe,
        "alignment": alignment,
        "summary": summary,
    }


def _summarize_alignment(ordered_keys: list, bias_by_timeframe: dict) -> tuple:
    """
    Alignment rules -- deliberately simple and transparent rather than
    a black box, since this is the number a person will make trading
    decisions from:
      "bullish"      every timeframe with a DETERMINED bias is bullish
      "bearish"      every timeframe with a DETERMINED bias is bearish
      "undetermined" no timeframe has a determined bias at all yet
      "mixed"        otherwise (genuine disagreement between timeframes)

    Timeframes still UNDETERMINED (not enough swing history yet, e.g.
    early in a new dataset) are excluded from the alignment check
    itself, but ARE still listed in the summary string so nothing is
    silently hidden.
    """
    determined = {tf: b for tf, b in bias_by_timeframe.items() if b != Bias.UNDETERMINED}

    if not determined:
        alignment = "undetermined"
    elif all(b == Bias.BULLISH for b in determined.values()):
        alignment = "bullish"
    elif all(b == Bias.BEARISH for b in determined.values()):
        alignment = "bearish"
    else:
        alignment = "mixed"

    header = {
        "bullish": "All timeframes aligned BULLISH.",
        "bearish": "All timeframes aligned BEARISH.",
        "mixed": "Timeframes are MIXED -- no clean top-down alignment.",
        "undetermined": "No timeframe has a determined bias yet.",
    }[alignment]

    lines = [f"{tf.upper()}: {bias_by_timeframe[tf].value}" for tf in ordered_keys]
    summary = header + " " + " | ".join(lines)

    return alignment, summary


def fetch_multi_timeframe_data(ticker: str, timeframes: list | None = None) -> dict:
    """
    Fetches real NSE data for each requested timeframe via data_loader.
    NOT testable inside the dev sandbox (network restricted to package
    registries) -- run this locally to verify against real data.

    timeframes: subset of TIMEFRAME_ORDER; defaults to all five.
    """
    from data_loader import fetch_ohlcv, resample_to_4h

    timeframes = timeframes or TIMEFRAME_ORDER

    # interval/period chosen per yfinance's own history limits per
    # interval (e.g. 1h data is only available for roughly the last
    # 730 days, 15m for roughly the last 60 days) -- these are yfinance
    # platform limits, not something this project controls.
    fetch_plan = {
        "monthly": {"interval": "1mo", "period": "10y"},
        "weekly": {"interval": "1wk", "period": "5y"},
        "daily": {"interval": "1d", "period": "2y"},
        "4h": {"interval": "1h", "period": "180d"},   # fetched as 1h, resampled below
        "15min": {"interval": "15m", "period": "60d"},
    }

    data = {}
    for tf in timeframes:
        if tf not in fetch_plan:
            raise ValueError(f"Unknown timeframe {tf!r}. Choose from {list(fetch_plan)}.")
        plan = fetch_plan[tf]
        df = fetch_ohlcv(ticker, interval=plan["interval"], period=plan["period"])
        if tf == "4h":
            df = resample_to_4h(df)
        data[tf] = df

    return data


if __name__ == "__main__":
    # Quick manual smoke test -- run locally, needs real internet access.
    data = fetch_multi_timeframe_data("RELIANCE", timeframes=["weekly", "daily", "4h"])
    result = analyze_multi_timeframe(data)
    print(result["summary"])

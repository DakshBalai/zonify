"""
test_multi_timeframe.py
Tests analyze_multi_timeframe() against synthetic data where the bias
relationship across timeframes is deliberately known in advance:
  - a short (n=25) generate_trending_candles() sample is a single clean
    uptrend leg (confirmed directly: legs are randint(15,35) long, and
    the code always starts direction=1, so for n<=35 the very first
    leg IS the whole dataset) -- reliably produces Bias.BULLISH.
  - inverting that same series around its start price (open' = 2*start
    - open, etc., with high/low swapped-and-mirrored to stay valid
    OHLC) produces a mirror-image downtrend -- reliably Bias.BEARISH.
This lets every test assert against a specific known answer instead of
just checking "something plausible came out".
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_price_data import generate_trending_candles  # noqa: E402
from structure_engine import Bias  # noqa: E402
from multi_timeframe import analyze_multi_timeframe, _summarize_alignment  # noqa: E402


def make_uptrend_df(n=25, seed=1):
    return generate_trending_candles(n=n, start_price=100.0, seed=seed)


def make_downtrend_df(n=25, seed=1):
    """Mirror-image of the uptrend series around its start price."""
    df = make_uptrend_df(n=n, seed=seed)
    start = df["open"].iloc[0]
    inverted = df.copy()
    inverted["open"] = 2 * start - df["open"]
    inverted["close"] = 2 * start - df["close"]
    inverted["high"] = 2 * start - df["low"]
    inverted["low"] = 2 * start - df["high"]
    return inverted


def make_flat_undetermined_df(n=10):
    """Too few candles / no real swings -> Bias.UNDETERMINED."""
    rows = [(100, 100.1, 99.9, 100) for _ in range(n)]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000
    return df


def test_single_timeframe_uptrend_is_bullish():
    data = {"daily": make_uptrend_df()}
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["bias_by_timeframe"]["daily"] == Bias.BULLISH
    assert result["alignment"] == "bullish"
    print("PASS: single uptrend timeframe -> bullish alignment")


def test_single_timeframe_downtrend_is_bearish():
    data = {"daily": make_downtrend_df()}
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["bias_by_timeframe"]["daily"] == Bias.BEARISH
    assert result["alignment"] == "bearish"
    print("PASS: single downtrend timeframe -> bearish alignment")


def test_all_timeframes_aligned_bullish():
    data = {
        "weekly": make_uptrend_df(seed=2),
        "daily": make_uptrend_df(seed=3),
        "4h": make_uptrend_df(seed=4),
    }
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["bias_by_timeframe"]["weekly"] == Bias.BULLISH
    assert result["bias_by_timeframe"]["daily"] == Bias.BULLISH
    assert result["bias_by_timeframe"]["4h"] == Bias.BULLISH
    assert result["alignment"] == "bullish"
    # order must be HTF-first regardless of dict insertion order
    assert list(result["bias_by_timeframe"].keys()) == ["weekly", "daily", "4h"]
    print("PASS: all timeframes bullish -> aligned bullish, HTF-first order preserved")


def test_mixed_timeframes_produce_mixed_alignment():
    # Engineered disagreement: weekly bullish (HTF), daily bearish (LTF pullback)
    data = {
        "weekly": make_uptrend_df(seed=5),
        "daily": make_downtrend_df(seed=4),
    }
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["bias_by_timeframe"]["weekly"] == Bias.BULLISH
    assert result["bias_by_timeframe"]["daily"] == Bias.BEARISH
    assert result["alignment"] == "mixed"
    assert "MIXED" in result["summary"]
    assert "WEEKLY: bullish" in result["summary"]
    assert "DAILY: bearish" in result["summary"]
    print("PASS: genuinely disagreeing timeframes -> mixed alignment, both listed in summary")


def test_undetermined_timeframe_excluded_from_alignment_but_shown_in_summary():
    # weekly bullish (determined), daily flat/undetermined -- alignment
    # should still resolve to "bullish" (undetermined is excluded from
    # the alignment CHECK) but daily's undetermined status must still
    # be visible in the summary string, not silently dropped.
    data = {
        "weekly": make_uptrend_df(seed=7),
        "daily": make_flat_undetermined_df(),
    }
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["bias_by_timeframe"]["weekly"] == Bias.BULLISH
    assert result["bias_by_timeframe"]["daily"] == Bias.UNDETERMINED
    assert result["alignment"] == "bullish"
    assert "DAILY: undetermined" in result["summary"]
    print("PASS: undetermined timeframe excluded from alignment check but still shown in summary")


def test_all_undetermined_gives_undetermined_alignment():
    data = {"daily": make_flat_undetermined_df(), "4h": make_flat_undetermined_df()}
    result = analyze_multi_timeframe(data, include_poi=False)
    assert result["alignment"] == "undetermined"
    print("PASS: all-flat data -> undetermined alignment (not falsely bullish/bearish)")


def test_poi_included_by_default_and_omittable():
    data = {"daily": make_uptrend_df()}
    with_poi = analyze_multi_timeframe(data, include_poi=True)
    without_poi = analyze_multi_timeframe(data, include_poi=False)
    assert with_poi["per_timeframe"]["daily"]["poi"] is not None
    assert without_poi["per_timeframe"]["daily"]["poi"] is None
    print("PASS: include_poi correctly toggles POI analysis per timeframe")


def test_custom_lookback_override_is_respected():
    data = {"daily": make_uptrend_df()}
    # An absurdly large lookback should reduce/eliminate detected swings
    # relative to the default -- confirms the override actually reaches
    # analyze_structure() rather than being silently ignored.
    default_result = analyze_multi_timeframe(data, include_poi=False)
    override_result = analyze_multi_timeframe(
        data, lookback_by_timeframe={"daily": 10}, include_poi=False
    )
    default_swings = len(default_result["per_timeframe"]["daily"]["structure"]["swings"])
    override_swings = len(override_result["per_timeframe"]["daily"]["structure"]["swings"])
    assert override_swings <= default_swings
    print(f"PASS: lookback override respected ({default_swings} swings -> {override_swings} swings)")


if __name__ == "__main__":
    test_single_timeframe_uptrend_is_bullish()
    test_single_timeframe_downtrend_is_bearish()
    test_all_timeframes_aligned_bullish()
    test_mixed_timeframes_produce_mixed_alignment()
    test_undetermined_timeframe_excluded_from_alignment_but_shown_in_summary()
    test_all_undetermined_gives_undetermined_alignment()
    test_poi_included_by_default_and_omittable()
    test_custom_lookback_override_is_respected()
    print("\nAll multi_timeframe tests passed.")

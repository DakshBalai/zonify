"""
test_data_loader.py
Tests everything in data_loader.py that does NOT require reaching
Yahoo Finance's servers (blocked in this dev sandbox by network policy).

Covers:
  - normalize_nse_ticker(): suffix handling
  - _flatten_columns(): the MultiIndex quirk confirmed in yfinance 1.7.0
  - resample_to_4h(): the actual quant-correctness-critical piece --
    session-anchored 4H bucket boundaries and correct OHLCV aggregation

The real network fetch (fetch_ohlcv() actually calling yf.download()
and getting real candles back) is NOT and CANNOT be tested here -- that
part is verified locally, see run_locally.md / the chat instructions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import (  # noqa: E402
    NSE_TZ,
    _clean_ohlcv,
    _flatten_columns,
    normalize_nse_ticker,
    resample_to_4h,
)


def make_synthetic_1h_nse_data(n_days=5, seed=42) -> pd.DataFrame:
    """
    Builds synthetic 1-hour OHLCV data with NSE's real intraday candle
    schedule: 7 hourly candles per session, at 09:15, 10:15, 11:15,
    12:15, 13:15, 14:15, 15:15 IST (the 15:15 candle covers the last,
    shorter 15-minute stub of the session up to the 15:30 close -- this
    is exactly how yfinance buckets NSE 1h data in practice). Shaped to
    match the clean output of fetch_ohlcv(..., interval="1h"), so it
    exercises resample_to_4h() the same way real data would.
    """
    rng = np.random.default_rng(seed)
    session_times = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]

    timestamps = []
    trading_days = pd.bdate_range("2024-01-01", periods=n_days, tz=NSE_TZ)
    for day in trading_days:
        for t in session_times:
            hh, mm = t.split(":")
            timestamps.append(day + pd.Timedelta(hours=int(hh), minutes=int(mm)))

    idx = pd.DatetimeIndex(timestamps, name="date")
    n = len(idx)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = price
    close = price + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.1, 0.05, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.1, 0.05, n))
    volume = rng.integers(1000, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_normalize_nse_ticker():
    assert normalize_nse_ticker("RELIANCE") == "RELIANCE.NS"
    assert normalize_nse_ticker("reliance") == "RELIANCE.NS"
    assert normalize_nse_ticker("RELIANCE.NS") == "RELIANCE.NS"
    assert normalize_nse_ticker("  tcs  ") == "TCS.NS"
    # BSE tickers left alone, not force-converted to NSE
    assert normalize_nse_ticker("RELIANCE.BO") == "RELIANCE.BO"
    print("PASS: normalize_nse_ticker")


def test_flatten_columns():
    # Simulate the real MultiIndex shape confirmed from yfinance 1.7.0
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["RELIANCE.NS"]])
    df = pd.DataFrame(np.zeros((3, 5)), columns=cols)
    flat = _flatten_columns(df)
    assert list(flat.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert not isinstance(flat.columns, pd.MultiIndex)

    # Already-flat input should pass through unchanged
    plain = pd.DataFrame(np.zeros((3, 2)), columns=["Open", "Close"])
    assert list(_flatten_columns(plain).columns) == ["Open", "Close"]
    print("PASS: _flatten_columns")


def test_clean_ohlcv_drops_zero_volume_placeholder_rows():
    # Confirmed directly against real yfinance data: NSE trading-
    # calendar dates with no actual trade data still come back as a
    # row -- volume=0, open=high=low=close all pinned to the prior
    # close (a zero-range candle). Structurally meaningless and, left
    # in, can produce a degenerate zero-width Order Block.
    idx = pd.DatetimeIndex([
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
    ])
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 101.0, 103.0],
        "High": [101.0, 101.0, 101.0, 104.0],
        "Low": [99.0, 101.0, 101.0, 102.0],
        "Close": [100.5, 101.0, 101.0, 103.5],
        "Volume": [5000, 0, 6000, 7000],
    }, index=idx)
    cleaned = _clean_ohlcv(df)
    assert len(cleaned) == 3
    assert 0 not in cleaned["volume"].values
    print("PASS: _clean_ohlcv drops zero-volume placeholder rows")


def test_clean_ohlcv_keeps_low_but_nonzero_volume_rows():
    # A genuinely thin trading day (nonzero volume) must NOT be
    # dropped -- only the volume==0 placeholder case is data-quality
    # noise, not "low volume" in general.
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.5],
        "Close": [100.5, 101.5], "Volume": [5000, 3],
    }, index=idx)
    cleaned = _clean_ohlcv(df)
    assert len(cleaned) == 2
    print("PASS: _clean_ohlcv keeps a genuinely thin (nonzero-volume) trading day")


def test_resample_bucket_count_and_boundaries():
    df_1h = make_synthetic_1h_nse_data(n_days=3)
    df_4h = resample_to_4h(df_1h)

    # 7 hourly candles/day -> ceil(7/4) = 2 buckets/day -> 6 buckets for 3 days
    assert len(df_4h) == 6, f"expected 6 buckets, got {len(df_4h)}"

    day1 = pd.Timestamp("2024-01-01", tz=NSE_TZ)
    expected_starts = [
        day1 + pd.Timedelta(hours=9, minutes=15),
        day1 + pd.Timedelta(hours=13, minutes=15),
    ]
    actual_starts = list(df_4h.index[:2])
    assert actual_starts == expected_starts, (
        f"bucket boundaries wrong.\nexpected: {expected_starts}\nactual:   {actual_starts}"
    )

    # First bucket each day = 09:15,10:15,11:15,12:15 (4 candles)
    # Second bucket each day = 13:15,14:15,15:15 (3 candles, session's short tail)
    print("PASS: resample bucket count and boundaries")


def test_resample_ohlcv_aggregation_correctness():
    df_1h = make_synthetic_1h_nse_data(n_days=1)
    df_4h = resample_to_4h(df_1h)

    first_bucket_source = df_1h.iloc[0:4]   # 09:15-12:15
    second_bucket_source = df_1h.iloc[4:7]  # 13:15-15:15

    b1 = df_4h.iloc[0]
    assert b1["open"] == first_bucket_source["open"].iloc[0]
    assert b1["close"] == first_bucket_source["close"].iloc[-1]
    assert b1["high"] == first_bucket_source["high"].max()
    assert b1["low"] == first_bucket_source["low"].min()
    assert b1["volume"] == first_bucket_source["volume"].sum()

    b2 = df_4h.iloc[1]
    assert b2["open"] == second_bucket_source["open"].iloc[0]
    assert b2["close"] == second_bucket_source["close"].iloc[-1]
    assert b2["high"] == second_bucket_source["high"].max()
    assert b2["low"] == second_bucket_source["low"].min()
    assert b2["volume"] == second_bucket_source["volume"].sum()

    print("PASS: resample OHLCV aggregation correctness")


def test_resample_no_data_leakage_across_days():
    # A 4H bucket must never blend candles from two different trading
    # days -- e.g. day 1's 15:15 candle must not merge with day 2's
    # 09:15 candle just because they're both "close in time".
    df_1h = make_synthetic_1h_nse_data(n_days=2)
    df_4h = resample_to_4h(df_1h)

    days_present = sorted(set(ts.normalize() for ts in df_4h.index))
    assert len(days_present) == 2

    for day in days_present:
        day_buckets = df_4h[df_4h.index.normalize() == day]
        assert len(day_buckets) == 2, f"day {day} should have exactly 2 buckets, got {len(day_buckets)}"
    print("PASS: no cross-day data leakage in resampling")


def test_resample_handles_empty_input():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty.index = pd.DatetimeIndex([], tz=NSE_TZ, name="date")
    result = resample_to_4h(empty)
    assert result.empty
    print("PASS: resample handles empty input gracefully")


if __name__ == "__main__":
    test_normalize_nse_ticker()
    test_flatten_columns()
    test_clean_ohlcv_drops_zero_volume_placeholder_rows()
    test_clean_ohlcv_keeps_low_but_nonzero_volume_rows()
    test_resample_bucket_count_and_boundaries()
    test_resample_ohlcv_aggregation_correctness()
    test_resample_no_data_leakage_across_days()
    test_resample_handles_empty_input()
    print("\nAll data_loader tests passed.")

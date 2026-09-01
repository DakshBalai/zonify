"""
test_session_model.py
Hand-built intraday candle sequences with known-in-advance answers,
same convention as test_poi_engine.py / test_data_loader.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session_model import (  # noqa: E402
    DailyRangeFormation,
    classify_daily_range_formation,
    find_po3_setups,
)


def intraday_candles(rows):
    """rows: list of (timestamp_str, open, high, low, close)."""
    timestamps = [pd.Timestamp(r[0], tz="Asia/Kolkata") for r in rows]
    df = pd.DataFrame(
        [r[1:] for r in rows], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(timestamps, name="date"),
    )
    df["volume"] = 1000
    return df


# ---------------------------------------------------------------------------
# classify_daily_range_formation
# ---------------------------------------------------------------------------

def test_classify_olhc_when_low_prints_before_high():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 95, 96),      # low of the day (95) -- prints first
        ("2024-01-01 10:15", 96, 97, 96, 96.5),
        ("2024-01-01 11:15", 96.5, 105, 96, 104),    # high of the day (105) -- prints later
    ])
    result = classify_daily_range_formation(df)
    day = pd.Timestamp("2024-01-01", tz="Asia/Kolkata")
    assert result[day] == DailyRangeFormation.OLHC
    print("PASS: low printing before high classifies as OLHC")


def test_classify_ohlc_when_high_prints_before_low():
    df = intraday_candles([
        ("2024-01-02 09:15", 100, 105, 99, 104),     # high of the day (105) -- prints first
        ("2024-01-02 10:15", 104, 104.5, 100, 101),
        ("2024-01-02 11:15", 101, 101.5, 95, 96),    # low of the day (95) -- prints later
    ])
    result = classify_daily_range_formation(df)
    day = pd.Timestamp("2024-01-02", tz="Asia/Kolkata")
    assert result[day] == DailyRangeFormation.OHLC
    print("PASS: high printing before low classifies as OHLC")


def test_classify_undetermined_when_high_and_low_in_same_candle():
    df = intraday_candles([
        ("2024-01-03 09:15", 100, 100.5, 99.5, 100),
        ("2024-01-03 10:15", 100, 106, 94, 100),     # this single candle has BOTH the day's high and low
        ("2024-01-03 11:15", 100, 101, 99, 100),
    ])
    result = classify_daily_range_formation(df)
    day = pd.Timestamp("2024-01-03", tz="Asia/Kolkata")
    assert result[day] == DailyRangeFormation.UNDETERMINED
    print("PASS: high and low landing in the same candle classifies as undetermined")


def test_classify_handles_multiple_days_independently():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 95, 96),
        ("2024-01-01 10:15", 96, 105, 96, 104),
        ("2024-01-02 09:15", 100, 105, 99, 104),
        ("2024-01-02 10:15", 104, 104.5, 95, 96),
    ])
    result = classify_daily_range_formation(df)
    assert len(result) == 2
    assert result[pd.Timestamp("2024-01-01", tz="Asia/Kolkata")] == DailyRangeFormation.OLHC
    assert result[pd.Timestamp("2024-01-02", tz="Asia/Kolkata")] == DailyRangeFormation.OHLC
    print("PASS: classify_daily_range_formation handles multiple days independently")


# ---------------------------------------------------------------------------
# find_po3_setups
# ---------------------------------------------------------------------------

def test_po3_bullish_manipulation_expects_bearish_distribution():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),        # accumulation range candle: (99, 101)
        ("2024-01-01 10:15", 100, 103, 99.5, 100.5),    # wicks above 101 (manipulation), closes at 100.5 (inside)
        ("2024-01-01 11:15", 100.5, 100.6, 98, 98.5),   # entry candle -- distribution starts down
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert len(setups) == 1
    s = setups[0]
    assert s.accumulation_low == 99 and s.accumulation_high == 101
    assert s.manipulation_index == 1
    assert s.manipulation_direction == "bullish"
    assert s.distribution_direction == "bearish"
    assert s.stop_price == 103    # the manipulation candle's own high
    assert s.entry_index == 2
    assert s.entry_price == 98.5  # entry candle's close
    print("PASS: a bullish manipulation (swept the range high) expects bearish distribution")


def test_po3_bearish_manipulation_expects_bullish_distribution():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),
        ("2024-01-01 10:15", 100, 100.5, 97, 99.5),   # wicks below 99 but closes at 99.5 (inside)
        ("2024-01-01 11:15", 99.5, 103, 99.4, 102),
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert len(setups) == 1
    s = setups[0]
    assert s.manipulation_direction == "bearish"
    assert s.distribution_direction == "bullish"
    assert s.stop_price == 97
    assert s.entry_price == 102
    print("PASS: a bearish manipulation (swept the range low) expects bullish distribution")


def test_po3_no_setup_when_range_never_swept():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),
        ("2024-01-01 10:15", 100, 100.8, 99.2, 100.3),
        ("2024-01-01 11:15", 100.3, 100.9, 99.5, 100.1),
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert setups == []
    print("PASS: no PO3 setup when the accumulation range is never swept")


def test_po3_no_setup_when_manipulation_is_the_last_candle_of_the_day():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),
        ("2024-01-01 10:15", 100, 103, 99.5, 100.5),  # manipulation, but no candle left today to enter on
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert setups == []
    print("PASS: no PO3 setup when the manipulation candle is the last one of the session")


def test_po3_only_one_setup_per_day():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),
        ("2024-01-01 10:15", 100, 103, 99.5, 100.5),   # 1st manipulation (bullish)
        ("2024-01-01 11:15", 100.5, 100.6, 98, 98.5),  # entry
        ("2024-01-01 12:15", 98.5, 99, 96, 96.5),      # wicks even further -- would ALSO qualify, must be ignored
        ("2024-01-01 13:15", 96.5, 97, 96, 96.8),
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert len(setups) == 1
    assert setups[0].manipulation_index == 1
    print("PASS: only the first manipulation of the day produces a setup")


def test_po3_independent_across_days():
    df = intraday_candles([
        ("2024-01-01 09:15", 100, 101, 99, 100),
        ("2024-01-01 10:15", 100, 103, 99.5, 100.5),
        ("2024-01-01 11:15", 100.5, 100.6, 98, 98.5),
        ("2024-01-02 09:15", 50, 51, 49, 50),
        ("2024-01-02 10:15", 50, 50.5, 47, 49.5),
        ("2024-01-02 11:15", 49.5, 53, 49.4, 52),
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert len(setups) == 2
    days = {s.day for s in setups}
    assert len(days) == 2, "each day's setup must use that day's OWN accumulation range"
    print("PASS: PO3 setups are computed independently per day, not blended across days")


def test_po3_empty_on_daily_bars_with_no_intraday_ordering():
    # One candle per "day" -- there's no intraday opening range to define.
    df = intraday_candles([
        ("2024-01-01 00:00", 100, 105, 95, 102),
        ("2024-01-02 00:00", 102, 108, 100, 106),
        ("2024-01-03 00:00", 106, 110, 104, 108),
    ])
    setups = find_po3_setups(df, opening_range_bars=1)
    assert setups == []
    print("PASS: find_po3_setups is a no-op on daily bars (each day is a single candle)")


def test_po3_empty_on_empty_dataframe():
    df = intraday_candles([])
    assert find_po3_setups(df) == []
    assert classify_daily_range_formation(df) == {}
    print("PASS: both functions handle an empty DataFrame gracefully")


if __name__ == "__main__":
    test_classify_olhc_when_low_prints_before_high()
    test_classify_ohlc_when_high_prints_before_low()
    test_classify_undetermined_when_high_and_low_in_same_candle()
    test_classify_handles_multiple_days_independently()
    test_po3_bullish_manipulation_expects_bearish_distribution()
    test_po3_bearish_manipulation_expects_bullish_distribution()
    test_po3_no_setup_when_range_never_swept()
    test_po3_no_setup_when_manipulation_is_the_last_candle_of_the_day()
    test_po3_only_one_setup_per_day()
    test_po3_independent_across_days()
    test_po3_empty_on_daily_bars_with_no_intraday_ordering()
    test_po3_empty_on_empty_dataframe()
    print("\nAll session_model tests passed.")

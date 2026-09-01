"""
test_fundamentals.py
Tests for the parts of fundamentals.py that don't need real network
access -- compute_yearly_metrics() and format_profile(). fetch_ticker_profile()
itself is NOT tested here, same reasoning as data_loader.fetch_ohlcv():
it needs real yfinance access, verified manually (see chat/README).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fundamentals import TickerProfile, YearlyMetrics, compute_yearly_metrics, format_profile  # noqa: E402


def daily_df(rows):
    """rows: list of (date_str, open, high, low, close)."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz="Asia/Kolkata") for r in rows], name="date")
    df = pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000
    return df


# ---------------------------------------------------------------------------
# compute_yearly_metrics
# ---------------------------------------------------------------------------

def test_compute_yearly_metrics_empty_dataframe():
    empty = daily_df([])
    assert compute_yearly_metrics(empty) == []
    print("PASS: compute_yearly_metrics handles an empty DataFrame")


def test_compute_yearly_metrics_single_year():
    df = daily_df([
        ("2024-01-02", 100, 105, 95, 102),
        ("2024-06-15", 102, 120, 90, 110),
        ("2024-12-30", 110, 115, 108, 112),
    ])
    metrics = compute_yearly_metrics(df)
    assert len(metrics) == 1
    m = metrics[0]
    assert m.year == 2024
    assert m.open == 100 and m.close == 112
    assert m.high == 120 and m.low == 90
    assert m.return_pct == (112 - 100) / 100 * 100
    assert m.range_pct == (120 - 90) / 90 * 100
    print("PASS: compute_yearly_metrics computes correct OHLC/return/range for one year")


def test_compute_yearly_metrics_multiple_years_independent():
    df = daily_df([
        ("2023-01-02", 100, 110, 95, 105),
        ("2023-12-29", 105, 108, 100, 106),
        ("2024-01-02", 106, 112, 104, 108),
        ("2024-12-30", 108, 130, 106, 125),
    ])
    metrics = compute_yearly_metrics(df)
    assert [m.year for m in metrics] == [2023, 2024]
    assert metrics[0].open == 100 and metrics[0].close == 106
    assert metrics[1].open == 106 and metrics[1].close == 125
    print("PASS: compute_yearly_metrics keeps years independent and in order")


def test_compute_yearly_metrics_negative_return():
    df = daily_df([
        ("2024-01-02", 100, 105, 95, 100),
        ("2024-12-30", 90, 92, 85, 80),
    ])
    metrics = compute_yearly_metrics(df)
    assert metrics[0].return_pct == (80 - 100) / 100 * 100
    assert metrics[0].return_pct < 0
    print("PASS: compute_yearly_metrics correctly reports a negative yearly return")


# ---------------------------------------------------------------------------
# format_profile
# ---------------------------------------------------------------------------

def make_profile(**overrides):
    defaults = dict(
        ticker="RELIANCE.NS",
        all_time_high=1600.0, all_time_high_date=pd.Timestamp("2026-01-05"),
        all_time_low=3.4, all_time_low_date=pd.Timestamp("1996-01-29"),
        yearly_metrics=[
            YearlyMetrics(year=2023, open=100, high=120, low=90, close=110, return_pct=10.0, range_pct=33.3),
            YearlyMetrics(year=2024, open=110, high=140, low=105, close=130, return_pct=18.2, range_pct=33.3),
        ],
        dividends=pd.Series([5.0, 5.5], index=pd.DatetimeIndex([pd.Timestamp("2024-08-19"), pd.Timestamp("2025-08-14")])),
        splits=pd.Series([2.0], index=pd.DatetimeIndex([pd.Timestamp("2024-10-28")])),
        earnings_dates=pd.DataFrame({"EPS Estimate": [12.0], "Reported EPS": [13.0]}, index=[pd.Timestamp("2026-01-16")]),
        info={"longName": "Reliance Industries Limited", "sector": "Energy", "industry": "Oil & Gas", "marketCap": 17_000_000_000_000, "trailingPE": 23.7},
    )
    defaults.update(overrides)
    return TickerProfile(**defaults)


def test_format_profile_includes_all_time_high_and_low():
    profile = make_profile()
    text = format_profile(profile)
    assert "All-time high: 1600.00" in text
    assert "All-time low:  3.40" in text
    print("PASS: format_profile includes all-time high and low")


def test_format_profile_handles_missing_all_time_high():
    profile = make_profile(all_time_high=None, all_time_high_date=None, all_time_low=None, all_time_low_date=None)
    text = format_profile(profile)
    assert "All-time high/low: unavailable" in text
    print("PASS: format_profile handles missing all-time high/low without crashing")


def test_format_profile_limits_yearly_metrics_to_n_years():
    profile = make_profile(yearly_metrics=[
        YearlyMetrics(year=y, open=1, high=1, low=1, close=1, return_pct=0, range_pct=0) for y in range(2015, 2025)
    ])
    text = format_profile(profile, n_years=3)
    assert "2022:" in text and "2023:" in text and "2024:" in text
    assert "2015:" not in text
    print("PASS: format_profile shows only the last n_years of yearly metrics")


def test_format_profile_handles_no_dividends_or_splits():
    profile = make_profile(dividends=pd.Series(dtype=float), splits=pd.Series(dtype=float))
    text = format_profile(profile)
    assert "Dividends: none recorded" in text
    assert "Splits: none recorded" in text
    print("PASS: format_profile reports 'none recorded' rather than crashing on empty dividends/splits")


def test_format_profile_handles_missing_earnings_dates():
    profile = make_profile(earnings_dates=None)
    text = format_profile(profile)
    assert "Earnings history: not available for this ticker" in text
    print("PASS: format_profile handles a None earnings_dates without crashing")


def test_format_profile_handles_empty_info():
    profile = make_profile(info={})
    text = format_profile(profile)
    assert "RELIANCE.NS -- ? / ?" in text
    print("PASS: format_profile falls back gracefully when info is empty")


if __name__ == "__main__":
    test_compute_yearly_metrics_empty_dataframe()
    test_compute_yearly_metrics_single_year()
    test_compute_yearly_metrics_multiple_years_independent()
    test_compute_yearly_metrics_negative_return()
    test_format_profile_includes_all_time_high_and_low()
    test_format_profile_handles_missing_all_time_high()
    test_format_profile_limits_yearly_metrics_to_n_years()
    test_format_profile_handles_no_dividends_or_splits()
    test_format_profile_handles_missing_earnings_dates()
    test_format_profile_handles_empty_info()
    print("\nAll fundamentals tests passed.")

"""
data_loader.py
Real-market data layer for NSE (India) tickers, built on yfinance.

This is the ONLY module in the project that talks to an external data
source. Everything downstream (structure_engine.py, multi_timeframe.py,
the backtester, the Streamlit app) consumes the plain OHLCV DataFrame
this module returns, so none of that code needs to know or care that
the data came from Yahoo Finance specifically.

Two jobs:
  1. fetch_ohlcv()      -- download candles for an NSE ticker/timeframe,
                            normalize the result into a clean, predictable
                            DataFrame (lowercase columns, no MultiIndex,
                            consistent tz handling).
  2. resample_to_4h()   -- yfinance/Yahoo does not offer a native 4-hour
                            interval, so we fetch 1-hour candles and build
                            4H candles ourselves. This has to be done
                            carefully: a naive pandas .resample("4h") uses
                            midnight-aligned UTC buckets, which do NOT line
                            up with NSE's actual trading session and would
                            produce meaningless, misaligned candles. See
                            the function docstring for the fix.

SANDBOX NOTE: this dev sandbox's network is restricted to package
registries (pypi, github, etc) and cannot reach Yahoo Finance's API
(query1/query2.finance.yahoo.com). That was verified directly here --
yf.download() correctly raises/returns empty with a 403 host-not-allowed
error, exactly as expected, which is *why* fetch_ohlcv() has clean
try/except handling around the download call rather than assuming it
always succeeds.

Everything in this module that does NOT require reaching Yahoo's
servers (ticker normalization, column flattening/renaming, the 4H
resampling math) IS tested in this sandbox, against synthetic
1-hour data shaped exactly like what yfinance returns for NSE
tickers (see tests/test_data_loader.py). The actual network fetch is
verified by you locally -- instructions at the bottom of this file
and in the chat.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

NSE_SUFFIX = ".NS"
NSE_TZ = "Asia/Kolkata"

# NSE cash-market session: 09:15 to 15:30 IST.
NSE_SESSION_START = "09:15"
NSE_SESSION_END = "15:30"

# yfinance interval strings we allow through this project. yfinance
# supports more (2m, 90m, ...) but we constrain to the set the SMC
# engine's multi-timeframe orchestration actually uses, so a typo
# doesn't silently hit yfinance's own defaults.
#
# yfinance's OWN history limits per interval (not something this
# project controls, and tighter the lower you go): 1h ~730 days,
# 30m/15m/5m ~60 days, 1m ~7 days. The 1m limit in particular means a
# 1h->1min top-down drill-down (see top_down.py) only ever has about a
# week of history to backtest against.
SUPPORTED_INTERVALS = {"1mo", "1wk", "1d", "1h", "30m", "15m", "5m", "1m"}


def normalize_nse_ticker(ticker: str) -> str:
    """
    Ensures an NSE ticker carries the '.NS' suffix yfinance requires,
    without double-appending it if the caller already included it (or
    accidentally passed a BSE '.BO' ticker, which we leave alone since
    that's a deliberate choice, not a mistake).
    """
    ticker = ticker.strip().upper()
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    return f"{ticker}{NSE_SUFFIX}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance's yf.download() returns a MultiIndex column DataFrame
    (e.g. ('Close', 'RELIANCE.NS')) even for a single ticker in the
    installed version (1.7.0) -- confirmed directly in this sandbox,
    not assumed. This collapses that down to plain 'Close', 'Open', etc.
    Safe to call on an already-flat DataFrame too (no-op).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a raw yfinance DataFrame into the exact shape
    structure_engine.py expects: lowercase open/high/low/close/volume
    columns, a clean DatetimeIndex localized to NSE's timezone, sorted,
    de-duplicated, with any all-NaN rows (which yfinance sometimes
    returns for the most recent, still-forming candle) dropped, and
    zero-volume placeholder rows dropped.

    The zero-volume case is a real data quality issue, confirmed
    directly: yfinance returns a row for every NSE trading-calendar
    date even on days with no actual trade data for a given ticker
    (exactly 5 such rows appeared identically across RELIANCE, TCS,
    SBIN, HDFCBANK, and INFY over the same 2-year window -- almost
    certainly the same handful of exchange holidays/feed gaps, not
    ticker-specific illiquidity). These rows have volume=0 and
    open=high=low=close all pinned to the prior close, which makes
    them a ZERO-RANGE candle -- structurally meaningless, but not
    harmless: this exact shape was caught producing a degenerate
    zero-width Order Block (found the fresh Order Block on SBIN's
    daily chart was a straight line, not a zone, tracing back to one
    of these rows). A real trading day for a liquid NSE stock always
    has nonzero volume, so filtering on volume==0 removes only these
    placeholder rows, never a genuinely low-volume (but real) session.
    """
    df = _flatten_columns(df)

    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()

    df.index.name = "date"

    # yfinance returns tz-aware timestamps for NSE tickers already in
    # Asia/Kolkata; this just makes that explicit and handles the case
    # (e.g. daily/weekly/monthly bars) where the index comes back
    # tz-naive instead.
    if df.index.tz is None:
        df.index = df.index.tz_localize(NSE_TZ)
    else:
        df.index = df.index.tz_convert(NSE_TZ)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    if "volume" in df.columns:
        df = df[df["volume"] != 0]

    return df


def fetch_ohlcv(
    ticker: str,
    interval: str = "1d",
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Downloads OHLCV candles for one NSE ticker and returns a clean
    DataFrame indexed by tz-aware (Asia/Kolkata) timestamp with columns
    open/high/low/close/volume -- the exact format structure_engine.py's
    analyze_structure() consumes.

    ticker:   e.g. "RELIANCE" or "RELIANCE.NS" (suffix auto-added)
    interval: one of SUPPORTED_INTERVALS. Note '4h' is NOT in this list
              on purpose -- yfinance has no native 4H interval. Fetch
              '1h' and call resample_to_4h() on the result instead.
    period:   yfinance shorthand window, e.g. "60d", "2y", "max".
              Mutually exclusive with start/end (yfinance rule, not ours).
    start/end: explicit "YYYY-MM-DD" bounds, alternative to `period`.

    Raises ValueError for bad inputs (caught early, before ever hitting
    the network). Raises RuntimeError if yfinance returns nothing --
    wrong ticker, no internet, or (in this sandbox) the network policy
    blocking Yahoo's API -- with a message that tells you which.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"interval={interval!r} not supported here. "
            f"Use one of {sorted(SUPPORTED_INTERVALS)}. "
            f"For 4H candles, fetch '1h' and call resample_to_4h()."
        )
    if period and (start or end):
        raise ValueError("Pass either `period` OR `start`/`end`, not both.")

    nse_ticker = normalize_nse_ticker(ticker)

    kwargs = {"interval": interval, "auto_adjust": True, "progress": False}
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    try:
        raw = yf.download(nse_ticker, **kwargs)
    except Exception as exc:  # network errors, yfinance internal errors, etc.
        raise RuntimeError(
            f"yfinance download failed for {nse_ticker!r} ({interval}). "
            f"Original error: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {nse_ticker!r} ({interval}, "
            f"period={period!r}, start={start!r}, end={end!r}). "
            f"Check the ticker is correct and that you have internet "
            f"access (this call cannot succeed inside the dev sandbox)."
        )

    return _clean_ohlcv(raw)


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 4-hour candles from 1-hour NSE candles.

    THE PROBLEM WITH df.resample("4h"): pandas aligns resample buckets
    to midnight UTC by default (00:00, 04:00, 08:00, ...). NSE trades
    09:15-15:30 IST, so midnight-aligned buckets slice straight through
    the middle of the session at arbitrary points that have nothing to
    do with how a trader actually reads a 4H chart -- the resulting
    candles would be structurally meaningless, and worse, would shift
    depending on what time zone the data happened to be in.

    THE FIX: anchor buckets to the session open (09:15) for each
    trading day independently. Candle 1 covers 09:15-13:15, candle 2
    covers the remainder of the session, 13:15-15:30 (only 2h15m --
    NSE's ~6h15m session doesn't divide evenly by 4h, so the last
    candle of the day is intentionally shorter). This is standard
    practice for building intraday HTF candles on a fixed-length
    session and matches how charting platforms build session-anchored
    4H bars.

    Expects df_1h to already be the clean, tz-aware output of
    fetch_ohlcv(ticker, interval="1h", ...).
    """
    if df_1h.empty:
        return df_1h.copy()

    df = df_1h.copy()
    session_date = df.index.tz_convert(NSE_TZ).normalize()
    session_open = session_date + pd.Timedelta(NSE_SESSION_START + ":00")

    hours_since_open = (df.index - session_open) / pd.Timedelta(hours=1)
    bucket_number = (hours_since_open // 4).astype(int)
    bucket_start = session_open + bucket_number * pd.Timedelta(hours=4)

    df["_bucket"] = bucket_start

    agg = df.groupby("_bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg.index.name = "date"
    return agg


if __name__ == "__main__":
    # Quick manual smoke test -- run this locally (see chat for the
    # exact command), NOT in the dev sandbox, since it needs real
    # internet access to Yahoo Finance.
    df_1h = fetch_ohlcv("RELIANCE", interval="1h", period="60d")
    print(f"Fetched {len(df_1h)} 1H candles for RELIANCE.NS")
    print(df_1h.head())

    df_4h = resample_to_4h(df_1h)
    print(f"\nResampled to {len(df_4h)} 4H candles")
    print(df_4h.head())

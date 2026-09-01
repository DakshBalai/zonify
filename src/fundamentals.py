"""
fundamentals.py
Reference/context data for a ticker: how it got here (all-time high/
low, yearly price metrics) and what's driven it (corporate actions --
dividends, splits -- and earnings history). This is layered ON TOP OF
data_loader.py, not a replacement for it: OHLCV price/structure
analysis stays completely separate from this reference layer, so a gap
or quirk in yfinance's fundamentals coverage can never affect
structure/POI detection.

Deliberately NOT wired into poi_engine's zone significance/scoring --
"does being near the all-time high make a zone more important" is a
real, testable hypothesis, but it's a NEW claim that needs its own
backtest pass (same discipline as everything else in this project),
not something to bolt on silently. This module only supplies the
reference data; using it to weight zones is a separate, later step.

Two honest caveats, confirmed directly against real NSE tickers:
  - yfinance's fundamentals coverage for NSE (.NS) tickers is
    inconsistent -- info fields, earnings dates, and financials can be
    missing or sparse, especially for smaller-cap names (confirmed
    directly: SUNPHARMA.NS returned a real (nonempty) earnings_dates
    frame, but coverage is NOT guaranteed generally). Every fetch here
    is defensive: it returns whatever's actually available and reports
    what's missing, rather than assuming a fixed shape.
  - Earnings-date timestamps carry US-market timezone artifacts from
    yfinance (stripped here -- see fetch_ticker_profile), and reported
    EPS for a stock with a split history comes back scaled to the
    CURRENT adjusted share count, not the nominal EPS actually reported
    that quarter -- consistent with this project's own
    auto_adjust=True price series, but worth knowing if you cross-
    reference against a contemporary news headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from data_loader import normalize_nse_ticker


@dataclass
class YearlyMetrics:
    year: int
    open: float
    high: float
    low: float
    close: float
    return_pct: float   # close-to-close return for the year
    range_pct: float      # (high-low)/low -- a simple volatility proxy


@dataclass
class TickerProfile:
    ticker: str
    all_time_high: float | None
    all_time_high_date: pd.Timestamp | None
    all_time_low: float | None
    all_time_low_date: pd.Timestamp | None
    yearly_metrics: list[YearlyMetrics]
    dividends: pd.Series               # date -> per-share dividend amount
    splits: pd.Series                    # date -> split ratio
    earnings_dates: pd.DataFrame | None   # date-indexed EPS estimate/reported/surprise, or None
    info: dict                            # whatever yfinance's .info returned -- may be sparse


def compute_yearly_metrics(daily_df: pd.DataFrame) -> list[YearlyMetrics]:
    """
    Groups daily OHLCV by calendar year and computes simple, honest
    price-derived yearly stats -- no fundamentals data needed, so this
    always works regardless of yfinance's fundamentals coverage for
    this ticker.
    """
    if daily_df.empty:
        return []

    years = daily_df.index.year
    metrics = []
    for year in sorted(set(years)):
        year_df = daily_df[years == year]
        open_ = float(year_df["open"].iloc[0])
        close = float(year_df["close"].iloc[-1])
        high = float(year_df["high"].max())
        low = float(year_df["low"].min())
        metrics.append(YearlyMetrics(
            year=int(year), open=open_, high=high, low=low, close=close,
            return_pct=(close - open_) / open_ * 100 if open_ else 0.0,
            range_pct=(high - low) / low * 100 if low else 0.0,
        ))
    return metrics


def fetch_ticker_profile(ticker: str) -> TickerProfile:
    """
    Fetches all-time high/low + yearly metrics from max-period daily
    OHLCV (via data_loader.fetch_ohlcv -- always available if the
    ticker itself is valid) plus dividends/splits/earnings/info via
    yfinance's Ticker object -- best-effort, may come back partially
    empty for smaller/less-covered names. Not testable inside a
    hermetic test suite (needs real network access), same as
    data_loader.fetch_ohlcv itself -- see compute_yearly_metrics() and
    format_profile() for the pieces that ARE unit-tested.
    """
    from data_loader import fetch_ohlcv

    nse_ticker = normalize_nse_ticker(ticker)
    daily_df = fetch_ohlcv(ticker, interval="1d", period="max")

    if daily_df.empty:
        ath = ath_date = atl = atl_date = None
    else:
        ath, ath_date = float(daily_df["high"].max()), daily_df["high"].idxmax()
        atl, atl_date = float(daily_df["low"].min()), daily_df["low"].idxmin()

    yf_ticker = yf.Ticker(nse_ticker)

    try:
        dividends = yf_ticker.dividends
    except Exception:
        dividends = pd.Series(dtype=float)

    try:
        splits = yf_ticker.splits
    except Exception:
        splits = pd.Series(dtype=float)

    try:
        earnings_dates = yf_ticker.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            earnings_dates = earnings_dates.copy()
            earnings_dates.index = earnings_dates.index.tz_localize(None).normalize()
    except Exception:
        earnings_dates = None

    try:
        info = yf_ticker.info or {}
    except Exception:
        info = {}

    return TickerProfile(
        ticker=nse_ticker,
        all_time_high=ath, all_time_high_date=ath_date,
        all_time_low=atl, all_time_low_date=atl_date,
        yearly_metrics=compute_yearly_metrics(daily_df),
        dividends=dividends, splits=splits,
        earnings_dates=earnings_dates, info=info,
    )


def format_profile(profile: TickerProfile, n_years: int = 5) -> str:
    """Human-readable summary -- the CLI/report view of a TickerProfile."""
    lines = [f"=== {profile.ticker} ==="]

    name = profile.info.get("longName", profile.ticker)
    sector = profile.info.get("sector", "?")
    industry = profile.info.get("industry", "?")
    lines.append(f"{name} -- {sector} / {industry}")

    market_cap = profile.info.get("marketCap")
    if market_cap:
        lines.append(f"Market cap: Rs {market_cap / 1e7:,.0f} crore")
    pe = profile.info.get("trailingPE")
    if pe:
        lines.append(f"Trailing P/E: {pe:.1f}")

    if profile.all_time_high is not None:
        lines.append(f"All-time high: {profile.all_time_high:.2f} on {profile.all_time_high_date.date()}")
        lines.append(f"All-time low:  {profile.all_time_low:.2f} on {profile.all_time_low_date.date()}")
    else:
        lines.append("All-time high/low: unavailable")

    if profile.yearly_metrics:
        lines.append(f"\nYearly (last {n_years}):")
        for m in profile.yearly_metrics[-n_years:]:
            lines.append(
                f"  {m.year}: O={m.open:.2f} H={m.high:.2f} L={m.low:.2f} C={m.close:.2f} "
                f"return={m.return_pct:+.1f}% range={m.range_pct:.1f}%"
            )

    if profile.dividends is not None and len(profile.dividends):
        last_date = profile.dividends.index[-1]
        lines.append(f"\nDividends: {len(profile.dividends)} recorded, most recent {last_date.date()} = {profile.dividends.iloc[-1]}")
    else:
        lines.append("\nDividends: none recorded")

    if profile.splits is not None and len(profile.splits):
        history = ", ".join(f"{d.date()} ({r:.0f}:1)" for d, r in profile.splits.items())
        lines.append(f"Splits: {len(profile.splits)} recorded -- {history}")
    else:
        lines.append("Splits: none recorded")

    if profile.earnings_dates is not None and not profile.earnings_dates.empty:
        lines.append(f"\nEarnings history: {len(profile.earnings_dates)} entries available (dates + EPS estimate/reported/surprise)")
        lines.append("  NOTE: reported EPS is split-adjusted to match this project's price series, not the nominal figure originally reported.")
    else:
        lines.append("\nEarnings history: not available for this ticker")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    profile = fetch_ticker_profile(ticker)
    print(format_profile(profile))

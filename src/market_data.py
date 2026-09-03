"""
market_data.py
Lightweight LTP/change quote service -- deliberately separate from
data_loader.py's full OHLC history fetching.

WHY THIS IS A SEPARATE MODULE (see app.py's architecture rule in its own
docstring): the header market strip and the screener's live price column
need to refresh every ~20s without triggering a full historical-candle
download + SMC re-analysis for 50 tickers on every tick. This module only
ever asks Yahoo for the last couple of daily bars (one batched request for
however many symbols are requested) and returns LTP/change -- nothing here
computes structure, POIs, or a score. That stays entirely in
structure_engine.py/poi_engine.py/screener.py, run on a much slower,
explicitly-triggered cadence.

Freshness is intraday-daily-bar freshness, not tick-level: yfinance's own
daily bar for "today" updates periodically during the session, not on
every trade. Callers should label this honestly (e.g. "Updated HH:MM:SS
IST", not "LIVE tick data") -- see app.py's render_market_strip().
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from data_loader import NSE_TZ as _NSE_TZ_NAME, normalize_nse_ticker

NSE_TZ = ZoneInfo(_NSE_TZ_NAME)   # data_loader.NSE_TZ is a plain tz *name* string (pandas accepts that); real code needs an actual tzinfo


@dataclass
class Quote:
    label: str          # display name, e.g. "NIFTY 50" or "RELIANCE"
    symbol: str          # underlying yahoo symbol, e.g. "^NSEI" or "RELIANCE.NS"
    ltp: float
    prev_close: float
    change: float
    change_pct: float
    as_of: datetime       # timestamp of the underlying bar this LTP came from


class QuoteFetchError(RuntimeError):
    """Raised when the underlying batched quote request fails entirely
    (network down, provider outage, etc). Callers should catch this and
    fall back to the last known-good quotes, clearly labeled stale --
    never silently keep showing an old price as if it were current."""


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str, single: bool) -> pd.DataFrame | None:
    if single:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        return raw[symbol]
    return None


def fetch_quotes(symbols: dict[str, str]) -> dict[str, Quote]:
    """
    symbols: {display_label: yahoo_symbol}, e.g. {"NIFTY 50": "^NSEI"} or
    {"RELIANCE": "RELIANCE.NS", ...}. One single batched yf.download() call
    for every symbol requested, regardless of count.

    Returns {label: Quote} -- a symbol yfinance couldn't resolve is simply
    omitted (never fabricated), so callers should check for missing keys.

    Raises QuoteFetchError if the batched request fails outright (e.g. no
    network) -- as opposed to a partial/empty response for one symbol,
    which just omits that symbol.
    """
    if not symbols:
        return {}

    tickers = list(symbols.values())
    try:
        raw = yf.download(
            tickers, period="5d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker", threads=True,
        )
    except Exception as exc:
        raise QuoteFetchError(f"Batched quote fetch failed for {len(tickers)} symbol(s): {exc}") from exc

    if raw is None or raw.empty:
        raise QuoteFetchError(f"No data returned for {len(tickers)} symbol(s) -- provider may be unreachable.")

    single = len(tickers) == 1
    quotes: dict[str, Quote] = {}
    for label, symbol in symbols.items():
        sub = _extract_symbol_frame(raw, symbol, single)
        if sub is None or "Close" not in sub.columns:
            continue
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        last = sub.iloc[-1]
        prev = sub.iloc[-2] if len(sub) > 1 else last
        ltp = float(last["Close"])
        prev_close = float(prev["Close"])
        change = ltp - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        as_of = sub.index[-1]
        as_of = as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else as_of
        quotes[label] = Quote(
            label=label, symbol=symbol, ltp=ltp, prev_close=prev_close,
            change=change, change_pct=change_pct, as_of=as_of,
        )
    return quotes


# Yahoo Finance symbols for the major NSE/BSE benchmark indices this
# project's users actually watch. NIFTY FIN SERVICE's Yahoo symbol
# (^CNXFIN) is less consistently populated than the other three -- if
# yfinance ever stops returning it, fetch_quotes() just omits it rather
# than erroring the whole strip out.
INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY FIN SERVICE": "^CNXFIN",
}


def fetch_index_quotes() -> dict[str, Quote]:
    return fetch_quotes(INDEX_SYMBOLS)


def fetch_ticker_quotes(tickers: list[str]) -> dict[str, Quote]:
    """tickers: bare NSE symbols, e.g. ["RELIANCE", "TCS"] (".NS" added automatically).
    Returns {bare_ticker: Quote}, and Quote.label is that SAME bare ticker
    too (not the yahoo symbol) -- fetch_quotes() labels each Quote by
    whatever key its `symbols` dict used, so passing {bare: yahoo_symbol}
    directly (rather than a symbol->symbol map re-keyed afterward) gets
    both right in one pass."""
    return fetch_quotes({t: normalize_nse_ticker(t) for t in tickers})


def now_ist() -> datetime:
    return datetime.now(timezone.utc).astimezone(NSE_TZ)

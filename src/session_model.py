"""
session_model.py
Daily-session models: which extreme of the day's range prints first
(the "OHLC vs OLHC" read), and the Power-of-Three read of a session --
Accumulation (an opening range), Manipulation (a sweep beyond it), and
Distribution (the real move, opposite the sweep).

Unlike structure_engine.py, nothing here is swing-based -- these models
key off the CALENDAR DAY boundary and a session's own opening range,
not a fractal swing point. Both require INTRADAY data (1h, 15min, or a
resampled timeframe like 4h) to know how a day's own high/low unfolded
in time; a single daily/weekly/monthly bar carries no such ordering and
both functions here are no-ops on it (each "day" is one candle, so
there's nothing to sequence).

IMPORTANT ASYMMETRY between the two pieces here:
  - find_po3_setups() is a real, causal, backtestable signal: the
    manipulation sweep is known the moment it happens, so a trade can
    actually be entered off it in real time.
  - classify_daily_range_formation() is NOT a forward signal on its own
    -- you only know whether a day was OLHC or OHLC once BOTH extremes
    have printed, which is usually not until the day is over. Using it
    to predict that SAME day's direction would be lookahead. It's
    included here as a descriptive/diagnostic lens (e.g. for checking
    whether a day's formation type is coherent with how it closed, or
    for next-day research), not as something to plug into the
    backtester as an entry trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class DailyRangeFormation(Enum):
    OLHC = "olhc"   # low printed first, then the high -- a "buy day" read
    OHLC = "ohlc"   # high printed first, then the low -- a "sell day" read
    UNDETERMINED = "undetermined"  # high and low landed in the same candle


@dataclass
class PO3Setup:
    day: pd.Timestamp
    accumulation_low: float
    accumulation_high: float
    manipulation_index: int      # the candle whose wick swept beyond the accumulation range
    manipulation_direction: str   # "bullish" or "bearish" -- direction of the SWEEP (the fakeout side)
    distribution_direction: str   # the opposite -- the expected real move
    entry_index: int              # the candle right after manipulation; its close is the entry
    entry_price: float
    stop_price: float             # the manipulation candle's own wick extreme


def _day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # NOT index.normalize().values -- .values on a tz-aware DatetimeIndex
    # strips the timezone into naive UTC datetime64, which then fails to
    # equal a tz-aware pd.Timestamp day key. Keep it as a DatetimeIndex.
    return index.normalize()


def classify_daily_range_formation(df: pd.DataFrame) -> dict:
    """
    Groups intraday candles by calendar day and determines, for each
    day, whether the day's HIGH or LOW printed first in time. See the
    module docstring's ASYMMETRY note -- this is descriptive, not a
    same-day forward signal.

    Returns {day_timestamp: DailyRangeFormation}.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {}

    days = _day_key(df.index)
    highs, lows = df["high"].values, df["low"].values

    result = {}
    for day in pd.unique(days):
        mask = days == day
        day_highs = highs[mask]
        day_lows = lows[mask]
        high_pos = int(day_highs.argmax())
        low_pos = int(day_lows.argmin())

        if high_pos == low_pos:
            result[day] = DailyRangeFormation.UNDETERMINED
        elif low_pos < high_pos:
            result[day] = DailyRangeFormation.OLHC
        else:
            result[day] = DailyRangeFormation.OHLC

    return result


def find_po3_setups(df: pd.DataFrame, opening_range_bars: int = 1) -> list[PO3Setup]:
    """
    For each calendar day: the first `opening_range_bars` candles
    define the day's ACCUMULATION range. Scans the REST of that day's
    candles for the first one that wicks beyond the range without
    closing beyond it -- the MANIPULATION sweep, exactly IDM's own
    wick-not-close rule (structure_engine.detect_structure_events),
    just anchored to the session's opening range instead of a swing
    point. The candle right after it is the entry point for the
    anticipated DISTRIBUTION move in the opposite direction, stopped
    beyond the manipulation candle's own wick.

    A day contributes at most one PO3 setup -- the first manipulation
    found each day, matching how a trader would only take the first
    genuine stop-run per session, not every subsequent wick.

    Silently returns [] on daily/weekly/monthly data, where each "day"
    is a single candle and `len(day_positions) <= opening_range_bars`
    is always true -- there's no intraday opening range to define.

    Also a no-op (returns []) if df's index isn't a DatetimeIndex at
    all, since there's no calendar day to group by.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return []

    days = _day_key(df.index)
    opens = df["open"].values
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    setups = []

    for day in pd.unique(days):
        day_positions = np.flatnonzero(days == day)
        if len(day_positions) <= opening_range_bars:
            continue

        acc_positions = day_positions[:opening_range_bars]
        acc_low = float(lows[acc_positions].min())
        acc_high = float(highs[acc_positions].max())

        for j in day_positions[opening_range_bars:]:
            swept_high = highs[j] > acc_high and closes[j] <= acc_high
            swept_low = lows[j] < acc_low and closes[j] >= acc_low
            if not swept_high and not swept_low:
                continue

            manipulation_direction = "bullish" if swept_high else "bearish"
            # A bullish manipulation (swept the range's HIGH) implies
            # the real move -- distribution -- goes DOWN; a bearish
            # manipulation (swept the LOW) implies it goes UP.
            distribution_direction = "bearish" if manipulation_direction == "bullish" else "bullish"
            stop_price = float(highs[j]) if manipulation_direction == "bullish" else float(lows[j])

            entry_index = j + 1
            if entry_index >= n or days[entry_index] != day:
                break  # no candle left in this session to enter on

            setups.append(PO3Setup(
                day=day, accumulation_low=acc_low, accumulation_high=acc_high,
                manipulation_index=int(j), manipulation_direction=manipulation_direction,
                distribution_direction=distribution_direction,
                entry_index=int(entry_index), entry_price=float(closes[entry_index]), stop_price=stop_price,
            ))
            break  # one setup per day

    return setups

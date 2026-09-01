"""
structure_engine.py
The core algorithm: converts subjective SMC (Smart Money Concept) chart
reading into precise, testable rules.

Definitions used (confirmed with the project owner, a real trader):
  - Swing point: a fractal -- a candle whose high (for a swing high) or
    low (for a swing low) is more extreme than the N candles on either
    side of it. This is the raw material everything else is built from.
  - Structural bias: BULLISH if the swing sequence shows higher highs
    AND higher lows; BEARISH if lower highs AND lower lows; otherwise
    UNDETERMINED (ranging / unclear).
  - BOS (Break of Structure): a candle's CLOSE breaks beyond the most
    recent relevant swing point in the direction of the current bias --
    confirms trend CONTINUATION. Requires a body close, not just a wick
    (per project convention: "bos and choch are only valid when body
    closes").
  - CHoCH (Change of Character): a candle's CLOSE breaks beyond the
    most recent relevant swing point AGAINST the current bias -- signals
    a potential trend REVERSAL. Also requires a body close.
  - IDM (Inducement / liquidity grab): price WICKS beyond a minor swing
    point (sweeping the stops/liquidity resting there) WITHOUT closing
    beyond it, then reverses back the other way. Wick-based per project
    convention ("for idm we can take wick"), specifically because a
    liquidity grab is defined by price *reaching* a level to trigger
    stops, not by the candle *closing* there.

All of this operates on ONE timeframe's candle data at a time --
multi-timeframe orchestration (monthly -> weekly -> daily -> 4H -> 15min)
is handled separately in multi_timeframe.py, by running this same engine
on each timeframe's data independently.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNDETERMINED = "undetermined"


@dataclass
class SwingPoint:
    index: int          # row position in the candle dataframe
    price: float
    kind: str            # "high" or "low"
    is_minor: bool = False  # minor swings are candidates for IDM sweeps


@dataclass
class StructureEvent:
    index: int
    event_type: str       # "BOS" or "CHoCH" or "IDM"
    direction: str         # "bullish" or "bearish"
    price: float
    # The swing point this event originates from: for BOS/CHoCH, the
    # swing whose price got body-closed beyond; for IDM, the swing
    # whose wick got swept. Used to draw a line from that swing point
    # to the candle where the event fired (see chart.py).
    source_swing_index: int = None
    source_swing_price: float = None


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> list[SwingPoint]:
    """
    Fractal-based swing detection: a swing high at position i needs
    high[i] to be the strictly highest high among [i-lookback, i+lookback].
    Same logic (inverted) for swing lows. `lookback=3` is a reasonable
    default -- tighter values (e.g. 2) catch more, noisier swings;
    looser values (e.g. 5) catch fewer, more significant ones. This is
    tunable per timeframe (see multi_timeframe.py).
    """
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    swings = []

    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback: i + lookback + 1]
        window_lows = lows[i - lookback: i + lookback + 1]

        if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
            swings.append(SwingPoint(index=i, price=highs[i], kind="high"))
        if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
            swings.append(SwingPoint(index=i, price=lows[i], kind="low"))

    swings.sort(key=lambda s: s.index)
    return swings


def classify_bias(swings: list[SwingPoint]) -> Bias:
    """
    Looks at the most recent two swing highs and two swing lows: higher
    highs + higher lows = bullish structure; lower highs + lower lows =
    bearish. Anything else (e.g. a higher high but a lower low) is
    genuinely ambiguous structure, not a coin-flip -- reported as such
    rather than forced into a direction.
    """
    recent_highs = [s for s in swings if s.kind == "high"][-2:]
    recent_lows = [s for s in swings if s.kind == "low"][-2:]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return Bias.UNDETERMINED

    higher_highs = recent_highs[-1].price > recent_highs[-2].price
    higher_lows = recent_lows[-1].price > recent_lows[-2].price
    lower_highs = recent_highs[-1].price < recent_highs[-2].price
    lower_lows = recent_lows[-1].price < recent_lows[-2].price

    if higher_highs and higher_lows:
        return Bias.BULLISH
    if lower_highs and lower_lows:
        return Bias.BEARISH
    return Bias.UNDETERMINED


def detect_structure_events(df: pd.DataFrame, swings: list[SwingPoint]) -> tuple[list[StructureEvent], "Bias"]:
    """
    Walks forward through the candles, tracking the prevailing bias and
    the nearest still-unbroken swing high/low, and flags BOS, CHoCH,
    and IDM events as they occur.

    IMPORTANT DESIGN POINT: bias is a STATE that only changes when a
    CHoCH actually occurs -- it is NOT recomputed from scratch every
    candle by re-examining raw swing patterns. This matches how SMC
    actually works: CHoCH *is* the event that flips the bias; BOS is
    the event that confirms the CURRENT bias is continuing. An earlier
    version of this function recalculated bias fresh every candle from
    the last two swing highs/lows, which made bias flip erratically on
    every minor pullback swing and produced far too many false CHoCH
    signals -- a good example of a subtle-but-important correctness bug.

    Returns both the event list and the final bias, since the bias at
    the end of the loop naturally IS "the current bias."
    """
    events = []
    closes = df["close"].values
    highs, lows = df["high"].values, df["low"].values

    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    broken_high_idx = set()
    broken_low_idx = set()
    swept_high_idx = set()
    swept_low_idx = set()

    current_bias = Bias.UNDETERMINED

    for i in range(len(df)):
        prior_highs = [s for s in swing_highs if s.index < i and s.index not in broken_high_idx]
        prior_lows = [s for s in swing_lows if s.index < i and s.index not in broken_low_idx]
        if not prior_highs or not prior_lows:
            continue

        nearest_high = prior_highs[-1]
        nearest_low = prior_lows[-1]

        # --- BOS / CHoCH: candle body CLOSE breaking the nearest unbroken swing ---
        if closes[i] > nearest_high.price:
            if current_bias == Bias.BULLISH:
                event_type = "BOS"  # continuing an already-bullish structure
            else:
                event_type = "CHoCH"  # was bearish/undetermined -- this break FLIPS the bias
                current_bias = Bias.BULLISH
            events.append(StructureEvent(
                index=i, event_type=event_type, direction="bullish", price=closes[i],
                source_swing_index=nearest_high.index, source_swing_price=nearest_high.price,
            ))
            broken_high_idx.add(nearest_high.index)

        elif closes[i] < nearest_low.price:
            if current_bias == Bias.BEARISH:
                event_type = "BOS"
            else:
                event_type = "CHoCH"
                current_bias = Bias.BEARISH
            events.append(StructureEvent(
                index=i, event_type=event_type, direction="bearish", price=closes[i],
                source_swing_index=nearest_low.index, source_swing_price=nearest_low.price,
            ))
            broken_low_idx.add(nearest_low.index)

        # --- IDM: wick sweeps the nearest unbroken swing WITHOUT a body close beyond it ---
        if (nearest_high.index not in swept_high_idx
                and highs[i] > nearest_high.price and closes[i] <= nearest_high.price):
            events.append(StructureEvent(
                index=i, event_type="IDM", direction="bearish", price=highs[i],
                source_swing_index=nearest_high.index, source_swing_price=nearest_high.price,
            ))
            swept_high_idx.add(nearest_high.index)

        if (nearest_low.index not in swept_low_idx
                and lows[i] < nearest_low.price and closes[i] >= nearest_low.price):
            events.append(StructureEvent(
                index=i, event_type="IDM", direction="bullish", price=lows[i],
                source_swing_index=nearest_low.index, source_swing_price=nearest_low.price,
            ))
            swept_low_idx.add(nearest_low.index)

    return events, current_bias


def analyze_structure(df: pd.DataFrame, lookback: int = 3) -> dict:
    """Convenience wrapper: runs the full pipeline and returns everything
    needed for both visualization and the current top-line bias."""
    swings = find_swing_points(df, lookback=lookback)
    events, current_bias = detect_structure_events(df, swings)

    last_event = events[-1] if events else None

    return {
        "swings": swings,
        "events": events,
        "current_bias": current_bias,
        "last_event": last_event,
    }

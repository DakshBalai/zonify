"""
poi_engine.py
Points of Interest: Fair Value Gaps, Order Blocks, and Mitigation Blocks.

This is layered ON TOP of structure_engine.py -- it consumes swings and
StructureEvents from analyze_structure() rather than re-deriving its own
notion of structure, so there's exactly one source of truth for "what
counts as a break" or "what counts as a swing" across the whole project.

Definitions used (confirmed with the project owner, a real trader):

  - FVG (Fair Value Gap): the classic 3-candle imbalance. For candles
    at positions i-2, i-1, i:
      bullish FVG: low[i] > high[i-2]   -- zone = (high[i-2], low[i])
      bearish FVG: high[i] < low[i-2]   -- zone = (high[i], low[i-2])
    Any nonzero gap is detected, but only flagged VALID if it sits in
    the "right" half of the current dealing range: a bullish FVG must
    be in DISCOUNT (below the 50% equilibrium of the most recent swing
    high<->swing low range), a bearish FVG must be in PREMIUM (above
    it) -- since a bullish FVG in premium (or vice versa) is priced
    "expensive"/"cheap" in the wrong direction and is far less likely
    to actually cause the reaction the concept predicts.

  - Order Block: the last opposite-colored candle immediately before
    the impulsive run that causes a BOS or CHoCH. Zone = that candle's
    full high-low range (not just the body). One OB per BOS/CHoCH
    event -- built directly off structure_engine's already-verified
    event detection, not a separate "impulse" detector.

  - Mitigation Block: same idea as an Order Block, but for a FAILED
    attempt rather than a successful one -- the last opposite-colored
    candle before a push toward a swing level that does NOT close
    beyond it (no BOS/CHoCH) and then reverses. This is exactly what
    structure_engine's IDM events already capture (wick sweep, no
    body close beyond, then reversal), so Mitigation Blocks are built
    off IDM events the same way Order Blocks are built off BOS/CHoCH.

  - Mitigation (the verb): a POI (any of the three above) is
    considered MITIGATED the first time price trades back into its
    zone at all (any overlap between a later candle's high-low range
    and the zone) -- at that point it's "used up" and no longer
    treated as active.

  - Premium/Discount equilibrium: computed from the CURRENT DEALING
    RANGE, defined as the most recent swing high <-> most recent swing
    low as of the candle being evaluated -- using only swings known up
    to that point (no lookahead), since this same range calculation
    will later feed the backtester and must never see the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FVG:
    start_index: int   # index of the first candle of the 3-candle pattern (i-2)
    end_index: int      # index of the third candle (i) -- the gap "confirms" here
    direction: str        # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    valid: bool             # True if in the correct premium/discount half
    mitigated: bool = False
    mitigated_index: int = None


@dataclass
class OrderBlock:
    index: int             # the OB candle itself
    direction: str            # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    source_event_index: int   # the BOS/CHoCH event this OB was derived from
    mitigated: bool = False
    mitigated_index: int = None


@dataclass
class MitigationBlock:
    index: int
    direction: str
    zone_low: float
    zone_high: float
    source_event_index: int   # the IDM event this MB was derived from
    mitigated: bool = False
    mitigated_index: int = None


# ---------------------------------------------------------------------------
# Premium / Discount
# ---------------------------------------------------------------------------

def compute_dealing_range(swings: list, as_of_index: int):
    """
    Most recent swing high <-> most recent swing low, using ONLY swings
    with index < as_of_index (strictly before the candle being
    evaluated) -- deliberately excludes same-bar and future swings so
    this can never leak lookahead into FVG validity or, later, the
    backtester.

    Returns (range_low, range_high, equilibrium) or None if there
    isn't yet at least one swing high and one swing low available.
    """
    prior_highs = [s for s in swings if s.kind == "high" and s.index < as_of_index]
    prior_lows = [s for s in swings if s.kind == "low" and s.index < as_of_index]
    if not prior_highs or not prior_lows:
        return None

    latest_high = prior_highs[-1].price
    latest_low = prior_lows[-1].price
    range_low, range_high = min(latest_high, latest_low), max(latest_high, latest_low)
    equilibrium = (range_low + range_high) / 2
    return range_low, range_high, equilibrium


def _zone_side(zone_low: float, zone_high: float, equilibrium: float) -> str:
    """'premium' if the zone's midpoint sits above equilibrium, else 'discount'."""
    zone_mid = (zone_low + zone_high) / 2
    return "premium" if zone_mid > equilibrium else "discount"


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

def find_fvgs(df: pd.DataFrame, swings: list) -> list[FVG]:
    highs, lows = df["high"].values, df["low"].values
    fvgs = []

    for i in range(2, len(df)):
        # bullish: candle i's low is above candle (i-2)'s high -> gap
        if lows[i] > highs[i - 2]:
            zone_low, zone_high = highs[i - 2], lows[i]
            dr = compute_dealing_range(swings, as_of_index=i)
            valid = dr is not None and _zone_side(zone_low, zone_high, dr[2]) == "discount"
            fvgs.append(FVG(
                start_index=i - 2, end_index=i, direction="bullish",
                zone_low=zone_low, zone_high=zone_high, valid=valid,
            ))

        # bearish: candle i's high is below candle (i-2)'s low -> gap
        if highs[i] < lows[i - 2]:
            zone_low, zone_high = highs[i], lows[i - 2]
            dr = compute_dealing_range(swings, as_of_index=i)
            valid = dr is not None and _zone_side(zone_low, zone_high, dr[2]) == "premium"
            fvgs.append(FVG(
                start_index=i - 2, end_index=i, direction="bearish",
                zone_low=zone_low, zone_high=zone_high, valid=valid,
            ))

    return fvgs


# ---------------------------------------------------------------------------
# Order Blocks / Mitigation Blocks -- shared "last opposite candle" scan
# ---------------------------------------------------------------------------

def _find_last_opposite_candle(df: pd.DataFrame, before_index: int, impulse_direction: str) -> int | None:
    """
    Scans backward from `before_index - 1`, skipping candles that match
    the impulsive move's own color, and returns the index of the first
    candle found with the OPPOSITE color -- that's the order-block /
    mitigation-block candle. Returns None if it runs off the start of
    the data without finding one (can happen near the very start of
    the dataset).

    impulse_direction: "bullish" (impulse candles close > open) or
    "bearish" (impulse candles close < open).
    """
    opens, closes = df["open"].values, df["close"].values
    i = before_index - 1
    while i >= 0:
        is_bullish_candle = closes[i] > opens[i]
        matches_impulse = is_bullish_candle if impulse_direction == "bullish" else not is_bullish_candle
        if not matches_impulse:
            return i
        i -= 1
    return None


def find_order_blocks(df: pd.DataFrame, events: list) -> list[OrderBlock]:
    """One Order Block per BOS/CHoCH event, per project convention."""
    highs, lows = df["high"].values, df["low"].values
    blocks = []

    for event in events:
        if event.event_type not in ("BOS", "CHoCH"):
            continue
        ob_index = _find_last_opposite_candle(df, event.index, event.direction)
        if ob_index is None:
            continue
        blocks.append(OrderBlock(
            index=ob_index, direction=event.direction,
            zone_low=lows[ob_index], zone_high=highs[ob_index],
            source_event_index=event.index,
        ))
    return blocks


def find_mitigation_blocks(df: pd.DataFrame, events: list) -> list[MitigationBlock]:
    """
    One Mitigation Block per IDM event -- an IDM is, by construction,
    a failed break attempt (wick beyond a swing, no close beyond, then
    reversal), which is exactly what a Mitigation Block's origin is.
    """
    highs, lows = df["high"].values, df["low"].values
    blocks = []

    for event in events:
        if event.event_type != "IDM":
            continue
        mb_index = _find_last_opposite_candle(df, event.index, event.direction)
        if mb_index is None:
            continue
        blocks.append(MitigationBlock(
            index=mb_index, direction=event.direction,
            zone_low=lows[mb_index], zone_high=highs[mb_index],
            source_event_index=event.index,
        ))
    return blocks


# ---------------------------------------------------------------------------
# Mitigation tracking (shared across FVG / OB / MB)
# ---------------------------------------------------------------------------

def apply_mitigation(df: pd.DataFrame, pois: list, formation_index_attr: str = "end_index") -> None:
    """
    Mutates each POI in place: sets .mitigated=True and .mitigated_index
    to the first later candle whose high-low range overlaps the POI's
    zone at all. Works for FVG (formation_index_attr="end_index"),
    OrderBlock/MitigationBlock (formation_index_attr="index") -- pass
    the right attribute name for the list you're processing.
    """
    highs, lows = df["high"].values, df["low"].values
    n = len(df)

    for poi in pois:
        formed_at = getattr(poi, formation_index_attr)
        for j in range(formed_at + 1, n):
            if lows[j] <= poi.zone_high and highs[j] >= poi.zone_low:
                poi.mitigated = True
                poi.mitigated_index = j
                break


def analyze_poi(df: pd.DataFrame, swings: list, events: list) -> dict:
    """Convenience wrapper: runs FVG/OB/MB detection + mitigation tracking."""
    fvgs = find_fvgs(df, swings)
    order_blocks = find_order_blocks(df, events)
    mitigation_blocks = find_mitigation_blocks(df, events)

    apply_mitigation(df, fvgs, formation_index_attr="end_index")
    apply_mitigation(df, order_blocks, formation_index_attr="index")
    apply_mitigation(df, mitigation_blocks, formation_index_attr="index")

    return {
        "fvgs": fvgs,
        "order_blocks": order_blocks,
        "mitigation_blocks": mitigation_blocks,
    }

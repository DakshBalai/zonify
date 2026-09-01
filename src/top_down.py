"""
top_down.py
The top-down HTF-zone -> LTF-entry workflow: mark Points of Interest on
a higher timeframe, tagged with which timeframe and POI type they came
from, then drill down to a lower timeframe and only take an entry where
a Market Structure Shift (MSS -- a CHoCH) plus the Fair Value Gap it
produces both occur INSIDE one of those HTF zones.

Standard drill-down pairs (per project owner):
    daily -> 1h, 30min
    4h    -> 15min, 5min
    1h    -> 5min, 1min

This module invents no new detection primitive -- HTFZone is just a
timestamped, tagged wrapper around POIs analyze_poi() already produces,
and find_top_down_entries() is a FILTER over FVGs analyze_poi() already
produces on the lower timeframe. The actual trade (entry/stop/target)
is still simulated by backtester.backtest_pois() exactly like a plain
FVG trade -- this module only decides WHICH FVGs qualify, so a result
here can be compared directly against the plain "FVG" baseline.

Cross-timeframe alignment is done by TIMESTAMP, not row index -- an HTF
and LTF DataFrame covering the same calendar period have different row
counts, so "was this HTF zone already formed (and not yet mitigated)
at the moment of this LTF event" has to be asked in calendar time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from structure_engine import Bias

# HTF timeframe -> the LTF timeframes to drill down into for entries.
HTF_TO_LTF = {
    "daily": ["1h", "30min"],
    "4h": ["15min", "5min"],
    "1h": ["5min", "1min"],
}

# MitigationBlock is deliberately excluded -- the backtester's own
# accumulated evidence (across every timeframe and ticker tested) is
# that it's a consistently negative-expectancy signal; treating it as
# an HTF "zone of interest" would inject a known-weak filter into the
# top-down entry logic rather than a genuine confluence.
HTF_ZONE_POI_TYPES = ("fvgs", "order_blocks", "extreme_order_blocks", "breaker_blocks")
_POI_TYPE_LABELS = {
    "fvgs": "FVG",
    "order_blocks": "OrderBlock",
    "extreme_order_blocks": "ExtremeOB",
    "breaker_blocks": "BreakerBlock",
}


@dataclass
class HTFZone:
    timeframe: str          # e.g. "daily", "4h", "1h"
    poi_type: str             # "FVG" / "OrderBlock" / "ExtremeOB" / "BreakerBlock"
    direction: str             # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    formed_at: pd.Timestamp
    mitigated_at: pd.Timestamp | None   # None if still live as of the end of the HTF data


def _poi_formation_index(poi) -> int:
    formed_at = getattr(poi, "end_index", None)
    return formed_at if formed_at is not None else poi.index


def collect_htf_zones(htf_df: pd.DataFrame, structure_result: dict, poi_result: dict, timeframe: str) -> list[HTFZone]:
    """
    Builds the timestamped, tagged "zones of interest" list from one
    higher timeframe's already-computed structure + POI: every
    FVG/OrderBlock/ExtremeOB/BreakerBlock ALIGNED WITH THIS TIMEFRAME'S
    OWN CURRENT BIAS -- the top-down premise is that you only draw on
    zones in the direction you already believe the HTF is heading, not
    every zone regardless of direction.

    Only VALID FVGs are included (an invalid one is a known-wrong-side
    setup, same rule the backtester already applies elsewhere).
    Returns [] if the HTF has no determined bias yet.
    """
    bias = structure_result["current_bias"]
    if bias == Bias.UNDETERMINED:
        return []
    bias_direction = bias.value

    zones = []
    for key in HTF_ZONE_POI_TYPES:
        pois = poi_result.get(key, [])
        if key == "fvgs":
            pois = [p for p in pois if p.valid]
        for poi in pois:
            if poi.direction != bias_direction:
                continue
            formed_at = htf_df.index[_poi_formation_index(poi)]
            mitigated_at = htf_df.index[poi.mitigated_index] if poi.mitigated else None
            zones.append(HTFZone(
                timeframe=timeframe, poi_type=_POI_TYPE_LABELS[key], direction=poi.direction,
                zone_low=poi.zone_low, zone_high=poi.zone_high,
                formed_at=formed_at, mitigated_at=mitigated_at,
            ))
    return zones


def find_top_down_entries(
    ltf_df: pd.DataFrame,
    ltf_structure_result: dict,
    ltf_fvgs: list,
    htf_zones: list,
    max_lookback_bars: int = 20,
    zone_lookback_bars: int = 40,
) -> list:
    """
    Filters ltf_fvgs down to the "MSS + FVG" top-down entry combo: an
    FVG qualifies only if BOTH of these hold --

      1. It formed within `max_lookback_bars` candles AFTER a CHoCH (a
         Market Structure Shift) of the SAME direction on this same LTF
         timeframe -- i.e. it's part of the impulsive leg the shift
         produced, not a coincidentally-timed, unrelated gap.
      2. Within the `zone_lookback_bars` candles leading up to that
         CHoCH, LTF price actually TRADED INTO an HTF zone of the same
         direction that was already formed and not yet mitigated at
         that moment (checked by real timestamp, since HTF and LTF bars
         don't share a row-index scale) -- i.e. the shift happened
         shortly after price swept into a higher-timeframe zone, not
         somewhere unrelated on the chart.

    Condition 2 checks candle high/low OVERLAP with the zone (the same
    touch test apply_mitigation() uses elsewhere), not whether some
    single fractal swing's exact pivot price lands inside it -- the
    latter is far too strict a coincidence to require in practice (an
    HTF zone can be a narrow band, and a confirmed swing's precise
    extreme landing inside it is a much rarer event than price having
    simply traded through it on the way to reversing).

    Returns the qualifying subset of ltf_fvgs -- feed it straight into
    backtester.backtest_pois() exactly like a plain FVG list, so a
    result is directly comparable against the unfiltered "FVG" baseline.
    """
    events = ltf_structure_result["events"]
    choch_events = [e for e in events if e.event_type == "CHoCH"]
    lows, highs = ltf_df["low"].values, ltf_df["high"].values

    qualifying = []
    for fvg in ltf_fvgs:
        candidates = [
            c for c in choch_events
            if c.direction == fvg.direction and c.index <= fvg.start_index <= c.index + max_lookback_bars
        ]
        if not candidates:
            continue
        choch = max(candidates, key=lambda c: c.index)  # nearest preceding CHoCH

        matching_zones = [z for z in htf_zones if z.direction == fvg.direction]
        if not matching_zones:
            continue

        window_start = max(0, choch.index - zone_lookback_bars)
        touched_a_zone = False
        for k in range(window_start, choch.index + 1):
            candle_time = ltf_df.index[k]
            for zone in matching_zones:
                if candle_time < zone.formed_at:
                    continue  # zone didn't exist yet -- no lookahead
                if zone.mitigated_at is not None and candle_time > zone.mitigated_at:
                    continue  # zone was already used up by this candle
                if lows[k] <= zone.zone_high and highs[k] >= zone.zone_low:
                    touched_a_zone = True
                    break
            if touched_a_zone:
                break

        if touched_a_zone:
            qualifying.append(fvg)

    return qualifying

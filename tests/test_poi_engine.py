"""
test_poi_engine.py
Every test here uses hand-constructed candles where the "correct"
answer is known in advance -- same rigor as test_data_loader.py and
the original structure_engine verification. No synthetic-random data
here on purpose: for exact geometric rules like FVG gaps and premium/
discount sides, we want to know precisely what SHOULD be detected,
not just that something plausible came out.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import SwingPoint, StructureEvent  # noqa: E402
from poi_engine import (  # noqa: E402
    compute_dealing_range,
    find_fvgs,
    find_order_blocks,
    find_mitigation_blocks,
    apply_mitigation,
)


def candles(rows):
    """rows: list of (open, high, low, close) tuples -> OHLCV DataFrame."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000
    return df


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------

def test_bullish_fvg_detected_with_correct_zone():
    # candle0 high=10, candle1 whatever, candle2 low=12 -> gap (10,12)
    df = candles([
        (9, 10, 8, 9.5),    # i=0
        (9.5, 13, 9.5, 12.5),  # i=1 (impulsive)
        (12.5, 14, 12, 13.5),  # i=2, low(12) > high[0](10) -> bullish FVG
    ])
    fvgs = find_fvgs(df, swings=[])  # no swings -> dealing range unknown -> valid=False, but zone must still be right
    bullish = [f for f in fvgs if f.direction == "bullish"]
    assert len(bullish) == 1, bullish
    f = bullish[0]
    assert f.start_index == 0 and f.end_index == 2
    assert f.zone_low == 10 and f.zone_high == 12
    print("PASS: bullish FVG detected with correct zone")


def test_bearish_fvg_detected_with_correct_zone():
    # candle0 low=12, candle2 high=10 -> gap (10,12), bearish
    df = candles([
        (13, 14, 12, 12.5),
        (12.5, 12.5, 9, 9.5),
        (9.5, 10, 8, 8.5),   # high(10) < low[0](12) -> bearish FVG
    ])
    fvgs = find_fvgs(df, swings=[])
    bearish = [f for f in fvgs if f.direction == "bearish"]
    assert len(bearish) == 1, bearish
    f = bearish[0]
    assert f.start_index == 0 and f.end_index == 2
    assert f.zone_low == 10 and f.zone_high == 12
    print("PASS: bearish FVG detected with correct zone")


def test_no_fvg_when_candles_overlap():
    # Overlapping ranges, no gap anywhere
    df = candles([
        (10, 11, 9, 10.5),
        (10.5, 11.5, 9.5, 10),
        (10, 11, 9, 10.5),
    ])
    fvgs = find_fvgs(df, swings=[])
    assert fvgs == [], fvgs
    print("PASS: no false-positive FVG on overlapping candles")


def test_fvg_premium_discount_validity():
    # Build a dealing range: swing low=100 (index 0), swing high=120 (index 5)
    # equilibrium = 110. A bullish FVG below 110 -> valid. Above -> invalid.
    # The FVG must form AFTER index 5 so the dealing range is actually
    # known at that point (no-lookahead: a swing at index 5 isn't
    # "seen" by compute_dealing_range until as_of_index > 5).
    swings = [
        SwingPoint(index=0, price=100.0, kind="low"),
        SwingPoint(index=5, price=120.0, kind="high"),
    ]

    # 6 filler candles (indices 0-5) just to get past the swings, then
    # the 3-candle FVG pattern at indices 6,7,8.
    filler = [(110, 111, 109, 110.5)] * 6

    # Bullish FVG entirely in discount (zone 103-105, well below equilibrium=110)
    df_discount = candles(filler + [
        (102, 103, 101, 102.5),   # i=6
        (102.5, 106, 102, 105.5),  # i=7
        (105.5, 108, 105, 107),    # i=8, low=105 > high[6]=103 -> bullish FVG zone (103,105), mid=104 < 110 -> discount
    ])
    fvgs = find_fvgs(df_discount, swings=swings)
    bullish = [f for f in fvgs if f.direction == "bullish" and f.start_index == 6]
    assert len(bullish) == 1, bullish
    assert bullish[0].valid is True, "bullish FVG in discount should be valid"

    # Bullish FVG entirely in premium (zone well above 110, e.g. 116-118)
    df_premium = candles(filler + [
        (115, 116, 114, 115.5),   # i=6
        (115.5, 119, 115, 118.5),  # i=7
        (118.5, 121, 118, 120),    # i=8, low=118 > high[6]=116 -> bullish FVG zone (116,118), mid=117 > 110 -> premium
    ])
    fvgs2 = find_fvgs(df_premium, swings=swings)
    bullish2 = [f for f in fvgs2 if f.direction == "bullish" and f.start_index == 6]
    assert len(bullish2) == 1, bullish2
    assert bullish2[0].valid is False, "bullish FVG in premium should be INVALID"

    print("PASS: FVG premium/discount validity filter")


def test_dealing_range_excludes_lookahead():
    swings = [
        SwingPoint(index=0, price=100.0, kind="low"),
        SwingPoint(index=10, price=120.0, kind="high"),  # this is AFTER as_of_index=5
    ]
    # as_of_index=5: only the swing at index=0 exists yet (low=100), no
    # swing high yet at all -> dealing range must be None (not allowed
    # to see the index=10 high from the future)
    result = compute_dealing_range(swings, as_of_index=5)
    assert result is None, f"expected None (no lookahead), got {result}"

    result2 = compute_dealing_range(swings, as_of_index=11)
    assert result2 is not None
    assert result2 == (100.0, 120.0, 110.0)
    print("PASS: dealing range correctly excludes future swings (no lookahead)")


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def test_order_block_finds_last_opposite_candle():
    # red, green, green, green(closes beyond swing high -> BOS at index 3)
    df = candles([
        (10, 10.2, 9.5, 9.6),   # i=0 red (close<open)
        (9.6, 10.5, 9.6, 10.4),  # i=1 green
        (10.4, 11.2, 10.4, 11.0),  # i=2 green
        (11.0, 12.0, 11.0, 11.8),  # i=3 green, this is the BOS close candle
    ])
    events = [StructureEvent(index=3, event_type="BOS", direction="bullish", price=11.8,
                              source_swing_index=0, source_swing_price=10.5)]
    obs = find_order_blocks(df, events)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.index == 0, f"expected OB at the single red candle (index 0), got {ob.index}"
    assert ob.zone_low == 9.5 and ob.zone_high == 10.2
    print("PASS: order block finds last opposite candle before impulsive BOS run")


def test_order_block_skips_multiple_same_color_candles():
    # red, red, green, green, green(BOS at index 4) -> OB should be the
    # LAST red before the green run starts, i.e. index 1, not index 0
    df = candles([
        (10.5, 10.6, 10.0, 10.2),  # i=0 red
        (10.2, 10.3, 9.8, 9.9),    # i=1 red (this one, closest to the run)
        (9.9, 10.6, 9.9, 10.5),    # i=2 green
        (10.5, 11.2, 10.5, 11.0),  # i=3 green
        (11.0, 12.0, 11.0, 11.9),  # i=4 green, BOS close
    ])
    events = [StructureEvent(index=4, event_type="BOS", direction="bullish", price=11.9,
                              source_swing_index=0, source_swing_price=10.6)]
    obs = find_order_blocks(df, events)
    assert obs[0].index == 1, f"expected OB at index 1, got {obs[0].index}"
    print("PASS: order block correctly picks the LAST opposite candle, skipping through same-color run")


# ---------------------------------------------------------------------------
# Mitigation Blocks
# ---------------------------------------------------------------------------

def test_mitigation_block_derived_from_idm_event():
    # green, green -> IDM sweep candle at index 2 (bearish IDM = wicks
    # below a swing low without closing below, per structure_engine).
    # MB should be the last opposite (red... wait IDM direction bullish
    # means it swept a swing LOW, i.e. price dipped down -- impulse
    # direction into that sweep is bearish, so MB = last GREEN candle
    # before the bearish dip).
    df = candles([
        (10.0, 10.6, 9.9, 10.5),   # i=0 green
        (10.5, 10.6, 9.6, 9.7),    # i=1 red (impulse down toward the swing low)
        (9.7, 9.8, 9.3, 9.6),      # i=2 red, wicks below swing low but closes above it -> IDM candle
    ])
    events = [StructureEvent(index=2, event_type="IDM", direction="bullish", price=9.3,
                              source_swing_index=0, source_swing_price=9.9)]
    mbs = find_mitigation_blocks(df, events)
    assert len(mbs) == 1
    mb = mbs[0]
    # impulse_direction passed is event.direction="bullish" per our
    # convention (IDM event.direction describes the REVERSAL direction,
    # i.e. the sweep-down's aftermath) -- scan looks for the last
    # candle NOT matching "bullish", starting just before index 2.
    # index 1 is red (not bullish) -> immediately returned as MB.
    assert mb.index == 1, f"expected MB at index 1, got {mb.index}"
    print("PASS: mitigation block correctly derived from IDM event")


# ---------------------------------------------------------------------------
# Mitigation tracking
# ---------------------------------------------------------------------------

def test_apply_mitigation_flags_first_touch():
    from poi_engine import FVG
    fvg = FVG(start_index=0, end_index=2, direction="bullish", zone_low=10, zone_high=12, valid=True)
    df = candles([
        (9, 10, 8, 9.5),
        (9.5, 13, 9.5, 12.5),
        (12.5, 14, 12, 13.5),  # formation candle (end_index=2)
        (13.5, 14, 13, 13.8),  # i=3, no overlap with (10,12)
        (13.8, 13.9, 11, 11.5),  # i=4, low=11 dips into zone (10,12) -> mitigated here
        (11.5, 12, 10.5, 11.8),  # i=5
    ])
    apply_mitigation(df, [fvg], formation_index_attr="end_index")
    assert fvg.mitigated is True
    assert fvg.mitigated_index == 4, f"expected mitigation at index 4, got {fvg.mitigated_index}"
    print("PASS: apply_mitigation flags the first candle that touches the zone")


def test_apply_mitigation_leaves_untouched_poi_unmitigated():
    from poi_engine import FVG
    fvg = FVG(start_index=0, end_index=2, direction="bullish", zone_low=10, zone_high=12, valid=True)
    df = candles([
        (9, 10, 8, 9.5),
        (9.5, 13, 9.5, 12.5),
        (12.5, 14, 12, 13.5),
        (13.5, 16, 13.5, 15.5),  # stays well above the zone
        (15.5, 17, 15, 16.5),
    ])
    apply_mitigation(df, [fvg], formation_index_attr="end_index")
    assert fvg.mitigated is False
    assert fvg.mitigated_index is None
    print("PASS: apply_mitigation correctly leaves an untouched POI unmitigated")


if __name__ == "__main__":
    test_bullish_fvg_detected_with_correct_zone()
    test_bearish_fvg_detected_with_correct_zone()
    test_no_fvg_when_candles_overlap()
    test_fvg_premium_discount_validity()
    test_dealing_range_excludes_lookahead()
    test_order_block_finds_last_opposite_candle()
    test_order_block_skips_multiple_same_color_candles()
    test_mitigation_block_derived_from_idm_event()
    test_apply_mitigation_flags_first_touch()
    test_apply_mitigation_leaves_untouched_poi_unmitigated()
    print("\nAll poi_engine tests passed.")

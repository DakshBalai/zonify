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

from structure_engine import SwingPoint, StructureEvent, analyze_structure  # noqa: E402
from poi_engine import (  # noqa: E402
    FVG,
    OrderBlock,
    compute_dealing_range,
    find_fvgs,
    find_order_blocks,
    find_extreme_order_blocks,
    find_mitigation_blocks,
    find_breaker_blocks,
    find_idm_confluence_pois,
    apply_mitigation,
    analyze_poi,
    analyze_poi_swing_tier,
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
# Extreme Order Blocks
# ---------------------------------------------------------------------------

def test_extreme_order_block_picks_most_extreme_candle_in_the_run():
    # Two reds before the green run: index0 low=9.5 (the true extreme),
    # index1 low=9.8 (nearer to the run, less extreme) -- the extreme OB
    # must reach past the nearer red (index1) to find the deeper one.
    df = candles([
        (10.5, 10.6, 9.5, 9.6),    # i=0 red (close 9.6 < open 10.5), low=9.5
        (10.3, 10.4, 9.8, 9.9),    # i=1 red (close 9.9 < open 10.3), low=9.8
        (9.9, 10.6, 9.9, 10.5),    # i=2 green
        (10.5, 11.2, 10.5, 11.0),  # i=3 green
        (11.0, 12.0, 11.0, 11.9),  # i=4 green, BOS close
    ])
    events = [StructureEvent(index=4, event_type="BOS", direction="bullish", price=11.9,
                              source_swing_index=0, source_swing_price=10.6)]
    regular = find_order_blocks(df, events)
    extreme = find_extreme_order_blocks(df, events)
    assert regular[0].index == 1, f"regular OB should be the nearer red (index 1), got {regular[0].index}"
    assert extreme[0].index == 0, f"extreme OB should be the deeper red (index 0), got {extreme[0].index}"
    assert extreme[0].zone_low == 9.5
    print("PASS: extreme order block reaches past the nearer candle to the true extreme of the run")


def test_extreme_order_block_matches_regular_when_run_has_one_candle():
    df = candles([
        (10, 10.2, 9.5, 9.6),
        (9.6, 10.5, 9.6, 10.4),
        (10.4, 11.2, 10.4, 11.0),
        (11.0, 12.0, 11.0, 11.8),
    ])
    events = [StructureEvent(index=3, event_type="BOS", direction="bullish", price=11.8,
                              source_swing_index=0, source_swing_price=10.5)]
    regular = find_order_blocks(df, events)
    extreme = find_extreme_order_blocks(df, events)
    assert extreme[0].index == regular[0].index == 0
    print("PASS: extreme order block matches the regular OB when the opposite run is only one candle long")


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
# Breaker Blocks
# ---------------------------------------------------------------------------

def test_breaker_block_created_when_ob_far_edge_is_closed_through_for_two_bars():
    # Bullish OB zone (9.5, 10.2) at index 0. Far edge for a bullish OB
    # is zone_low=9.5. TWO consecutive closes below 9.5 (default
    # confirm_bars=2) -> the OB fails and becomes a BEARISH breaker.
    ob = OrderBlock(index=0, direction="bullish", zone_low=9.5, zone_high=10.2, source_event_index=3)
    df = candles([
        (10.0, 10.2, 9.5, 9.6),    # i=0, the OB candle itself
        (9.6, 11.0, 9.6, 10.8),    # i=1
        (10.8, 12.0, 10.5, 11.8),  # i=2
        (11.8, 12.5, 11.5, 12.2),  # i=3, the BOS close candle
        (12.2, 12.3, 9.0, 9.0),    # i=4, closes at 9.0 -- through the OB floor (9.5), 1st confirming close
        (9.0, 9.1, 8.7, 8.8),      # i=5, still below 9.5 -- 2nd confirming close -> breaker confirmed here
    ])
    breakers = find_breaker_blocks(df, [ob])
    assert len(breakers) == 1
    b = breakers[0]
    assert b.direction == "bearish", "a failed bullish OB must flip to a bearish breaker"
    assert b.zone_low == 9.5 and b.zone_high == 10.2, "breaker keeps the OB's original zone"
    assert b.index == 5, f"expected confirmation at index 5, got {b.index}"
    assert b.source_order_block_index == 0
    print("PASS: breaker block created and flipped after 2 consecutive confirming closes")


def test_breaker_block_flips_bearish_ob_to_bullish():
    # Bearish OB zone (10.0, 10.5); far edge = zone_high = 10.5. Two
    # consecutive closes above 10.5 invalidate it -> flips BULLISH.
    ob = OrderBlock(index=0, direction="bearish", zone_low=10.0, zone_high=10.5, source_event_index=2)
    df = candles([
        (10.4, 10.5, 10.0, 10.1),  # i=0, OB candle
        (10.1, 10.2, 9.0, 9.2),    # i=1
        (9.2, 9.3, 8.5, 8.7),      # i=2, CHoCH close candle
        (8.7, 11.0, 8.6, 10.9),    # i=3, closes above 10.5 -- 1st confirming close
        (10.9, 11.2, 10.8, 11.1),  # i=4, still above 10.5 -- 2nd confirming close
    ])
    breakers = find_breaker_blocks(df, [ob])
    assert len(breakers) == 1
    assert breakers[0].direction == "bullish"
    assert breakers[0].index == 4
    print("PASS: a failed bearish OB flips to a bullish breaker")


def test_no_breaker_block_when_ob_is_never_invalidated():
    # Price wicks into the zone (ordinary mitigation) but never CLOSES
    # through the far edge -- no breaker should be created.
    ob = OrderBlock(index=0, direction="bullish", zone_low=9.5, zone_high=10.2, source_event_index=1)
    df = candles([
        (10.0, 10.2, 9.5, 9.6),
        (9.6, 11.0, 9.6, 10.8),
        (10.8, 11.0, 9.4, 9.7),   # wicks to 9.4 (below zone_low) but closes at 9.7 (still inside)
    ])
    breakers = find_breaker_blocks(df, [ob])
    assert breakers == []
    print("PASS: no breaker block when the OB is only wicked into, never closed through")


def test_breaker_block_rejects_single_bar_fakeout():
    # A single close beyond the far edge, immediately reclaimed on the
    # very next candle -- this is exactly the noisy wick-and-reclaim
    # pattern confirmed against real NSE data (see find_breaker_blocks
    # docstring). Must NOT produce a breaker with the confirm_bars=2
    # default, since the "break" never actually held.
    ob = OrderBlock(index=0, direction="bullish", zone_low=9.5, zone_high=10.2, source_event_index=1)
    df = candles([
        (10.0, 10.2, 9.5, 9.6),
        (9.6, 9.7, 9.4, 9.5),
        (9.5, 9.6, 9.0, 9.0),     # i=2, single close below 9.5 -- a fakeout
        (9.0, 10.6, 9.0, 10.5),   # i=3, immediately reclaims back above the zone
    ])
    breakers = find_breaker_blocks(df, [ob])
    assert breakers == [], "a single-candle close, immediately reclaimed, must not confirm a breaker"
    print("PASS: a single-bar fakeout close does not confirm a breaker block")


def test_breaker_block_confirm_bars_is_configurable():
    # The same single-close fakeout DOES count as a breaker if the
    # caller explicitly asks for confirm_bars=1 -- confirms the
    # parameter is actually wired through, not hardcoded.
    ob = OrderBlock(index=0, direction="bullish", zone_low=9.5, zone_high=10.2, source_event_index=1)
    df = candles([
        (10.0, 10.2, 9.5, 9.6),
        (9.6, 9.7, 9.4, 9.5),
        (9.5, 9.6, 9.0, 9.0),     # i=2, single close below 9.5
        (9.0, 10.6, 9.0, 10.5),
    ])
    breakers = find_breaker_blocks(df, [ob], confirm_bars=1)
    assert len(breakers) == 1
    assert breakers[0].index == 2
    print("PASS: confirm_bars is configurable and defaults to requiring more than 1 bar")


def test_breaker_block_mitigation_tracks_from_confirmation_candle():
    # After the breaker confirms at index 3 (2nd confirming close), a
    # later candle trades back into the zone -- that's the breaker's own
    # mitigation (the retest entry), tracked from the CONFIRMATION index
    # forward, not from the original OB index.
    ob = OrderBlock(index=0, direction="bullish", zone_low=9.5, zone_high=10.2, source_event_index=1)
    df = candles([
        (10.0, 10.2, 9.5, 9.6),
        (9.6, 9.7, 9.4, 9.5),
        (9.5, 9.6, 9.0, 9.0),     # i=2, 1st confirming close
        (9.0, 9.2, 8.8, 8.9),     # i=3, 2nd confirming close -> breaker confirmed here
        (8.9, 9.1, 8.7, 8.9),     # i=4, still below the zone
        (8.9, 9.8, 8.8, 9.7),     # i=5, high=9.8 re-enters zone (9.5, 10.2) -> breaker mitigated here
    ])
    breakers = find_breaker_blocks(df, [ob])
    apply_mitigation(df, breakers, formation_index_attr="index")
    assert len(breakers) == 1
    assert breakers[0].index == 3
    assert breakers[0].mitigated is True
    assert breakers[0].mitigated_index == 5
    print("PASS: breaker block mitigation is tracked forward from its own confirmation candle")


# ---------------------------------------------------------------------------
# analyze_poi_swing_tier
# ---------------------------------------------------------------------------

def test_analyze_poi_swing_tier_uses_swing_structure_inputs():
    # Not testing fractal detection here -- just confirming
    # analyze_poi_swing_tier() actually feeds swing_structure /
    # swing_structure_events through to analyze_poi(), rather than
    # silently using the internal-tier swings/events instead.
    df = candles([
        (95, 96, 94, 95), (95, 97, 94, 96), (96, 98, 95, 97), (97, 102, 96, 101),
        (101, 103, 100, 102), (102, 112, 101, 111), (111, 112, 94, 95),
    ])
    structure_result = analyze_structure(df, lookback=1)
    swing_poi = analyze_poi_swing_tier(df, structure_result)
    manual = analyze_poi(df, structure_result["swing_structure"], structure_result["swing_structure_events"])
    assert [ob.index for ob in swing_poi["order_blocks"]] == [ob.index for ob in manual["order_blocks"]]
    assert set(swing_poi.keys()) == {
        "fvgs", "order_blocks", "extreme_order_blocks", "mitigation_blocks", "breaker_blocks",
    }
    print("PASS: analyze_poi_swing_tier derives every POI from swing-tier structure, not internal")


# ---------------------------------------------------------------------------
# IDM POI confluence
# ---------------------------------------------------------------------------

def test_idm_confluence_finds_nearest_qualifying_poi_above_bullish_sweep():
    idm = StructureEvent(index=5, event_type="IDM", direction="bullish", price=95.0,
                          source_swing_index=1, source_swing_price=94.0)
    near_ob = OrderBlock(index=2, direction="bullish", zone_low=100.0, zone_high=102.0, source_event_index=1)
    far_ob = OrderBlock(index=1, direction="bullish", zone_low=110.0, zone_high=112.0, source_event_index=0)
    poi_result = {"fvgs": [], "order_blocks": [near_ob, far_ob], "mitigation_blocks": [], "breaker_blocks": []}

    confluence = find_idm_confluence_pois([idm], poi_result)
    assert confluence[idm.index] is near_ob, "should pick the NEAREST qualifying POI, not just any"
    print("PASS: IDM confluence picks the nearest qualifying POI above a bullish sweep")


def test_idm_confluence_excludes_poi_on_wrong_side():
    idm = StructureEvent(index=5, event_type="IDM", direction="bullish", price=95.0,
                          source_swing_index=1, source_swing_price=94.0)
    below_ob = OrderBlock(index=2, direction="bullish", zone_low=80.0, zone_high=82.0, source_event_index=1)
    poi_result = {"fvgs": [], "order_blocks": [below_ob], "mitigation_blocks": [], "breaker_blocks": []}

    confluence = find_idm_confluence_pois([idm], poi_result)
    assert idm.index not in confluence
    print("PASS: IDM confluence excludes a POI on the wrong side of the sweep")


def test_idm_confluence_excludes_poi_formed_after_idm():
    idm = StructureEvent(index=5, event_type="IDM", direction="bullish", price=95.0,
                          source_swing_index=1, source_swing_price=94.0)
    later_ob = OrderBlock(index=6, direction="bullish", zone_low=100.0, zone_high=102.0, source_event_index=6)
    poi_result = {"fvgs": [], "order_blocks": [later_ob], "mitigation_blocks": [], "breaker_blocks": []}

    confluence = find_idm_confluence_pois([idm], poi_result)
    assert idm.index not in confluence, "a POI formed at/after the sweep is lookahead, must be excluded"
    print("PASS: IDM confluence excludes a POI that formed at or after the sweep (no lookahead)")


def test_idm_confluence_excludes_already_mitigated_poi():
    idm = StructureEvent(index=5, event_type="IDM", direction="bullish", price=95.0,
                          source_swing_index=1, source_swing_price=94.0)
    used_up_ob = OrderBlock(index=2, direction="bullish", zone_low=100.0, zone_high=102.0, source_event_index=1,
                             mitigated=True, mitigated_index=4)
    poi_result = {"fvgs": [], "order_blocks": [used_up_ob], "mitigation_blocks": [], "breaker_blocks": []}

    confluence = find_idm_confluence_pois([idm], poi_result)
    assert idm.index not in confluence, "a POI already used up before the sweep doesn't count"
    print("PASS: IDM confluence excludes a POI already mitigated before the sweep")


def test_idm_confluence_uses_fvg_end_index_not_index_attr():
    idm = StructureEvent(index=5, event_type="IDM", direction="bearish", price=100.0,
                          source_swing_index=1, source_swing_price=101.0)
    fvg = FVG(start_index=1, end_index=3, direction="bearish", zone_low=90.0, zone_high=92.0, valid=True)
    poi_result = {"fvgs": [fvg], "order_blocks": [], "mitigation_blocks": [], "breaker_blocks": []}

    confluence = find_idm_confluence_pois([idm], poi_result)
    assert confluence[idm.index] is fvg
    print("PASS: IDM confluence reads an FVG's formation point from end_index, not index")


def test_idm_confluence_empty_when_no_idm_events():
    poi_result = {"fvgs": [], "order_blocks": [], "mitigation_blocks": [], "breaker_blocks": []}
    events = [StructureEvent(index=1, event_type="BOS", direction="bullish", price=100.0)]
    assert find_idm_confluence_pois(events, poi_result) == {}
    print("PASS: IDM confluence returns empty when there are no IDM events at all")


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
    test_extreme_order_block_picks_most_extreme_candle_in_the_run()
    test_extreme_order_block_matches_regular_when_run_has_one_candle()
    test_mitigation_block_derived_from_idm_event()
    test_breaker_block_created_when_ob_far_edge_is_closed_through_for_two_bars()
    test_breaker_block_flips_bearish_ob_to_bullish()
    test_no_breaker_block_when_ob_is_never_invalidated()
    test_breaker_block_rejects_single_bar_fakeout()
    test_breaker_block_confirm_bars_is_configurable()
    test_breaker_block_mitigation_tracks_from_confirmation_candle()
    test_analyze_poi_swing_tier_uses_swing_structure_inputs()
    test_idm_confluence_finds_nearest_qualifying_poi_above_bullish_sweep()
    test_idm_confluence_excludes_poi_on_wrong_side()
    test_idm_confluence_excludes_poi_formed_after_idm()
    test_idm_confluence_excludes_already_mitigated_poi()
    test_idm_confluence_uses_fvg_end_index_not_index_attr()
    test_idm_confluence_empty_when_no_idm_events()
    test_apply_mitigation_flags_first_touch()
    test_apply_mitigation_leaves_untouched_poi_unmitigated()
    print("\nAll poi_engine tests passed.")

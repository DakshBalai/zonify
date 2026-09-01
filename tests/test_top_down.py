"""
test_top_down.py
Hand-built structure/POI scenarios with known-in-advance answers, same
convention as test_poi_engine.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import Bias, StructureEvent  # noqa: E402
from poi_engine import FVG, OrderBlock, MitigationBlock  # noqa: E402
from top_down import HTFZone, collect_htf_zones, find_top_down_entries  # noqa: E402


def make_df(n, freq="1h", start="2024-01-01 09:15"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="Asia/Kolkata")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)


def make_price_df(lows, highs, freq="1h", start="2024-01-01 09:15"):
    """Builds a DataFrame with explicit per-candle low/high (open/close pinned to the midpoint)."""
    idx = pd.date_range(start, periods=len(lows), freq=freq, tz="Asia/Kolkata")
    mid = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
    return pd.DataFrame({"open": mid, "high": highs, "low": lows, "close": mid, "volume": 1}, index=idx)


# ---------------------------------------------------------------------------
# collect_htf_zones
# ---------------------------------------------------------------------------

def test_collect_htf_zones_empty_when_bias_undetermined():
    df = make_df(10)
    structure_result = {"current_bias": Bias.UNDETERMINED}
    poi_result = {"fvgs": [], "order_blocks": [], "extreme_order_blocks": [], "breaker_blocks": []}
    assert collect_htf_zones(df, structure_result, poi_result, "daily") == []
    print("PASS: collect_htf_zones returns empty when bias is undetermined")


def test_collect_htf_zones_filters_by_bias_direction():
    df = make_df(10)
    structure_result = {"current_bias": Bias.BULLISH}
    bullish_ob = OrderBlock(index=1, direction="bullish", zone_low=100, zone_high=102, source_event_index=2)
    bearish_ob = OrderBlock(index=3, direction="bearish", zone_low=110, zone_high=112, source_event_index=4)
    poi_result = {"fvgs": [], "order_blocks": [bullish_ob, bearish_ob], "extreme_order_blocks": [], "breaker_blocks": []}
    zones = collect_htf_zones(df, structure_result, poi_result, "daily")
    assert len(zones) == 1
    assert zones[0].direction == "bullish"
    assert zones[0].zone_low == 100 and zones[0].zone_high == 102
    print("PASS: collect_htf_zones keeps only zones matching the HTF's own current bias")


def test_collect_htf_zones_excludes_invalid_fvgs():
    df = make_df(10)
    structure_result = {"current_bias": Bias.BULLISH}
    valid_fvg = FVG(start_index=0, end_index=2, direction="bullish", zone_low=100, zone_high=102, valid=True)
    invalid_fvg = FVG(start_index=3, end_index=5, direction="bullish", zone_low=105, zone_high=107, valid=False)
    poi_result = {"fvgs": [valid_fvg, invalid_fvg], "order_blocks": [], "extreme_order_blocks": [], "breaker_blocks": []}
    zones = collect_htf_zones(df, structure_result, poi_result, "daily")
    assert len(zones) == 1
    assert zones[0].zone_low == 100
    print("PASS: collect_htf_zones excludes invalid FVGs")


def test_collect_htf_zones_excludes_mitigation_blocks():
    df = make_df(10)
    structure_result = {"current_bias": Bias.BULLISH}
    poi_result = {
        "fvgs": [], "order_blocks": [], "extreme_order_blocks": [], "breaker_blocks": [],
        "mitigation_blocks": [MitigationBlock(index=1, direction="bullish", zone_low=100, zone_high=102, source_event_index=2)],
    }
    zones = collect_htf_zones(df, structure_result, poi_result, "daily")
    assert zones == []
    print("PASS: collect_htf_zones never includes mitigation_blocks even if present in poi_result")


def test_collect_htf_zones_computes_real_timestamps():
    df = make_df(10, freq="1D", start="2024-01-01")
    structure_result = {"current_bias": Bias.BULLISH}
    ob = OrderBlock(index=2, direction="bullish", zone_low=100, zone_high=102, source_event_index=3,
                     mitigated=True, mitigated_index=5)
    poi_result = {"fvgs": [], "order_blocks": [ob], "extreme_order_blocks": [], "breaker_blocks": []}
    zones = collect_htf_zones(df, structure_result, poi_result, "daily")
    assert zones[0].formed_at == df.index[2]
    assert zones[0].mitigated_at == df.index[5]
    print("PASS: collect_htf_zones converts row indices to real timestamps")


def test_collect_htf_zones_unmitigated_has_none_mitigated_at():
    df = make_df(10)
    structure_result = {"current_bias": Bias.BULLISH}
    ob = OrderBlock(index=2, direction="bullish", zone_low=100, zone_high=102, source_event_index=3)
    poi_result = {"fvgs": [], "order_blocks": [ob], "extreme_order_blocks": [], "breaker_blocks": []}
    zones = collect_htf_zones(df, structure_result, poi_result, "daily")
    assert zones[0].mitigated_at is None
    print("PASS: an unmitigated HTF zone has mitigated_at=None (still live)")


# ---------------------------------------------------------------------------
# find_top_down_entries
# ---------------------------------------------------------------------------

def _lows_highs(n, default_low=200.0, default_high=201.0):
    return [default_low] * n, [default_high] * n


def test_top_down_entry_qualifies_when_all_conditions_met():
    lows, highs = _lows_highs(10)
    lows[1], highs[1] = 98.5, 99.5   # candle at index 1 trades into the zone (98, 100)
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=3, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}

    htf_zone = HTFZone(
        timeframe="daily", poi_type="OrderBlock", direction="bullish",
        zone_low=98.0, zone_high=100.0, formed_at=ltf_df.index[0], mitigated_at=None,
    )
    entries = find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone])
    assert entries == [fvg]
    print("PASS: an FVG qualifies when CHoCH + a prior zone-touch conditions are both met")


def test_top_down_entry_rejected_without_a_preceding_choch():
    lows, highs = _lows_highs(10)
    lows[1], highs[1] = 98.5, 99.5
    ltf_df = make_price_df(lows, highs)
    structure_result = {"events": []}
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[0], None)
    assert find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone]) == []
    print("PASS: no CHoCH at all -> no top-down entry")


def test_top_down_entry_rejected_when_choch_too_far_before_fvg():
    lows, highs = _lows_highs(30)
    lows[1], highs[1] = 98.5, 99.5
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=2, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=20, end_index=22, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[0], None)
    entries = find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone], max_lookback_bars=10)
    assert entries == []
    print("PASS: an FVG far outside max_lookback_bars after the CHoCH does not qualify")


def test_top_down_entry_rejected_when_price_never_touched_the_zone():
    lows, highs = _lows_highs(10)  # never dips anywhere near (98, 100)
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=3, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[0], None)
    assert find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone]) == []
    print("PASS: price never touching any HTF zone before the CHoCH -> no top-down entry")


def test_top_down_entry_rejected_when_touch_outside_zone_lookback_window():
    lows, highs = _lows_highs(30)
    lows[0], highs[0] = 98.5, 99.5   # touches the zone, but 25 candles before the CHoCH
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=25, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=26, end_index=28, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[0], None)
    entries = find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone], zone_lookback_bars=20)
    assert entries == []
    print("PASS: a zone touch outside zone_lookback_bars before the CHoCH does not qualify")


def test_top_down_entry_rejected_when_zone_not_yet_formed():
    lows, highs = _lows_highs(10)
    lows[1], highs[1] = 98.5, 99.5
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=3, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    # the zone forms AFTER the only candle that touches it -- lookahead, must be excluded
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[2], None)
    assert find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone]) == []
    print("PASS: an HTF zone that forms AFTER the touching candle does not qualify (no lookahead)")


def test_top_down_entry_rejected_when_zone_already_mitigated():
    lows, highs = _lows_highs(10)
    lows[1], highs[1] = 98.5, 99.5
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=3, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    # zone was already mitigated (used up) before the touching candle at index 1
    htf_zone = HTFZone("daily", "OrderBlock", "bullish", 98.0, 100.0, ltf_df.index[0], ltf_df.index[0])
    assert find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone]) == []
    print("PASS: an HTF zone already mitigated before the touch does not qualify")


def test_top_down_entry_rejected_on_direction_mismatch():
    lows, highs = _lows_highs(10)
    lows[1], highs[1] = 98.5, 99.5
    ltf_df = make_price_df(lows, highs)
    choch = StructureEvent(index=3, event_type="CHoCH", direction="bullish", price=105.0,
                            source_swing_index=0, source_swing_price=104.0)
    fvg = FVG(start_index=4, end_index=6, direction="bullish", zone_low=106, zone_high=108, valid=True)
    structure_result = {"events": [choch]}
    htf_zone = HTFZone("daily", "OrderBlock", "bearish", 98.0, 100.0, ltf_df.index[0], None)  # wrong direction
    assert find_top_down_entries(ltf_df, structure_result, [fvg], [htf_zone]) == []
    print("PASS: a zone of the opposite direction never qualifies, regardless of price overlap")


if __name__ == "__main__":
    test_collect_htf_zones_empty_when_bias_undetermined()
    test_collect_htf_zones_filters_by_bias_direction()
    test_collect_htf_zones_excludes_invalid_fvgs()
    test_collect_htf_zones_excludes_mitigation_blocks()
    test_collect_htf_zones_computes_real_timestamps()
    test_collect_htf_zones_unmitigated_has_none_mitigated_at()
    test_top_down_entry_qualifies_when_all_conditions_met()
    test_top_down_entry_rejected_without_a_preceding_choch()
    test_top_down_entry_rejected_when_choch_too_far_before_fvg()
    test_top_down_entry_rejected_when_price_never_touched_the_zone()
    test_top_down_entry_rejected_when_touch_outside_zone_lookback_window()
    test_top_down_entry_rejected_when_zone_not_yet_formed()
    test_top_down_entry_rejected_when_zone_already_mitigated()
    test_top_down_entry_rejected_on_direction_mismatch()
    print("\nAll top_down tests passed.")

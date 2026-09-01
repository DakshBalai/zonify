"""
test_screener.py
Tests for the pure-logic pieces of screener.py that need no network
access. screen_ticker() itself needs real network access
(fetch_multi_timeframe_data) and is not unit-tested here, same
reasoning as data_loader.fetch_ohlcv().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poi_engine import OrderBlock  # noqa: E402
from screener import ScreenerZone, find_fresh_zone, preview_stop_target  # noqa: E402


def ob(index, direction, zone_low, zone_high, mitigated=False):
    return OrderBlock(index=index, direction=direction, zone_low=zone_low, zone_high=zone_high,
                       source_event_index=index, mitigated=mitigated)


def test_prefers_extreme_ob_over_order_block():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bullish", 100, 102)],
        "order_blocks": [ob(6, "bullish", 200, 202)],
    }
    zone = find_fresh_zone(poi_result, "bullish")
    assert zone.poi_type == "ExtremeOB"
    assert zone.zone_low == 100
    print("PASS: prefers ExtremeOB over OrderBlock when both exist")


def test_falls_back_to_order_block_when_no_extreme_ob():
    poi_result = {"extreme_order_blocks": [], "order_blocks": [ob(6, "bullish", 200, 202)]}
    zone = find_fresh_zone(poi_result, "bullish")
    assert zone.poi_type == "OrderBlock"
    assert zone.zone_low == 200
    print("PASS: falls back to OrderBlock when no ExtremeOB is available")


def test_filters_by_direction():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bearish", 100, 102)],
        "order_blocks": [],
    }
    assert find_fresh_zone(poi_result, "bullish") is None
    print("PASS: filters out zones of the wrong direction")


def test_filters_out_mitigated_zones():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bullish", 100, 102, mitigated=True)],
        "order_blocks": [],
    }
    assert find_fresh_zone(poi_result, "bullish") is None
    print("PASS: filters out already-mitigated zones")


def test_picks_most_recently_formed_among_candidates():
    poi_result = {
        "extreme_order_blocks": [
            ob(5, "bullish", 100, 102),
            ob(20, "bullish", 150, 152),
            ob(12, "bullish", 130, 132),
        ],
        "order_blocks": [],
    }
    zone = find_fresh_zone(poi_result, "bullish")
    assert zone.zone_low == 150
    print("PASS: picks the most recently formed (highest index) qualifying zone")


def test_returns_none_when_nothing_qualifies():
    poi_result = {"extreme_order_blocks": [], "order_blocks": []}
    assert find_fresh_zone(poi_result, "bullish") is None
    print("PASS: returns None when no zone qualifies at all")


def test_missing_keys_treated_as_empty():
    assert find_fresh_zone({}, "bullish") is None
    print("PASS: a poi_result missing the expected keys is treated as having no zones")


# ---------------------------------------------------------------------------
# preview_stop_target
# ---------------------------------------------------------------------------

def test_preview_bullish_zone_uses_own_edges_not_current_price():
    # entry = zone_high (near edge), stop = zone_low (far edge) --
    # NOT current_price, which can be arbitrarily far from an
    # unmitigated zone (see preview_stop_target's docstring).
    zone = ScreenerZone(timeframe="daily", poi_type="OrderBlock", direction="bullish", zone_low=95, zone_high=100)
    entry, stop, target = preview_stop_target(zone, current_price=105, reward_r=2.0)
    assert entry == 100 and stop == 95
    # risk = 100-95 = 5, reward_r=2 -> target = 100 + 10 = 110
    assert target == 110
    print("PASS: bullish zone preview uses its OWN edges for entry/stop, not the live price")


def test_preview_bearish_zone_uses_own_edges_not_current_price():
    zone = ScreenerZone(timeframe="daily", poi_type="ExtremeOB", direction="bearish", zone_low=95, zone_high=100)
    entry, stop, target = preview_stop_target(zone, current_price=90, reward_r=2.0)
    assert entry == 95 and stop == 100
    # risk = 100-95 = 5, reward_r=2 -> target = 95 - 10 = 85
    assert target == 85
    print("PASS: bearish zone preview uses its OWN edges for entry/stop, not the live price")


def test_preview_invalidated_when_price_already_through_stop():
    # bullish zone but price is already BELOW zone_low (the stop side) -- no valid preview
    zone = ScreenerZone(timeframe="daily", poi_type="OrderBlock", direction="bullish", zone_low=95, zone_high=100)
    entry, stop, target = preview_stop_target(zone, current_price=90, reward_r=2.0)
    assert entry == 100 and stop == 95
    assert target == entry  # no risk to project a target from -- zone already invalidated
    print("PASS: preview falls back gracefully when price has already invalidated the zone")


if __name__ == "__main__":
    test_prefers_extreme_ob_over_order_block()
    test_falls_back_to_order_block_when_no_extreme_ob()
    test_filters_by_direction()
    test_filters_out_mitigated_zones()
    test_picks_most_recently_formed_among_candidates()
    test_returns_none_when_nothing_qualifies()
    test_missing_keys_treated_as_empty()
    test_preview_bullish_zone_uses_own_edges_not_current_price()
    test_preview_bearish_zone_uses_own_edges_not_current_price()
    test_preview_invalidated_when_price_already_through_stop()
    print("\nAll screener tests passed.")

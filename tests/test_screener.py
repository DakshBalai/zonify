"""
test_screener.py
Tests for _find_fresh_zone(), the one piece of screener.py that's pure
logic with no network dependency. screen_ticker() itself needs real
network access (fetch_multi_timeframe_data) and is not unit-tested
here, same reasoning as data_loader.fetch_ohlcv().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poi_engine import OrderBlock  # noqa: E402
from screener import _find_fresh_zone  # noqa: E402


def ob(index, direction, zone_low, zone_high, mitigated=False):
    return OrderBlock(index=index, direction=direction, zone_low=zone_low, zone_high=zone_high,
                       source_event_index=index, mitigated=mitigated)


def test_prefers_extreme_ob_over_order_block():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bullish", 100, 102)],
        "order_blocks": [ob(6, "bullish", 200, 202)],
    }
    zone = _find_fresh_zone(poi_result, "bullish")
    assert zone.poi_type == "ExtremeOB"
    assert zone.zone_low == 100
    print("PASS: prefers ExtremeOB over OrderBlock when both exist")


def test_falls_back_to_order_block_when_no_extreme_ob():
    poi_result = {"extreme_order_blocks": [], "order_blocks": [ob(6, "bullish", 200, 202)]}
    zone = _find_fresh_zone(poi_result, "bullish")
    assert zone.poi_type == "OrderBlock"
    assert zone.zone_low == 200
    print("PASS: falls back to OrderBlock when no ExtremeOB is available")


def test_filters_by_direction():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bearish", 100, 102)],
        "order_blocks": [],
    }
    assert _find_fresh_zone(poi_result, "bullish") is None
    print("PASS: filters out zones of the wrong direction")


def test_filters_out_mitigated_zones():
    poi_result = {
        "extreme_order_blocks": [ob(5, "bullish", 100, 102, mitigated=True)],
        "order_blocks": [],
    }
    assert _find_fresh_zone(poi_result, "bullish") is None
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
    zone = _find_fresh_zone(poi_result, "bullish")
    assert zone.zone_low == 150
    print("PASS: picks the most recently formed (highest index) qualifying zone")


def test_returns_none_when_nothing_qualifies():
    poi_result = {"extreme_order_blocks": [], "order_blocks": []}
    assert _find_fresh_zone(poi_result, "bullish") is None
    print("PASS: returns None when no zone qualifies at all")


def test_missing_keys_treated_as_empty():
    assert _find_fresh_zone({}, "bullish") is None
    print("PASS: a poi_result missing the expected keys is treated as having no zones")


if __name__ == "__main__":
    test_prefers_extreme_ob_over_order_block()
    test_falls_back_to_order_block_when_no_extreme_ob()
    test_filters_by_direction()
    test_filters_out_mitigated_zones()
    test_picks_most_recently_formed_among_candidates()
    test_returns_none_when_nothing_qualifies()
    test_missing_keys_treated_as_empty()
    print("\nAll screener tests passed.")

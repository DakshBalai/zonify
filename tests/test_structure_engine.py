"""
test_structure_engine.py
Tests for structure_engine.py's swing-structure filtering, using hand-
built SwingPoint lists and candle sequences where the correct answer is
known in advance -- same convention as test_poi_engine.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import SwingPoint, detect_structure_events, filter_swing_structure  # noqa: E402


def sp(index, price, kind):
    return SwingPoint(index=index, price=price, kind=kind)


def as_tuples(swings):
    return [(s.index, s.price, s.kind) for s in swings]


def candles(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000
    return df


# ---------------------------------------------------------------------------
# filter_swing_structure
# ---------------------------------------------------------------------------

def test_filter_swing_structure_empty():
    assert filter_swing_structure([]) == []
    print("PASS: filter_swing_structure handles empty input")


def test_filter_swing_structure_single_swing():
    swings = [sp(0, 100, "high")]
    assert filter_swing_structure(swings) == swings
    print("PASS: filter_swing_structure passes through a single swing unchanged")


def test_filter_swing_structure_already_alternating_is_unchanged():
    swings = [sp(0, 90, "low"), sp(5, 100, "high"), sp(10, 95, "low"), sp(15, 110, "high")]
    assert filter_swing_structure(swings) == swings
    print("PASS: an already-alternating zigzag passes through unchanged")


def test_filter_swing_structure_collapses_to_most_extreme_high():
    # Two highs in a row before a low forms -- only the higher one is
    # structurally significant; the first was never really "the" top.
    swings = [sp(0, 90, "low"), sp(5, 100, "high"), sp(8, 105, "high"), sp(12, 95, "low")]
    assert as_tuples(filter_swing_structure(swings)) == [
        (0, 90, "low"), (8, 105, "high"), (12, 95, "low"),
    ]
    print("PASS: consecutive same-kind swings collapse to the most extreme one")


def test_filter_swing_structure_keeps_first_extreme_even_with_a_dip_between():
    # A LOWER high (92) forms after the real high (100) but before the
    # eventual low -- it must not displace the genuine extreme just
    # because it comes later in time.
    swings = [sp(0, 90, "low"), sp(5, 100, "high"), sp(8, 92, "high"), sp(12, 85, "low")]
    assert as_tuples(filter_swing_structure(swings)) == [
        (0, 90, "low"), (5, 100, "high"), (12, 85, "low"),
    ]
    print("PASS: a later but LESS extreme same-kind swing does not displace the real extreme")


def test_filter_swing_structure_collapses_runs_on_both_sides():
    swings = [
        sp(0, 100, "high"), sp(3, 105, "high"),                   # -> keep 105
        sp(6, 90, "low"), sp(9, 85, "low"), sp(11, 95, "low"),     # -> keep 85 (lowest)
        sp(14, 110, "high"),
    ]
    assert as_tuples(filter_swing_structure(swings)) == [
        (3, 105, "high"), (9, 85, "low"), (14, 110, "high"),
    ]
    print("PASS: filter_swing_structure collapses runs on both highs and lows")


# ---------------------------------------------------------------------------
# Integration: swing-tier structure avoids the noisy-minor-swing problem
# ---------------------------------------------------------------------------

def test_swing_tier_produces_fewer_events_than_internal_tier():
    """
    Raw fractal swings: a minor high (100) gets immediately exceeded by
    a bigger high (110) before any low forms in between -- exactly the
    case filter_swing_structure() collapses away. Feeding the raw list
    into detect_structure_events() fires TWO bullish breaks (through
    the minor 100 high, then again through the real 110 high); feeding
    the filtered list fires only ONE -- the minor high was never a real
    swing on this tier, so price breaking through it in passing isn't a
    structure event at all.
    """
    df = candles([
        (95, 96, 94, 95),     # i=0
        (95, 97, 94, 96),     # i=1
        (96, 98, 95, 97),     # i=2
        (97, 102, 96, 101),   # i=3, close=101 breaks the minor 100 high
        (101, 103, 100, 102),  # i=4
        (102, 112, 101, 111),  # i=5, close=111 breaks the real 110 high
        (111, 112, 94, 95),    # i=6
    ])

    raw_swings = [sp(0, 90, "low"), sp(2, 100, "high"), sp(4, 110, "high"), sp(6, 95, "low")]
    swing_structure = filter_swing_structure(raw_swings)
    assert as_tuples(swing_structure) == [(0, 90, "low"), (4, 110, "high"), (6, 95, "low")]

    internal_events, _ = detect_structure_events(df, raw_swings)
    swing_events, _ = detect_structure_events(df, swing_structure)

    assert len(internal_events) == 2, f"expected 2 internal-tier events, got {len(internal_events)}"
    assert [e.index for e in internal_events] == [3, 5]

    assert len(swing_events) == 1, f"expected 1 swing-tier event, got {len(swing_events)}"
    assert swing_events[0].index == 5
    print("PASS: swing-tier structure skips the break of a swing that was never structurally significant")


if __name__ == "__main__":
    test_filter_swing_structure_empty()
    test_filter_swing_structure_single_swing()
    test_filter_swing_structure_already_alternating_is_unchanged()
    test_filter_swing_structure_collapses_to_most_extreme_high()
    test_filter_swing_structure_keeps_first_extreme_even_with_a_dip_between()
    test_filter_swing_structure_collapses_runs_on_both_sides()
    test_swing_tier_produces_fewer_events_than_internal_tier()
    print("\nAll structure_engine tests passed.")

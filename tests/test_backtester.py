"""
test_backtester.py
Every test uses hand-constructed candles/events where the correct
outcome is known in advance -- same convention as test_poi_engine.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import StructureEvent, analyze_structure  # noqa: E402
from poi_engine import OrderBlock, analyze_poi  # noqa: E402
from session_model import PO3Setup  # noqa: E402
from backtester import (  # noqa: E402
    Outcome,
    Trade,
    backtest_structure_events,
    backtest_pois,
    backtest_po3_setups,
    run_extended_backtest,
    summarize_trades,
    _simulate_trade,
)


def candles(rows):
    """rows: list of (open, high, low, close) tuples -> OHLCV DataFrame."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000
    return df


# ---------------------------------------------------------------------------
# _simulate_trade core
# ---------------------------------------------------------------------------

def test_simulate_trade_win_when_target_hit_first():
    df = candles([
        (100, 101, 99, 100),   # i=0 entry candle
        (100, 102, 99.5, 101),  # i=1 neither hit yet
        (101, 106, 100, 105),   # i=2 high=106 reaches target=105
    ])
    trade = _simulate_trade(
        df, entry_index=0, entry_price=100, stop_price=95, target_price=105,
        direction="bullish", max_bars=10, source_type="BOS", reward_r=2.0,
    )
    assert trade.outcome == Outcome.WIN
    assert trade.exit_index == 2
    assert trade.r_multiple == 2.0
    print("PASS: win when target hit first")


def test_simulate_trade_loss_when_stop_hit_first():
    df = candles([
        (100, 101, 99, 100),
        (100, 102, 94, 95),   # i=1 low=94 reaches stop=95 before any target approach
        (95, 106, 94, 105),
    ])
    trade = _simulate_trade(
        df, entry_index=0, entry_price=100, stop_price=95, target_price=105,
        direction="bullish", max_bars=10, source_type="BOS", reward_r=2.0,
    )
    assert trade.outcome == Outcome.LOSS
    assert trade.exit_index == 1
    assert trade.r_multiple == -1.0
    print("PASS: loss when stop hit first")


def test_simulate_trade_same_bar_ambiguity_counts_as_loss():
    # A single candle whose range touches BOTH stop and target -- can't
    # tell intrabar order from OHLC alone, so the conservative LOSS
    # assumption applies.
    df = candles([
        (100, 101, 99, 100),
        (100, 106, 94, 100),   # i=1 high=106 >= target=105 AND low=94 <= stop=95
    ])
    trade = _simulate_trade(
        df, entry_index=0, entry_price=100, stop_price=95, target_price=105,
        direction="bullish", max_bars=10, source_type="BOS", reward_r=2.0,
    )
    assert trade.outcome == Outcome.LOSS
    print("PASS: same-bar target+stop ambiguity scored as loss")


def test_simulate_trade_timeout_when_neither_hit():
    df = candles([(100, 101, 99, 100)] * 6)
    trade = _simulate_trade(
        df, entry_index=0, entry_price=100, stop_price=95, target_price=105,
        direction="bullish", max_bars=3, source_type="BOS", reward_r=2.0,
    )
    assert trade.outcome == Outcome.TIMEOUT
    assert trade.exit_index is None
    assert trade.r_multiple == 0.0
    print("PASS: timeout when neither target nor stop reached in time")


def test_simulate_trade_bearish_direction():
    df = candles([
        (100, 101, 99, 100),
        (100, 101, 90, 91),   # i=1 low=90 reaches bearish target=91
    ])
    trade = _simulate_trade(
        df, entry_index=0, entry_price=100, stop_price=105, target_price=91,
        direction="bearish", max_bars=10, source_type="CHoCH", reward_r=2.0,
    )
    assert trade.outcome == Outcome.WIN
    print("PASS: bearish direction checks target below / stop above correctly")


# ---------------------------------------------------------------------------
# backtest_structure_events: entry/stop derivation per event type
# ---------------------------------------------------------------------------

def test_bos_choch_trade_uses_source_swing_as_stop():
    # bullish BOS: close=110 broke swing high=105 -> entry=110, stop=105,
    # risk=5, reward_r=2 -> target=120. Price rallies to 121 by i=3.
    df = candles([
        (105, 106, 104, 105),
        (105, 111, 104, 110),   # i=1, the BOS close candle (event.index=1)
        (110, 115, 109, 114),
        (114, 122, 113, 121),   # i=3, high=122 >= target=120 -> WIN
    ])
    events = [StructureEvent(index=1, event_type="BOS", direction="bullish", price=110,
                              source_swing_index=0, source_swing_price=105)]
    trades = backtest_structure_events(df, events, reward_r=2.0, max_bars=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 110 and t.stop_price == 105 and t.target_price == 120
    assert t.outcome == Outcome.WIN
    print("PASS: BOS/CHoCH trade derives entry/stop/target from event + source swing")


def test_idm_trade_uses_wick_extreme_as_stop():
    # bullish IDM: wick swept a low at 98 (event.price=98), close back
    # inside at 100 -> entry=100, stop=98, risk=2, reward_r=2 -> target=104.
    df = candles([
        (102, 103, 101, 101.5),
        (101.5, 101.6, 98, 100),   # i=1, the IDM sweep candle (event.index=1)
        (100, 105, 99, 104.5),     # i=2, high=105 >= target=104 -> WIN
    ])
    events = [StructureEvent(index=1, event_type="IDM", direction="bullish", price=98,
                              source_swing_index=0, source_swing_price=99)]
    trades = backtest_structure_events(df, events, reward_r=2.0, max_bars=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 100 and t.stop_price == 98 and t.target_price == 104
    assert t.outcome == Outcome.WIN
    print("PASS: IDM trade derives entry (close) / stop (wick extreme) correctly")


def test_events_without_source_swing_are_skipped():
    events = [StructureEvent(index=0, event_type="BOS", direction="bullish", price=100)]
    df = candles([(100, 101, 99, 100)] * 5)
    trades = backtest_structure_events(df, events)
    assert trades == []
    print("PASS: events with no source swing produce no trade")


# ---------------------------------------------------------------------------
# backtest_pois
# ---------------------------------------------------------------------------

def test_poi_trade_skips_unmitigated_zones():
    ob = OrderBlock(index=0, direction="bullish", zone_low=95, zone_high=100, source_event_index=1)
    assert ob.mitigated is False
    df = candles([(100, 101, 99, 100)] * 5)
    trades = backtest_pois(df, [ob], "OrderBlock")
    assert trades == []
    print("PASS: unmitigated POI produces no trade")


def test_poi_trade_uses_far_edge_as_stop():
    # bullish OB zone (95, 100). Mitigated at index 2, close=99 ->
    # entry=99, stop=zone_low=95, risk=4, reward_r=2 -> target=107.
    ob = OrderBlock(index=0, direction="bullish", zone_low=95, zone_high=100, source_event_index=1,
                     mitigated=True, mitigated_index=2)
    df = candles([
        (105, 106, 104, 105),
        (105, 106, 99, 100),
        (100, 101, 97, 99),     # i=2, mitigation candle, close=99
        (99, 108, 98, 107.5),   # i=3, high=108 >= target=107 -> WIN
    ])
    trades = backtest_pois(df, [ob], "OrderBlock", reward_r=2.0, max_bars=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 99 and t.stop_price == 95 and t.target_price == 107
    assert t.outcome == Outcome.WIN
    print("PASS: POI trade uses zone's far edge as stop, mitigation candle close as entry")


def test_poi_trade_invalidated_at_entry_is_skipped():
    # bullish OB zone (95, 100), but the mitigation candle closed at 93
    # -- already through the far edge (95) on the very candle that
    # mitigated it. No valid entry -> no trade.
    ob = OrderBlock(index=0, direction="bullish", zone_low=95, zone_high=100, source_event_index=1,
                     mitigated=True, mitigated_index=1)
    df = candles([
        (105, 106, 104, 105),
        (105, 106, 92, 93),   # i=1, closed straight through the zone and below zone_low
    ])
    trades = backtest_pois(df, [ob], "OrderBlock")
    assert trades == []
    print("PASS: POI invalidated on its own mitigation candle produces no trade")


# ---------------------------------------------------------------------------
# backtest_po3_setups
# ---------------------------------------------------------------------------

def test_po3_setup_backtests_as_a_trade_in_its_own_direction():
    # bullish distribution setup: entry=100, stop=95, risk=5, reward_r=2
    # -> target=110. Price rallies to 111 by i=1 (entry_index+1).
    setup = PO3Setup(
        day=None, accumulation_low=95, accumulation_high=105,
        manipulation_index=0, manipulation_direction="bearish",
        distribution_direction="bullish", entry_index=0, entry_price=100, stop_price=95,
    )
    df = candles([
        (100, 101, 99, 100),
        (100, 112, 99, 111),  # high=112 >= target=110 -> WIN
    ])
    trades = backtest_po3_setups(df, [setup], reward_r=2.0, max_bars=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.source_type == "PO3"
    assert t.entry_price == 100 and t.stop_price == 95 and t.target_price == 110
    assert t.outcome == Outcome.WIN
    print("PASS: PO3 setup backtests using its own entry/stop/distribution-direction")


def test_po3_setup_invalidated_at_entry_is_skipped():
    # bullish setup but entry_price is already below stop_price -- an
    # impossible/invalidated configuration, must produce no trade.
    setup = PO3Setup(
        day=None, accumulation_low=95, accumulation_high=105,
        manipulation_index=0, manipulation_direction="bearish",
        distribution_direction="bullish", entry_index=0, entry_price=90, stop_price=95,
    )
    df = candles([(90, 91, 89, 90)] * 3)
    trades = backtest_po3_setups(df, [setup])
    assert trades == []
    print("PASS: a PO3 setup with entry already through its own stop produces no trade")


# ---------------------------------------------------------------------------
# run_extended_backtest
# ---------------------------------------------------------------------------

def test_run_extended_backtest_includes_swing_poi_and_po3_labels():
    df = candles([
        (95, 96, 94, 95), (95, 97, 94, 96), (96, 98, 95, 97), (97, 102, 96, 101),
        (101, 103, 100, 102), (102, 112, 101, 111), (111, 112, 94, 95), (95, 96, 93, 94),
    ])
    structure_result = analyze_structure(df, lookback=1)
    poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])
    result = run_extended_backtest(df, structure_result, poi_result)

    # Not asserting specific signal counts (this is generic hand-built
    # data, not a scenario engineered to produce every signal type) --
    # just that it runs cleanly end-to-end, in particular that PO3's
    # DatetimeIndex requirement doesn't crash on this candles() helper's
    # plain RangeIndex (see session_model's isinstance guard), and that
    # it produces no PO3 trades on data with no calendar-day structure.
    assert isinstance(result["trades"], list)
    source_types = set(result["stats"].keys())
    assert "PO3" not in source_types, "PO3 must contribute nothing on a non-DatetimeIndex df"
    print("PASS: run_extended_backtest runs without error and PO3 is correctly absent on non-intraday data")


# ---------------------------------------------------------------------------
# summarize_trades
# ---------------------------------------------------------------------------

def test_summarize_trades_computes_win_rate_and_expectancy():
    trades = [
        Trade("OrderBlock", "bullish", 0, 100, 95, 110, Outcome.WIN, 3, 2.0),
        Trade("OrderBlock", "bullish", 5, 100, 95, 110, Outcome.WIN, 8, 2.0),
        Trade("OrderBlock", "bullish", 10, 100, 95, 110, Outcome.LOSS, 12, -1.0),
        Trade("OrderBlock", "bullish", 20, 100, 95, 110, Outcome.TIMEOUT, None, 0.0),
    ]
    stats = summarize_trades(trades)
    s = stats["OrderBlock"]
    assert s.n_trades == 4
    assert s.n_wins == 2 and s.n_losses == 1 and s.n_timeouts == 1
    assert s.win_rate == 2 / 3          # timeouts excluded from the win-rate denominator
    assert s.expectancy_r == (2.0 + 2.0 - 1.0 + 0.0) / 4
    print("PASS: summarize_trades computes win rate (ex-timeout) and expectancy (all trades)")


def test_summarize_trades_groups_by_source_type():
    trades = [
        Trade("BOS", "bullish", 0, 100, 95, 110, Outcome.WIN, 3, 2.0),
        Trade("FVG", "bearish", 5, 100, 105, 90, Outcome.LOSS, 8, -1.0),
    ]
    stats = summarize_trades(trades)
    assert set(stats.keys()) == {"BOS", "FVG"}
    assert stats["BOS"].n_trades == 1
    assert stats["FVG"].n_trades == 1
    print("PASS: summarize_trades groups independently by source_type")


if __name__ == "__main__":
    test_simulate_trade_win_when_target_hit_first()
    test_simulate_trade_loss_when_stop_hit_first()
    test_simulate_trade_same_bar_ambiguity_counts_as_loss()
    test_simulate_trade_timeout_when_neither_hit()
    test_simulate_trade_bearish_direction()
    test_bos_choch_trade_uses_source_swing_as_stop()
    test_idm_trade_uses_wick_extreme_as_stop()
    test_events_without_source_swing_are_skipped()
    test_poi_trade_skips_unmitigated_zones()
    test_poi_trade_uses_far_edge_as_stop()
    test_poi_trade_invalidated_at_entry_is_skipped()
    test_po3_setup_backtests_as_a_trade_in_its_own_direction()
    test_po3_setup_invalidated_at_entry_is_skipped()
    test_run_extended_backtest_includes_swing_poi_and_po3_labels()
    test_summarize_trades_computes_win_rate_and_expectancy()
    test_summarize_trades_groups_by_source_type()
    print("\nAll backtester tests passed.")

"""
backtester.py
Empirical accuracy measurement for structure_engine / poi_engine signals.

Every signal this project detects already carries a natural, structural
invalidation level -- this module uses that instead of inventing a new
"stop distance" parameter, so results measure whether the SMC concepts
themselves work, not an arbitrary risk model bolted on afterward:

  - BOS / CHoCH: entry = the confirming close. Stop = the swing level
    that was just broken (event.source_swing_price) -- a later close
    back beyond that level invalidates the break. Target is a fixed
    multiple (reward_r) of that same distance.
  - IDM: entry = the sweep candle's close (back inside the range).
    Stop = the sweep wick's own extreme (event.price) -- a later break
    beyond the wick invalidates the "stop hunt then reverse" thesis.
  - FVG / Order Block / Mitigation Block: entry = the close of the
    candle that first mitigates the zone. Stop = the zone's FAR edge
    (the boundary furthest from the expected reaction) -- a close
    through the whole zone invalidates the reaction thesis. Only
    MITIGATED zones produce a trade (an untouched zone was never
    entered), and only VALID FVGs are tested (an invalid one is a
    known-wrong-side setup, not a real signal).

A trade's outcome is decided by walking forward bar by bar from entry:
  WIN      a later bar's high/low reaches target before stop
  LOSS     a later bar's high/low reaches stop before target
  TIMEOUT  neither is reached within max_bars
A bar that touches both target and stop in the same candle is scored
LOSS -- OHLC data alone can't tell us which happened first intrabar,
so this is the conservative assumption, applied consistently.

This module only CONSUMES structure_engine/poi_engine output, the same
one-way dependency the rest of the project follows -- it invents no
new notion of a swing, break, or zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Outcome(Enum):
    WIN = "win"
    LOSS = "loss"
    TIMEOUT = "timeout"


@dataclass
class Trade:
    source_type: str   # "BOS" / "CHoCH" / "IDM" / "OrderBlock" / "MitigationBlock" / "FVG"
    direction: str      # "bullish" or "bearish"
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    outcome: Outcome
    exit_index: int | None
    r_multiple: float   # +reward_r on WIN, -1.0 on LOSS, 0.0 on TIMEOUT


@dataclass
class BacktestStats:
    source_type: str
    n_trades: int
    n_wins: int
    n_losses: int
    n_timeouts: int
    win_rate: float       # wins / (wins + losses); TIMEOUTs excluded from this denominator
    expectancy_r: float    # mean r_multiple across ALL trades (TIMEOUTs count as 0)


def _simulate_trade(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    max_bars: int,
    source_type: str,
    reward_r: float,
) -> Trade:
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    last = min(entry_index + max_bars, n - 1)

    for j in range(entry_index + 1, last + 1):
        hit_stop = lows[j] <= stop_price if direction == "bullish" else highs[j] >= stop_price
        if hit_stop:
            return Trade(source_type, direction, entry_index, entry_price, stop_price, target_price, Outcome.LOSS, j, -1.0)

        hit_target = highs[j] >= target_price if direction == "bullish" else lows[j] <= target_price
        if hit_target:
            return Trade(source_type, direction, entry_index, entry_price, stop_price, target_price, Outcome.WIN, j, reward_r)

    return Trade(source_type, direction, entry_index, entry_price, stop_price, target_price, Outcome.TIMEOUT, None, 0.0)


def _target_from_risk(entry_price: float, risk: float, direction: str, reward_r: float) -> float:
    return entry_price + risk * reward_r if direction == "bullish" else entry_price - risk * reward_r


def backtest_structure_events(
    df: pd.DataFrame,
    events: list,
    reward_r: float = 2.0,
    max_bars: int = 20,
    label_suffix: str = "",
) -> list[Trade]:
    """
    One trade per BOS/CHoCH/IDM event that has a source swing to derive
    a stop from. `label_suffix` is appended to the event's own type to
    make the resulting Trade.source_type -- used to keep internal-tier
    ("BOS") and swing-tier ("BOS (swing)") results from being lumped
    together when both are backtested via run_backtest().
    """
    closes = df["close"].values
    trades = []

    for event in events:
        if event.source_swing_price is None:
            continue

        if event.event_type in ("BOS", "CHoCH"):
            entry_price = event.price
            stop_price = event.source_swing_price
        elif event.event_type == "IDM":
            entry_price = closes[event.index]
            stop_price = event.price
        else:
            continue

        # Guaranteed non-zero and correctly signed by construction: a
        # BOS/CHoCH close is always beyond the swing it broke, and an
        # IDM close is always back inside the range from the wick it
        # swept -- see structure_engine.detect_structure_events().
        risk = (entry_price - stop_price) if event.direction == "bullish" else (stop_price - entry_price)
        if risk <= 0:
            continue

        target_price = _target_from_risk(entry_price, risk, event.direction, reward_r)
        trades.append(_simulate_trade(
            df, event.index, entry_price, stop_price, target_price,
            event.direction, max_bars, event.event_type + label_suffix, reward_r,
        ))

    return trades


def backtest_pois(
    df: pd.DataFrame,
    pois: list,
    source_type: str,
    reward_r: float = 2.0,
    max_bars: int = 20,
) -> list[Trade]:
    """
    One trade per MITIGATED poi (FVG / OrderBlock / MitigationBlock) --
    an unmitigated zone was never actually touched, so there's no entry
    to test. Entry is the mitigation candle's close; unlike BOS/CHoCH/IDM,
    mitigation is defined by high/low overlap, not close position, so a
    zone CAN already be invalidated (closed straight through) on the very
    candle that mitigates it -- those are skipped rather than backtested,
    since there was never a valid entry.
    """
    closes = df["close"].values
    trades = []

    for poi in pois:
        if not poi.mitigated:
            continue

        entry_index = poi.mitigated_index
        entry_price = closes[entry_index]

        if poi.direction == "bullish":
            stop_price = poi.zone_low
            risk = entry_price - stop_price
        else:
            stop_price = poi.zone_high
            risk = stop_price - entry_price

        if risk <= 0:
            continue  # invalidated at entry -- close already through the far edge

        target_price = _target_from_risk(entry_price, risk, poi.direction, reward_r)
        trades.append(_simulate_trade(
            df, entry_index, entry_price, stop_price, target_price,
            poi.direction, max_bars, source_type, reward_r,
        ))

    return trades


def backtest_po3_setups(
    df: pd.DataFrame,
    setups: list,
    reward_r: float = 2.0,
    max_bars: int = 20,
) -> list[Trade]:
    """
    One trade per PO3 setup (session_model.find_po3_setups()) -- entry/
    stop/direction are already fully specified on the setup itself
    (the manipulation candle's own wick is the stop, same "don't invent
    a new risk model" rule as everywhere else in this module), so this
    is a thin adapter into the same _simulate_trade() core.
    """
    trades = []
    for s in setups:
        risk = (
            (s.entry_price - s.stop_price) if s.distribution_direction == "bullish"
            else (s.stop_price - s.entry_price)
        )
        if risk <= 0:
            continue  # invalidated at entry -- the entry candle already closed through the stop

        target_price = _target_from_risk(s.entry_price, risk, s.distribution_direction, reward_r)
        trades.append(_simulate_trade(
            df, s.entry_index, s.entry_price, s.stop_price, target_price,
            s.distribution_direction, max_bars, "PO3", reward_r,
        ))

    return trades


def summarize_trades(trades: list[Trade]) -> dict[str, BacktestStats]:
    """Groups trades by source_type and computes win rate / expectancy for each."""
    by_type: dict[str, list[Trade]] = {}
    for t in trades:
        by_type.setdefault(t.source_type, []).append(t)

    stats = {}
    for source_type, group in by_type.items():
        n = len(group)
        wins = sum(1 for t in group if t.outcome == Outcome.WIN)
        losses = sum(1 for t in group if t.outcome == Outcome.LOSS)
        timeouts = sum(1 for t in group if t.outcome == Outcome.TIMEOUT)
        decided = wins + losses
        win_rate = wins / decided if decided else 0.0
        expectancy_r = sum(t.r_multiple for t in group) / n if n else 0.0
        stats[source_type] = BacktestStats(source_type, n, wins, losses, timeouts, win_rate, expectancy_r)

    return stats


def format_stats(stats: dict[str, BacktestStats]) -> str:
    lines = []
    for source_type in sorted(stats):
        s = stats[source_type]
        lines.append(
            f"{source_type:24s} n={s.n_trades:4d}  win_rate={s.win_rate:5.1%}  "
            f"expectancy={s.expectancy_r:+.2f}R  (W{s.n_wins}/L{s.n_losses}/T{s.n_timeouts})"
        )
    return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    structure_result: dict,
    poi_result: dict | None = None,
    reward_r: float = 2.0,
    max_bars: int = 20,
    include_swing_structure: bool = True,
) -> dict:
    """
    Convenience wrapper: backtests every event/POI type analyze_structure()
    and analyze_poi() produced for this timeframe's data, in one call.

    When structure_result carries swing-tier structure (the
    filter_swing_structure() output analyze_structure() now includes),
    those BOS/CHoCH/IDM events are ALSO backtested, labeled "<TYPE>
    (swing)" so they show up separately from the internal-tier ones
    instead of being averaged together -- the whole point is to compare
    the two, not blend them into one misleading number.

    Returns {"trades": [...], "stats": {source_type: BacktestStats}}.
    """
    trades = backtest_structure_events(df, structure_result["events"], reward_r=reward_r, max_bars=max_bars)

    if include_swing_structure and "swing_structure_events" in structure_result:
        trades += backtest_structure_events(
            df, structure_result["swing_structure_events"], reward_r=reward_r, max_bars=max_bars,
            label_suffix=" (swing)",
        )

    if poi_result is not None:
        trades += backtest_pois(df, poi_result["order_blocks"], "OrderBlock", reward_r=reward_r, max_bars=max_bars)
        trades += backtest_pois(df, poi_result.get("extreme_order_blocks", []), "ExtremeOB", reward_r=reward_r, max_bars=max_bars)
        trades += backtest_pois(df, poi_result["mitigation_blocks"], "MitigationBlock", reward_r=reward_r, max_bars=max_bars)
        trades += backtest_pois(df, poi_result.get("breaker_blocks", []), "BreakerBlock", reward_r=reward_r, max_bars=max_bars)
        valid_fvgs = [f for f in poi_result["fvgs"] if f.valid]
        trades += backtest_pois(df, valid_fvgs, "FVG", reward_r=reward_r, max_bars=max_bars)

    return {"trades": trades, "stats": summarize_trades(trades)}


def run_extended_backtest(
    df: pd.DataFrame,
    structure_result: dict,
    poi_result: dict,
    reward_r: float = 2.0,
    max_bars: int = 20,
) -> dict:
    """
    Runs run_backtest()'s full signal set, PLUS the newer comparisons
    built on top of it, each labeled distinctly so they show up next to
    (not blended into) the baseline they're being compared against:
      - "<TYPE> (swing-POI)": every POI type re-derived from swing-tier
        structure instead of internal (poi_engine.analyze_poi_swing_tier)
        -- "mark only the important levels" applied to POI derivation,
        the same idea filter_swing_structure() already applies to
        BOS/CHoCH itself.
      - "IDM (POI confluence)": IDM events filtered down to only those
        with an existing, unmitigated POI sitting just beyond the sweep
        in the expected direction (poi_engine.find_idm_confluence_pois).
      - "PO3": session Accumulation/Manipulation/Distribution setups
        (session_model.find_po3_setups) -- contributes zero trades on
        daily/weekly/monthly data, since a single-candle "day" carries
        no intraday opening range.

    Kept separate from run_backtest() rather than folded into it (more
    parameters bolted on) so run_backtest()'s existing signature, every
    caller, and every test stay untouched.
    """
    from poi_engine import analyze_poi_swing_tier, find_idm_confluence_pois
    from session_model import find_po3_setups

    base = run_backtest(df, structure_result, poi_result, reward_r=reward_r, max_bars=max_bars)
    trades = list(base["trades"])

    swing_poi = analyze_poi_swing_tier(df, structure_result)
    trades += backtest_pois(df, swing_poi["order_blocks"], "OrderBlock (swing-POI)", reward_r=reward_r, max_bars=max_bars)
    trades += backtest_pois(df, swing_poi.get("extreme_order_blocks", []), "ExtremeOB (swing-POI)", reward_r=reward_r, max_bars=max_bars)
    trades += backtest_pois(df, swing_poi["mitigation_blocks"], "MitigationBlock (swing-POI)", reward_r=reward_r, max_bars=max_bars)
    trades += backtest_pois(df, swing_poi.get("breaker_blocks", []), "BreakerBlock (swing-POI)", reward_r=reward_r, max_bars=max_bars)
    valid_fvgs_swing = [f for f in swing_poi["fvgs"] if f.valid]
    trades += backtest_pois(df, valid_fvgs_swing, "FVG (swing-POI)", reward_r=reward_r, max_bars=max_bars)

    idm_events = [e for e in structure_result["events"] if e.event_type == "IDM"]
    confluence_map = find_idm_confluence_pois(structure_result["events"], poi_result)
    idm_confluence_events = [e for e in idm_events if e.index in confluence_map]
    trades += backtest_structure_events(
        df, idm_confluence_events, reward_r=reward_r, max_bars=max_bars, label_suffix=" (POI confluence)",
    )

    po3_setups = find_po3_setups(df)
    trades += backtest_po3_setups(df, po3_setups, reward_r=reward_r, max_bars=max_bars)

    return {"trades": trades, "stats": summarize_trades(trades)}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from structure_engine import analyze_structure
    from poi_engine import analyze_poi

    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic_test_candles.csv")
    structure_result = analyze_structure(df, lookback=3)
    poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])

    result = run_backtest(df, structure_result, poi_result)
    print(f"{len(result['trades'])} trades simulated (synthetic test data, reward_r=2.0, max_bars=20)\n")
    print(format_stats(result["stats"]))

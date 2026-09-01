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
) -> list[Trade]:
    """One trade per BOS/CHoCH/IDM event that has a source swing to derive a stop from."""
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
            event.direction, max_bars, event.event_type, reward_r,
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
            f"{source_type:16s} n={s.n_trades:4d}  win_rate={s.win_rate:5.1%}  "
            f"expectancy={s.expectancy_r:+.2f}R  (W{s.n_wins}/L{s.n_losses}/T{s.n_timeouts})"
        )
    return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    structure_result: dict,
    poi_result: dict | None = None,
    reward_r: float = 2.0,
    max_bars: int = 20,
) -> dict:
    """
    Convenience wrapper: backtests every event/POI type analyze_structure()
    and analyze_poi() produced for this timeframe's data, in one call.

    Returns {"trades": [...], "stats": {source_type: BacktestStats}}.
    """
    trades = backtest_structure_events(df, structure_result["events"], reward_r=reward_r, max_bars=max_bars)

    if poi_result is not None:
        trades += backtest_pois(df, poi_result["order_blocks"], "OrderBlock", reward_r=reward_r, max_bars=max_bars)
        trades += backtest_pois(df, poi_result["mitigation_blocks"], "MitigationBlock", reward_r=reward_r, max_bars=max_bars)
        valid_fvgs = [f for f in poi_result["fvgs"] if f.valid]
        trades += backtest_pois(df, valid_fvgs, "FVG", reward_r=reward_r, max_bars=max_bars)

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

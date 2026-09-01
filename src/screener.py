"""
screener.py
The actual "screener" -- scans a universe of tickers and ranks them
using ONLY the signals this project has backtested and found to
actually work (see scripts/basket_backtest.py and
scripts/top_down_backtest.py for the numbers): HTF bias alignment, a
fresh (unmitigated) Order Block or Extreme Order Block on the daily
timeframe in that same direction, and -- for the top tier -- a live
top-down MSS+FVG entry already active on the 4h->15min drill-down.

This is deliberately narrow, and deliberately NOT everything the
project can detect. BOS, CHoCH, IDM, standalone FVG, MitigationBlock,
PO3, and the swing-tier structure variants are all still fully
computed and available through analyze_multi_timeframe() / analyze_poi()
/ backtester.py -- nothing was removed. This module just doesn't use
them for RANKING, because the 20-ticker basket backtest found they
don't reliably add edge on their own. If future backtesting changes
that for one of them, extending the tiers below is a small, local
change to this one file -- it never requires touching detection code.

Result tiers, simple and transparent on purpose (the same "not a black
box" principle multi_timeframe.py's own alignment logic follows):
  STRONG  HTF bias aligned (daily+4h) AND a fresh, bias-aligned
          ExtremeOB/OrderBlock on daily AND a live top-down MSS+FVG
          entry already active on 4h->15min right now.
  SETUP   HTF bias aligned AND a fresh, bias-aligned ExtremeOB/
          OrderBlock on daily, but no live LTF trigger yet -- worth
          watching, not yet actionable.
  (anything else -- mixed/undetermined bias, or no fresh zone at all
   -- is not surfaced; screen_ticker() returns None for it.)
"""

from __future__ import annotations

from dataclasses import dataclass

from structure_engine import analyze_structure
from poi_engine import analyze_poi
from multi_timeframe import DEFAULT_LOOKBACK, analyze_multi_timeframe, fetch_multi_timeframe_data
from top_down import collect_htf_zones, find_top_down_entries
from backtester import target_from_risk

# ExtremeOB is preferred over a plain OrderBlock when both exist --
# every basket backtest run so far found it the stronger of the two.
_ZONE_PREFERENCE = (("extreme_order_blocks", "ExtremeOB"), ("order_blocks", "OrderBlock"))


@dataclass
class ScreenerZone:
    timeframe: str
    poi_type: str    # "ExtremeOB" or "OrderBlock"
    direction: str
    zone_low: float
    zone_high: float


@dataclass
class ScreenerResult:
    ticker: str
    direction: str          # "bullish" or "bearish"
    tier: str                 # "STRONG" or "SETUP"
    daily_zone: ScreenerZone
    live_ltf_entry: bool       # a top-down MSS+FVG entry is currently unmitigated on 4h->15min
    current_price: float
    prev_close: float
    change_pct: float
    entry_price: float          # zone's near edge -- where price would first touch it
    stop_price: float          # zone's far edge -- the SAME rule backtester.backtest_pois() uses
    target_price: float         # target_from_risk() at reward_r -- the SAME formula the backtest uses
    reward_r: float


def find_fresh_zone(poi_result: dict, direction: str) -> ScreenerZone | None:
    """
    Picks the most recently formed, still-unmitigated, direction-
    aligned zone -- ExtremeOB checked first (see _ZONE_PREFERENCE),
    falling back to a plain OrderBlock if there's no fresh ExtremeOB.
    """
    for key, poi_type in _ZONE_PREFERENCE:
        candidates = [p for p in poi_result.get(key, []) if p.direction == direction and not p.mitigated]
        if candidates:
            latest = max(candidates, key=lambda p: p.index)
            return ScreenerZone(
                timeframe="daily", poi_type=poi_type, direction=direction,
                zone_low=latest.zone_low, zone_high=latest.zone_high,
            )
    return None


def _has_live_top_down_entry(data: dict, htf_result: dict) -> bool:
    """
    Checks whether find_top_down_entries() (the same mechanism
    scripts/top_down_backtest.py validated) currently has an
    unmitigated qualifying FVG on the 4h->15min drill-down -- i.e. an
    entry a trader could act on RIGHT NOW, not just a historical one.
    """
    htf_zones = collect_htf_zones(data["4h"], htf_result["structure"], htf_result["poi"], "4h")

    ltf_lookback = DEFAULT_LOOKBACK.get("15min", 4)
    ltf_structure_result = analyze_structure(data["15min"], lookback=ltf_lookback)
    ltf_poi_result = analyze_poi(data["15min"], ltf_structure_result["swings"], ltf_structure_result["events"])
    valid_ltf_fvgs = [f for f in ltf_poi_result["fvgs"] if f.valid]

    entries = find_top_down_entries(data["15min"], ltf_structure_result, valid_ltf_fvgs, htf_zones)
    return any(not f.mitigated for f in entries)


def preview_stop_target(zone: ScreenerZone, current_price: float, reward_r: float) -> tuple[float, float, float]:
    """
    Previews the entry/stop/target a trade off this zone WOULD use.
    Entry is the zone's OWN near edge (zone_high for bullish, zone_low
    for bearish) -- the point price would first touch approaching the
    zone -- and stop is the far edge, target = target_from_risk() at
    reward_r: the EXACT SAME rule backtester.backtest_pois() already
    uses for OrderBlock/ExtremeOB trades (entry near the zone, stop at
    its far edge), not a new formula invented for display purposes.

    Deliberately NOT current_price as the entry: an unmitigated zone
    can sit arbitrarily far from where price is trading right now (it
    hasn't been touched yet), so using the live quote as "entry"
    produced nonsensical, wildly distant targets -- caught by actually
    looking at a rendered screener row, not assumed. current_price is
    only used to check whether the zone has ALREADY been invalidated
    (price closed through the far edge) before a preview is shown.

    Returns (entry_price, stop_price, target_price).
    """
    if zone.direction == "bullish":
        entry_price, stop_price = zone.zone_high, zone.zone_low
        invalidated = current_price < stop_price
    else:
        entry_price, stop_price = zone.zone_low, zone.zone_high
        invalidated = current_price > stop_price

    risk = abs(entry_price - stop_price)
    if risk <= 0 or invalidated:
        return entry_price, stop_price, entry_price

    target_price = target_from_risk(entry_price, risk, zone.direction, reward_r)
    return entry_price, stop_price, target_price


def screen_ticker(ticker: str, reward_r: float = 2.0) -> ScreenerResult | None:
    """
    Runs the full pipeline for one ticker and returns a ScreenerResult
    if it qualifies for at least the SETUP tier; None otherwise (mixed/
    undetermined HTF bias, or no fresh zone). Needs real network access
    (fetch_multi_timeframe_data) -- not unit-tested here, same as
    data_loader.fetch_ohlcv itself; find_fresh_zone() is the piece
    that IS unit-tested (see tests/test_screener.py).
    """
    data = fetch_multi_timeframe_data(ticker, timeframes=["daily", "4h", "15min"])

    htf_result = analyze_multi_timeframe({"daily": data["daily"], "4h": data["4h"]})
    if htf_result["alignment"] not in ("bullish", "bearish"):
        return None
    direction = htf_result["alignment"]

    daily_zone = find_fresh_zone(htf_result["per_timeframe"]["daily"]["poi"], direction)
    if daily_zone is None:
        return None

    live_entry = _has_live_top_down_entry(data, htf_result["per_timeframe"]["4h"])
    tier = "STRONG" if live_entry else "SETUP"

    daily_close = data["daily"]["close"]
    current_price = float(daily_close.iloc[-1])
    prev_close = float(daily_close.iloc[-2]) if len(daily_close) > 1 else current_price
    change_pct = (current_price - prev_close) / prev_close * 100 if prev_close else 0.0

    entry_price, stop_price, target_price = preview_stop_target(daily_zone, current_price, reward_r)

    return ScreenerResult(
        ticker=ticker, direction=direction, tier=tier,
        daily_zone=daily_zone, live_ltf_entry=live_entry,
        current_price=current_price, prev_close=prev_close, change_pct=change_pct,
        entry_price=entry_price, stop_price=stop_price, target_price=target_price, reward_r=reward_r,
    )

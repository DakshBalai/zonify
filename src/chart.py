"""
chart.py
Interactive Plotly candlestick chart with market-structure events drawn
the way SMC charts actually look, not as floating scatter markers.

Design system (shared with app.py's CSS so the chart and the rest of
the UI read as one product, not a chart bolted onto a generic page):
  BG_PRIMARY / BG_CARD / BORDER / ACCENT / TEXT_* -- see constants below.
Directional color is used CONSISTENTLY across every zone/event type --
green for bullish, red for bearish -- and zone TYPES are told apart by
line style/weight instead of by hue, the same way a real SMC chart
layers many concepts without turning into a rainbow:
  OrderBlock       solid border
  ExtremeOB        solid border, thicker + a star in its label --
                   this is the strongest-validated signal in the
                   project's own backtests, and looks like it
  MitigationBlock  dotted, muted grey (the weakest-validated signal)
  BreakerBlock     dash-dot, colored by its OWN (flipped) direction
  BOS              thin solid line, directional color
  CHoCH            thicker solid line, directional color (a bias flip
                   is the more significant event)
  IDM              dotted, a muted violet -- neither bullish nor
                   bearish-green/red, since a sweep is a liquidity
                   mechanic, not a directional structural claim

Every category (each POI type, each event type, swing markers) is its
own legend entry with a shared `legendgroup`, and the layout sets
`legend.groupclick="togglegroup"` -- click one legend entry and every
trace/zone in that category shows or hides together. That's the "clean
experience" toggle: turn off what you don't want to see, per chart.

This module only ever CONSUMES the output of structure_engine.py
(analyze_structure()) / poi_engine.py (analyze_poi()) / top_down.py
(HTFZone) -- it doesn't recompute or reinterpret anything, so a chart
bug here can never be confused with a detection-logic bug. The one
exception is deliberately labeled as such: "BSL"/"SSL" liquidity lines
are just the most recent swing high/low, RELABELED with the SMC terms
for what they represent (a swing high is where buy-side stops/orders
rest above price; a swing low is where sell-side liquidity rests below
it) -- no new detection, just honest reuse of swings already computed.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Design tokens -- kept in one place and imported by app.py so the chart
# and the surrounding dashboard chrome use identical colors.
# ---------------------------------------------------------------------------
BG_PRIMARY = "#080C12"
BG_CARD = "#10161F"
BG_CARD_ALT = "#111820"
BORDER = "#202A36"
ACCENT = "#21D4B4"
BULLISH_COLOR = "#22C55E"   # professional green -- deliberately distinct from ACCENT (brand teal)
BEARISH_COLOR = "#EF4444"   # professional red
TEXT_PRIMARY = "#F5F7FA"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

IDM_COLOR = "#A78BFA"   # muted violet -- a sweep is a liquidity mechanic, not a directional claim
MB_COLOR = "#7C8698"     # muted grey -- MitigationBlock is this project's weakest-validated signal
LIQUIDITY_COLOR = "#5B6472"

EVENT_LINE_STYLE = {
    "BOS": {"dash": "solid", "width": 1.4},
    "CHoCH": {"dash": "solid", "width": 2.6},
    "IDM": {"dash": "dot", "width": 1.4},
}

FVG_BULLISH_FILL = "rgba(34,197,94,0.14)"
FVG_BEARISH_FILL = "rgba(239,68,68,0.14)"
FVG_INVALID_FILL = "rgba(148,163,184,0.08)"   # faint grey -- valid=False (wrong side of premium/discount)

# One legend entry per higher-timeframe, used by plot_multi_timeframe_zones()
# to keep "1H FVG" visually distinct from "4H FVG" even though both are FVGs.
HTF_TIMEFRAME_COLORS = {
    "monthly": "#C084FC", "weekly": "#818CF8", "daily": "#38BDF8",
    "4h": "#21D4B4", "1h": "#FBBF24",
}


def _event_color(event) -> str:
    if event.event_type == "IDM":
        return IDM_COLOR
    return BULLISH_COLOR if event.direction == "bullish" else BEARISH_COLOR


def _axis_style() -> dict:
    """Shared crosshair/spike + grid styling for both chart functions below."""
    spike = dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikedash="dot", spikethickness=1, spikecolor=TEXT_MUTED,
    )
    return dict(
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=True, linecolor=BORDER,
                   tickfont=dict(color=TEXT_SECONDARY, size=11), **spike),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=True, linecolor=BORDER,
                   tickfont=dict(color=TEXT_SECONDARY, size=11), side="right", **spike),
    )


def _base_layout(title: str, height: int, x_range=None) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(size=15, color=TEXT_PRIMARY, family=FONT_FAMILY)),
        template="plotly_dark",
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PRIMARY,
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=12),
        xaxis_rangeslider_visible=False,
        height=height,
        margin=dict(l=16, r=150, t=48, b=32),
        hovermode="x",
        legend=dict(
            groupclick="togglegroup", orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_SECONDARY),
        ),
    )
    axis_style = _axis_style()
    if x_range is not None:
        axis_style["xaxis"]["range"] = list(x_range)
        axis_style["xaxis"]["autorange"] = False
    layout.update(axis_style)
    return layout


def _default_visible_range(x, n_bars: int = 120):
    """
    A candlestick range with thousands of bars (e.g. months of 15min
    data) renders unreadably dense if Plotly's default "show
    everything" behavior is left alone. Returns an initial [start, end]
    x-range showing only the most recent `n_bars` candles -- the user
    can still zoom/pan to see the rest, this only sets what's visible
    on first render.
    """
    n = len(x)
    if n <= n_bars:
        return None
    return [x[n - n_bars], x[n - 1]]


def _zone_trace(x0, x1, y0, y1, fillcolor, line_color, line_width, dash, name, legendgroup, showlegend, hovertext):
    """One POI zone as a closed, fillable polygon trace -- not fig.add_shape(), specifically
    because shapes can't be shown/hidden by clicking a legend entry, and traces can."""
    return go.Scatter(
        x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
        mode="lines", fill="toself",
        fillcolor=fillcolor, line=dict(color=line_color, width=line_width, dash=dash),
        name=name, legendgroup=legendgroup, showlegend=showlegend,
        hoverinfo="text", hovertext=hovertext, hoveron="fills",
    )


def _add_poi_shapes(
    fig: go.Figure, x, poi_result: dict, show_invalid_fvgs: bool, max_extend: int,
    visible: dict, seen_groups: set,
) -> None:
    """
    Draws FVGs as translucent filled zones, and Order Blocks / Extreme
    Order Blocks / Mitigation Blocks / Breaker Blocks as outlined zones
    -- see the module docstring for the line-style convention that
    tells each zone TYPE apart while direction stays green/red
    throughout. Each zone extends from its formation candle to its
    mitigation candle, or -- if still unmitigated/"live" -- forward by
    at most `max_extend` candles (capped at the last candle on the chart).
    """
    n = len(x)

    def _end_x(formation_index: int, mitigated: bool, mitigated_index):
        if mitigated:
            return x[mitigated_index]
        capped = min(formation_index + max_extend, n - 1)
        return x[capped]

    def _add(group, zones, x0_fn, color_fn, fill_fn, width, dash, label_fn):
        is_visible = True if visible.get(group, True) else "legendonly"
        first = group not in seen_groups
        for i, poi in enumerate(zones):
            x0 = x[x0_fn(poi)]
            x1 = _end_x(x0_fn(poi), poi.mitigated, poi.mitigated_index)
            trace = _zone_trace(
                x0, x1, poi.zone_low, poi.zone_high,
                fillcolor=fill_fn(poi), line_color=color_fn(poi), line_width=width, dash=dash,
                name=group, legendgroup=group, showlegend=(first and i == 0),
                hovertext=label_fn(poi),
            )
            trace.visible = is_visible
            fig.add_trace(trace)
        if zones:
            seen_groups.add(group)

    fvgs = [f for f in poi_result.get("fvgs", []) if f.valid or show_invalid_fvgs]
    _add(
        "FVG", fvgs, lambda f: f.end_index,
        lambda f: "rgba(148,163,184,0.5)" if not f.valid else (BULLISH_COLOR if f.direction == "bullish" else BEARISH_COLOR),
        lambda f: FVG_INVALID_FILL if not f.valid else (FVG_BULLISH_FILL if f.direction == "bullish" else FVG_BEARISH_FILL),
        0, "solid",
        lambda f: f"FVG ({f.direction}){' — invalid (wrong side of premium/discount)' if not f.valid else ''}",
    )

    _add(
        "OrderBlock", poi_result.get("order_blocks", []), lambda o: o.index,
        lambda o: BULLISH_COLOR if o.direction == "bullish" else BEARISH_COLOR,
        lambda o: "rgba(0,0,0,0)", 1.2, "solid",
        lambda o: f"Order Block ({o.direction})",
    )

    _add(
        "ExtremeOB", poi_result.get("extreme_order_blocks", []), lambda o: o.index,
        lambda o: BULLISH_COLOR if o.direction == "bullish" else BEARISH_COLOR,
        lambda o: "rgba(0,0,0,0)", 2.2, "solid",
        lambda o: f"★ Extreme OB ({o.direction}) — high-confidence zone",
    )

    _add(
        "MitigationBlock", poi_result.get("mitigation_blocks", []), lambda m: m.index,
        lambda m: MB_COLOR, lambda m: "rgba(0,0,0,0)", 1, "dot",
        lambda m: f"Mitigation Block ({m.direction})",
    )

    _add(
        "BreakerBlock", poi_result.get("breaker_blocks", []), lambda b: b.index,
        lambda b: BULLISH_COLOR if b.direction == "bullish" else BEARISH_COLOR,
        lambda b: "rgba(0,0,0,0)", 1.4, "dashdot",
        lambda b: f"Breaker Block ({b.direction})",
    )


def _add_liquidity_lines(fig: go.Figure, x, swings: list, visible: dict) -> None:
    """
    BSL (Buy-Side Liquidity) / SSL (Sell-Side Liquidity): the most
    recent swing high / swing low, relabeled with the SMC terms for
    what they represent -- resting stops/orders above the last swing
    high, and below the last swing low. NOT a new detector: reuses
    whichever swings analyze_structure() already found. Deliberately a
    simplification (the single most recent of each kind, not filtered
    for "still structurally unbroken") -- good enough as a reference
    line, not a claim of precise liquidity-pool tracking.
    """
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    is_visible = True if visible.get("Liquidity", True) else "legendonly"
    n = len(x)

    if highs:
        last_high = max(highs, key=lambda s: s.index)
        x1 = x[min(last_high.index + max(1, (n - last_high.index) // 3), n - 1)]
        fig.add_trace(go.Scatter(
            x=[x[last_high.index], x1], y=[last_high.price, last_high.price],
            mode="lines", line=dict(color=LIQUIDITY_COLOR, width=1, dash="dot"),
            name="Liquidity", legendgroup="Liquidity", showlegend=True, visible=is_visible,
            hoverinfo="text", hovertext=f"BSL (buy-side liquidity) {last_high.price:.2f}",
        ))
    if lows:
        last_low = max(lows, key=lambda s: s.index)
        x1 = x[min(last_low.index + max(1, (n - last_low.index) // 3), n - 1)]
        fig.add_trace(go.Scatter(
            x=[x[last_low.index], x1], y=[last_low.price, last_low.price],
            mode="lines", line=dict(color=LIQUIDITY_COLOR, width=1, dash="dot"),
            name="Liquidity", legendgroup="Liquidity", showlegend=False, visible=is_visible,
            hoverinfo="text", hovertext=f"SSL (sell-side liquidity) {last_low.price:.2f}",
        ))


def _add_trade_setup(fig: go.Figure, x, trade_setup: dict) -> None:
    """
    Overlays an entry/stop/target preview: shaded risk region (entry to
    stop) and reward region (entry to target), plus three labeled
    lines. trade_setup: {"entry": float, "stop": float, "target": float,
    "direction": "bullish"|"bearish"} -- e.g. from screener.ScreenerResult
    or a backtester.Trade. Purely a display of numbers already computed
    elsewhere; this function invents no prices.
    """
    entry, stop, target = trade_setup["entry"], trade_setup["stop"], trade_setup["target"]
    x0, x1 = x[0], x[-1]

    risk_fill = "rgba(239,68,68,0.08)"
    reward_fill = "rgba(34,197,94,0.08)"

    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0], y=[entry, entry, stop, stop, entry],
        mode="none", fill="toself", fillcolor=risk_fill,
        name="Risk", legendgroup="TradeSetup", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0], y=[entry, entry, target, target, entry],
        mode="none", fill="toself", fillcolor=reward_fill,
        name="Reward", legendgroup="TradeSetup", showlegend=False, hoverinfo="skip",
    ))

    for label, price, color, dash in (
        ("Target", target, BULLISH_COLOR, "dash"),
        ("Entry", entry, ACCENT, "solid"),
        ("Stop", stop, BEARISH_COLOR, "dash"),
    ):
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[price, price], mode="lines",
            line=dict(color=color, width=1.4, dash=dash),
            name="Trade Setup", legendgroup="TradeSetup", showlegend=(label == "Entry"),
            hoverinfo="text", hovertext=f"{label}: {price:.2f}",
        ))
        fig.add_annotation(
            x=x1, y=price, text=f" {label} {price:.2f}", showarrow=False,
            xanchor="left", font=dict(size=10, color=color, family=FONT_FAMILY),
        )


def plot_structure(
    df,
    result: dict,
    poi_result: dict | None = None,
    title: str = "Market Structure",
    show_swing_markers: bool = True,
    show_invalid_fvgs: bool = False,
    poi_max_extend: int = 40,
    dark_theme: bool = True,
    visible: dict | None = None,
    show_liquidity: bool = False,
    trade_setup: dict | None = None,
    height: int = 700,
) -> go.Figure:
    """
    df:     the same OHLCV DataFrame passed into analyze_structure()
    result: the dict returned by structure_engine.analyze_structure(df)
            (keys: swings, events, current_bias, last_event, ...)
    visible: optional {category: bool} controlling which categories are
        shown by default -- "Swings", "BOS", "CHoCH", "IDM", "FVG",
        "OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock",
        "Liquidity". Every category defaults to True (visible) if
        omitted; setting one to False draws it hidden but still
        toggleable from the legend (it isn't removed, just unchecked).
    show_liquidity: draw BSL/SSL reference lines (see _add_liquidity_lines).
    trade_setup: optional {"entry", "stop", "target", "direction"} to
        overlay a live trade preview (see _add_trade_setup).

    Returns a plotly Figure -- call .show() locally, or in Streamlit
    use st.plotly_chart(fig, use_container_width=True). Every category
    is independently toggleable by clicking its legend entry.
    """
    visible = visible or {}
    x = df.index

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                increasing_line_color=BULLISH_COLOR, decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR, decreasing_fillcolor=BEARISH_COLOR,
                increasing_line_width=1, decreasing_line_width=1,
                name="price", showlegend=False,
            )
        ]
    )

    if show_swing_markers:
        swing_highs = [s for s in result["swings"] if s.kind == "high"]
        swing_lows = [s for s in result["swings"] if s.kind == "low"]
        vis = True if visible.get("Swings", True) else "legendonly"
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_highs], y=[s.price for s in swing_highs],
            mode="markers", marker=dict(symbol="circle", size=4.5, color=TEXT_MUTED,
                                          line=dict(width=1, color=TEXT_SECONDARY)),
            name="Swings", legendgroup="Swings", showlegend=True, visible=vis,
            hoverinfo="text", hovertext="swing high",
        ))
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_lows], y=[s.price for s in swing_lows],
            mode="markers", marker=dict(symbol="circle", size=4.5, color=TEXT_MUTED,
                                          line=dict(width=1, color=TEXT_SECONDARY)),
            name="Swings", legendgroup="Swings", showlegend=False, visible=vis,
            hoverinfo="text", hovertext="swing low",
        ))

    if show_liquidity:
        _add_liquidity_lines(fig, x, result["swings"], visible)

    # Each event is drawn the way a trader actually marks it on a real
    # SMC chart: a HORIZONTAL line held at the swing point's price
    # level, running flat from the swing candle across to the candle
    # that broke/swept it, then a short vertical "elbow" dropping down
    # to the actual break price.
    event_group_seen = set()
    for event in result["events"]:
        if event.source_swing_index is None:
            continue

        x0 = x[event.source_swing_index]
        x1 = x[event.index]
        level = event.source_swing_price
        style = EVENT_LINE_STYLE[event.event_type]
        color = _event_color(event)
        group = event.event_type
        first = group not in event_group_seen
        event_group_seen.add(group)
        vis = True if visible.get(group, True) else "legendonly"

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[level, level],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            name=group, legendgroup=group, showlegend=first, visible=vis,
            hoverinfo="text", hovertext=f"{event.event_type} ({event.direction})",
        ))
        fig.add_trace(go.Scatter(
            x=[x1, x1], y=[level, event.price],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            name=group, legendgroup=group, showlegend=False, visible=vis,
            hoverinfo="skip",
        ))

    if poi_result is not None:
        _add_poi_shapes(fig, x, poi_result, show_invalid_fvgs, poi_max_extend, visible=visible, seen_groups=set())

    if trade_setup is not None:
        _add_trade_setup(fig, x, trade_setup)

    fig.update_layout(**_base_layout(title, height, x_range=_default_visible_range(x)))
    return fig


def plot_multi_timeframe_zones(
    ltf_df,
    ltf_result: dict,
    htf_zones: list,
    title: str = "Top-Down Entry Chart",
    show_swing_markers: bool = False,
    dark_theme: bool = True,
    visible: dict | None = None,
    max_extend: int = 60,
    height: int = 700,
) -> go.Figure:
    """
    Plots the LTF (entry-timeframe) candlestick chart with HTF zones
    from OTHER, higher timeframes overlaid -- each one tagged and
    color-coded by ITS OWN source timeframe (see top_down.HTFZone),
    so a zone reads unambiguously as e.g. "1H FVG" or "1D OrderBlock"
    rather than just "FVG". Every {timeframe} {poi_type} combination is
    its own legend entry -- click one to show/hide every zone of that
    timeframe+type at once, e.g. turn off "4H FVG" while keeping "1D
    OrderBlock" on.

    ltf_df, ltf_result: the entry timeframe's own OHLCV + analyze_structure()
        output (so its own swings/BOS/CHoCH/IDM can still be drawn if wanted).
    htf_zones: list of top_down.HTFZone -- typically the concatenation of
        collect_htf_zones() calls across every HTF you want overlaid.
    visible: optional {"{TIMEFRAME} {POI_TYPE}": bool} -- e.g.
        {"4H FVG": False} to start that one category hidden.
    """
    visible = visible or {}
    x = ltf_df.index
    n = len(x)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=ltf_df["open"], high=ltf_df["high"], low=ltf_df["low"], close=ltf_df["close"],
                increasing_line_color=BULLISH_COLOR, decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR, decreasing_fillcolor=BEARISH_COLOR,
                increasing_line_width=1, decreasing_line_width=1,
                name="price", showlegend=False,
            )
        ]
    )

    if show_swing_markers:
        swing_highs = [s for s in ltf_result["swings"] if s.kind == "high"]
        swing_lows = [s for s in ltf_result["swings"] if s.kind == "low"]
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_highs] + [x[s.index] for s in swing_lows],
            y=[s.price for s in swing_highs] + [s.price for s in swing_lows],
            mode="markers", marker=dict(symbol="circle", size=4, color=TEXT_MUTED),
            name="Swings", legendgroup="Swings", showlegend=True,
            visible=True if visible.get("Swings", True) else "legendonly",
            hoverinfo="skip",
        ))

    seen_groups = set()
    drawn_x0s = []
    for zone in htf_zones:
        # A zone already mitigated entirely before this LTF chart even
        # starts has nothing left to show here -- drawing it anyway
        # would give it an x1 before x[0], dragging the WHOLE chart's
        # x-axis autorange backward to include a rectangle nobody
        # should see (this was a real bug, caught by inspecting a
        # rendered chart: the x-axis started over a year before the
        # LTF candles themselves).
        if zone.mitigated_at is not None and zone.mitigated_at < x[0]:
            continue

        group = f"{zone.timeframe.upper()} {zone.poi_type}"
        color = HTF_TIMEFRAME_COLORS.get(zone.timeframe, TEXT_SECONDARY)
        if zone.poi_type == "FVG":
            fill = FVG_BULLISH_FILL if zone.direction == "bullish" else FVG_BEARISH_FILL
        else:
            fill = "rgba(0,0,0,0)"

        x0 = max(zone.formed_at, x[0])
        if zone.mitigated_at is not None and zone.mitigated_at <= x[-1]:
            x1 = zone.mitigated_at
        else:
            # still live (or mitigated beyond this LTF chart's own range)
            # -- extend forward by max_extend LTF candles, capped at the
            # last candle, same "don't draw a full-width band" rule
            # plot_structure() uses.
            start_pos = x.searchsorted(x0)
            capped_pos = min(start_pos + max_extend, n - 1)
            x1 = x[capped_pos]

        first = group not in seen_groups
        seen_groups.add(group)
        drawn_x0s.append(x0)
        fig.add_trace(_zone_trace(
            x0, x1, zone.zone_low, zone.zone_high,
            fillcolor=fill, line_color=color, line_width=1.6,
            dash="solid" if zone.poi_type in ("OrderBlock", "ExtremeOB") else ("dashdot" if zone.poi_type == "BreakerBlock" else "dot"),
            name=group, legendgroup=group, showlegend=first,
            hovertext=f"{group} ({zone.direction})",
        ))
        fig.data[-1].visible = True if visible.get(group, True) else "legendonly"

    # Unlike a plain price chart, the whole point here is showing zones
    # relative to price -- default to a window wide enough to include
    # the earliest still-drawn zone, not just the most recent candles,
    # but still capped so it doesn't regress to the "thousands of
    # unreadable candles" problem on a long LTF history.
    if drawn_x0s:
        earliest_zone_pos = min(x.searchsorted(min(drawn_x0s)), n - 1)
        window_start = max(0, earliest_zone_pos - 10, n - 500)
        x_range = [x[window_start], x[-1]]
    else:
        x_range = _default_visible_range(x)

    fig.update_layout(**_base_layout(title, height, x_range=x_range))
    return fig


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from structure_engine import analyze_structure
    from poi_engine import analyze_poi

    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic_test_candles.csv")
    result = analyze_structure(df, lookback=3)
    poi_result = analyze_poi(df, result["swings"], result["events"])

    fig = plot_structure(df, result, poi_result=poi_result, title="Structure + POI Chart (synthetic test data)", show_liquidity=True)
    out_path = Path(__file__).resolve().parents[1] / "data" / "structure_chart_preview.png"
    fig.write_image(str(out_path), width=1600, height=700, scale=2)
    print(f"Saved preview to {out_path}")

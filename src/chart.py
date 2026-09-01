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


# ---------------------------------------------------------------------------
# Lightweight Charts (TradingView) renderer
#
# WHY THIS EXISTS ALONGSIDE THE PLOTLY FUNCTIONS ABOVE: inside Streamlit,
# a Plotly figure is rendered server-side once per rerun -- there is no
# live Python<->JS bridge, so a Plotly chart cannot dynamically refit its
# price axis as the user scroll-zooms or drags the time axis (Plotly's
# own y-autorange considers the FULL dataset, not the currently-visible
# window, and there is no callback to recompute it client-side without a
# custom component). That produced exactly the complaints checked
# directly against a live rendered chart: compressed candles, huge empty
# vertical space, no real scroll-to-zoom, no smooth drag-to-pan. Rather
# than fight Plotly's architecture for something it isn't built for,
# this renders TradingView's own Lightweight Charts library instead, via
# a self-contained HTML+JS component (st.components.v1.html on the
# caller's side) -- which handles auto-fit price scale, scroll-zoom,
# drag-pan, and crosshair OHLC natively, with zero custom JS event
# plumbing needed beyond initialization.
#
# This is presentation only: it consumes the exact same
# analyze_structure()/analyze_poi()/HTFZone objects the Plotly path
# does and invents no new zone, event, or price data.
# ---------------------------------------------------------------------------

import json as _json
from string import Template as _Template


def _unix_seconds(index) -> list[int]:
    """DatetimeIndex -> list of UNIX seconds (Lightweight Charts' native time format)."""
    return (index.astype("int64") // 1_000_000_000).tolist()


def _candle_payload(df) -> tuple[list[dict], dict]:
    times = _unix_seconds(df.index)
    candles = [
        {"time": t, "open": round(float(o), 4), "high": round(float(h), 4),
         "low": round(float(l), 4), "close": round(float(c), 4)}
        for t, o, h, l, c in zip(times, df["open"], df["high"], df["low"], df["close"])
    ]
    volumes = {t: float(v) for t, v in zip(times, df["volume"])}
    return candles, volumes


def _lwc_zone_payload(poi_result: dict, x_times: list, max_extend: int, visible: dict) -> list[dict]:
    """Same zones _add_poi_shapes() draws, as JSON-serializable {t1,t2,p1,p2,color,fill,label} dicts."""
    n = len(x_times)
    zones = []

    def end_time(formation_index, mitigated, mitigated_index):
        if mitigated:
            return x_times[mitigated_index]
        return x_times[min(formation_index + max_extend, n - 1)]

    def add(category, items, index_fn, color_fn, fill_fn, label_fn):
        if not visible.get(category, True):
            return
        for poi in items:
            idx = index_fn(poi)
            zones.append({
                "t1": x_times[idx], "t2": end_time(idx, poi.mitigated, poi.mitigated_index),
                "p1": poi.zone_low, "p2": poi.zone_high,
                "color": color_fn(poi), "fill": fill_fn(poi), "label": label_fn(poi),
            })

    valid_fvgs = [f for f in poi_result.get("fvgs", []) if f.valid]
    add("FVG", valid_fvgs, lambda f: f.end_index,
        lambda f: BULLISH_COLOR if f.direction == "bullish" else BEARISH_COLOR,
        lambda f: FVG_BULLISH_FILL if f.direction == "bullish" else FVG_BEARISH_FILL,
        lambda f: "FVG")
    add("OrderBlock", poi_result.get("order_blocks", []), lambda o: o.index,
        lambda o: BULLISH_COLOR if o.direction == "bullish" else BEARISH_COLOR,
        lambda o: "rgba(34,197,94,0.10)" if o.direction == "bullish" else "rgba(239,68,68,0.10)",
        lambda o: "OB")
    add("ExtremeOB", poi_result.get("extreme_order_blocks", []), lambda o: o.index,
        lambda o: BULLISH_COLOR if o.direction == "bullish" else BEARISH_COLOR,
        lambda o: "rgba(34,197,94,0.16)" if o.direction == "bullish" else "rgba(239,68,68,0.16)",
        lambda o: "★ Extreme OB")
    add("MitigationBlock", poi_result.get("mitigation_blocks", []), lambda m: m.index,
        lambda m: MB_COLOR, lambda m: "rgba(124,134,152,0.10)", lambda m: "MB")
    add("BreakerBlock", poi_result.get("breaker_blocks", []), lambda b: b.index,
        lambda b: BULLISH_COLOR if b.direction == "bullish" else BEARISH_COLOR,
        lambda b: "rgba(34,197,94,0.10)" if b.direction == "bullish" else "rgba(239,68,68,0.10)",
        lambda b: "Breaker")

    return zones


def _lwc_event_lines(events: list, visible: dict, max_per_type: int = 3) -> list[dict]:
    """Price lines are full-width on Lightweight Charts (no time bounds), so
    rendering every historical BOS/CHoCH/IDM ever detected floods the price
    axis with stacked labels. Keep only the most recent few per event type --
    the underlying `events` list (and detection logic) is untouched."""
    by_type: dict[str, list] = {}
    for event in events:
        if event.source_swing_index is None or not visible.get(event.event_type, True):
            continue
        by_type.setdefault(event.event_type, []).append(event)

    lines = []
    for event_type, evs in by_type.items():
        evs.sort(key=lambda e: e.index, reverse=True)
        for event in evs[:max_per_type]:
            color = IDM_COLOR if event.event_type == "IDM" else _event_color(event)
            arrow = "↑" if event.direction == "bullish" else "↓"
            lines.append({
                "price": event.source_swing_price, "color": color,
                "width": 2 if event.event_type == "CHoCH" else 1,
                "style": 2 if event.event_type == "IDM" else 0,   # LightweightCharts.LineStyle: 0=Solid, 2=Dotted
                "title": f"{event.event_type} {arrow}",
            })
    return lines


def _lwc_liquidity_lines(swings: list, visible: dict) -> list[dict]:
    if not visible.get("Liquidity", True):
        return []
    lines = []
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if highs:
        s = max(highs, key=lambda s: s.index)
        lines.append({"price": s.price, "color": LIQUIDITY_COLOR, "title": "BSL"})
    if lows:
        s = max(lows, key=lambda s: s.index)
        lines.append({"price": s.price, "color": LIQUIDITY_COLOR, "title": "SSL"})
    return lines


def _lwc_markers(swings: list, x_times: list, visible: dict) -> list[dict]:
    if not visible.get("Swings", True):
        return []
    markers = [
        {"time": x_times[s.index], "position": "aboveBar" if s.kind == "high" else "belowBar",
         "color": TEXT_MUTED, "shape": "circle", "text": ""}
        for s in swings
    ]
    markers.sort(key=lambda m: m["time"])
    return markers


_CHART_HTML = _Template(r"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  html, body { margin:0; padding:0; background:$bg_primary; overflow:hidden; }
  #wrap { width:100%; height:${height}px; font-family:$font_family; }
  #toolbar { display:flex; align-items:center; gap:6px; padding:6px 10px; background:$bg_card; border-bottom:1px solid $border; height:24px; }
  #toolbar button { background:transparent; color:$text_secondary; border:1px solid $border; border-radius:6px;
    font-size:11px; padding:3px 10px; cursor:pointer; font-family:$font_family; font-weight:600; }
  #toolbar button:hover { border-color:$accent; color:$accent; }
  #toolbar button.active { background:$accent; color:#06110D; border-color:$accent; }
  #toolbar .label { color:$text_muted; font-size:10px; font-weight:700; letter-spacing:0.05em; margin-right:2px; }
  #chart-container { width:100%; height:calc(${height}px - 38px); position:relative; }
  #ohlc-info { position:absolute; top:8px; left:10px; z-index:5; font-size:11px; color:$text_secondary;
    background:rgba(16,22,31,0.85); border:1px solid $border; border-radius:6px; padding:5px 10px;
    pointer-events:none; line-height:1.5; font-variant-numeric:tabular-nums; white-space:nowrap; }
  #ohlc-info b { color:$text_primary; }
  #ohlc-info .up { color:$bullish; } #ohlc-info .down { color:$bearish; }
</style></head>
<body>
<div id="wrap">
  <div id="toolbar">
    <span class="label">VISIBLE</span>
    <button data-n="50">50</button><button data-n="100" class="active">100</button>
    <button data-n="200">200</button><button data-n="500">500</button>
    <button id="reset-btn">Reset View</button>
  </div>
  <div id="chart-container"><div id="ohlc-info"></div></div>
</div>
<script>
(function() {
  const candles = $candles_json, volumes = $volumes_json, zones = $zones_json,
        eventLines = $event_lines_json, liquidityLines = $liquidity_lines_json,
        markers = $markers_json, tradeSetup = $trade_setup_json,
        defaultBars = $default_bars, title = $title_json;

  const container = document.getElementById('chart-container');
  const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth, height: container.clientHeight,
    layout: { background:{color:'$bg_primary'}, textColor:'$text_secondary', fontFamily:'$font_family' },
    grid: { vertLines:{color:'$border'}, horzLines:{color:'$border'} },
    rightPriceScale: { borderColor:'$border' },
    timeScale: { borderColor:'$border', timeVisible:true, secondsVisible:false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: { mouseWheel:true, pressedMouseMove:true, horzTouchDrag:true, vertTouchDrag:false },
    handleScale: { mouseWheel:true, pinch:true, axisPressedMouseMove:true },
  });

  const series = chart.addCandlestickSeries({
    upColor:'$bullish', downColor:'$bearish', borderVisible:false,
    wickUpColor:'$bullish', wickDownColor:'$bearish',
  });
  window.__chart = chart; window.__series = series;  // debug/test introspection hook
  series.setData(candles);
  if (markers.length) series.setMarkers(markers);

  class RectRenderer {
    constructor(p1,p2,fill,stroke,label){ this._p1=p1; this._p2=p2; this._fill=fill; this._stroke=stroke; this._label=label; }
    draw(target) {
      target.useBitmapCoordinateSpace(scope => {
        if (this._p1.x==null||this._p1.y==null||this._p2.x==null||this._p2.y==null) return;
        const ctx = scope.context;
        const x1=this._p1.x*scope.horizontalPixelRatio, y1=this._p1.y*scope.verticalPixelRatio;
        const x2=this._p2.x*scope.horizontalPixelRatio, y2=this._p2.y*scope.verticalPixelRatio;
        const rx=Math.min(x1,x2), ry=Math.min(y1,y2), rw=Math.abs(x2-x1), rh=Math.abs(y2-y1);
        ctx.fillStyle=this._fill; ctx.fillRect(rx,ry,rw,rh);
        ctx.strokeStyle=this._stroke; ctx.lineWidth=Math.max(1,scope.verticalPixelRatio); ctx.strokeRect(rx,ry,rw,rh);
        if (this._label && rw > 26*scope.horizontalPixelRatio) {
          ctx.fillStyle=this._stroke; ctx.font=(10*scope.verticalPixelRatio)+'px $font_family';
          ctx.fillText(this._label, rx+3*scope.horizontalPixelRatio, ry+11*scope.verticalPixelRatio);
        }
      });
    }
  }
  class RectPaneView {
    constructor(src){ this._src=src; this._p1={x:null,y:null}; this._p2={x:null,y:null}; }
    update() {
      const y1=this._src.series.priceToCoordinate(this._src.p1price);
      const y2=this._src.series.priceToCoordinate(this._src.p2price);
      const x1=chart.timeScale().timeToCoordinate(this._src.t1);
      const x2=chart.timeScale().timeToCoordinate(this._src.t2);
      this._p1={x:x1,y:y1}; this._p2={x:x2,y:y2};
    }
    renderer(){ return new RectRenderer(this._p1,this._p2,this._src.fill,this._src.color,this._src.label); }
  }
  class RectPrimitive {
    constructor(series,t1,t2,p1price,p2price,fill,color,label){
      this.series=series; this.t1=t1; this.t2=t2; this.p1price=p1price; this.p2price=p2price;
      this.fill=fill; this.color=color; this.label=label; this._views=[new RectPaneView(this)];
    }
    updateAllViews(){ this._views.forEach(v=>v.update()); }
    paneViews(){ return this._views; }
  }

  zones.forEach(z => series.attachPrimitive(new RectPrimitive(series, z.t1, z.t2, z.p1, z.p2, z.fill, z.color, z.label)));

  eventLines.forEach(l => series.createPriceLine({ price:l.price, color:l.color, lineWidth:l.width, lineStyle:l.style, axisLabelVisible:true, title:l.title }));
  liquidityLines.forEach(l => series.createPriceLine({ price:l.price, color:l.color, lineWidth:1, lineStyle:2, axisLabelVisible:true, title:l.title }));
  if (tradeSetup) {
    series.createPriceLine({ price:tradeSetup.entry, color:'$accent', lineWidth:1, lineStyle:0, axisLabelVisible:true, title:'Entry' });
    series.createPriceLine({ price:tradeSetup.stop, color:'$bearish', lineWidth:1, lineStyle:2, axisLabelVisible:true, title:'Stop' });
    series.createPriceLine({ price:tradeSetup.target, color:'$bullish', lineWidth:1, lineStyle:2, axisLabelVisible:true, title:'Target' });
  }

  function setVisibleBars(n) {
    const total = candles.length;
    if (!total) return;
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, total - n), to: total - 1 + 2 });
  }
  setVisibleBars(defaultBars);

  document.querySelectorAll('#toolbar button[data-n]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#toolbar button[data-n]').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      setVisibleBars(parseInt(btn.dataset.n, 10));
    });
  });
  document.getElementById('reset-btn').addEventListener('click', () => {
    document.querySelectorAll('#toolbar button[data-n]').forEach(b=>b.classList.remove('active'));
    document.querySelector('#toolbar button[data-n="' + defaultBars + '"]')?.classList.add('active');
    setVisibleBars(defaultBars);
  });

  const infoBox = document.getElementById('ohlc-info');
  function fmt(n){ return n.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
  function updateInfo(param) {
    let d = candles[candles.length-1], t = d.time;
    if (param && param.time) {
      const data = param.seriesData.get(series);
      if (data) { d = data; t = param.time; }
    }
    const dt = new Date(t*1000);
    const dateStr = dt.toLocaleDateString(undefined,{day:'2-digit',month:'short',year:'numeric'});
    const vol = volumes[t];
    const cls = d.close >= d.open ? 'up' : 'down';
    infoBox.innerHTML = '<b>'+title+'</b> &nbsp; '+dateStr+
      '<br>O <span class="'+cls+'">'+fmt(d.open)+'</span> H <span class="'+cls+'">'+fmt(d.high)+'</span> '+
      'L <span class="'+cls+'">'+fmt(d.low)+'</span> C <span class="'+cls+'">'+fmt(d.close)+'</span>'+
      (vol !== undefined ? ('<br>Volume '+Math.round(vol).toLocaleString()) : '');
  }
  updateInfo(null);
  chart.subscribeCrosshairMove(updateInfo);

  new ResizeObserver(entries => {
    if (!entries.length) return;
    const { width, height } = entries[0].contentRect;
    chart.applyOptions({ width, height });
  }).observe(container);
})();
</script>
</body></html>
""")


def render_lightweight_chart(
    df,
    result: dict,
    poi_result: dict | None = None,
    title: str = "Market Structure",
    visible: dict | None = None,
    show_liquidity: bool = False,
    trade_setup: dict | None = None,
    height: int = 700,
    default_visible_bars: int = 100,
) -> str:
    """
    Builds a self-contained HTML+JS document embedding a TradingView
    Lightweight Charts candlestick chart with this project's SMC zones/
    events overlaid. Pass the returned string to
    streamlit.components.v1.html(html, height=height). See the module
    docstring's "Lightweight Charts (TradingView) renderer" section for
    why this exists alongside the Plotly functions above.
    """
    visible = visible or {}
    x_times = _unix_seconds(df.index)
    candles, volumes = _candle_payload(df)

    zones = _lwc_zone_payload(poi_result, x_times, max_extend=max(40, default_visible_bars // 2), visible=visible) if poi_result else []
    event_lines = _lwc_event_lines(result["events"], visible)
    liquidity_lines = _lwc_liquidity_lines(result["swings"], visible) if show_liquidity else []
    markers = _lwc_markers(result["swings"], x_times, visible)

    return _CHART_HTML.substitute(
        bg_primary=BG_PRIMARY, bg_card=BG_CARD, border=BORDER, accent=ACCENT,
        bullish=BULLISH_COLOR, bearish=BEARISH_COLOR, text_primary=TEXT_PRIMARY,
        text_secondary=TEXT_SECONDARY, text_muted=TEXT_MUTED, font_family=FONT_FAMILY,
        height=height, default_bars=default_visible_bars,
        candles_json=_json.dumps(candles), volumes_json=_json.dumps(volumes),
        zones_json=_json.dumps(zones), event_lines_json=_json.dumps(event_lines),
        liquidity_lines_json=_json.dumps(liquidity_lines), markers_json=_json.dumps(markers),
        trade_setup_json=_json.dumps(trade_setup), title_json=_json.dumps(title),
    )


def render_lightweight_multi_timeframe_chart(
    ltf_df,
    ltf_result: dict,
    htf_zones: list,
    title: str = "Top-Down Entry Chart",
    height: int = 700,
    default_visible_bars: int = 100,
    visible: dict | None = None,
) -> str:
    """
    Same renderer as render_lightweight_chart(), for the Top-Down page's
    HTF-zones-on-LTF-candles view. htf_zones (top_down.HTFZone) already
    carry real pd.Timestamp bounds, so -- unlike the old Plotly version,
    which had to map row-index positions across differently-sized HTF/LTF
    DataFrames -- this just converts those timestamps straight to UNIX
    seconds. One shared HTML/JS template for both pages, per instruction:
    fix the shared chart component rather than maintaining two.
    """
    import bisect

    visible = visible or {}
    x_times = _unix_seconds(ltf_df.index)
    candles, volumes = _candle_payload(ltf_df)

    n = len(x_times)
    zones = []
    for zone in htf_zones:
        group = f"{zone.timeframe.upper()} {zone.poi_type}"
        if not visible.get(group, True):
            continue

        formed_ts = int(zone.formed_at.timestamp())
        if formed_ts > x_times[-1] or (zone.mitigated_at is not None and int(zone.mitigated_at.timestamp()) < x_times[0]):
            continue  # zone's whole active window falls outside this LTF chart's range

        # HTF zone timestamps rarely land exactly on an LTF candle -- find
        # the nearest LTF candle at/after each boundary (bisect on the
        # already-ascending x_times), same "clamp to what's actually on
        # this chart" rule the old Plotly renderer used.
        start_idx = min(bisect.bisect_left(x_times, formed_ts), n - 1)
        t1 = x_times[start_idx]

        if zone.mitigated_at is not None:
            mitigated_ts = int(zone.mitigated_at.timestamp())
            end_idx = min(bisect.bisect_left(x_times, mitigated_ts), n - 1)
        else:
            end_idx = min(start_idx + max(40, default_visible_bars // 2), n - 1)
        t2 = x_times[end_idx]

        color = HTF_TIMEFRAME_COLORS.get(zone.timeframe, TEXT_SECONDARY)
        fill = (FVG_BULLISH_FILL if zone.direction == "bullish" else FVG_BEARISH_FILL) if zone.poi_type == "FVG" else "rgba(148,163,184,0.08)"
        zones.append({"t1": t1, "t2": t2, "p1": zone.zone_low, "p2": zone.zone_high, "color": color, "fill": fill, "label": group})

    markers = _lwc_markers(ltf_result["swings"], x_times, {"Swings": visible.get("Swings", False)})

    return _CHART_HTML.substitute(
        bg_primary=BG_PRIMARY, bg_card=BG_CARD, border=BORDER, accent=ACCENT,
        bullish=BULLISH_COLOR, bearish=BEARISH_COLOR, text_primary=TEXT_PRIMARY,
        text_secondary=TEXT_SECONDARY, text_muted=TEXT_MUTED, font_family=FONT_FAMILY,
        height=height, default_bars=default_visible_bars,
        candles_json=_json.dumps(candles), volumes_json=_json.dumps(volumes),
        zones_json=_json.dumps(zones), event_lines_json="[]",
        liquidity_lines_json="[]", markers_json=_json.dumps(markers),
        trade_setup_json="null", title_json=_json.dumps(title),
    )


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

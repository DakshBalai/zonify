"""
chart.py
Interactive Plotly candlestick chart with market-structure events drawn
the way SMC charts actually look, not as floating scatter markers.

Convention (per project owner):
  - BOS / CHoCH: a straight line connecting the swing point that got
    broken directly to the candle where the break closed -- solid line.
    CHoCH is drawn slightly thicker than BOS since it's the more
    significant event (a bias flip vs. a continuation).
  - IDM: a straight line connecting the swept swing point directly to
    the candle that swept it (wick) -- dotted line, since a sweep is a
    "grab and reverse", not a structural break.
  - Direction still carries meaning via color: green = bullish event,
    red = bearish event -- same convention a trader reading a real SMC
    chart would expect.

Every category (each POI type, each event type, swing markers) is its
own legend entry with a shared `legendgroup`, and the layout sets
`legend.groupclick="togglegroup"` -- click one legend entry and every
trace/zone in that category shows or hides together. That's the "clean
experience" toggle: turn off what you don't want to see, per chart.

This module only ever CONSUMES the output of structure_engine.py
(analyze_structure()) / poi_engine.py (analyze_poi()) / top_down.py
(HTFZone) -- it doesn't recompute or reinterpret anything, so a chart
bug here can never be confused with a detection-logic bug.
"""

from __future__ import annotations

import plotly.graph_objects as go

BULLISH_COLOR = "#26A69A"   # teal-green, matches typical dark-theme candle up-color
BEARISH_COLOR = "#EF5350"   # red, matches typical dark-theme candle down-color
IDM_COLOR = "#B39DDB"       # muted purple, visually distinct from BOS/CHoCH

EVENT_LINE_STYLE = {
    "BOS": {"dash": "solid", "width": 1.5},
    "CHoCH": {"dash": "solid", "width": 2.5},
    "IDM": {"dash": "dot", "width": 1.5},
}

FVG_BULLISH_COLOR = "rgba(38,166,154,0.18)"   # translucent teal fill
FVG_BEARISH_COLOR = "rgba(239,83,80,0.18)"    # translucent red fill
FVG_INVALID_COLOR = "rgba(150,150,150,0.10)"  # faint grey -- valid=False (wrong side of premium/discount)
OB_BULLISH_BORDER = "#26A69A"
OB_BEARISH_BORDER = "#EF5350"
EOB_BULLISH_BORDER = "#00E676"  # brighter green -- ExtremeOB is the strongest-performing signal, deserves to stand out
EOB_BEARISH_BORDER = "#FF1744"
MB_BORDER = "#B39DDB"
BRK_BULLISH_BORDER = "#FFA726"  # orange -- visually distinct from OB's teal/red so a flipped zone reads at a glance
BRK_BEARISH_BORDER = "#FF7043"

# One legend entry per higher-timeframe, used by plot_multi_timeframe_zones()
# to keep "1H FVG" visually distinct from "4H FVG" even though both are FVGs.
HTF_TIMEFRAME_COLORS = {
    "monthly": "#E040FB", "weekly": "#7C4DFF", "daily": "#448AFF",
    "4h": "#00E5FF", "1h": "#FFD740",
}


def _event_color(event) -> str:
    if event.event_type == "IDM":
        return IDM_COLOR
    return BULLISH_COLOR if event.direction == "bullish" else BEARISH_COLOR


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
    Draws FVGs as translucent filled zones (grey if invalid -- i.e. on
    the wrong side of premium/discount -- unless show_invalid_fvgs=
    True), Order Blocks / Extreme Order Blocks / Mitigation Blocks /
    Breaker Blocks as outlined zones (solid border for OB, long-dash
    for ExtremeOB, dotted for MB, dash-dot for Breaker -- matching the
    BOS/CHoCH-solid vs IDM-dotted convention used for the event lines).
    Each zone extends from its formation candle to its mitigation
    candle, or -- if still unmitigated/"live" -- forward by at most
    `max_extend` candles (capped at the last candle on the chart).
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
        lambda f: "rgba(150,150,150,0.6)" if not f.valid else (BULLISH_COLOR if f.direction == "bullish" else BEARISH_COLOR),
        lambda f: FVG_INVALID_COLOR if not f.valid else (FVG_BULLISH_COLOR if f.direction == "bullish" else FVG_BEARISH_COLOR),
        0, "solid",
        lambda f: f"FVG ({f.direction}){' -- invalid' if not f.valid else ''}",
    )

    _add(
        "OrderBlock", poi_result.get("order_blocks", []), lambda o: o.index,
        lambda o: OB_BULLISH_BORDER if o.direction == "bullish" else OB_BEARISH_BORDER,
        lambda o: "rgba(0,0,0,0)", 1, "solid",
        lambda o: f"OrderBlock ({o.direction})",
    )

    _add(
        "ExtremeOB", poi_result.get("extreme_order_blocks", []), lambda o: o.index,
        lambda o: EOB_BULLISH_BORDER if o.direction == "bullish" else EOB_BEARISH_BORDER,
        lambda o: "rgba(0,0,0,0)", 1.5, "longdash",
        lambda o: f"ExtremeOB ({o.direction})",
    )

    _add(
        "MitigationBlock", poi_result.get("mitigation_blocks", []), lambda m: m.index,
        lambda m: MB_BORDER, lambda m: "rgba(0,0,0,0)", 1, "dot",
        lambda m: f"MitigationBlock ({m.direction})",
    )

    _add(
        "BreakerBlock", poi_result.get("breaker_blocks", []), lambda b: b.index,
        lambda b: BRK_BULLISH_BORDER if b.direction == "bullish" else BRK_BEARISH_BORDER,
        lambda b: "rgba(0,0,0,0)", 1, "dashdot",
        lambda b: f"BreakerBlock ({b.direction})",
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
) -> go.Figure:
    """
    df:     the same OHLCV DataFrame passed into analyze_structure()
    result: the dict returned by structure_engine.analyze_structure(df)
            (keys: swings, events, current_bias, last_event, ...)
    visible: optional {category: bool} controlling which categories are
        shown by default -- "Swings", "BOS", "CHoCH", "IDM", "FVG",
        "OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock".
        Every category defaults to True (visible) if omitted; setting
        one to False draws it hidden but still toggleable from the
        legend (it isn't removed, just starts unchecked).

    Returns a plotly Figure -- call .show() locally, or in Streamlit
    use st.plotly_chart(fig, use_container_width=True). Every category
    is independently toggleable by clicking its legend entry.
    """
    visible = visible or {}

    def _vis(category: str):
        return True if visible.get(category, True) else "legendonly"

    x = df.index

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR,
                name="price", showlegend=False,
            )
        ]
    )

    if show_swing_markers:
        swing_highs = [s for s in result["swings"] if s.kind == "high"]
        swing_lows = [s for s in result["swings"] if s.kind == "low"]
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_highs], y=[s.price for s in swing_highs],
            mode="markers", marker=dict(symbol="circle", size=5, color="rgba(255,193,7,0.85)",
                                          line=dict(width=1, color="rgba(255,193,7,1)")),
            name="Swings", legendgroup="Swings", showlegend=True, visible=_vis("Swings"),
            hoverinfo="text", hovertext="swing high",
        ))
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_lows], y=[s.price for s in swing_lows],
            mode="markers", marker=dict(symbol="circle", size=5, color="rgba(255,193,7,0.85)",
                                          line=dict(width=1, color="rgba(255,193,7,1)")),
            name="Swings", legendgroup="Swings", showlegend=False, visible=_vis("Swings"),
            hoverinfo="text", hovertext="swing low",
        ))

    # Each event is drawn the way a trader actually marks it on a real
    # SMC chart (per reference screenshot): a HORIZONTAL line held at
    # the swing point's price level, running flat from the swing
    # candle across to the candle that broke/swept it, then a short
    # vertical "elbow" dropping down to the actual break price.
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

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[level, level],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            name=group, legendgroup=group, showlegend=first, visible=_vis(group),
            hoverinfo="text", hovertext=f"{event.event_type} ({event.direction})",
        ))
        fig.add_trace(go.Scatter(
            x=[x1, x1], y=[level, event.price],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            name=group, legendgroup=group, showlegend=False, visible=_vis(group),
            hoverinfo="skip",
        ))

    if poi_result is not None:
        _add_poi_shapes(fig, x, poi_result, show_invalid_fvgs, poi_max_extend, visible=visible, seen_groups=set())

    template = "plotly_dark" if dark_theme else "plotly_white"
    fig.update_layout(
        title=title,
        template=template,
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=40, r=140, t=50, b=40),
        legend=dict(groupclick="togglegroup", orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    )
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
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR,
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
            mode="markers", marker=dict(symbol="circle", size=4, color="rgba(255,193,7,0.7)"),
            name="Swings", legendgroup="Swings", showlegend=True,
            visible=True if visible.get("Swings", True) else "legendonly",
            hoverinfo="skip",
        ))

    seen_groups = set()
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
        color = HTF_TIMEFRAME_COLORS.get(zone.timeframe, "#9E9E9E")
        if zone.poi_type == "FVG":
            fill = "rgba(38,166,154,0.12)" if zone.direction == "bullish" else "rgba(239,83,80,0.12)"
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
        fig.add_trace(_zone_trace(
            x0, x1, zone.zone_low, zone.zone_high,
            fillcolor=fill, line_color=color, line_width=1.5,
            dash="solid" if zone.poi_type in ("OrderBlock", "ExtremeOB") else ("dashdot" if zone.poi_type == "BreakerBlock" else "dot"),
            name=group, legendgroup=group, showlegend=first,
            hovertext=f"{group} ({zone.direction})",
        ))
        fig.data[-1].visible = True if visible.get(group, True) else "legendonly"

    template = "plotly_dark" if dark_theme else "plotly_white"
    fig.update_layout(
        title=title,
        template=template,
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=40, r=160, t=50, b=40),
        legend=dict(groupclick="togglegroup", orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    )
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

    fig = plot_structure(df, result, poi_result=poi_result, title="Structure + POI Chart (synthetic test data)")
    out_path = Path(__file__).resolve().parents[1] / "data" / "structure_chart_preview.png"
    fig.write_image(str(out_path), width=1600, height=700, scale=2)
    print(f"Saved preview to {out_path}")

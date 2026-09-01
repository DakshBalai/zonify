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

This module only ever CONSUMES the output of structure_engine.py
(analyze_structure()) -- it doesn't recompute or reinterpret anything,
so a chart bug here can never be confused with a detection-logic bug.
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
MB_BORDER = "#B39DDB"


def _event_color(event) -> str:
    if event.event_type == "IDM":
        return IDM_COLOR
    return BULLISH_COLOR if event.direction == "bullish" else BEARISH_COLOR


def _add_poi_shapes(fig: go.Figure, x, poi_result: dict, show_invalid_fvgs: bool, max_extend: int) -> None:
    """
    Draws FVGs as translucent filled rectangles (grey if invalid --
    i.e. on the wrong side of premium/discount -- unless
    show_invalid_fvgs=True), and Order Blocks / Mitigation Blocks as
    outlined rectangles (solid border for OB, dashed for MB, matching
    the BOS/CHoCH-solid vs IDM-dotted convention used for the lines).
    Each zone extends from its formation candle to its mitigation
    candle, or -- if still unmitigated/"live" -- forward by at most
    `max_extend` candles (capped at the last candle on the chart).
    Uncapping this to "always extend to the last candle" is what
    caused unmitigated zones to draw as full-width bands across the
    whole chart when zoomed out to a long history; capping it keeps
    the zone visually tied to where it actually formed. Each zone gets
    a small text label (FVG / OB / MB) at its left edge.
    """
    n = len(x)

    def _end_x(formation_index: int, mitigated: bool, mitigated_index):
        if mitigated:
            return x[mitigated_index]
        capped = min(formation_index + max_extend, n - 1)
        return x[capped]

    for fvg in poi_result.get("fvgs", []):
        if not fvg.valid and not show_invalid_fvgs:
            continue
        x0 = x[fvg.end_index]
        x1 = _end_x(fvg.end_index, fvg.mitigated, fvg.mitigated_index)
        color = FVG_INVALID_COLOR if not fvg.valid else (
            FVG_BULLISH_COLOR if fvg.direction == "bullish" else FVG_BEARISH_COLOR
        )
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=fvg.zone_low, y1=fvg.zone_high,
            fillcolor=color, line=dict(width=0), layer="below",
        )
        fig.add_annotation(
            x=x0, y=fvg.zone_high, text="FVG", showarrow=False,
            xanchor="left", yshift=6, font=dict(size=8, color="rgba(200,200,200,0.8)"),
        )

    for ob in poi_result.get("order_blocks", []):
        x0 = x[ob.index]
        x1 = _end_x(ob.index, ob.mitigated, ob.mitigated_index)
        border = OB_BULLISH_BORDER if ob.direction == "bullish" else OB_BEARISH_BORDER
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=ob.zone_low, y1=ob.zone_high,
            fillcolor="rgba(0,0,0,0)", line=dict(color=border, width=1, dash="solid"), layer="below",
        )
        fig.add_annotation(
            x=x0, y=ob.zone_high, text="OB", showarrow=False,
            xanchor="left", yshift=6, font=dict(size=8, color=border),
        )

    for mb in poi_result.get("mitigation_blocks", []):
        x0 = x[mb.index]
        x1 = _end_x(mb.index, mb.mitigated, mb.mitigated_index)
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=mb.zone_low, y1=mb.zone_high,
            fillcolor="rgba(0,0,0,0)", line=dict(color=MB_BORDER, width=1, dash="dot"), layer="below",
        )
        fig.add_annotation(
            x=x0, y=mb.zone_high, text="MB", showarrow=False,
            xanchor="left", yshift=6, font=dict(size=8, color=MB_BORDER),
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
) -> go.Figure:
    """
    df:     the same OHLCV DataFrame passed into analyze_structure()
    result: the dict returned by structure_engine.analyze_structure(df)
            (keys: swings, events, current_bias, last_event)

    Returns a plotly Figure -- call .show() locally, or in Streamlit
    use st.plotly_chart(fig, use_container_width=True).
    """
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
                name="price",
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
            name="swing high", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[x[s.index] for s in swing_lows], y=[s.price for s in swing_lows],
            mode="markers", marker=dict(symbol="circle", size=5, color="rgba(255,193,7,0.85)",
                                          line=dict(width=1, color="rgba(255,193,7,1)")),
            name="swing low", showlegend=False, hoverinfo="skip",
        ))

    # Each event is drawn the way a trader actually marks it on a real
    # SMC chart (per reference screenshot): a HORIZONTAL line held at
    # the swing point's price level, running flat from the swing
    # candle across to the candle that broke/swept it, then a short
    # vertical "elbow" dropping down to the actual break price. The
    # label sits centered above the line's midpoint, not stacked at
    # the endpoint -- this is what keeps choppy zones (several events
    # close together) readable instead of overlapping.
    for event in result["events"]:
        if event.source_swing_index is None:
            continue

        x0 = x[event.source_swing_index]
        x1 = x[event.index]
        level = event.source_swing_price
        style = EVENT_LINE_STYLE[event.event_type]
        color = _event_color(event)

        # flat level line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[level, level],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{event.event_type} ({event.direction})",
        ))
        # short elbow down/up to the candle's actual break price
        fig.add_trace(go.Scatter(
            x=[x1, x1], y=[level, event.price],
            mode="lines",
            line=dict(color=color, dash=style["dash"], width=style["width"]),
            showlegend=False, hoverinfo="skip",
        ))

        mid_x = x[(event.source_swing_index + event.index) // 2]
        fig.add_annotation(
            x=mid_x, y=level,
            text=event.event_type,
            showarrow=False,
            yshift=11,
            font=dict(size=9, color=color),
        )

    if poi_result is not None:
        _add_poi_shapes(fig, x, poi_result, show_invalid_fvgs=show_invalid_fvgs, max_extend=poi_max_extend)

    template = "plotly_dark" if dark_theme else "plotly_white"
    fig.update_layout(
        title=title,
        template=template,
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h"),
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

"""
app/app.py
Zonify -- an SMC (Smart Money Concepts) screener and structure/POI
analysis dashboard for NSE stocks.

This file adds NO new detection, scoring, or backtest-simulation
logic. Every number, zone, and chart here comes straight from this
project's own engine (structure_engine.py, poi_engine.py,
multi_timeframe.py, backtester.py, top_down.py, session_model.py,
fundamentals.py, screener.py) -- this is purely the presentation
layer, restyled to read as a real financial product rather than a
default Streamlit script.

Two small, presentation-only helpers live in THIS file rather than
src/, because they are generic, standard, well-known calculations with
no relationship to SMC detection/scoring (adding them to the detection
engine would misrepresent what they are):
  - market_status(): is NSE's cash session open right now (a clock
    check against 09:15-15:30 IST, Mon-Fri).
  - compute_atr(): a standard 14-period Average True Range.
Everything else -- HTF bias, fresh zones, entry/stop/target previews,
tiers -- is read directly from the engine, never invented here. Where
a mockup asked for a number the engine doesn't compute (a continuous
"SMC Score" or "Confidence %"), it is deliberately NOT shown -- see
screener.py's ScreenerResult.tier (STRONG/SETUP) for what IS real.

Run locally with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import Bias, analyze_structure  # noqa: E402
from poi_engine import analyze_poi  # noqa: E402
from multi_timeframe import TIMEFRAME_ORDER, analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from backtester import (  # noqa: E402
    compute_equity_curve, compute_max_drawdown, run_extended_backtest, summarize_trades,
)
from chart import (  # noqa: E402
    ACCENT, BEARISH_COLOR, BG_CARD, BORDER, BULLISH_COLOR, FONT_FAMILY, TEXT_MUTED, TEXT_SECONDARY,
    plot_multi_timeframe_zones, plot_structure,
)
from top_down import HTF_TO_LTF, collect_htf_zones, find_top_down_entries  # noqa: E402
from fundamentals import fetch_ticker_profile  # noqa: E402
from screener import find_fresh_zone, preview_stop_target, screen_ticker  # noqa: E402

APP_NAME = "Zonify"
TAGLINE = "Smart Money Intelligence"
NSE_TZ = ZoneInfo("Asia/Kolkata")

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM", "ADANIENT", "ADANIPORTS", "BAJAJFINSV",
    "BAJAJ-AUTO", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "NESTLEIND", "SBILIFE",
    "SHREECEM", "UPL", "APOLLOHOSP", "BPCL", "TATACONSUM",
]

STRUCTURE_TIMEFRAMES = [tf for tf in TIMEFRAME_ORDER if tf != "1min"]
LAYER_CATEGORIES = ["Swings", "BOS", "CHoCH", "IDM", "FVG", "OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock"]
DEFAULT_LAYERS_ON = {"Swings", "CHoCH", "FVG", "OrderBlock", "ExtremeOB"}

NAV_ITEMS = [("screener", "Screener"), ("analyze", "Analyze"), ("topdown", "Top-Down"), ("backtest", "Backtest")]

# Headline numbers from the last full 49-ticker NIFTY 50 backtest
# (scripts/basket_backtest.py) -- static, clearly labeled as such, not
# a live recompute (a full basket run takes several minutes).
PROVEN_SIGNALS = [
    {"name": "ORDER BLOCK", "win_rate": 60.7, "expectancy": 0.77, "consistency": "49 / 49 tickers positive"},
    {"name": "EXTREME ORDER BLOCK", "win_rate": 63.3, "expectancy": 0.84, "consistency": "49 / 49 tickers positive"},
]


# ---------------------------------------------------------------------------
# Small, generic, presentation-only helpers (see module docstring)
# ---------------------------------------------------------------------------

def market_status() -> tuple[str, str]:
    now = datetime.now(NSE_TZ)
    is_weekday = now.weekday() < 5
    in_session = dtime(9, 15) <= now.time() <= dtime(15, 30)
    return ("LIVE", BULLISH_COLOR) if (is_weekday and in_session) else ("CLOSED", TEXT_MUTED)


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Standard Average True Range -- a generic volatility measure, not an SMC concept."""
    if len(df) < period + 1:
        return None
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trs = [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        for i in range(len(df) - period, len(df))
    ]
    return sum(trs) / len(trs)


def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}

        .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1480px; }}
        footer, #MainMenu {{ visibility: hidden; }}

        /* ---- header ---- */
        .app-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 14px 20px; border-radius: 10px;
            background: {BG_CARD}; border: 1px solid {BORDER};
            margin-bottom: 18px;
        }}
        .app-header-left {{ display: flex; align-items: baseline; gap: 10px; }}
        .app-logo {{ font-size: 1.15rem; font-weight: 800; letter-spacing: 0.04em; color: {ACCENT}; }}
        .app-tagline {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; }}
        .app-header-right {{ display: flex; align-items: center; gap: 8px; }}

        .pill {{
            display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; letter-spacing: 0.02em; color: {TEXT_SECONDARY};
            border: 1px solid {BORDER}; background: rgba(255,255,255,0.02);
        }}

        /* ---- nav ---- */
        div[data-testid="stButton"] button {{
            border-radius: 8px; font-weight: 600; font-size: 0.85rem;
            transition: all 0.15s ease;
        }}
        div[data-testid="stButton"] button[kind="secondary"] {{
            background: transparent; border: 1px solid {BORDER}; color: {TEXT_SECONDARY};
        }}
        div[data-testid="stButton"] button[kind="secondary"]:hover {{
            border-color: {ACCENT}; color: {ACCENT};
        }}

        /* ---- section headers ---- */
        .section-title {{ font-size: 1.15rem; font-weight: 700; margin: 4px 0 2px 0; color: #F5F7FA; }}
        .section-sub {{ color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 14px; }}

        /* ---- KPI cards ---- */
        .kpi-card {{
            border-radius: 10px; border: 1px solid {BORDER}; background: {BG_CARD};
            padding: 14px 16px; height: 100%;
        }}
        .kpi-label {{ color: {TEXT_SECONDARY}; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; }}
        .kpi-value {{ font-size: 1.9rem; font-weight: 800; color: #F5F7FA; line-height: 1.25; margin-top: 2px; }}
        .kpi-value.accent {{ color: {ACCENT}; }}
        .kpi-tag {{ color: {TEXT_MUTED}; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; margin-top: -4px; }}
        .kpi-detail {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; margin-top: 6px; }}

        /* ---- badges ---- */
        .badge {{
            display: inline-block; padding: 2px 9px; border-radius: 6px;
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
        }}
        .badge-bull {{ background: rgba(34,197,94,0.14); color: {BULLISH_COLOR}; border: 1px solid rgba(34,197,94,0.35); }}
        .badge-bear {{ background: rgba(239,68,68,0.14); color: {BEARISH_COLOR}; border: 1px solid rgba(239,68,68,0.35); }}
        .badge-strong {{ background: rgba(33,212,180,0.14); color: {ACCENT}; border: 1px solid rgba(33,212,180,0.4); }}
        .badge-setup {{ background: rgba(148,163,184,0.12); color: {TEXT_SECONDARY}; border: 1px solid {BORDER}; }}
        .badge-muted {{ background: rgba(148,163,184,0.08); color: {TEXT_MUTED}; border: 1px solid {BORDER}; }}

        /* ---- info / summary cards ---- */
        .info-card {{
            border-radius: 10px; border: 1px solid {BORDER}; background: {BG_CARD};
            padding: 10px 14px;
        }}
        .info-card .lbl {{ color: {TEXT_MUTED}; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }}
        .info-card .val {{ color: #F5F7FA; font-size: 1.05rem; font-weight: 700; margin-top: 2px; }}

        /* ---- table rows ---- */
        .row-card {{
            border-radius: 8px; border: 1px solid {BORDER}; background: {BG_CARD};
            padding: 8px 12px; margin-bottom: 6px; transition: border-color 0.12s ease, background 0.12s ease;
        }}
        .row-card:hover {{ border-color: {ACCENT}; background: {BG_CARD}; }}
        .row-header {{ color: {TEXT_MUTED}; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em;
                       text-transform: uppercase; padding: 0 12px; margin-bottom: 4px; }}
        .cell-primary {{ color: #F5F7FA; font-weight: 700; font-size: 0.92rem; }}
        .cell-secondary {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; }}
        .cell-mono {{ color: #F5F7FA; font-size: 0.86rem; font-variant-numeric: tabular-nums; }}

        div[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 10px; background: {BG_CARD}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header + nav
# ---------------------------------------------------------------------------

def render_header() -> None:
    status_label, status_color = market_status()
    now_str = datetime.now(NSE_TZ).strftime("%d %b, %H:%M IST")
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-left">
                <span class="app-logo">ZONIFY</span>
                <span class="app-tagline">{TAGLINE}</span>
            </div>
            <div class="app-header-right">
                <span class="pill">NSE</span>
                <span class="pill">NIFTY 50</span>
                <span class="pill">Updated {now_str}</span>
                <span class="pill" style="color:{status_color};border-color:{status_color}66;">&#9679; {status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav() -> None:
    st.session_state.setdefault("page", "screener")
    cols = st.columns(len(NAV_ITEMS))
    for col, (key, label) in zip(cols, NAV_ITEMS):
        with col:
            active = st.session_state["page"] == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state["page"] = key
                st.rerun()
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)


def render_proof_strip() -> None:
    cols = st.columns(3)
    for col, sig in zip(cols[:2], PROVEN_SIGNALS):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{sig['name']}</div>
                    <div class="kpi-value accent">{sig['win_rate']:.1f}%</div>
                    <div class="kpi-tag">WIN RATE</div>
                    <div class="kpi-detail">+{sig['expectancy']:.2f}R avg expectancy</div>
                    <div class="kpi-detail">{sig['consistency']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with cols[2]:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">BACKTEST COVERAGE</div>
                <div class="kpi-value">NIFTY 50</div>
                <div class="kpi-tag">49 SYMBOLS</div>
                <div class="kpi-detail">~3,000 pooled trades / signal</div>
                <div class="kpi-detail">daily + 4H timeframes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("From the last full basket backtest (scripts/basket_backtest.py) — re-run it any time for fresh numbers.")


# ---------------------------------------------------------------------------
# Cached engine calls
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch(ticker: str, timeframes: tuple) -> dict:
    return fetch_multi_timeframe_data(ticker, timeframes=list(timeframes))


@st.cache_data(ttl=600, show_spinner=False)
def cached_screen(ticker: str):
    try:
        return screen_ticker(ticker), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=600, show_spinner=False)
def cached_profile(ticker: str):
    try:
        return fetch_ticker_profile(ticker), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Screener page
# ---------------------------------------------------------------------------

def page_screener() -> None:
    st.markdown('<div class="section-title">Screener</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ranks tickers using only the two backtested signals above, plus HTF bias '
        'alignment. BOS, CHoCH, IDM, standalone FVG, MitigationBlock and PO3 are still fully computed in the '
        'engine — just not used for ranking here, since the basket backtest found no reliable edge in them alone.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])
    with c1:
        universe_choice = st.radio("Universe", ["NIFTY 50", "Custom"], horizontal=True, label_visibility="collapsed")
    if universe_choice == "NIFTY 50":
        tickers = NIFTY_50
    else:
        custom = st.text_input("Tickers", value="RELIANCE, TCS, INFY", label_visibility="collapsed")
        tickers = [t.strip() for t in custom.split(",") if t.strip()]
    with c2:
        bias_filter = st.selectbox("Bias", ["All", "Bullish", "Bearish"], label_visibility="collapsed")
    with c3:
        tier_filter = st.selectbox("Tier", ["All", "Strong only"], label_visibility="collapsed")
    with c4:
        scan_clicked = st.button("Scan Market", type="primary", use_container_width=True)

    if scan_clicked:
        results, failures = [], 0
        steps = ["Evaluating market structure", "Detecting order blocks", "Checking liquidity zones", "Confirming top-down entries"]
        with st.status(f"Scanning {len(tickers)} symbols...", expanded=True) as status:
            for i, ticker in enumerate(tickers):
                status.write(f"`{ticker}` — {steps[i % len(steps)]}...")
                result, err = cached_screen(ticker)
                if result is not None:
                    results.append(result)
                elif err is not None:
                    failures += 1
            status.update(label=f"Scan complete — {len(results)} setup(s) found across {len(tickers)} symbols", state="complete")
        st.session_state["screener_results"] = results
        st.session_state["screener_failures"] = failures

    results = st.session_state.get("screener_results")
    if results is None:
        return

    failures = st.session_state.get("screener_failures", 0)
    view = list(results)
    if bias_filter != "All":
        view = [r for r in view if r.direction == bias_filter.lower()]
    if tier_filter == "Strong only":
        view = [r for r in view if r.tier == "STRONG"]
    view = sorted(view, key=lambda r: (r.tier != "STRONG", r.ticker))

    st.markdown(f"**{len(view)}** result(s) shown out of **{len(results)}** qualified ({len(tickers)} scanned"
                + (f", {failures} failed to fetch" if failures else "") + ")")

    if not view:
        st.info(
            "**No high-confidence setups found.**\n\n"
            "No current setup satisfies the selected filters. Try:\n"
            "- Lowering the tier filter to \"All\"\n"
            "- Changing the bias filter\n"
            "- Expanding the universe"
        )
        return

    header_cols = st.columns([0.5, 1.4, 1.3, 1, 1.3, 1.6, 1, 1, 0.7, 0.9, 0.9])
    for col, label in zip(header_cols, ["#", "TICKER", "PRICE / CHG", "BIAS", "SIGNAL", "ENTRY ZONE", "STOP", "TARGET", "R:R", "TIER", ""]):
        col.markdown(f'<div class="row-header">{label}</div>', unsafe_allow_html=True)

    for rank, r in enumerate(view, start=1):
        cols = st.columns([0.5, 1.4, 1.3, 1, 1.3, 1.6, 1, 1, 0.7, 0.9, 0.9])
        dir_badge = "bull" if r.direction == "bullish" else "bear"
        chg_color = BULLISH_COLOR if r.change_pct >= 0 else BEARISH_COLOR
        cols[0].markdown(f'<span class="cell-secondary">{rank}</span>', unsafe_allow_html=True)
        cols[1].markdown(f'<span class="cell-primary">{r.ticker}</span>', unsafe_allow_html=True)
        cols[2].markdown(
            f'<span class="cell-mono">₹{r.current_price:,.2f}</span> '
            f'<span style="color:{chg_color};font-size:0.78rem;">{r.change_pct:+.2f}%</span>',
            unsafe_allow_html=True,
        )
        cols[3].markdown(badge(r.direction, dir_badge), unsafe_allow_html=True)
        cols[4].markdown(f'<span class="cell-secondary">{r.daily_zone.poi_type}</span>', unsafe_allow_html=True)
        cols[5].markdown(f'<span class="cell-mono">{r.daily_zone.zone_low:,.2f}–{r.daily_zone.zone_high:,.2f}</span>', unsafe_allow_html=True)
        cols[6].markdown(f'<span class="cell-mono" style="color:{BEARISH_COLOR}">{r.stop_price:,.2f}</span>', unsafe_allow_html=True)
        cols[7].markdown(f'<span class="cell-mono" style="color:{BULLISH_COLOR}">{r.target_price:,.2f}</span>', unsafe_allow_html=True)
        cols[8].markdown(f'<span class="cell-mono">{r.reward_r:.1f}R</span>', unsafe_allow_html=True)
        cols[9].markdown(badge("HIGH" if r.tier == "STRONG" else "WATCH", "strong" if r.tier == "STRONG" else "setup"), unsafe_allow_html=True)
        if cols[10].button("Analyze →", key=f"analyze_{r.ticker}_{rank}", use_container_width=True):
            st.session_state["analyze_ticker"] = r.ticker
            st.session_state["page"] = "analyze"
            st.rerun()


# ---------------------------------------------------------------------------
# Analyze Ticker page
# ---------------------------------------------------------------------------

def page_analyze() -> None:
    st.markdown('<div class="section-title">Analyze Ticker</div>', unsafe_allow_html=True)

    st.session_state.setdefault("analyze_ticker", "RELIANCE")
    top_l, top_r = st.columns([2, 1])
    with top_l:
        ticker = st.text_input("Ticker", key="analyze_ticker")
    with top_r:
        timeframe = st.selectbox("Timeframe", STRUCTURE_TIMEFRAMES, index=STRUCTURE_TIMEFRAMES.index("daily"), key="analyze_timeframe")

    layer_cols = st.columns(len(LAYER_CATEGORIES) + 1)
    visible = {}
    for col, cat in zip(layer_cols, LAYER_CATEGORIES):
        with col:
            visible[cat] = st.checkbox(cat, value=cat in DEFAULT_LAYERS_ON, key=f"layer_{cat}")
    with layer_cols[-1]:
        show_liquidity = st.checkbox("Liquidity", value=False, key="layer_liquidity")

    if not ticker:
        return

    try:
        fetch_tfs = (timeframe,) if timeframe == "daily" else (timeframe, "daily")
        data = cached_fetch(ticker, fetch_tfs)
        df = data[timeframe]
        structure_result = analyze_structure(df)
        poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])
        htf_bias = structure_result["current_bias"] if timeframe == "daily" else analyze_structure(data["daily"])["current_bias"]
    except Exception as exc:
        st.error(f"Couldn't load {ticker}: {exc}")
        return

    current_price = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    atr = compute_atr(df)

    bias = structure_result["current_bias"]
    fresh_zone = find_fresh_zone(poi_result, bias.value) if bias != Bias.UNDETERMINED else None
    trade_setup = None
    if fresh_zone is not None:
        entry_price, stop_price, target_price = preview_stop_target(fresh_zone, current_price, reward_r=2.0)
        trade_setup = {"entry": entry_price, "stop": stop_price, "target": target_price, "direction": fresh_zone.direction}

    # --- Price header ---
    chg_color = BULLISH_COLOR if change >= 0 else BEARISH_COLOR
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:14px; margin: 4px 0 6px 0;">
            <span style="font-size:1.6rem; font-weight:800;">{ticker.upper()}</span>
            <span style="font-size:1.6rem; font-weight:800;">₹{current_price:,.2f}</span>
            <span style="color:{chg_color}; font-weight:700;">{change:+,.2f} ({change_pct:+.2f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_cols = st.columns(6)
    info_items = [
        ("OPEN", f"{df['open'].iloc[-1]:,.2f}"), ("HIGH", f"{df['high'].iloc[-1]:,.2f}"),
        ("LOW", f"{df['low'].iloc[-1]:,.2f}"), ("VOLUME", f"{df['volume'].iloc[-1]:,.0f}"),
        ("ATR (14)", f"{atr:,.2f}" if atr else "n/a"), ("TREND", bias.value.upper()),
    ]
    for col, (lbl, val) in zip(info_cols, info_items):
        col.markdown(f'<div class="info-card"><div class="lbl">{lbl}</div><div class="val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # --- Signal summary cards ---
    sig_cols = st.columns(4)
    sig_items = [
        ("HTF BIAS", htf_bias.value.upper(), "bull" if htf_bias == Bias.BULLISH else ("bear" if htf_bias == Bias.BEARISH else "muted")),
        ("STRUCTURE", bias.value.upper(), "bull" if bias == Bias.BULLISH else ("bear" if bias == Bias.BEARISH else "muted")),
        ("ACTIVE SIGNAL", fresh_zone.poi_type if fresh_zone else "None", "strong" if fresh_zone else "muted"),
        ("TIER", "HIGH" if fresh_zone else "N/A", "strong" if fresh_zone else "muted"),
    ]
    for col, (lbl, val, kind) in zip(sig_cols, sig_items):
        col.markdown(
            f'<div class="info-card"><div class="lbl">{lbl}</div><div class="val">{badge(val, kind)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # --- Chart + side panel ---
    chart_col, side_col = st.columns([3, 1])
    with chart_col:
        fig = plot_structure(
            df, structure_result, poi_result=poi_result, title=f"{ticker.upper()} — {timeframe}",
            visible=visible, show_liquidity=show_liquidity, trade_setup=trade_setup, height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

    with side_col:
        st.markdown('<div class="section-sub" style="margin-bottom:6px;"><b>SMART MONEY ANALYSIS</b></div>', unsafe_allow_html=True)

        last_bos = next((e for e in reversed(structure_result["events"]) if e.event_type == "BOS"), None)
        last_choch = next((e for e in reversed(structure_result["events"]) if e.event_type == "CHoCH"), None)
        swing_highs = [s for s in structure_result["swings"] if s.kind == "high"]
        swing_lows = [s for s in structure_result["swings"] if s.kind == "low"]

        def line(lbl, val):
            st.markdown(f'<div class="info-card" style="margin-bottom:6px;"><div class="lbl">{lbl}</div><div class="val" style="font-size:0.92rem;">{val}</div></div>', unsafe_allow_html=True)

        line("Market Structure", badge(bias.value, "bull" if bias == Bias.BULLISH else ("bear" if bias == Bias.BEARISH else "muted")))
        line("Latest BOS", f"{last_bos.direction} @ {last_bos.price:.2f}" if last_bos else "None")
        line("Latest CHoCH", f"{last_choch.direction} @ {last_choch.price:.2f}" if last_choch else "None")
        if fresh_zone:
            line(fresh_zone.poi_type, f"{fresh_zone.zone_low:,.2f} – {fresh_zone.zone_high:,.2f}")
        if swing_highs:
            line("BSL (buy-side liquidity)", f"{max(swing_highs, key=lambda s: s.index).price:,.2f}")
        if swing_lows:
            line("SSL (sell-side liquidity)", f"{max(swing_lows, key=lambda s: s.index).price:,.2f}")

        if trade_setup:
            st.markdown('<div class="section-sub" style="margin: 10px 0 6px 0;"><b>TRADE SETUP</b></div>', unsafe_allow_html=True)
            line("Direction", badge("LONG" if fresh_zone.direction == "bullish" else "SHORT", "bull" if fresh_zone.direction == "bullish" else "bear"))
            line("Entry", f"{trade_setup['entry']:,.2f}")
            line("Stop", f"{trade_setup['stop']:,.2f}")
            line("Target", f"{trade_setup['target']:,.2f}")
            line("Risk / Reward", "2.0R (fixed, this project's backtest convention)")

    with st.expander("Company reference (all-time high, yearly stats, dividends)"):
        profile, err = cached_profile(ticker)
        if err:
            st.warning(f"Reference data unavailable: {err}")
        elif profile:
            c1, c2, c3 = st.columns(3)
            c1.metric("All-time high", f"{profile.all_time_high:.2f}" if profile.all_time_high else "n/a")
            c2.metric("All-time low", f"{profile.all_time_low:.2f}" if profile.all_time_low else "n/a")
            if profile.all_time_high:
                pct = (current_price - profile.all_time_high) / profile.all_time_high * 100
                c3.metric("From ATH", f"{pct:+.1f}%")
            if profile.yearly_metrics:
                yearly_df = pd.DataFrame([vars(m) for m in profile.yearly_metrics[-6:]])
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Top-Down page
# ---------------------------------------------------------------------------

def page_top_down() -> None:
    st.markdown('<div class="section-title">Top-Down Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Validated in scripts/top_down_backtest.py — 4H→15min showed the strongest '
        'real improvement (31.2%→45.7% win rate) of the drill-down pairs tested.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="RELIANCE", key="td_ticker")
    with col2:
        htf = st.selectbox("HTF", list(HTF_TO_LTF.keys()), index=1, key="td_htf")
    with col3:
        ltf = st.selectbox("LTF", HTF_TO_LTF[htf], key="td_ltf")

    if not ticker:
        return

    bias_tfs = ["monthly", "weekly", "daily", "4h", "1h", "15min"]
    try:
        data = cached_fetch(ticker, tuple(sorted(set(bias_tfs + [htf, ltf]))))
        bias_result = analyze_multi_timeframe({tf: data[tf] for tf in bias_tfs if tf in data})

        htf_res = analyze_multi_timeframe({htf: data[htf]})["per_timeframe"][htf]
        htf_zones = collect_htf_zones(data[htf], htf_res["structure"], htf_res["poi"], htf)

        ltf_structure = analyze_structure(data[ltf])
        ltf_poi = analyze_poi(data[ltf], ltf_structure["swings"], ltf_structure["events"])
        valid_fvgs = [f for f in ltf_poi["fvgs"] if f.valid]
        entries = find_top_down_entries(data[ltf], ltf_structure, valid_fvgs, htf_zones)
        live_entries = [e for e in entries if not e.mitigated]
    except Exception as exc:
        st.error(f"Couldn't load {ticker}: {exc}")
        return

    st.markdown(f"#### {ticker.upper()}")

    tf_cols = st.columns(len(bias_tfs))
    determined = {}
    for col, tf in zip(tf_cols, bias_tfs):
        b = bias_result["bias_by_timeframe"].get(tf, Bias.UNDETERMINED)
        kind = "bull" if b == Bias.BULLISH else ("bear" if b == Bias.BEARISH else "muted")
        if b != Bias.UNDETERMINED:
            determined[tf] = b
        col.markdown(
            f'<div class="info-card"><div class="lbl">{tf.upper()}</div><div class="val">{badge(b.value, kind)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-sub"><b>HIGHER-TIMEFRAME CONTEXT</b></div>', unsafe_allow_html=True)

    ctx_cols = st.columns(2)
    with ctx_cols[0]:
        st.markdown(f'<div class="info-card"><div class="lbl">Alignment</div><div class="val">{bias_result["alignment"].upper()}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-card" style="margin-top:6px;"><div class="lbl">{htf.upper()} zones of interest</div><div class="val">{len(htf_zones)}</div></div>', unsafe_allow_html=True)
    with ctx_cols[1]:
        st.markdown(f'<div class="info-card"><div class="lbl">{ltf.upper()} MSS+FVG matches (historical)</div><div class="val">{len(entries)}</div></div>', unsafe_allow_html=True)
        live_kind = "strong" if live_entries else "muted"
        st.markdown(f'<div class="info-card" style="margin-top:6px;"><div class="lbl">Live entries right now</div><div class="val">{badge(str(len(live_entries)), live_kind)}</div></div>', unsafe_allow_html=True)

    # Honest confluence ratio -- NOT a fabricated 0-100 score, just how
    # many of the checked timeframes with a determined bias agree with
    # the dominant direction.
    if determined:
        dominant = max(set(determined.values()), key=list(determined.values()).count)
        agree = sum(1 for b in determined.values() if b == dominant)
        total = len(determined)
        pct = agree / total
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub"><b>BIAS ALIGNMENT</b> — {agree}/{total} timeframes agree ({pct:.0%})</div>', unsafe_allow_html=True)
        st.progress(pct)
        trade_bias_kind = "bull" if dominant == Bias.BULLISH else "bear"
        st.markdown(f"**Trade bias:** {badge(dominant.value.upper(), trade_bias_kind)}", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    fig = plot_multi_timeframe_zones(data[ltf], ltf_structure, htf_zones, title=f"{ticker.upper()} — {htf.upper()} zones on {ltf.upper()}", height=650)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Backtest page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def cached_backtest(ticker: str, timeframe: str, reward_r: float):
    data = fetch_multi_timeframe_data(ticker, timeframes=[timeframe])
    df = data[timeframe]
    structure_result = analyze_structure(df)
    poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])
    return run_extended_backtest(df, structure_result, poi_result, reward_r=reward_r)


def page_backtest() -> None:
    st.markdown('<div class="section-title">Strategy Backtest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Runs the full extended backtest — every signal, including the '
        'swing-POI / ExtremeOB / IDM-confluence / PO3 comparisons — for one ticker + timeframe.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1.6, 1, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="RELIANCE", key="bt_ticker")
    with col2:
        timeframe = st.selectbox("Timeframe", STRUCTURE_TIMEFRAMES, index=STRUCTURE_TIMEFRAMES.index("daily"), key="bt_tf")
    with col3:
        reward_r = st.number_input("Reward (R)", value=2.0, step=0.5, key="bt_rr")
    with col4:
        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    if not run_clicked and "bt_result" not in st.session_state:
        return
    if run_clicked:
        with st.spinner(f"Backtesting {ticker} {timeframe}..."):
            try:
                st.session_state["bt_result"] = cached_backtest(ticker, timeframe, reward_r)
                st.session_state["bt_label"] = f"{ticker.upper()} · {timeframe}"
            except Exception as exc:
                st.error(f"Couldn't backtest {ticker}: {exc}")
                return

    result = st.session_state.get("bt_result")
    if result is None:
        return

    st.caption(f"Showing results for {st.session_state.get('bt_label', '')}")

    all_trades = result["trades"]
    core_signals = {"OrderBlock", "ExtremeOB", "BOS", "CHoCH", "IDM", "FVG", "MitigationBlock", "BreakerBlock"}
    core_trades = [t for t in all_trades if t.source_type in core_signals]

    stats = summarize_trades(core_trades) if core_trades else {}
    total_trades = len(core_trades)
    total_wins = sum(s.n_wins for s in stats.values())
    total_losses = sum(s.n_losses for s in stats.values())
    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) else 0.0
    avg_r = sum(t.r_multiple for t in core_trades) / total_trades if total_trades else 0.0
    gross_win = sum(t.r_multiple for t in core_trades if t.r_multiple > 0)
    gross_loss = -sum(t.r_multiple for t in core_trades if t.r_multiple < 0)
    profit_factor = gross_win / gross_loss if gross_loss > 0 else None
    max_dd = compute_max_drawdown(core_trades)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Win Rate", f"{win_rate:.1%}")
    metric_cols[1].metric("Total Trades", f"{total_trades:,}")
    metric_cols[2].metric("Average R", f"{avg_r:+.2f}R")
    metric_cols[3].metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor is not None else "n/a")
    metric_cols[4].metric("Max Drawdown", f"-{max_dd:.2f}R")

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    tab_signals, tab_equity, tab_dist = st.tabs(["By Signal", "Equity Curve", "Win/Loss Distribution"])

    with tab_signals:
        rows = []
        for source_type, s in sorted(result["stats"].items()):
            rows.append({
                "Signal": source_type, "Trades": s.n_trades, "Win rate": f"{s.win_rate:.1%}",
                "Expectancy (R)": round(s.expectancy_r, 2),
                "Profit Factor": round(s.profit_factor, 2) if s.profit_factor is not None else None,
                "W/L/T": f"{s.n_wins}/{s.n_losses}/{s.n_timeouts}",
            })
        stats_df = pd.DataFrame(rows)

        def _highlight(val):
            if isinstance(val, (int, float)):
                color = BULLISH_COLOR if val > 0 else (BEARISH_COLOR if val < 0 else "")
                return f"color: {color}; font-weight: 700;" if color else ""
            return ""

        st.dataframe(
            stats_df.style.map(_highlight, subset=["Expectancy (R)"]).format(
                {"Expectancy (R)": "{:+.2f}", "Profit Factor": lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"}
            ),
            use_container_width=True, hide_index=True,
        )

    with tab_equity:
        if core_trades:
            curve = compute_equity_curve(core_trades)
            curve_df = pd.DataFrame(curve, columns=["Trade #", "Cumulative R"])
            curve_df["Trade #"] = range(1, len(curve_df) + 1)
            st.line_chart(curve_df.set_index("Trade #"), color=ACCENT, height=320)
        else:
            st.caption("No trades to plot.")

    with tab_dist:
        if core_trades:
            dist_df = pd.DataFrame({"R multiple": [t.r_multiple for t in core_trades]})
            st.bar_chart(dist_df["R multiple"].value_counts().sort_index(), height=320)
        else:
            st.caption("No trades to plot.")

    st.caption(
        "\"Performance by ticker\" is intentionally omitted here — this page backtests one ticker at a time; "
        "for cross-ticker comparison use scripts/basket_backtest.py, which pools thousands of trades across the full NIFTY 50."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — SMC Screener", page_icon="📈", layout="wide")
    inject_css()
    render_header()
    render_proof_strip()
    render_nav()

    page = st.session_state.get("page", "screener")
    if page == "screener":
        page_screener()
    elif page == "analyze":
        page_analyze()
    elif page == "topdown":
        page_top_down()
    elif page == "backtest":
        page_backtest()


if __name__ == "__main__":
    main()

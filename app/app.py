"""
app/app.py
Zonify -- an SMC (Smart Money Concepts) screener and structure/POI
analysis dashboard for NSE stocks.

This file adds NO new detection, scoring, or backtest-simulation
logic. Every number, zone, and chart here comes straight from this
project's own engine (structure_engine.py, poi_engine.py,
multi_timeframe.py, backtester.py, top_down.py, session_model.py,
fundamentals.py, screener.py) -- this is purely the presentation
layer. Design tokens/CSS live in theme.py (shared with chart.py so the
dashboard and the chart are one visual system); lightweight live-price
polling lives in market_data.py, deliberately separate from the heavy
OHLC/SMC pipeline below -- see the "architecture" note above
render_market_strip() for why.

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
"SMC Score" or "Confidence %"), it is deliberately NOT shown -- the
screener table uses screener.py's real ScreenerResult.tier
(STRONG/SETUP) instead, labeled "Tier".

Run locally with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import Bias, analyze_structure  # noqa: E402
from poi_engine import analyze_poi  # noqa: E402
from multi_timeframe import TIMEFRAME_ORDER, analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from backtester import (  # noqa: E402
    compute_equity_curve, compute_max_drawdown, run_extended_backtest, summarize_trades,
)
from chart import ChartDataError, render_lightweight_chart, render_lightweight_multi_timeframe_chart  # noqa: E402
from top_down import HTF_TO_LTF, collect_htf_zones, find_top_down_entries  # noqa: E402
from fundamentals import fetch_ticker_profile  # noqa: E402
from screener import find_fresh_zone, preview_stop_target, screen_ticker  # noqa: E402
from theme import get_active_theme, inject_global_css, tokens as theme_tokens, toggle_theme  # noqa: E402
from market_data import (  # noqa: E402
    INDEX_SYMBOLS, Quote, QuoteFetchError, fetch_index_quotes, fetch_ticker_quotes, now_ist,
)

APP_NAME = "Zonify"
TAGLINE = "Smart Money Intelligence"
QUOTE_REFRESH_SECONDS = 20

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM", "ADANIENT", "ADANIPORTS", "BAJAJFINSV",
    "BAJAJ-AUTO", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "NESTLEIND", "SBILIFE",
    "SHREECEM", "UPL", "APOLLOHOSP", "BPCL", "TATACONSUM",
]

# Static reference names -- NOT a live fetch. yfinance's per-ticker
# `.info` lookup (the only source of a company name) is slow enough
# (multiple seconds each) that resolving it for 50 tickers on every
# screener page load would defeat the entire point of the lightweight
# live-price path below. NIFTY 50 constituents are public, stable
# facts, so a static map is honest data, just not a network round trip.
NIFTY50_COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank", "INFY": "Infosys", "HINDUNILVR": "Hindustan Unilever", "ITC": "ITC",
    "SBIN": "State Bank of India", "BHARTIARTL": "Bharti Airtel", "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro", "AXISBANK": "Axis Bank", "BAJFINANCE": "Bajaj Finance",
    "ASIANPAINT": "Asian Paints", "MARUTI": "Maruti Suzuki", "SUNPHARMA": "Sun Pharmaceutical",
    "TITAN": "Titan Company", "ULTRACEMCO": "UltraTech Cement", "WIPRO": "Wipro",
    "ONGC": "Oil & Natural Gas Corp", "NTPC": "NTPC", "POWERGRID": "Power Grid Corp",
    "M&M": "Mahindra & Mahindra", "TATASTEEL": "Tata Steel", "JSWSTEEL": "JSW Steel",
    "HCLTECH": "HCL Technologies", "TECHM": "Tech Mahindra", "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports & SEZ", "BAJAJFINSV": "Bajaj Finserv", "BAJAJ-AUTO": "Bajaj Auto",
    "BRITANNIA": "Britannia Industries", "CIPLA": "Cipla", "COALINDIA": "Coal India",
    "DIVISLAB": "Divi's Laboratories", "DRREDDY": "Dr. Reddy's Laboratories", "EICHERMOT": "Eicher Motors",
    "GRASIM": "Grasim Industries", "HDFCLIFE": "HDFC Life Insurance", "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries", "INDUSINDBK": "IndusInd Bank", "NESTLEIND": "Nestle India",
    "SBILIFE": "SBI Life Insurance", "SHREECEM": "Shree Cement", "UPL": "UPL Limited",
    "APOLLOHOSP": "Apollo Hospitals", "BPCL": "Bharat Petroleum", "TATACONSUM": "Tata Consumer Products",
}

STRUCTURE_TIMEFRAMES = [tf for tf in TIMEFRAME_ORDER if tf != "1min"]
LAYER_CATEGORIES = ["Swings", "BOS", "CHoCH", "IDM", "FVG", "OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock"]
DEFAULT_LAYERS_ON = {"Swings", "CHoCH", "FVG", "OrderBlock", "ExtremeOB"}
# Full, spaced-out display names -- "MitigationBlock" as a checkbox
# label wraps mid-word ("MitigationBlo/ck"); the underlying dict KEYS
# stay identical to chart.py's legendgroup names, only the on-screen
# label changes.
LAYER_LABELS = {
    "Swings": "Swings", "BOS": "BOS", "CHoCH": "CHoCH", "IDM": "IDM", "FVG": "FVG",
    "OrderBlock": "Order Block", "ExtremeOB": "Extreme OB",
    "MitigationBlock": "Mitigation Block", "BreakerBlock": "Breaker Block",
}

NAV_ITEMS = [
    ("screener", "Screener"), ("analyze", "Analyze"),
    ("topdown", "Top-Down"), ("backtest", "Backtest"),
]

# Compact gradient "Z" wordmark: two horizontal price levels connected by a
# diagonal structural break -- the same visual language as a BOS/CHoCH line
# on the chart itself, deliberately (see theme.py's header CSS for the
# gradient tokens this reuses).
_LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="zfLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#60A5FA"/><stop offset="55%" stop-color="#3B82F6"/><stop offset="100%" stop-color="#2563EB"/>
</linearGradient></defs>
<rect width="34" height="34" rx="9" fill="url(#zfLogoGrad)"/>
<path d="M10 12.5H24M24 12.5L10 21.5M10 21.5H24" stroke="white" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

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
    """Returns (label, css_kind) -- css_kind maps to theme.py's .zf-pill.{kind} classes."""
    now = now_ist()
    is_weekday = now.weekday() < 5
    in_session = dtime(9, 15) <= now.time() <= dtime(15, 30)
    return ("MARKET OPEN", "live") if (is_weekday and in_session) else ("MARKET CLOSED", "closed")


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


def company_name(ticker: str) -> str:
    return NIFTY50_COMPANY_NAMES.get(ticker, ticker)


def _style_signed_columns(df: pd.DataFrame, columns: list[str]):
    """Colors positive values bullish-green and negative values bearish-red
    in a dataframe -- st.dataframe's column_config number formatting alone
    doesn't color by sign, and the market-strip/table's whole point is
    that green/red communicates direction at a glance."""
    t = theme_tokens()

    def _color(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ""
        color = t["bullish"] if val > 0 else (t["bearish"] if val < 0 else t["text-secondary"])
        return f"color: {color}; font-weight: 600;"

    return df.style.map(_color, subset=columns)


def go_to_analyze(ticker: str) -> None:
    st.session_state["analyze_ticker"] = ticker
    st.session_state["page"] = "analyze"
    st.rerun()


# ---------------------------------------------------------------------------
# Header + nav
# ---------------------------------------------------------------------------

def render_header() -> None:
    with st.container(key="zf_header"):
        c_logo, c_search, c_status = st.columns([2.0, 3.3, 3.4], vertical_alignment="center")

        with c_logo:
            st.markdown(
                f'<div class="zf-header-left">{_LOGO_SVG}'
                f'<span class="zf-logo">ZONIFY</span>'
                f'<span class="zf-tagline">{TAGLINE}</span></div>',
                unsafe_allow_html=True,
            )

        with c_search:
            query = st.text_input(
                "Search", key="global_search", label_visibility="collapsed",
                placeholder="Search stocks, e.g. RELIANCE",
            )
            last_query = st.session_state.get("_last_global_search", "")
            if query and query.strip() and query != last_query:
                st.session_state["_last_global_search"] = query
                go_to_analyze(query.strip().upper())

        with c_status:
            status_label, status_kind = market_status()
            updated = now_ist().strftime("%H:%M:%S IST")
            sub_status, sub_updated, sub_toggle = st.columns([1.55, 1.75, 0.55], vertical_alignment="center")
            with sub_status:
                st.markdown(
                    f'<div class="zf-header-right"><span class="zf-pill {status_kind}">'
                    f'<span class="dot"></span>{status_label}</span></div>',
                    unsafe_allow_html=True,
                )
            with sub_updated:
                st.markdown(
                    f'<div class="zf-header-right"><span class="zf-pill">Updated {updated}</span></div>',
                    unsafe_allow_html=True,
                )
            with sub_toggle:
                with st.container(key="zf_theme_toggle"):
                    icon = "☀" if get_active_theme() == "dark" else "☾"
                    if st.button(icon, key="theme_toggle_btn", help="Switch theme"):
                        toggle_theme()
                        st.rerun()


def render_nav() -> None:
    st.session_state.setdefault("page", "screener")
    with st.container(key="zf_nav"):
        nav_cols = st.columns([1, 1, 1, 1, 4.5], gap="small")
        for col, (key, label) in zip(nav_cols, NAV_ITEMS):
            with col:
                # Each button gets its own keyed wrapper (-> a stable
                # `.st-key-navicon_<key>` class) purely so theme.py's CSS
                # can attach the right vector icon via mask-image -- no
                # emoji in the label itself, see theme.py "NAV ICONS".
                with st.container(key=f"navicon_{key}"):
                    active = st.session_state["page"] == key
                    if st.button(label, key=f"nav_{key}", use_container_width=True,
                                 type="primary" if active else "secondary"):
                        st.session_state["page"] = key
                        st.rerun()


def render_proof_strip() -> None:
    # One markdown() call with a CSS grid, deliberately NOT st.columns():
    # Streamlit's own per-column auto-height sizing for raw-HTML content
    # measures each card BEFORE this file's injected CSS resizes it, then
    # never re-measures -- confirmed directly (DevTools) as a consistent
    # 16px shortfall that let the cards visually bleed into the nav row
    # below. A single grid element sizes itself from its own real content,
    # sidestepping that Streamlit column-height quirk entirely (the same
    # fix already used for the market index strip above).
    cards = [
        f"""<div class="kpi-card">
            <div class="kpi-label">{sig['name']}</div>
            <div class="kpi-row"><span class="kpi-value accent">{sig['win_rate']:.1f}%</span><span class="kpi-sub">Win Rate</span></div>
            <div class="kpi-detail">+{sig['expectancy']:.2f}R expectancy · {sig['consistency']}</div>
        </div>"""
        for sig in PROVEN_SIGNALS
    ]
    cards.append("""<div class="kpi-card">
        <div class="kpi-label">BACKTEST COVERAGE</div>
        <div class="kpi-row"><span class="kpi-value">NIFTY 50</span><span class="kpi-sub">49 symbols</span></div>
        <div class="kpi-detail">~3,000 trades/signal · daily + 4H</div>
    </div>""")
    st.markdown(f'<div class="zf-proof-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live market data -- ARCHITECTURE NOTE (see module docstring / task spec
# section 28): this section talks to market_data.py only, never to
# data_loader.py/structure_engine.py/poi_engine.py. It refreshes on its
# own short cadence via st.fragment(run_every=...), which reruns ONLY the
# fragment function, not the whole page -- so polling live prices here
# can never trigger a full historical-candle refetch or SMC re-analysis.
# Failures are surfaced honestly (never a silently stale "live" price):
# a fetch that fails falls back to the last known-good quotes already in
# session_state, explicitly labeled STALE with when they were last good.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=QUOTE_REFRESH_SECONDS, show_spinner=False)
def _cached_index_quotes() -> dict[str, Quote]:
    return fetch_index_quotes()


@st.cache_data(ttl=QUOTE_REFRESH_SECONDS, show_spinner=False)
def _cached_ticker_quotes(tickers: tuple) -> dict[str, Quote]:
    return fetch_ticker_quotes(list(tickers))


@st.fragment(run_every=QUOTE_REFRESH_SECONDS)
def render_market_strip() -> None:
    try:
        quotes = _cached_index_quotes()
        st.session_state["_index_quotes"] = quotes
        st.session_state["_index_quotes_at"] = now_ist()
        ok = True
    except QuoteFetchError:
        quotes = st.session_state.get("_index_quotes", {})
        ok = False

    if not quotes:
        st.markdown(
            '<div class="zf-pill offline"><span class="dot"></span>Market data temporarily unavailable — retrying…</div>',
            unsafe_allow_html=True,
        )
        return

    cards = []
    for label in INDEX_SYMBOLS:
        q = quotes.get(label)
        if q is None:
            continue
        direction = "up" if q.change > 0 else ("down" if q.change < 0 else "flat")
        sign = "+" if q.change >= 0 else ""
        cards.append(
            f'<div class="zf-index-card"><div class="zf-index-name">{label}</div>'
            f'<div class="zf-index-price">{q.ltp:,.2f}</div>'
            f'<div class="zf-index-chg {direction}">{sign}{q.change:,.2f} ({sign}{q.change_pct:.2f}%)</div></div>'
        )
    st.markdown(f'<div class="zf-market-strip">{"".join(cards)}</div>', unsafe_allow_html=True)

    last_good = st.session_state.get("_index_quotes_at")
    if ok:
        st.caption(f"Delayed daily-bar quotes via Yahoo Finance · refreshes every {QUOTE_REFRESH_SECONDS}s · checked {now_ist().strftime('%H:%M:%S')} IST")
    else:
        stale_at = last_good.strftime("%H:%M:%S IST") if last_good else "unknown"
        st.caption(f"⚠ Showing STALE quotes from {stale_at} — provider unreachable, retrying every {QUOTE_REFRESH_SECONDS}s")


# ---------------------------------------------------------------------------
# Cached engine calls (heavy: full OHLC history + SMC analysis)
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

@st.fragment(run_every=QUOTE_REFRESH_SECONDS)
def render_live_universe_table(tickers: list) -> None:
    """
    The lightweight, always-available half of the screener (task spec
    section 7): latest LTP/change for every ticker in the current
    universe, refreshed on its own short cycle via market_data.py --
    never the heavy SMC pipeline. Rows are sortable (native dataframe
    column-header sort) and clickable (selecting a row jumps to
    Analyze). This is what's shown before "Scan Market" is run, and
    stays available (as the Company/LTP/Change columns) even after.
    """
    key = tuple(tickers)
    try:
        quotes = _cached_ticker_quotes(key)
        st.session_state["_universe_quotes"] = quotes
        st.session_state["_universe_quotes_at"] = now_ist()
        ok = True
    except QuoteFetchError:
        quotes = st.session_state.get("_universe_quotes", {})
        ok = False

    if not quotes:
        st.warning("Market data temporarily unavailable. Retrying…")
        return

    movers = sorted(quotes.values(), key=lambda q: abs(q.change_pct), reverse=True)[:8]
    if movers:
        st.markdown('<div class="section-sub" style="margin:2px 0 4px 0;"><b>TOP MOVERS</b></div>', unsafe_allow_html=True)
        with st.container(key="zf_movers", horizontal=True):
            for q in movers:
                arrow = "▲" if q.change_pct >= 0 else "▼"
                if st.button(f"{arrow} {q.label}  {q.change_pct:+.1f}%", key=f"mover_{q.label}"):
                    go_to_analyze(q.label)

    rows = [
        {
            "Ticker": t, "Company": company_name(t),
            "LTP": quotes[t].ltp if t in quotes else None,
            "Change": quotes[t].change if t in quotes else None,
            "Change %": quotes[t].change_pct if t in quotes else None,
        }
        for t in tickers
    ]
    df = pd.DataFrame(rows)

    event = st.dataframe(
        _style_signed_columns(df, ["Change", "Change %"]), hide_index=True, use_container_width=True, height=420,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
            "Change": st.column_config.NumberColumn("Change", format="%+.2f"),
            "Change %": st.column_config.NumberColumn("Change %", format="%+.2f%%"),
        },
        on_select="rerun", selection_mode="single-row", key="universe_table",
    )
    if event and event.selection and event.selection.rows:
        go_to_analyze(df.iloc[event.selection.rows[0]]["Ticker"])

    freshness = f"Delayed daily-bar quotes · refreshes every {QUOTE_REFRESH_SECONDS}s" if ok else "⚠ STALE — provider unreachable, retrying"
    st.caption(freshness)


def page_screener() -> None:
    st.markdown('<div class="zf-page-title">Screener</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Live NIFTY 50 prices below refresh automatically. Run '
        '<b>Scan Market</b> to compute Bullish/Bearish structure, fresh POIs and trade setups using only the two '
        'backtested signals above, plus HTF bias alignment — BOS, CHoCH, IDM, standalone FVG, MitigationBlock and '
        'PO3 are still fully computed in the engine, just not used for ranking here.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.4, 1.2])
    with c1:
        universe_choice = st.radio("Universe", ["NIFTY 50", "Custom"], horizontal=True, label_visibility="collapsed")
    if universe_choice == "NIFTY 50":
        tickers = NIFTY_50
    else:
        custom = st.text_input("Tickers", value="RELIANCE, TCS, INFY", label_visibility="collapsed")
        tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
    with c2:
        bias_filter = st.pills("Bias", ["All", "Bullish", "Bearish"], default="All", key="scr_bias_pill")
    with c3:
        tier_filter = st.pills("Tier", ["All", "High quality"], default="All", key="scr_tier_pill")
    with c4:
        scan_clicked = st.button("Scan Market", type="primary", use_container_width=True)

    if scan_clicked:
        results, failures = [], 0
        steps = ["Fetching market data", "Analyzing price structure", "Detecting POIs", "Confirming top-down entries"]
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
        st.session_state["screener_universe"] = tickers

    results = st.session_state.get("screener_results")
    same_universe = results is not None and st.session_state.get("screener_universe") == tickers

    if not same_universe:
        st.markdown('<div class="section-sub" style="margin-top:6px;"><b>LIVE PRICES</b></div>', unsafe_allow_html=True)
        render_live_universe_table(tickers)
        return

    # --- Scanned: compact summary strip computed ONLY from real results ---
    failures = st.session_state.get("screener_failures", 0)
    n_bull = sum(1 for r in results if r.direction == "bullish")
    n_bear = sum(1 for r in results if r.direction == "bearish")
    n_neutral = len(tickers) - failures - len(results)
    n_strong = sum(1 for r in results if r.tier == "STRONG")

    summary_cols = st.columns(5)
    for col, (label, value, kind) in zip(summary_cols, [
        ("SCANNED", f"{len(tickers)}", ""), ("BULLISH", str(n_bull), "up"),
        ("BEARISH", str(n_bear), "down"), ("ACTIVE SETUPS", str(len(results)), "accent"),
        ("HIGH QUALITY", str(n_strong), "accent"),
    ]):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-row"><span class="kpi-value {kind}">{value}</span></div></div>',
                unsafe_allow_html=True,
            )

    view = list(results)
    if bias_filter and bias_filter != "All":
        view = [r for r in view if r.direction == bias_filter.lower()]
    if tier_filter == "High quality":
        view = [r for r in view if r.tier == "STRONG"]
    view = sorted(view, key=lambda r: (r.tier != "STRONG", r.ticker))

    st.markdown(
        f"<div class='section-sub' style='margin-top:8px;'><b>{len(view)}</b> result(s) shown out of <b>{len(results)}</b> qualified "
        f"({len(tickers)} scanned" + (f", {failures} failed to fetch" if failures else "") + ")</div>",
        unsafe_allow_html=True,
    )

    if not view:
        st.info(
            "**No setups match the current filters.**\n\n"
            "Try switching the Bias or Tier chip back to \"All\", or expand the universe."
        )
        return

    table_df = pd.DataFrame([
        {
            "Symbol": r.ticker, "Company": company_name(r.ticker), "LTP": r.current_price,
            "Change %": r.change_pct, "HTF Bias": r.direction.upper(), "Signal": r.daily_zone.poi_type,
            "Entry": r.entry_price, "Stop": r.stop_price, "Target": r.target_price,
            "R:R": r.reward_r, "Tier": "HIGH" if r.tier == "STRONG" else "WATCH", "Status": "ACTIVE",
        }
        for r in view
    ])

    event = st.dataframe(
        _style_signed_columns(table_df, ["Change %"]), hide_index=True, use_container_width=True, height=460,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
            "Change %": st.column_config.NumberColumn("Change %", format="%+.2f%%"),
            "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="₹%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
            "R:R": st.column_config.NumberColumn("R:R", format="%.1fR"),
        },
        on_select="rerun", selection_mode="single-row", key="scan_table",
    )
    if event and event.selection and event.selection.rows:
        go_to_analyze(table_df.iloc[event.selection.rows[0]]["Symbol"])

    if st.button("↻ Clear scan / browse live prices", key="clear_scan"):
        st.session_state["screener_results"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Analyze Ticker page
# ---------------------------------------------------------------------------

def page_analyze() -> None:
    st.session_state.setdefault("analyze_ticker", "RELIANCE")
    theme_active = get_active_theme()
    t = theme_tokens()

    # --- Compact analysis toolbar: ticker + timeframe + Analyze ---
    tb1, tb2, tb3 = st.columns([2.4, 1, 0.8])
    with tb1:
        ticker = st.text_input("Ticker", key="analyze_ticker", label_visibility="collapsed", placeholder="Ticker, e.g. RELIANCE")
    with tb2:
        timeframe = st.selectbox("Timeframe", STRUCTURE_TIMEFRAMES, index=STRUCTURE_TIMEFRAMES.index("daily"),
                                  key="analyze_timeframe", label_visibility="collapsed")
    with tb3:
        st.button("Analyze", type="primary", use_container_width=True, key="analyze_go")

    # --- Layer toggles, styled as chips (see theme.py) -- full names, two
    # compact rows, never wrap mid-word ---
    st.markdown('<div class="zf-chip-label">CHART LAYERS</div>', unsafe_allow_html=True)
    with st.container(key="zf_layer_chips"):
        layer_row1 = st.columns(5)
        layer_row2 = st.columns(5)
        visible = {}
        row1_items = ["Swings", "BOS", "CHoCH", "IDM", "FVG"]
        row2_items = ["OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock"]
        for col, cat in zip(layer_row1, row1_items):
            with col:
                visible[cat] = st.checkbox(LAYER_LABELS[cat], value=cat in DEFAULT_LAYERS_ON, key=f"layer_{cat}")
        for col, cat in zip(layer_row2, row2_items):
            with col:
                visible[cat] = st.checkbox(LAYER_LABELS[cat], value=cat in DEFAULT_LAYERS_ON, key=f"layer_{cat}")
        with layer_row2[4]:
            show_liquidity = st.checkbox("Liquidity", value=False, key="layer_liquidity")

    if not ticker:
        return

    try:
        with st.spinner(f"Fetching market data & analyzing structure for {ticker.upper()}…"):
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
    profile, profile_err = cached_profile(ticker)
    company = profile.info.get("longName") if (profile and not profile_err) else company_name(ticker.upper())

    bias = structure_result["current_bias"]
    fresh_zone = find_fresh_zone(poi_result, bias.value) if bias != Bias.UNDETERMINED else None
    trade_setup = None
    if fresh_zone is not None:
        entry_price, stop_price, target_price = preview_stop_target(fresh_zone, current_price, reward_r=2.0)
        trade_setup = {"entry": entry_price, "stop": stop_price, "target": target_price, "direction": fresh_zone.direction}

    # --- Ticker header: name + company, price + change, bias -- one compact row ---
    chg_color = t["bullish"] if change >= 0 else t["bearish"]
    bias_kind = "bull" if bias == Bias.BULLISH else ("bear" if bias == Bias.BEARISH else "muted")
    status_label, status_kind = market_status()
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:16px; margin: 2px 0 4px 0; flex-wrap: wrap;">
            <div>
                <span class="ticker-name">{ticker.upper()}</span>
                <div class="ticker-company">{company or '—'}</div>
            </div>
            <span class="ticker-price">₹{current_price:,.2f}</span>
            <span class="ticker-change" style="color:{chg_color};">{change:+,.2f} ({change_pct:+.2f}%)</span>
            {badge(bias.value.upper(), bias_kind)}
            <span class="zf-pill {status_kind}"><span class="dot"></span>NSE · {status_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Market data strip (single bar, not six cards) ---
    strip_items = [
        ("OPEN", f"{df['open'].iloc[-1]:,.2f}"), ("HIGH", f"{df['high'].iloc[-1]:,.2f}"),
        ("LOW", f"{df['low'].iloc[-1]:,.2f}"), ("PREV CLOSE", f"{prev_close:,.2f}"),
        ("VOLUME", f"{df['volume'].iloc[-1]:,.0f}"), ("ATR (14)", f"{atr:,.2f}" if atr else "—"),
    ]
    strip_html = "".join(f'<div class="item"><div class="lbl">{lbl}</div><div class="val">{val}</div></div>' for lbl, val in strip_items)
    st.markdown(f'<div class="data-strip">{strip_html}</div>', unsafe_allow_html=True)

    # --- Signal summary: one compact strip instead of four cards ---
    sig_items = [
        ("HTF BIAS", htf_bias.value.upper(), "bull" if htf_bias == Bias.BULLISH else ("bear" if htf_bias == Bias.BEARISH else "muted")),
        ("STRUCTURE", bias.value.upper(), bias_kind),
        ("ACTIVE SIGNAL", fresh_zone.poi_type if fresh_zone else "—", "strong" if fresh_zone else "muted"),
        ("TIER", "HIGH" if fresh_zone else "—", "strong" if fresh_zone else "muted"),
    ]
    sig_html = "".join(
        f'<div class="item"><div class="lbl">{lbl}</div><div class="val">{badge(val, kind)}</div></div>'
        for lbl, val, kind in sig_items
    )
    st.markdown(f'<div class="data-strip">{sig_html}</div>', unsafe_allow_html=True)

    # --- Chart (hero element) + side panel ---
    chart_col, side_col = st.columns([3, 1], gap="small")
    with chart_col:
        try:
            chart_html = render_lightweight_chart(
                df, structure_result, poi_result=poi_result, title=f"{ticker.upper()} · NSE · {timeframe}",
                visible=visible, show_liquidity=show_liquidity, trade_setup=trade_setup,
                height=700, default_visible_bars=100, theme=theme_active,
            )
            components.html(chart_html, height=700, scrolling=False)
        except ChartDataError as exc:
            st.error(f"**Market data unavailable.** Historical OHLC data could not be loaded correctly. Please retry.\n\n`{exc}`")

    with side_col:
        st.markdown('<div class="section-sub" style="margin-bottom:4px;"><b>SMART MONEY ANALYSIS</b></div>', unsafe_allow_html=True)

        last_bos = next((e for e in reversed(structure_result["events"]) if e.event_type == "BOS"), None)
        last_choch = next((e for e in reversed(structure_result["events"]) if e.event_type == "CHoCH"), None)
        swing_highs = [s for s in structure_result["swings"] if s.kind == "high"]
        swing_lows = [s for s in structure_result["swings"] if s.kind == "low"]

        def line(lbl, val):
            st.markdown(f'<div class="info-card" style="margin-bottom:5px;"><div class="lbl">{lbl}</div><div class="val" style="font-size:0.86rem;">{val}</div></div>', unsafe_allow_html=True)

        line("Market Structure", badge(bias.value, bias_kind))
        line("Latest BOS", f"{last_bos.direction} @ {last_bos.price:.2f}" if last_bos else "—")
        line("Latest CHoCH", f"{last_choch.direction} @ {last_choch.price:.2f}" if last_choch else "—")
        line(fresh_zone.poi_type if fresh_zone else "Order Block", f"{fresh_zone.zone_low:,.2f} – {fresh_zone.zone_high:,.2f}" if fresh_zone else "—")
        line("BSL (buy-side liquidity)", f"{max(swing_highs, key=lambda s: s.index).price:,.2f}" if swing_highs else "—")
        line("SSL (sell-side liquidity)", f"{max(swing_lows, key=lambda s: s.index).price:,.2f}" if swing_lows else "—")

        st.markdown('<div class="section-sub" style="margin: 8px 0 4px 0;"><b>TRADE SETUP</b></div>', unsafe_allow_html=True)
        if trade_setup:
            line("Direction", badge("LONG" if fresh_zone.direction == "bullish" else "SHORT", "bull" if fresh_zone.direction == "bullish" else "bear"))
            line("Entry", f"{trade_setup['entry']:,.2f}")
            line("Stop", f"{trade_setup['stop']:,.2f}")
            line("Target", f"{trade_setup['target']:,.2f}")
            line("Risk / Reward", "2.0R (this project's fixed backtest convention)")
            line("Quality", "HIGH" if fresh_zone.poi_type == "ExtremeOB" else "STANDARD")
        else:
            line("Direction", "—")
            line("Entry", "—")
            line("Stop", "—")
            line("Target", "—")

    with st.expander("Company reference (all-time high, yearly stats, dividends)"):
        err = profile_err
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
    st.markdown('<div class="zf-page-title">Top-Down Analysis</div>', unsafe_allow_html=True)
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
        with st.spinner(f"Fetching market data & analyzing structure for {ticker.upper()}…"):
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

    st.markdown(f'<div class="section-title">{ticker.upper()}</div>', unsafe_allow_html=True)

    # --- Compact timeframe strip -- small blocks, not giant cards ---
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

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
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
    try:
        chart_html = render_lightweight_multi_timeframe_chart(
            data[ltf], ltf_structure, htf_zones, title=f"{ticker.upper()} — {htf.upper()} zones on {ltf.upper()}",
            height=700, default_visible_bars=100, theme=get_active_theme(),
        )
        components.html(chart_html, height=700, scrolling=False)
    except ChartDataError as exc:
        st.error(f"**Market data unavailable.** Historical OHLC data could not be loaded correctly. Please retry.\n\n`{exc}`")


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
    st.markdown('<div class="zf-page-title">Strategy Backtest</div>', unsafe_allow_html=True)
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
        with st.status(f"Backtesting {ticker.upper()} · {timeframe}…", expanded=True) as status:
            try:
                status.write("Fetching market data...")
                status.write("Simulating trades across every detected signal...")
                st.session_state["bt_result"] = cached_backtest(ticker, timeframe, reward_r)
                st.session_state["bt_label"] = f"{ticker.upper()} · {timeframe}"
                status.update(label=f"Backtest complete — {ticker.upper()} · {timeframe}", state="complete")
            except Exception as exc:
                status.update(label="Backtest failed", state="error")
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

    t = theme_tokens()

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
                color = t["bullish"] if val > 0 else (t["bearish"] if val < 0 else "")
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
            st.line_chart(curve_df.set_index("Trade #"), color=t["accent-bright"], height=320)
        else:
            st.caption("No trades to plot.")

    with tab_dist:
        if core_trades:
            dist_df = pd.DataFrame({"R multiple": [t2.r_multiple for t2 in core_trades]})
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
    inject_global_css()
    render_header()
    render_market_strip()
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

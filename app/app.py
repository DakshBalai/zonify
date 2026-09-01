"""
app/app.py
Zonify -- an SMC (Smart Money Concepts) screener and structure/POI
analysis dashboard for NSE stocks.

This file adds NO new detection logic. Every number, zone, and chart
here comes straight from this project's own engine
(structure_engine.py, poi_engine.py, multi_timeframe.py, backtester.py,
top_down.py, session_model.py, fundamentals.py, screener.py) -- this
is purely the presentation layer on top of work already validated by
scripts/basket_backtest.py and scripts/top_down_backtest.py.

Run locally with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_engine import analyze_structure  # noqa: E402
from poi_engine import analyze_poi  # noqa: E402
from multi_timeframe import TIMEFRAME_ORDER, analyze_multi_timeframe, fetch_multi_timeframe_data  # noqa: E402
from backtester import run_extended_backtest  # noqa: E402
from chart import plot_structure, plot_multi_timeframe_zones  # noqa: E402
from top_down import HTF_TO_LTF, collect_htf_zones, find_top_down_entries  # noqa: E402
from fundamentals import fetch_ticker_profile  # noqa: E402
from screener import screen_ticker  # noqa: E402

APP_NAME = "Zonify"
TAGLINE = "Smart-money structure & liquidity zones for NSE — every signal backtested, not assumed."

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM", "ADANIENT", "ADANIPORTS", "BAJAJFINSV",
    "BAJAJ-AUTO", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "NESTLEIND", "SBILIFE",
    "SHREECEM", "UPL", "APOLLOHOSP", "BPCL", "TATACONSUM",
]

STRUCTURE_TIMEFRAMES = [tf for tf in TIMEFRAME_ORDER if tf not in ("1min",)]

# Headline numbers from the last full 49-ticker NIFTY 50 backtest
# (scripts/basket_backtest.py) -- static, clearly labeled, shown as
# evidence rather than a live recompute (a full basket run takes
# several minutes; this is a dashboard, not the backtest runner).
PROVEN_SIGNALS = [
    {"name": "Order Block", "win_rate": 60.7, "expectancy": 0.77, "consistency": "49/49 tickers"},
    {"name": "Extreme OB", "win_rate": 63.3, "expectancy": 0.84, "consistency": "49/49 tickers"},
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .zonify-hero {
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(34,211,170,0.14), rgba(34,211,170,0.02));
            border: 1px solid rgba(34,211,170,0.25);
            margin-bottom: 1.4rem;
        }
        .zonify-hero h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #22D3AA, #7DE8CE);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .zonify-hero p {
            margin: 0.35rem 0 0 0;
            color: #9AA7B8;
            font-size: 0.98rem;
        }

        .proof-card {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            padding: 0.9rem 1.1rem;
        }
        .proof-card .metric-label { color: #9AA7B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .proof-card .metric-value { font-size: 1.6rem; font-weight: 700; color: #22D3AA; margin-top: 0.15rem; }
        .proof-card .metric-sub { color: #6B7688; font-size: 0.78rem; margin-top: 0.15rem; }

        .badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
        }
        .badge-strong { background: rgba(34,211,170,0.18); color: #22D3AA; border: 1px solid rgba(34,211,170,0.4); }
        .badge-setup  { background: rgba(255,193,7,0.15); color: #FFC107; border: 1px solid rgba(255,193,7,0.35); }
        .badge-bull   { background: rgba(38,166,154,0.18); color: #26A69A; border: 1px solid rgba(38,166,154,0.4); }
        .badge-bear   { background: rgba(239,83,80,0.18); color: #EF5350; border: 1px solid rgba(239,83,80,0.4); }

        .result-card {
            border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.025); padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        }
        .result-card .ticker { font-size: 1.15rem; font-weight: 700; color: #E6EDF3; }
        .result-card .zone-line { color: #9AA7B8; font-size: 0.88rem; margin-top: 0.3rem; }

        footer, #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="zonify-hero">
            <h1>{APP_NAME}</h1>
            <p>{TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_proof_strip() -> None:
    cols = st.columns(len(PROVEN_SIGNALS) + 1)
    for col, sig in zip(cols, PROVEN_SIGNALS):
        with col:
            st.markdown(
                f"""
                <div class="proof-card">
                    <div class="metric-label">{sig['name']}</div>
                    <div class="metric-value">{sig['win_rate']:.1f}% win</div>
                    <div class="metric-sub">+{sig['expectancy']:.2f}R avg · {sig['consistency']} positive</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with cols[-1]:
        st.markdown(
            """
            <div class="proof-card">
                <div class="metric-label">Universe backtested</div>
                <div class="metric-value">NIFTY 50</div>
                <div class="metric-sub">daily + 4H, ~3,000 pooled trades/signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("From the last full basket backtest (scripts/basket_backtest.py) — re-run it any time for fresh numbers.")


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


def page_screener() -> None:
    st.subheader("Screener")
    st.caption(
        "Ranks tickers using ONLY the two backtested signals above, plus HTF bias alignment -- "
        "every other detected signal (BOS, CHoCH, IDM, FVG, MitigationBlock, PO3) is still fully "
        "computed in the engine, just not used for ranking here."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        universe_choice = st.radio(
            "Universe", ["NIFTY 50", "Custom"], horizontal=True, label_visibility="collapsed",
        )
    if universe_choice == "NIFTY 50":
        tickers = NIFTY_50
        st.caption(f"Scanning all {len(tickers)} NIFTY 50 constituents.")
    else:
        custom = st.text_input("Tickers (comma-separated)", value="RELIANCE, TCS, INFY")
        tickers = [t.strip() for t in custom.split(",") if t.strip()]

    if st.button("Run screener", type="primary"):
        progress = st.progress(0.0, text="Starting...")
        results, failures = [], 0
        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Scanning {ticker} ({i + 1}/{len(tickers)})")
            result, err = cached_screen(ticker)
            if result is not None:
                results.append(result)
            elif err is not None:
                failures += 1
        progress.empty()

        st.session_state["screener_results"] = results
        st.session_state["screener_failures"] = failures

    results = st.session_state.get("screener_results")
    if results is not None:
        failures = st.session_state.get("screener_failures", 0)
        results = sorted(results, key=lambda r: (r.tier != "STRONG", r.ticker))
        st.markdown(f"**{len(results)}** qualified out of **{len(tickers)}** scanned" + (f" ({failures} failed to fetch)" if failures else ""))

        if not results:
            st.info("No tickers currently qualify for STRONG or SETUP. That's a real result, not an error -- it means no clean HTF-aligned, fresh-zone setup exists right now.")
        for r in results:
            tier_badge = "badge-strong" if r.tier == "STRONG" else "badge-setup"
            dir_badge = "badge-bull" if r.direction == "bullish" else "badge-bear"
            live = "🟢 live entry active" if r.live_ltf_entry else "⚪ watching for LTF trigger"
            st.markdown(
                f"""
                <div class="result-card">
                    <span class="ticker">{r.ticker}</span>
                    &nbsp; <span class="badge {tier_badge}">{r.tier}</span>
                    &nbsp; <span class="badge {dir_badge}">{r.direction}</span>
                    <div class="zone-line">
                        {r.daily_zone.poi_type} zone <b>{r.daily_zone.zone_low:.2f} – {r.daily_zone.zone_high:.2f}</b>
                        &nbsp;|&nbsp; price {r.current_price:.2f} &nbsp;|&nbsp; {live}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_analyze() -> None:
    st.subheader("Analyze a Ticker")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Ticker", value="RELIANCE")
    with col2:
        timeframe = st.selectbox("Timeframe", STRUCTURE_TIMEFRAMES, index=STRUCTURE_TIMEFRAMES.index("daily"))

    st.markdown("**Show on chart**")
    cats = ["Swings", "BOS", "CHoCH", "IDM", "FVG", "OrderBlock", "ExtremeOB", "MitigationBlock", "BreakerBlock"]
    default_on = {"Swings", "CHoCH", "FVG", "OrderBlock", "ExtremeOB"}
    cols = st.columns(len(cats))
    visible = {}
    for col, cat in zip(cols, cats):
        with col:
            visible[cat] = st.checkbox(cat, value=cat in default_on, key=f"vis_{cat}")

    if st.button("Load chart", type="primary"):
        with st.spinner(f"Fetching {ticker} {timeframe}..."):
            try:
                data = cached_fetch(ticker, (timeframe,))
                df = data[timeframe]
                structure_result = analyze_structure(df)
                poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])
            except Exception as exc:
                st.error(f"Couldn't load {ticker}: {exc}")
                return

        bias = structure_result["current_bias"].value
        bias_badge = "badge-bull" if bias == "bullish" else ("badge-bear" if bias == "bearish" else "badge-setup")
        st.markdown(f"Current bias: <span class='badge {bias_badge}'>{bias}</span>", unsafe_allow_html=True)

        fig = plot_structure(df, structure_result, poi_result=poi_result, title=f"{ticker} — {timeframe}", visible=visible)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Company reference (all-time high, yearly stats, dividends)"):
            profile, err = cached_profile(ticker)
            if err:
                st.warning(f"Reference data unavailable: {err}")
            elif profile:
                c1, c2, c3 = st.columns(3)
                c1.metric("All-time high", f"{profile.all_time_high:.2f}" if profile.all_time_high else "n/a")
                c2.metric("All-time low", f"{profile.all_time_low:.2f}" if profile.all_time_low else "n/a")
                if profile.all_time_high:
                    pct = (float(df["close"].iloc[-1]) - profile.all_time_high) / profile.all_time_high * 100
                    c3.metric("From ATH", f"{pct:+.1f}%")
                if profile.yearly_metrics:
                    yearly_df = pd.DataFrame([vars(m) for m in profile.yearly_metrics[-6:]])
                    st.dataframe(yearly_df, use_container_width=True, hide_index=True)


def page_top_down() -> None:
    st.subheader("Top-Down Entries")
    st.caption("HTF zones (Order Block / Extreme OB / FVG / Breaker) overlaid on the entry timeframe, tagged by source timeframe. Validated in scripts/top_down_backtest.py -- 4H→15min showed the strongest real improvement (31.2%→45.7% win rate).")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="RELIANCE", key="td_ticker")
    with col2:
        htf = st.selectbox("HTF", list(HTF_TO_LTF.keys()), index=1)
    with col3:
        ltf = st.selectbox("LTF", HTF_TO_LTF[htf])

    if st.button("Load top-down chart", type="primary"):
        with st.spinner(f"Fetching {ticker} {htf}/{ltf}..."):
            try:
                data = cached_fetch(ticker, (htf, ltf))
                htf_result = analyze_multi_timeframe({htf: data[htf]})
                htf_res = htf_result["per_timeframe"][htf]
                htf_zones = collect_htf_zones(data[htf], htf_res["structure"], htf_res["poi"], htf)

                ltf_structure = analyze_structure(data[ltf])
                ltf_poi = analyze_poi(data[ltf], ltf_structure["swings"], ltf_structure["events"])
                valid_fvgs = [f for f in ltf_poi["fvgs"] if f.valid]
                entries = find_top_down_entries(data[ltf], ltf_structure, valid_fvgs, htf_zones)
                live_entries = [e for e in entries if not e.mitigated]
            except Exception as exc:
                st.error(f"Couldn't load {ticker}: {exc}")
                return

        st.markdown(f"**{len(htf_zones)}** {htf.upper()} zones of interest · **{len(entries)}** historical MSS+FVG matches · **{len(live_entries)}** live right now")
        fig = plot_multi_timeframe_zones(data[ltf], ltf_structure, htf_zones, title=f"{ticker} — {htf.upper()} zones on {ltf.upper()}")
        st.plotly_chart(fig, use_container_width=True)


def page_backtest() -> None:
    st.subheader("Backtest")
    st.caption("Runs the full extended backtest (every signal, including the swing-POI/ExtremeOB/IDM-confluence/PO3 comparisons) for one ticker + timeframe.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="RELIANCE", key="bt_ticker")
    with col2:
        timeframe = st.selectbox("Timeframe", STRUCTURE_TIMEFRAMES, index=STRUCTURE_TIMEFRAMES.index("daily"), key="bt_tf")
    with col3:
        reward_r = st.number_input("Reward (R)", value=2.0, step=0.5)

    if st.button("Run backtest", type="primary"):
        with st.spinner(f"Backtesting {ticker} {timeframe}..."):
            try:
                data = cached_fetch(ticker, (timeframe,))
                df = data[timeframe]
                structure_result = analyze_structure(df)
                poi_result = analyze_poi(df, structure_result["swings"], structure_result["events"])
                result = run_extended_backtest(df, structure_result, poi_result, reward_r=reward_r)
            except Exception as exc:
                st.error(f"Couldn't backtest {ticker}: {exc}")
                return

        rows = []
        for source_type, s in sorted(result["stats"].items()):
            rows.append({
                "Signal": source_type, "Trades": s.n_trades, "Win rate": f"{s.win_rate:.1%}",
                "Expectancy (R)": round(s.expectancy_r, 2), "W/L/T": f"{s.n_wins}/{s.n_losses}/{s.n_timeouts}",
            })
        stats_df = pd.DataFrame(rows)

        def _highlight(val):
            if isinstance(val, (int, float)):
                color = "#22D3AA" if val > 0 else ("#EF5350" if val < 0 else "")
                return f"color: {color}; font-weight: 700;" if color else ""
            return ""

        st.dataframe(
            stats_df.style.map(_highlight, subset=["Expectancy (R)"]).format({"Expectancy (R)": "{:+.2f}"}),
            use_container_width=True, hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — SMC Screener", page_icon="📈", layout="wide")
    inject_css()
    render_header()
    render_proof_strip()

    tab_screener, tab_analyze, tab_topdown, tab_backtest = st.tabs(
        ["🔍  Screener", "📊  Analyze Ticker", "🔭  Top-Down Entries", "🧪  Backtest"]
    )
    with tab_screener:
        page_screener()
    with tab_analyze:
        page_analyze()
    with tab_topdown:
        page_top_down()
    with tab_backtest:
        page_backtest()


if __name__ == "__main__":
    main()

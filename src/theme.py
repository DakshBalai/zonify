"""
theme.py
Zonify's design-token system: one dict of CSS custom properties per mode
(dark/light), and a single stylesheet that restyles Streamlit's own chrome
(buttons, inputs, selects, checkboxes, tabs, dataframes, expanders,
metrics) against those tokens -- so every page and every component reads
as one product instead of colors hardcoded per-widget.

Nothing here talks to market data or does layout for a specific page --
app.py owns page structure, this module only owns "what do the tokens
resolve to, and how does Streamlit's default chrome map onto them."
"""

from __future__ import annotations

import streamlit as st

# Double-quoted (not single-quoted) around "Segoe UI" on purpose: chart.py
# embeds this string inside single-quoted JS string literals in its
# Lightweight Charts template (e.g. fontFamily:'$font_family') -- a
# single-quoted "'Segoe UI'" here would prematurely close that JS string
# and break the chart with a SyntaxError (caught directly: an empty
# chart, "Unexpected identifier 'Segoe'" in the console).
FONT_FAMILY = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'

BULLISH_DARK, BEARISH_DARK = "#22C55E", "#EF4444"
BULLISH_LIGHT, BEARISH_LIGHT = "#16A34A", "#DC2626"

DARK = {
    "bg-primary": "#070B12",
    "bg-header": "#0D131D",
    "surface": "#111827",
    "surface-elevated": "#162030",
    "surface-hover": "#1A2434",
    "border": "#1F2937",
    "border-strong": "#2A3A4F",
    "text-primary": "#F8FAFC",
    "text-secondary": "#94A3B8",
    "text-muted": "#64748B",
    "accent": "#3B82F6",
    "accent-bright": "#60A5FA",
    "accent-hover": "#2563EB",
    "accent-tint": "rgba(59,130,246,0.14)",
    "accent-tint-strong": "rgba(59,130,246,0.24)",
    "bullish": BULLISH_DARK,
    "bullish-tint": "rgba(34,197,94,0.14)",
    "bearish": BEARISH_DARK,
    "bearish-tint": "rgba(239,68,68,0.14)",
    "warning": "#F59E0B",
    "warning-tint": "rgba(245,158,11,0.14)",
    "shadow": "0 8px 24px rgba(0,0,0,0.35)",
    "shadow-sm": "0 2px 8px rgba(0,0,0,0.25)",
    "scheme": "dark",
}

LIGHT = {
    "bg-primary": "#F6F8FC",
    "bg-header": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface-elevated": "#FFFFFF",
    "surface-hover": "#F1F5F9",
    "border": "#E5EAF2",
    "border-strong": "#D6DEEA",
    "text-primary": "#0F172A",
    "text-secondary": "#64748B",
    "text-muted": "#94A3B8",
    "accent": "#2563EB",
    "accent-bright": "#3B82F6",
    "accent-hover": "#1D4ED8",
    "accent-tint": "rgba(37,99,235,0.08)",
    "accent-tint-strong": "rgba(37,99,235,0.16)",
    "bullish": BULLISH_LIGHT,
    "bullish-tint": "rgba(22,163,74,0.10)",
    "bearish": BEARISH_LIGHT,
    "bearish-tint": "rgba(220,38,38,0.10)",
    "warning": "#D97706",
    "warning-tint": "rgba(217,119,6,0.12)",
    "shadow": "0 8px 24px rgba(15,23,42,0.08)",
    "shadow-sm": "0 2px 8px rgba(15,23,42,0.06)",
    "scheme": "light",
}

PALETTES = {"dark": DARK, "light": LIGHT}


def get_active_theme() -> str:
    """Reads the theme from st.session_state, seeded on first load from the
    `?theme=` URL param so a full page reload doesn't reset it to dark --
    the practical, framework-native way to persist a preference in
    Streamlit without a custom JS/localStorage bridge."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = st.query_params.get("theme", "dark")
    if st.session_state["theme"] not in PALETTES:
        st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def set_active_theme(theme: str) -> None:
    st.session_state["theme"] = theme
    st.query_params["theme"] = theme


def toggle_theme() -> None:
    set_active_theme("light" if get_active_theme() == "dark" else "dark")


def tokens() -> dict:
    return PALETTES[get_active_theme()]


def inject_global_css() -> None:
    t = tokens()

    def v(name: str) -> str:
        return t[name]

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg-primary: {v('bg-primary')}; --bg-header: {v('bg-header')};
            --surface: {v('surface')}; --surface-elevated: {v('surface-elevated')}; --surface-hover: {v('surface-hover')};
            --border: {v('border')}; --border-strong: {v('border-strong')};
            --text-primary: {v('text-primary')}; --text-secondary: {v('text-secondary')}; --text-muted: {v('text-muted')};
            --accent: {v('accent')}; --accent-bright: {v('accent-bright')}; --accent-hover: {v('accent-hover')};
            --accent-tint: {v('accent-tint')}; --accent-tint-strong: {v('accent-tint-strong')};
            --bullish: {v('bullish')}; --bullish-tint: {v('bullish-tint')};
            --bearish: {v('bearish')}; --bearish-tint: {v('bearish-tint')};
            --warning: {v('warning')}; --warning-tint: {v('warning-tint')};
            --shadow: {v('shadow')}; --shadow-sm: {v('shadow-sm')};
            --font: {FONT_FAMILY};
            color-scheme: {v('scheme')};
        }}

        html, body, [class*="css"], .stApp {{ font-family: var(--font); }}
        .stApp {{ background: var(--bg-primary); transition: background 200ms ease; }}
        * {{ transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease; }}

        .block-container {{ padding-top: 14px; padding-bottom: 2rem; max-width: 1560px; }}
        footer, #MainMenu, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
        div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
        div[data-testid="stDecoration"] {{ display: none; }}

        ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 6px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

        a, a:visited {{ color: var(--accent-bright); }}

        /* ================= APP SHELL / HEADER =================
           Real Streamlit widgets (search input, theme toggle button) live
           inside the header, so it can't be raw position:fixed HTML the
           way a static header could -- st.container(key="zf_header")
           instead gives a stable `.st-key-zf_header` class on the actual
           widget wrapper (a documented, version-stable Streamlit
           mechanism), which this styles as a bordered bar in normal
           document flow. */
        .st-key-zf_header {{
            background: var(--bg-header); border: 1px solid var(--border); border-radius: 12px;
            padding: 10px 18px; margin-bottom: 10px; box-shadow: var(--shadow-sm);
        }}
        .st-key-zf_header div[data-testid="stHorizontalBlock"] {{ align-items: center; gap: 12px; }}
        .zf-header-left {{ display: flex; align-items: center; gap: 10px; min-width: 0; flex: 0 0 auto; }}
        .zf-logo {{ font-size: 21px; font-weight: 800; letter-spacing: 0.02em; color: var(--accent-bright); white-space: nowrap; }}
        .zf-tagline {{ font-size: 11px; color: var(--text-secondary); white-space: nowrap; border-left: 1px solid var(--border); padding-left: 10px; }}
        .zf-header-right {{ display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex: 0 0 auto; width: 100%; }}
        .st-key-zf_header div[data-testid="stTextInput"] input {{ height: 38px; }}

        .st-key-zf_theme_toggle button {{
            border-radius: 999px !important; width: 38px !important; height: 38px !important; min-height: 38px !important;
            padding: 0 !important; font-size: 16px !important; border: 1px solid var(--border) !important;
            background: var(--surface) !important; color: var(--text-secondary) !important;
        }}
        .st-key-zf_theme_toggle button:hover {{ border-color: var(--accent) !important; color: var(--accent-bright) !important; }}

        /* ================= NAV CONTAINER ================= */
        .st-key-zf_nav {{ margin: 2px 0 12px 0; }}

        /* ================= TOP MOVERS (horizontal-scroll button row) ================= */
        .st-key-zf_movers {{ overflow-x: auto; gap: 8px; padding: 2px 2px 8px 2px; flex-wrap: nowrap !important; }}
        .st-key-zf_movers div[data-testid="stButton"] button {{
            white-space: nowrap; background: var(--surface); border: 1px solid var(--border);
            color: var(--text-primary); font-weight: 700; font-size: 12.5px; height: 34px; min-height: 34px;
        }}
        .st-key-zf_movers div[data-testid="stButton"] button:hover {{ border-color: var(--accent); background: var(--accent-tint); }}

        .zf-pill {{
            display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px; border-radius: 999px;
            font-size: 11.5px; font-weight: 600; color: var(--text-secondary);
            border: 1px solid var(--border); background: var(--surface); white-space: nowrap;
        }}
        .zf-pill .dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
        .zf-pill.live {{ color: var(--bullish); border-color: var(--bullish-tint); background: var(--bullish-tint); }}
        .zf-pill.live .dot {{ background: var(--bullish); box-shadow: 0 0 0 3px var(--bullish-tint); }}
        .zf-pill.closed {{ color: var(--text-muted); }}
        .zf-pill.closed .dot {{ background: var(--text-muted); }}
        .zf-pill.stale {{ color: var(--warning); border-color: var(--warning-tint); background: var(--warning-tint); }}
        .zf-pill.stale .dot {{ background: var(--warning); }}
        .zf-pill.offline {{ color: var(--bearish); border-color: var(--bearish-tint); background: var(--bearish-tint); }}
        .zf-pill.offline .dot {{ background: var(--bearish); }}

        /* ================= MARKET STRIP ================= */
        .zf-market-strip {{
            display: flex; gap: 10px; overflow-x: auto; padding: 2px 2px 6px 2px; margin-bottom: 6px;
            scrollbar-width: thin;
        }}
        .zf-index-card {{
            flex: 0 0 auto; min-width: 168px; padding: 9px 16px; border-radius: 10px;
            background: var(--surface); border: 1px solid var(--border);
        }}
        .zf-index-name {{ font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.04em; white-space: nowrap; }}
        .zf-index-price {{ font-size: 17px; font-weight: 800; color: var(--text-primary); margin-top: 1px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
        .zf-index-chg {{ font-size: 12px; font-weight: 700; margin-top: 1px; white-space: nowrap; }}
        .zf-index-chg.up {{ color: var(--bullish); }} .zf-index-chg.down {{ color: var(--bearish); }} .zf-index-chg.flat {{ color: var(--text-muted); }}

        /* ================= NAV ================= */
        .zf-nav-row {{ display: flex; gap: 6px; margin: 4px 0 10px 0; flex-wrap: wrap; }}
        div[data-testid="stButton"] button {{
            border-radius: 8px; font-weight: 600; font-size: 13.5px;
            transition: all 150ms ease; padding: 0.3rem 0.9rem; min-height: 38px; height: 38px;
            white-space: nowrap;
        }}
        div[data-testid="stButton"] button[kind="secondary"] {{
            background: var(--surface); border: 1px solid var(--border); color: var(--text-secondary);
        }}
        div[data-testid="stButton"] button[kind="secondary"]:hover {{
            border-color: var(--accent); color: var(--accent-bright); background: var(--accent-tint);
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background: var(--accent) !important; border: 1px solid var(--accent) !important; color: #FFFFFF !important;
            box-shadow: 0 0 0 1px var(--accent-tint-strong);
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{ background: var(--accent-hover) !important; }}
        div[data-testid="stButton"] button:active {{ transform: translateY(1px); }}

        /* ================= TYPOGRAPHY / SECTIONS ================= */
        .zf-page-title {{ font-size: 22px; font-weight: 700; color: var(--text-primary); margin: 4px 0 2px 0; }}
        .section-title {{ font-size: 16px; font-weight: 700; margin: 2px 0 0 0; color: var(--text-primary); }}
        .section-sub {{ color: var(--text-secondary); font-size: 12.5px; margin-bottom: 8px; line-height: 1.5; }}

        /* ================= CARDS / METRICS ================= */
        .kpi-card {{
            border-radius: 10px; border: 1px solid var(--border); background: var(--surface);
            padding: 10px 16px; height: 100%; min-width: 0;
        }}
        .kpi-label {{ color: var(--text-muted); font-size: 10.5px; font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .kpi-row {{ display: flex; align-items: baseline; gap: 7px; margin-top: 2px; flex-wrap: wrap; }}
        .kpi-value {{ font-size: 20px; font-weight: 800; color: var(--text-primary); line-height: 1.2; }}
        .kpi-value.accent {{ color: var(--accent-bright); }}
        .kpi-value.up {{ color: var(--bullish); }} .kpi-value.down {{ color: var(--bearish); }}
        .kpi-sub {{ color: var(--text-muted); font-size: 11px; font-weight: 600; white-space: nowrap; }}
        .kpi-detail {{ color: var(--text-secondary); font-size: 11.5px; margin-top: 2px; }}

        .info-card {{ border-radius: 9px; border: 1px solid var(--border); background: var(--surface); padding: 6px 13px; }}
        .info-card .lbl {{ color: var(--text-muted); font-size: 10px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .info-card .val {{ color: var(--text-primary); font-size: 14px; font-weight: 700; margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

        .data-strip {{
            display: flex; flex-wrap: wrap; gap: 0; border: 1px solid var(--border); border-radius: 10px;
            background: var(--surface); overflow: hidden; margin: 4px 0;
        }}
        .data-strip .item {{ flex: 1 1 120px; min-width: 100px; padding: 7px 16px; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
        .data-strip .item:last-child {{ border-right: none; }}
        .data-strip .lbl {{ color: var(--text-muted); font-size: 10px; font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; }}
        .data-strip .val {{ color: var(--text-primary); font-size: 14.5px; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

        /* ================= TICKER HEADER ================= */
        .ticker-name {{ font-size: 26px; font-weight: 800; color: var(--text-primary); line-height: 1.1; }}
        .ticker-company {{ font-size: 12.5px; color: var(--text-secondary); margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 320px; }}
        .ticker-price {{ font-size: 26px; font-weight: 800; color: var(--text-primary); }}
        .ticker-change {{ font-size: 14px; font-weight: 700; white-space: nowrap; }}

        /* ================= BADGES ================= */
        .badge {{ display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; white-space: nowrap; }}
        .badge-bull {{ background: var(--bullish-tint); color: var(--bullish); border: 1px solid var(--bullish-tint); }}
        .badge-bear {{ background: var(--bearish-tint); color: var(--bearish); border: 1px solid var(--bearish-tint); }}
        .badge-strong {{ background: var(--accent-tint); color: var(--accent-bright); border: 1px solid var(--accent-tint-strong); }}
        .badge-setup {{ background: var(--warning-tint); color: var(--warning); border: 1px solid var(--warning-tint); }}
        .badge-muted {{ background: var(--surface-hover); color: var(--text-muted); border: 1px solid var(--border); }}

        /* ================= FILTER CHIPS (st.pills) ================= */
        div[data-testid="stPills"] label {{ font-size: 12.5px !important; white-space: nowrap; }}
        /* Pill buttons carry Streamlit's native theme colors as an INLINE
           style (from .streamlit/config.toml, fixed at server start) --
           our light/dark toggle is a separate, session-level system, so
           these need an explicit !important override per testid rather
           than relying on inherited/default button styling. */
        button[data-testid="stBaseButton-pills"] {{
            background-color: var(--surface) !important; color: var(--text-secondary) !important;
            border: 1px solid var(--border) !important;
        }}
        button[data-testid="stBaseButton-pills"]:hover {{ border-color: var(--accent) !important; color: var(--accent-bright) !important; }}
        button[data-testid="stBaseButton-pillsActive"] {{
            background-color: var(--accent-tint) !important; color: var(--accent-bright) !important;
            border: 1px solid var(--accent) !important;
        }}

        /* ================= ROW CARDS (legacy/simple lists) ================= */
        .row-card {{
            border-radius: 9px; border: 1px solid var(--border); background: var(--surface);
            padding: 9px 13px; margin-bottom: 6px; transition: border-color 150ms ease, transform 150ms ease;
        }}
        .row-card:hover {{ border-color: var(--accent); transform: translateY(-1px); }}
        .cell-primary {{ color: var(--text-primary); font-weight: 700; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .cell-secondary {{ color: var(--text-secondary); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .cell-mono {{ color: var(--text-primary); font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }}

        /* ================= FORM CONTROLS ================= */
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {{
            background: var(--surface) !important; border: 1px solid var(--border) !important;
            color: var(--text-primary) !important; border-radius: 8px !important;
        }}
        div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {{
            border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-tint) !important;
        }}
        button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{
            background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text-secondary) !important;
        }}
        button[data-testid="stNumberInputStepDown"]:hover, button[data-testid="stNumberInputStepUp"]:hover {{
            border-color: var(--accent) !important; color: var(--accent-bright) !important;
        }}
        div[data-baseweb="select"] > div {{
            background: var(--surface) !important; border-color: var(--border) !important; border-radius: 8px !important;
            color: var(--text-primary) !important;
        }}
        div[data-baseweb="select"]:focus-within > div {{ border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-tint) !important; }}
        ul[data-testid="stSelectboxVirtualDropdown"] {{ background: var(--surface-elevated) !important; border: 1px solid var(--border) !important; }}
        div[data-testid="stCheckbox"] {{ margin-top: -4px; margin-bottom: -4px; }}
        div[data-testid="stCheckbox"] label p {{ white-space: nowrap; font-size: 12.5px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; }}
        /* The checkbox box is a <span> with NO role/aria-checked of its own
           (those live on the sibling <input> that follows it in the DOM) --
           found by inspecting the actual rendered markup, not assumed. */
        label[data-baseweb="checkbox"] > span:first-child {{ background-color: var(--surface) !important; border-color: var(--border-strong) !important; }}
        label[data-baseweb="checkbox"] > span:first-child:has(~ input:checked) {{ background-color: var(--accent) !important; border-color: var(--accent) !important; }}
        div[data-testid="stRadio"] label p {{ font-size: 13px; white-space: nowrap; }}
        label[data-baseweb="radio"] > div:first-child {{ background: var(--surface) !important; border-color: var(--border-strong) !important; }}
        label[data-baseweb="radio"] > div:first-child > div {{ background: var(--accent) !important; }}
        label[data-baseweb="radio"] > div:first-child:has(~ input:checked) {{ border-color: var(--accent) !important; }}

        /* ================= TABS ================= */
        button[data-baseweb="tab"] {{ color: var(--text-secondary); font-weight: 600; font-size: 13px; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--accent-bright); }}
        div[data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}
        div[data-baseweb="tab-border"] {{ background-color: var(--border) !important; }}

        /* ================= EXPANDER ================= */
        div[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
        div[data-testid="stExpander"] summary {{ color: var(--text-primary); font-weight: 600; font-size: 13px; }}

        /* ================= DATAFRAME / TABLE ================= */
        div[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
        div[data-testid="stDataFrame"] [role="row"]:hover {{ background: var(--surface-hover) !important; }}

        /* ================= METRICS ================= */
        div[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }}
        div[data-testid="stMetricLabel"] {{ color: var(--text-muted) !important; font-size: 11px !important; }}
        div[data-testid="stMetricValue"] {{ color: var(--text-primary) !important; }}

        /* ================= PROGRESS ================= */
        /* The filled bar is nested 2 levels inside [role="progressbar"]
           (found by inspecting the rendered markup) and carries its color
           via Streamlit's native emotion-generated class, which a shallow
           `> div` override missed -- `*` catches it regardless of nesting
           depth, safe here since a progress bar has no text content to
           accidentally recolor. */
        div[data-testid="stProgress"] div[role="progressbar"] {{ background-color: var(--border) !important; }}
        div[data-testid="stProgress"] div[role="progressbar"] * {{ background-color: var(--accent) !important; }}

        /* ================= STATUS / SPINNER ================= */
        div[data-testid="stStatusWidget"], div[data-testid="stExpander"] > details {{ background: var(--surface); }}

        hr {{ margin: 0.5rem 0; border-color: var(--border); }}

        /* ================= RESPONSIVE ================= */
        @media (max-width: 1400px) {{
            .block-container {{ max-width: 100%; padding-left: 1.2rem; padding-right: 1.2rem; }}
        }}
        @media (max-width: 900px) {{
            .zf-header {{ padding: 0 12px; }}
            .zf-tagline {{ display: none; }}
            .ticker-name, .ticker-price {{ font-size: 20px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

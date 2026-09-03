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
            background: var(--bg-header); border: 1px solid var(--border); border-radius: 14px;
            padding: 19px 22px; margin-bottom: 18px; box-shadow: var(--shadow-sm);
        }}
        .st-key-zf_header div[data-testid="stHorizontalBlock"] {{ align-items: center; gap: 14px; }}
        .zf-header-left {{ display: flex; align-items: center; gap: 11px; min-width: 0; flex: 0 0 auto; }}
        .zf-header-left svg {{ flex: 0 0 auto; display: block; }}
        .zf-logo {{
            font-size: 22px; font-weight: 800; letter-spacing: 0.02em; white-space: nowrap;
            background: linear-gradient(90deg, #60A5FA, #3B82F6 55%, #2563EB);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
            color: var(--accent-bright); /* fallback if background-clip:text isn't supported */
        }}
        .zf-tagline {{ font-size: 11.5px; color: var(--text-secondary); white-space: nowrap; border-left: 1px solid var(--border); padding-left: 11px; }}
        .zf-header-right {{ display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex: 0 0 auto; width: 100%; }}
        .st-key-zf_header div[data-testid="stTextInput"] input {{
            height: 43px !important; padding-left: 40px !important; font-size: 13.5px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2394A3B8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important; background-position: 14px center !important;
        }}

        .st-key-zf_theme_toggle button {{
            border-radius: 999px !important; width: 40px !important; height: 40px !important; min-height: 40px !important;
            padding: 0 !important; font-size: 17px !important; border: 1px solid var(--border) !important;
            background: var(--surface) !important; color: var(--text-secondary) !important;
        }}
        .st-key-zf_theme_toggle button:hover {{ border-color: var(--accent) !important; color: var(--accent-bright) !important; }}

        /* ================= NAV CONTAINER =================
           Its own bordered row, well clear of the performance cards above
           (see .kpi-card's min-height fix below for why those two used to
           visually collide) -- a real 18px section gap, not a negative
           margin pulling it upward. */
        .st-key-zf_nav {{ margin: 14px 0 22px 0; }}
        .st-key-zf_nav div[data-testid="stHorizontalBlock"] {{ gap: 10px; }}

        /* ---- Nav icons: plain vector shapes via CSS mask, no emoji ----
           Each nav button sits in its own st.container(key=f"navicon_{{id}}")
           wrapper (app.py), giving a stable `.st-key-navicon_<id>` class to
           hang a `::before` mask-image icon on -- the icon inherits the
           button's current text color via `background-color: currentColor`
           masked by the SVG shape, so it's automatically correct in both
           the active (white-on-blue) and inactive (muted) states and both
           themes, with no separate color variant needed. */
        .st-key-zf_nav div[data-testid="stButton"] button {{
            display: inline-flex; align-items: center; justify-content: center; gap: 8px;
        }}
        .st-key-zf_nav div[data-testid="stButton"] button::before {{
            content: ""; display: inline-block; width: 15px; height: 15px; flex: 0 0 auto;
            background-color: currentColor; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
            -webkit-mask-position: center; mask-position: center; -webkit-mask-size: contain; mask-size: contain;
        }}
        .st-key-navicon_screener button::before {{
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='21' y1='4' x2='14' y2='4'/%3E%3Cline x1='10' y1='4' x2='3' y2='4'/%3E%3Cline x1='21' y1='12' x2='12' y2='12'/%3E%3Cline x1='8' y1='12' x2='3' y2='12'/%3E%3Cline x1='21' y1='20' x2='16' y2='20'/%3E%3Cline x1='12' y1='20' x2='3' y2='20'/%3E%3Cline x1='14' y1='2' x2='14' y2='6'/%3E%3Cline x1='8' y1='10' x2='8' y2='14'/%3E%3Cline x1='16' y1='18' x2='16' y2='22'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='21' y1='4' x2='14' y2='4'/%3E%3Cline x1='10' y1='4' x2='3' y2='4'/%3E%3Cline x1='21' y1='12' x2='12' y2='12'/%3E%3Cline x1='8' y1='12' x2='3' y2='12'/%3E%3Cline x1='21' y1='20' x2='16' y2='20'/%3E%3Cline x1='12' y1='20' x2='3' y2='20'/%3E%3Cline x1='14' y1='2' x2='14' y2='6'/%3E%3Cline x1='8' y1='10' x2='8' y2='14'/%3E%3Cline x1='16' y1='18' x2='16' y2='22'/%3E%3C/svg%3E");
        }}
        .st-key-navicon_analyze button::before {{
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='3' y1='21' x2='21' y2='21'/%3E%3Crect x='6' y='9' width='4' height='7'/%3E%3Cline x1='8' y1='5' x2='8' y2='9'/%3E%3Cline x1='8' y1='16' x2='8' y2='18'/%3E%3Crect x='14' y='4' width='4' height='9'/%3E%3Cline x1='16' y1='2' x2='16' y2='4'/%3E%3Cline x1='16' y1='13' x2='16' y2='16'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='3' y1='21' x2='21' y2='21'/%3E%3Crect x='6' y='9' width='4' height='7'/%3E%3Cline x1='8' y1='5' x2='8' y2='9'/%3E%3Cline x1='8' y1='16' x2='8' y2='18'/%3E%3Crect x='14' y='4' width='4' height='9'/%3E%3Cline x1='16' y1='2' x2='16' y2='4'/%3E%3Cline x1='16' y1='13' x2='16' y2='16'/%3E%3C/svg%3E");
        }}
        .st-key-navicon_topdown button::before {{
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black' stroke='none'%3E%3Cpolygon points='12,2 22,8 12,14 2,8'/%3E%3Cpolygon points='2,12.5 12,18.5 22,12.5 22,15.5 12,21.5 2,15.5' opacity='0.55'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black' stroke='none'%3E%3Cpolygon points='12,2 22,8 12,14 2,8'/%3E%3Cpolygon points='2,12.5 12,18.5 22,12.5 22,15.5 12,21.5 2,15.5' opacity='0.55'/%3E%3C/svg%3E");
        }}
        .st-key-navicon_backtest button::before {{
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black' stroke='none'%3E%3Crect x='4' y='14' width='4' height='7' rx='0.5'/%3E%3Crect x='10' y='8' width='4' height='13' rx='0.5'/%3E%3Crect x='16' y='3' width='4' height='18' rx='0.5'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black' stroke='none'%3E%3Crect x='4' y='14' width='4' height='7' rx='0.5'/%3E%3Crect x='10' y='8' width='4' height='13' rx='0.5'/%3E%3Crect x='16' y='3' width='4' height='18' rx='0.5'/%3E%3C/svg%3E");
        }}

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

        /* ================= MARKET STRIP =================
           A real responsive grid (not a fixed-min-width horizontal-scroll
           row) -- with exactly 4 index cards this fills the available
           width evenly instead of leaving dead space to the right, and
           reflows to 2 then 1 column on narrow viewports. */
        .zf-market-strip {{
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px;
        }}
        @media (max-width: 1000px) {{ .zf-market-strip {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 560px) {{ .zf-market-strip {{ grid-template-columns: 1fr; }} }}
        .zf-index-card {{
            min-width: 0; padding: 12px 16px; border-radius: 10px;
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

        /* ================= CARDS / METRICS =================
           A real grid, not st.columns(): Streamlit's per-column
           auto-height sizing for raw-HTML content measures each card
           BEFORE this injected CSS can resize it and never re-measures --
           confirmed directly (DevTools) as a consistent 16px shortfall
           that let cards visually bleed into the nav row below. See
           render_proof_strip() in app.py. */
        .zf-proof-strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        @media (max-width: 1000px) {{ .zf-proof-strip {{ grid-template-columns: 1fr; }} }}
        div[data-testid="stElementContainer"]:has(.zf-proof-strip) {{ min-height: 94px !important; }}
        .kpi-card {{
            border-radius: 10px; border: 1px solid var(--border); background: var(--surface);
            padding: 12px 16px; min-width: 0; box-sizing: border-box;
        }}
        .kpi-label {{ color: var(--text-muted); font-size: 10.5px; font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .kpi-row {{ display: flex; align-items: baseline; gap: 7px; margin-top: 2px; flex-wrap: wrap; }}
        .kpi-value {{ font-size: 20px; font-weight: 800; color: var(--text-primary); line-height: 1.2; }}
        .kpi-value.accent {{ color: var(--accent-bright); }}
        .kpi-value.up {{ color: var(--bullish); }} .kpi-value.down {{ color: var(--bearish); }}
        .kpi-sub {{ color: var(--text-muted); font-size: 11px; font-weight: 600; white-space: nowrap; }}
        .kpi-detail {{ color: var(--text-secondary); font-size: 11.5px; margin-top: 2px; }}

        .info-card {{ border-radius: 10px; border: 1px solid var(--border); background: var(--surface); padding: 7px 13px; }}
        .info-card .lbl {{ color: var(--text-muted); font-size: 10px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .info-card .val {{ color: var(--text-primary); font-size: 14px; font-weight: 700; margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

        .data-strip {{
            display: flex; flex-wrap: wrap; gap: 0; border: 1px solid var(--border); border-radius: 10px;
            background: var(--surface); overflow: hidden; margin: 4px 0;
        }}
        /* Same Streamlit quirk as .zf-proof-strip above: the wrapping
           stElementContainer's auto-height under-measures a raw-HTML
           st.markdown() block by a small, content-dependent amount (8px
           here) -- confirmed directly, not assumed -- letting this strip
           bleed into whatever renders right after it (the sidebar's
           "SMART MONEY ANALYSIS" heading). */
        div[data-testid="stElementContainer"]:has(> .stMarkdown .data-strip) {{ min-height: 60px !important; }}
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

        /* ---- Layer toggles as chips, not a default form ----
           No negative margins: each chip carries its own real padding/
           border, so the row's height is however tall a chip actually is
           -- spacing between the two rows comes from the column gap
           st.container(key="zf_layer_chips") sets below, not a hack
           pulling rows together. The checkbox box is a <span> with NO
           role/aria-checked of its own (those live on the sibling <input>
           that follows it in the DOM) -- found by inspecting the actual
           rendered markup, not assumed. */
        .zf-chip-label {{ color: var(--text-muted); font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em; margin: 10px 0 6px 0; }}
        .st-key-zf_layer_chips div[data-testid="stHorizontalBlock"] {{ row-gap: 8px; }}
        label[data-baseweb="checkbox"] {{
            display: inline-flex !important; align-items: center; gap: 6px;
            background: var(--surface) !important; border: 1px solid var(--border); border-radius: 999px;
            padding: 5px 12px 5px 9px; width: fit-content;
        }}
        label[data-baseweb="checkbox"]:has(input:checked) {{ background: var(--accent-tint) !important; border-color: var(--accent); }}
        div[data-testid="stCheckbox"] label p {{ white-space: nowrap; font-size: 12.5px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; }}
        label[data-baseweb="checkbox"]:has(input:checked) p {{ color: var(--accent-bright); font-weight: 600; }}
        label[data-baseweb="checkbox"] > span:first-child {{
            background-color: var(--surface) !important; border-color: var(--border-strong) !important; width: 15px !important; height: 15px !important;
        }}
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
            .st-key-zf_header {{ padding: 14px 16px; }}
            .zf-tagline {{ display: none; }}
            .ticker-name, .ticker-price {{ font-size: 20px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

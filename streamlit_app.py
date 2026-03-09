import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import time
import re
import html

# ─────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────
RATE_LIMIT_CONFIG = {
    "optimize":    {"max_calls": 10, "window_seconds": 60},
    "predict":     {"max_calls": 20, "window_seconds": 60},
    "save_db":     {"max_calls": 5,  "window_seconds": 60},
    "report":      {"max_calls": 15, "window_seconds": 60},
}

def _rl_key(action: str) -> str:
    return f"_rl_{action}"

def check_rate_limit(action: str) -> tuple[bool, str]:
    """Returns (allowed, message). Call before any expensive operation."""
    cfg = RATE_LIMIT_CONFIG.get(action, {"max_calls": 30, "window_seconds": 60})
    key = _rl_key(action)
    now = time.time()
    window = cfg["window_seconds"]
    max_calls = cfg["max_calls"]

    if key not in st.session_state:
        st.session_state[key] = []

    # Purge old timestamps
    st.session_state[key] = [t for t in st.session_state[key] if now - t < window]

    if len(st.session_state[key]) >= max_calls:
        remaining = int(window - (now - st.session_state[key][0]))
        return False, f"⏱️ Rate limit reached ({max_calls} requests/{window}s). Please wait {remaining}s."

    st.session_state[key].append(now)
    return True, ""


# ─────────────────────────────────────────────
#  INPUT SANITIZATION HELPERS
# ─────────────────────────────────────────────
def sanitize_text(value: str, max_len: int = 200) -> str:
    """Strip HTML/scripts, trim whitespace, enforce max length."""
    if not isinstance(value, str):
        return ""
    value = html.escape(value.strip())
    value = re.sub(r"[<>\"'%;()&+]", "", value)   # Remove shell/SQL metacharacters
    return value[:max_len]

def sanitize_numeric(value, min_val: float, max_val: float, default: float) -> float:
    """Clamp a numeric value to a safe range."""
    try:
        v = float(value)
        if not np.isfinite(v):
            return default
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default

def sanitize_int(value, min_val: int, max_val: int, default: int) -> int:
    try:
        v = int(value)
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default

def sanitize_df_edit(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize a user-edited ingredient dataframe."""
    df = df.copy()
    if "Ingredient" in df.columns:
        df["Ingredient"] = df["Ingredient"].astype(str).apply(lambda x: sanitize_text(x, 100))
    for col in ["CP", "Fiber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100).fillna(0)
    if "Energy" in df.columns:
        df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce").clip(0, 10000).fillna(0)
    if "Cost" in df.columns:
        df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").clip(0, 100_000).fillna(0)
    # Drop blank ingredient rows
    df = df[df["Ingredient"].str.strip() != ""]
    return df


# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Necstech Feed Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

    /* ================================
       THEME TOKENS
    ================================ */
    :root {
        --g900:#0a2417;
        --g800:#123f29;
        --g700:#1a6640;
        --g600:#208550;
        --g500:#28a462;
        --g400:#45c47c;
        --g300:#7ddba8;
        --g200:#b6eecf;
        --g100:#e4f8ef;
        --g50:#f3fdf7;
        /* accent */
        --amber:#d97706;
        --amber-light:#fff4d6;
        --red:#dc2626;
        /* surfaces */
        --bg:#f6f9f7;
        --surface:#ffffff;
        --surface-2:#f9fbfa;
        --surface-3:#eef5f1;
        /* borders */
        --border:#d6e6dd;
        --border-s:#bdd6c7;
        /* text */
        --tx:#0d2a1c;
        --tx-2:#355a49;
        --tx-3:#6c8f7f;
        --tx-inv:#ffffff;
        /* shadows */
        --shadow-s:0 1px 3px rgba(0,0,0,.05);
        --shadow-m:0 6px 20px rgba(0,0,0,.07);
        --shadow-l:0 12px 40px rgba(0,0,0,.12);
        /* radius */
        --r:12px;
        --r-lg:20px;
        --r-sm:8px;
    }

    /* ================================
       DARK MODE
    ================================ */
    [data-theme="dark"] {
        --bg:#0f1c16;
        --surface:#16261e;
        --surface-2:#1c3026;
        --surface-3:#223a2e;
        --border:#29463a;
        --border-s:#335a49;
        --tx:#e6f4ec;
        --tx-2:#b6d6c6;
        --tx-3:#8fb5a3;
        --shadow-s:none;
        --shadow-m:none;
        --shadow-l:none;
    }

    /* ================================
       BASE LAYOUT
    ================================ */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--tx);
    }
    .stApp {
        background: var(--bg);
        color: var(--tx);
    }
    .main .block-container {
        padding: 2rem 2rem 4rem;
        max-width: 1200px;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ================================
       SIDEBAR
    ================================ */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown { color: var(--tx-2) !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--tx) !important; font-size:1.1rem !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: var(--tx-3) !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 { color: var(--tx) !important; }

    /* ================================
       BUTTONS
    ================================ */
    .stButton > button {
        background: var(--g600);
        color: white;
        border: none;
        border-radius: 10px;
        padding: .55rem 1rem;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        transition: .2s;
        white-space: normal !important;
    }
    .stButton > button:hover {
        background: var(--g700);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: var(--g600) !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--g700) !important;
        box-shadow: 0 6px 20px rgba(32,133,80,.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--surface) !important;
        border: 1.5px solid var(--border-s) !important;
        color: var(--tx-2) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--g500) !important;
        color: var(--g700) !important;
        background: var(--surface-2) !important;
    }

    /* ================================
       METRICS
    ================================ */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: .7rem 1rem !important;
        box-shadow: var(--shadow-s) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: .76rem !important;
        font-weight: 600 !important;
        color: var(--tx-3) !important;
        text-transform: uppercase !important;
        letter-spacing: .04em !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.5rem !important;
        color: var(--tx) !important;
        white-space: normal !important;
        overflow: visible !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricDelta"] { font-size: .8rem !important; }

    /* ================================
       INPUTS
    ================================ */
    .stTextInput input,
    .stNumberInput input,
    input[type="number"],
    input[type="text"] {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--tx) !important;
    }
    .stSelectbox div[data-baseweb="select"],
    [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--tx) !important;
    }
    [data-baseweb="select"] span { color: var(--tx) !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stTextInput label, .stCheckbox label {
        font-size: .88rem !important;
        font-weight: 500 !important;
        color: var(--tx-2) !important;
    }

    /* ================================
       CARDS
    ================================ */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r);
        box-shadow: var(--shadow-s);
        padding: 1rem;
        transition: box-shadow .25s, transform .2s, background .25s;
    }
    .card:hover { background: var(--surface-2); box-shadow: var(--shadow-m); transform: translateY(-2px); }

    .feature-card {
        background: var(--surface);
        border-radius: var(--r);
        padding: 1.75rem 1.5rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-s);
        transition: box-shadow .25s, transform .2s, background .25s;
        height: 100%;
    }
    .feature-card:hover { background: var(--surface-2); box-shadow: var(--shadow-l); transform: translateY(-3px); }

    .feat-icon { width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.25rem; margin-bottom:1rem; }
    .feat-icon.green  { background: var(--g100); }
    .feat-icon.amber  { background: var(--amber-light); }
    .feat-icon.blue   { background: #dbeafe; }
    .feat-icon.purple { background: #ede9fe; }
    [data-theme="dark"] .feat-icon.green  { background: rgba(69,196,124,.15); }
    [data-theme="dark"] .feat-icon.amber  { background: rgba(217,119,6,.18); }
    [data-theme="dark"] .feat-icon.blue   { background: rgba(59,130,246,.18); }
    [data-theme="dark"] .feat-icon.purple { background: rgba(139,92,246,.18); }
    .feat-title { font-weight:600; font-size:.95rem; color:var(--tx); margin-bottom:.4rem; }
    .feat-body  { font-size:.85rem; color:var(--tx-3); line-height:1.6; }

    /* ================================
       HERO
    ================================ */
    .hero-wrap {
        background: linear-gradient(135deg, var(--g900) 0%, var(--g700) 55%, var(--g500) 100%);
        border-radius: var(--r-lg);
        padding: 4rem 3.5rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before { content:''; position:absolute; top:-60px; right:-80px; width:340px; height:340px; background:rgba(255,255,255,.04); border-radius:50%; }
    .hero-wrap::after  { content:''; position:absolute; bottom:-40px; left:30%; width:220px; height:220px; background:rgba(255,255,255,.03); border-radius:50%; }
    .hero-tag    { display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); color:var(--g300); font-size:.73rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; padding:.28rem .9rem; border-radius:20px; margin-bottom:1.2rem; }
    .hero-heading { font-family:'DM Serif Display',serif; font-size:3.1rem; color:#fff; line-height:1.15; margin-bottom:1.2rem; max-width:620px; }
    .hero-heading em { font-style:italic; color:var(--g300); }
    .hero-body   { font-size:1.03rem; color:rgba(255,255,255,.75); line-height:1.7; max-width:560px; margin-bottom:2rem; }
    .hero-stats  { display:flex; gap:2.5rem; margin-top:2.5rem; padding-top:2rem; border-top:1px solid rgba(255,255,255,.12); }
    .hero-stat-num   { font-family:'DM Serif Display',serif; font-size:2rem; color:#fff; line-height:1; }
    .hero-stat-label { font-size:.78rem; color:rgba(255,255,255,.55); text-transform:uppercase; letter-spacing:.05em; margin-top:.25rem; }

    /* ================================
       TYPOGRAPHY HELPERS
    ================================ */
    .section-heading { font-family:'DM Serif Display',serif; font-size:1.6rem; color:var(--tx); margin-bottom:.3rem; }
    .section-sub     { font-size:.9rem; color:var(--tx-3); margin-bottom:1.5rem; }
    .page-header     { margin-bottom:1.75rem; }
    .page-title      { font-family:'DM Serif Display',serif; font-size:2.2rem; color:var(--tx); line-height:1.2; margin-bottom:.4rem; }
    .page-desc       { font-size:.95rem; color:var(--tx-3); max-width:600px; }

    /* ================================
       HOW IT WORKS STEPS
    ================================ */
    .steps-card { background:var(--surface); border-radius:var(--r); padding:1.5rem 2rem; border:1px solid var(--border); box-shadow:var(--shadow-s); }
    .steps-row  { display:flex; gap:0; position:relative; }
    .step-item  { flex:1; text-align:center; padding:1.5rem 1rem; position:relative; }
    .step-num   { width:36px; height:36px; border-radius:50%; background:var(--g700); color:#fff; font-weight:700; font-size:.9rem; display:flex; align-items:center; justify-content:center; margin:0 auto .75rem; position:relative; z-index:2; }
    .step-title { font-weight:600; font-size:.9rem; color:var(--tx); }
    .step-body  { font-size:.8rem; color:var(--tx-3); margin-top:.3rem; line-height:1.5; }
    .step-connector { position:absolute; top:2.3rem; left:calc(50% + 18px); right:calc(-50% + 18px); height:2px; background:var(--g300); z-index:1; }

    /* ================================
       STAGE HEADER
    ================================ */
    .stage-header {
        background: linear-gradient(90deg, var(--g800), var(--g600));
        color: #fff;
        padding: .7rem 1.2rem;
        border-radius: var(--r-sm);
        margin: 1.2rem 0 .8rem;
        font-weight: 700;
        font-size: .9rem;
        letter-spacing: .02em;
        border-left: 4px solid var(--g300);
        box-shadow: var(--shadow-s);
    }

    /* ================================
       BREED CARDS
    ================================ */
    .breed-card-wrap {
        background: var(--surface);
        border-radius: var(--r);
        border: 1px solid var(--border);
        border-left: 4px solid var(--g500);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-s);
        transition: box-shadow .2s, background .2s;
    }
    .breed-card-wrap:hover { background: var(--surface-2); box-shadow: var(--shadow-m); }
    .breed-card-wrap * { color: #000000 !important; }

    /* ================================
       ALERT BANNERS
    ================================ */
    .alert-amber {
        background: var(--amber-light);
        border-left: 4px solid var(--amber);
        padding: .85rem 1.1rem;
        border-radius: var(--r-sm);
        margin: .75rem 0;
        font-size: .9rem;
        color: #451a03;
    }
    .alert-green {
        background: var(--g50);
        border-left: 4px solid var(--g500);
        padding: .85rem 1.1rem;
        border-radius: var(--r-sm);
        margin: .75rem 0;
        font-size: .9rem;
        color: var(--g800);
    }
    [data-theme="dark"] .alert-amber { background: rgba(217,119,6,.15); color: #fcd34d; border-left-color: var(--amber); }
    [data-theme="dark"] .alert-green { background: rgba(40,164,98,.14); color: var(--g300); border-left-color: var(--g400); }

    hr { border-color: var(--border-s) !important; margin: 1.5rem 0 !important; }

    /* ================================
       TABS
    ================================ */
    .stTabs [role="tablist"] {
        background: var(--surface);
        border-radius: 10px;
        padding: .3rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-s);
        gap: .2rem;
    }
    .stTabs [role="tab"] {
        border-radius: 7px !important;
        font-size: .85rem !important;
        font-weight: 500 !important;
        padding: .45rem 1rem !important;
        color: var(--tx-2) !important;
        transition: all .2s !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--g700) !important;
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface) !important;
        border-radius: 10px !important;
        padding: 1.75rem 1.5rem !important;
        border: 1px solid var(--border) !important;
        margin-top: .5rem !important;
        box-shadow: var(--shadow-s) !important;
    }
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span,
    .stTabs [data-baseweb="tab-panel"] div,
    .stTabs [data-baseweb="tab-panel"] label,
    .stTabs [data-baseweb="tab-panel"] li  { color: var(--tx-2) !important; }
    .stTabs [data-baseweb="tab-panel"] h1,
    .stTabs [data-baseweb="tab-panel"] h2,
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4  { color: var(--tx) !important; }

    /* ================================
       EXPANDER
    ================================ */
    [data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important; }
    [data-testid="stExpander"] summary { font-weight:600 !important; font-size:.9rem !important; color:var(--g700) !important; }
    [data-theme="dark"] [data-testid="stExpander"] summary { color: var(--g400) !important; }

    /* ================================
       DATA TABLE
    ================================ */
    [data-testid="stDataFrame"] { border-radius: var(--r-sm) !important; overflow: hidden !important; border: 1px solid var(--border-s) !important; }
    [data-testid="stDataFrame"] th { background: var(--surface-3) !important; color: var(--tx-2) !important; font-weight:600 !important; font-size:.81rem !important; text-transform:uppercase !important; letter-spacing:.04em !important; padding:.6rem .8rem !important; }
    [data-testid="stDataFrame"] td { color: var(--tx) !important; font-size:.88rem !important; padding:.5rem .8rem !important; white-space:normal !important; }

    /* ================================
       STREAMLIT ALERTS
    ================================ */
    [data-testid="stSuccess"] { background: var(--g50) !important; border-left: 4px solid var(--g500) !important; color: var(--g800) !important; border-radius: var(--r-sm) !important; }
    [data-testid="stInfo"]    { background: #eff6ff !important; border-left: 4px solid #3b82f6 !important; color: #1e3a5f !important; border-radius: var(--r-sm) !important; }
    [data-testid="stWarning"] { background: var(--amber-light) !important; border-left: 4px solid var(--amber) !important; color: #451a03 !important; border-radius: var(--r-sm) !important; }
    [data-testid="stError"]   { background: #fff0f0 !important; border-left: 4px solid var(--red) !important; color: #5c0a0a !important; border-radius: var(--r-sm) !important; }
    [data-theme="dark"] [data-testid="stSuccess"] { background: rgba(40,164,98,.14)  !important; color: var(--g200) !important; }
    [data-theme="dark"] [data-testid="stInfo"]    { background: rgba(59,130,246,.14) !important; color: #bfdbfe !important; }
    [data-theme="dark"] [data-testid="stWarning"] { background: rgba(217,119,6,.15)  !important; color: #fcd34d !important; }
    [data-theme="dark"] [data-testid="stError"]   { background: rgba(220,38,38,.14)  !important; color: #fca5a5 !important; }
    [data-testid="stCaptionContainer"] p { color: var(--tx-3) !important; font-size: .8rem !important; }

    /* ================================
       NUTRIENT PANEL
    ================================ */
    .nutrient-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-s);
    }
    .nutrient-panel-title {
        font-weight: 700;
        font-size: .88rem;
        color: var(--g700);
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        gap: .4rem;
    }
    [data-theme="dark"] .nutrient-panel-title { color: var(--g400); }

    /* ================================
       NUTRIENT CHIPS
    ================================ */
    .nutrient-chip {
        text-align: center;
        padding: .55rem .4rem;
        background: var(--g50);
        border: 1px solid var(--g200);
        border-radius: var(--r-sm);
        transition: background .2s;
    }
    [data-theme="dark"] .nutrient-chip { background: rgba(40,164,98,.12); border-color: rgba(69,196,124,.25); }
    .nutrient-chip-label { font-size:.63rem; font-weight:700; color:var(--tx-3); text-transform:uppercase; letter-spacing:.05em; }
    .nutrient-chip-value { font-size:.95rem; font-weight:800; color:var(--g700); margin-top:.15rem; }
    [data-theme="dark"] .nutrient-chip-label { color: var(--tx-3); }
    [data-theme="dark"] .nutrient-chip-value { color: var(--g400); }

    /* ================================
       BREED BADGE
    ================================ */
    .breed-badge {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        background: var(--g100);
        color: var(--g800);
        font-weight: 700;
        font-size: .82rem;
        padding: .28rem .85rem;
        border-radius: 20px;
        margin-bottom: .75rem;
        border: 1px solid var(--g200);
    }
    [data-theme="dark"] .breed-badge { background: rgba(40,164,98,.18); color: var(--g300); border-color: rgba(69,196,124,.3); }

    /* ================================
       FOOTER
    ================================ */
    .footer-wrap {
        background: var(--g900);
        color: rgba(255,255,255,.6);
        border-radius: var(--r-lg);
        padding: 2.5rem 3rem;
        margin-top: 3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .footer-brand { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#fff; }
    .footer-meta  { font-size:.77rem; margin-top:.3rem; }
    .footer-badges { display:flex; gap:.5rem; }
    .badge { padding:.25rem .75rem; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1); border-radius:20px; font-size:.71rem; color:rgba(255,255,255,.65); }

    /* ================================
       RESPONSIVE
    ================================ */
    @media (max-width:768px) {
        [data-testid="column"] { width:100% !important; flex:1 1 100% !important; }
        .stButton > button { width:100% !important; font-size:.88rem !important; }
        [data-testid="stMetricValue"] { font-size:1.2rem !important; }
        .hero-heading { font-size:2.1rem; }
        .hero-wrap { padding:2.5rem 1.5rem; }
    }
    @media (max-width:480px) {
        .main .block-container { padding:0 1rem 1.5rem; }
        [data-testid="stMetricValue"] { font-size:1rem !important; }
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING & ML
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    rabbit  = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle  = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

@st.cache_resource
def train_model(data):
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal",
               "Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)


# ─────────────────────────────────────────────
#  REFERENCE DATA
# ─────────────────────────────────────────────
def get_breed_database():
    rabbit_breeds = {
        "New Zealand White":{"Type":"Meat","Mature Weight (kg)":"4.5-5.5","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Commercial meat production","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"},
        "Californian":      {"Type":"Meat","Mature Weight (kg)":"4.0-5.0","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Meat and show","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"},
        "Flemish Giant":    {"Type":"Meat","Mature Weight (kg)":"6.0-10.0","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Large-scale meat production","Recommended CP (%)":"17-19","Market Age (weeks)":"14-16"},
        "Dutch":            {"Type":"Pet/Show","Mature Weight (kg)":"2.0-2.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Pets and breeding","Recommended CP (%)":"15-17","Market Age (weeks)":"8-10"},
        "Rex":              {"Type":"Meat/Fur","Mature Weight (kg)":"3.5-4.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Fur and meat","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"},
    }
    poultry_breeds = {
        "Broiler (Cobb 500)":   {"Type":"Meat","Mature Weight (kg)":"2.5-3.0","Growth Rate":"Very Fast","Feed Efficiency":"Excellent (FCR 1.6-1.8)","Best For":"Commercial meat production","Recommended CP (%)":"20-22","Market Age (weeks)":"5-6"},
        "Broiler (Ross 308)":   {"Type":"Meat","Mature Weight (kg)":"2.3-2.8","Growth Rate":"Very Fast","Feed Efficiency":"Excellent (FCR 1.65-1.85)","Best For":"Commercial meat production","Recommended CP (%)":"20-22","Market Age (weeks)":"5-6"},
        "Layer (Isa Brown)":    {"Type":"Eggs","Mature Weight (kg)":"1.8-2.0","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"High egg production (300+ eggs/year)","Recommended CP (%)":"16-18","Market Age (weeks)":"18-20 (point of lay)"},
        "Layer (Lohmann Brown)":{"Type":"Eggs","Mature Weight (kg)":"1.9-2.1","Growth Rate":"Moderate","Feed Efficiency":"Excellent","Best For":"Egg production (320+ eggs/year)","Recommended CP (%)":"16-18","Market Age (weeks)":"18-20 (point of lay)"},
        "Noiler":               {"Type":"Dual Purpose","Mature Weight (kg)":"2.0-2.5","Growth Rate":"Fast","Feed Efficiency":"Good","Best For":"Meat and eggs (Nigerian adapted)","Recommended CP (%)":"18-20","Market Age (weeks)":"12-16"},
        "Kuroiler":             {"Type":"Dual Purpose","Mature Weight (kg)":"2.5-3.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Free-range, dual purpose","Recommended CP (%)":"16-18","Market Age (weeks)":"14-18"},
        "Local Nigerian":       {"Type":"Dual Purpose","Mature Weight (kg)":"1.2-1.8","Growth Rate":"Slow","Feed Efficiency":"Moderate","Best For":"Free-range, disease resistant","Recommended CP (%)":"14-16","Market Age (weeks)":"20-24"},
    }
    cattle_breeds = {
        "White Fulani":          {"Type":"Beef/Dairy","Mature Weight (kg)":"300-450","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Milk and beef (Nigerian indigenous)","Recommended CP (%)":"14-16","Market Age (months)":"24-30"},
        "Red Bororo":            {"Type":"Beef","Mature Weight (kg)":"250-350","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Beef production (heat tolerant)","Recommended CP (%)":"13-15","Market Age (months)":"24-28"},
        "Sokoto Gudali":         {"Type":"Beef","Mature Weight (kg)":"350-500","Growth Rate":"Moderate-Fast","Feed Efficiency":"Good","Best For":"Beef (large frame)","Recommended CP (%)":"14-16","Market Age (months)":"24-30"},
        "N'Dama":                {"Type":"Beef/Draft","Mature Weight (kg)":"300-400","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Trypanosomiasis resistant","Recommended CP (%)":"12-14","Market Age (months)":"30-36"},
        "Muturu":                {"Type":"Beef/Draft","Mature Weight (kg)":"200-300","Growth Rate":"Slow","Feed Efficiency":"Moderate","Best For":"Small-holder, disease resistant","Recommended CP (%)":"12-14","Market Age (months)":"30-36"},
        "Holstein Friesian (Cross)":{"Type":"Dairy","Mature Weight (kg)":"450-650","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"High milk production","Recommended CP (%)":"16-18","Market Age (months)":"24-28"},
        "Brahman Cross":         {"Type":"Beef","Mature Weight (kg)":"400-550","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Beef (heat adapted)","Recommended CP (%)":"14-16","Market Age (months)":"20-24"},
    }
    return {"Rabbit": rabbit_breeds, "Poultry": poultry_breeds, "Cattle": cattle_breeds}


def get_nutrient_requirements():
    rabbit_nutrients = {
        "Grower (4-12 weeks)":    {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"12-16","Calcium (%)":"0.4-0.8","Phosphorus (%)":"0.3-0.5","Lysine (%)":"0.65-0.75","Feed Intake (g/day)":"80-120"},
        "Finisher (12-16 weeks)": {"Crude Protein (%)":"14-16","Energy (kcal/kg)":"2400-2600","Crude Fiber (%)":"14-18","Calcium (%)":"0.4-0.7","Phosphorus (%)":"0.3-0.5","Lysine (%)":"0.55-0.65","Feed Intake (g/day)":"120-180"},
        "Doe (Maintenance)":      {"Crude Protein (%)":"15-16","Energy (kcal/kg)":"2500-2600","Crude Fiber (%)":"14-16","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.4-0.5","Lysine (%)":"0.60-0.70","Feed Intake (g/day)":"100-150"},
        "Doe (Pregnant)":         {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2600-2800","Crude Fiber (%)":"12-15","Calcium (%)":"0.8-1.2","Phosphorus (%)":"0.5-0.7","Lysine (%)":"0.70-0.80","Feed Intake (g/day)":"150-200"},
        "Doe (Lactating)":        {"Crude Protein (%)":"17-19","Energy (kcal/kg)":"2700-3000","Crude Fiber (%)":"12-14","Calcium (%)":"1.0-1.5","Phosphorus (%)":"0.6-0.8","Lysine (%)":"0.75-0.90","Feed Intake (g/day)":"200-400"},
        "Buck (Breeding)":        {"Crude Protein (%)":"15-17","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"14-16","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.4-0.6","Lysine (%)":"0.65-0.75","Feed Intake (g/day)":"120-170"},
    }
    poultry_nutrients = {
        "Broiler Starter (0-3 weeks)":    {"Crude Protein (%)":"22-24","Energy (kcal/kg)":"3000-3200","Crude Fiber (%)":"3-4","Calcium (%)":"0.9-1.0","Phosphorus (%)":"0.45-0.50","Lysine (%)":"1.20-1.35","Methionine (%)":"0.50-0.55","Feed Intake (g/day)":"25-35"},
        "Broiler Grower (3-6 weeks)":     {"Crude Protein (%)":"20-22","Energy (kcal/kg)":"3100-3300","Crude Fiber (%)":"3-5","Calcium (%)":"0.85-0.95","Phosphorus (%)":"0.40-0.45","Lysine (%)":"1.05-1.20","Methionine (%)":"0.45-0.50","Feed Intake (g/day)":"80-120"},
        "Broiler Finisher (6+ weeks)":    {"Crude Protein (%)":"18-20","Energy (kcal/kg)":"3200-3400","Crude Fiber (%)":"3-5","Calcium (%)":"0.80-0.90","Phosphorus (%)":"0.35-0.40","Lysine (%)":"0.95-1.10","Methionine (%)":"0.40-0.45","Feed Intake (g/day)":"140-180"},
        "Layer Starter (0-6 weeks)":      {"Crude Protein (%)":"18-20","Energy (kcal/kg)":"2800-3000","Crude Fiber (%)":"3-5","Calcium (%)":"0.9-1.0","Phosphorus (%)":"0.45-0.50","Lysine (%)":"0.95-1.05","Methionine (%)":"0.40-0.45","Feed Intake (g/day)":"20-40"},
        "Layer Grower (6-18 weeks)":      {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2700-2900","Crude Fiber (%)":"4-6","Calcium (%)":"0.8-0.9","Phosphorus (%)":"0.40-0.45","Lysine (%)":"0.75-0.85","Methionine (%)":"0.35-0.40","Feed Intake (g/day)":"60-90"},
        "Layer Production (18+ weeks)":   {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2750-2900","Crude Fiber (%)":"4-6","Calcium (%)":"3.5-4.0","Phosphorus (%)":"0.35-0.40","Lysine (%)":"0.75-0.85","Methionine (%)":"0.38-0.42","Feed Intake (g/day)":"110-130"},
    }
    cattle_nutrients = {
        "Calf Starter (0-3 months)":{"Crude Protein (%)":"18-20","Energy (kcal/kg)":"3000-3200","Crude Fiber (%)":"8-12","Calcium (%)":"0.7-1.0","Phosphorus (%)":"0.4-0.6","TDN (%)":"72-78","Feed Intake (kg/day)":"0.5-1.5"},
        "Calf Grower (3-6 months)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2800-3000","Crude Fiber (%)":"10-15","Calcium (%)":"0.6-0.9","Phosphorus (%)":"0.35-0.50","TDN (%)":"68-74","Feed Intake (kg/day)":"2-4"},
        "Heifer (6-12 months)":     {"Crude Protein (%)":"14-16","Energy (kcal/kg)":"2600-2800","Crude Fiber (%)":"12-18","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.30-0.45","TDN (%)":"65-70","Feed Intake (kg/day)":"4-7"},
        "Bull (Breeding)":          {"Crude Protein (%)":"12-14","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"15-20","Calcium (%)":"0.4-0.7","Phosphorus (%)":"0.25-0.40","TDN (%)":"62-68","Feed Intake (kg/day)":"8-12"},
        "Cow (Dry)":                {"Crude Protein (%)":"10-12","Energy (kcal/kg)":"2400-2600","Crude Fiber (%)":"18-25","Calcium (%)":"0.4-0.6","Phosphorus (%)":"0.25-0.35","TDN (%)":"58-65","Feed Intake (kg/day)":"10-15"},
        "Cow (Lactating)":          {"Crude Protein (%)":"14-18","Energy (kcal/kg)":"2700-3000","Crude Fiber (%)":"15-22","Calcium (%)":"0.6-0.9","Phosphorus (%)":"0.35-0.50","TDN (%)":"68-75","Feed Intake (kg/day)":"12-20"},
        "Beef Finisher":            {"Crude Protein (%)":"12-14","Energy (kcal/kg)":"2800-3100","Crude Fiber (%)":"8-15","Calcium (%)":"0.5-0.7","Phosphorus (%)":"0.30-0.45","TDN (%)":"70-78","Feed Intake (kg/day)":"8-14"},
    }
    return {"Rabbit": rabbit_nutrients, "Poultry": poultry_nutrients, "Cattle": cattle_nutrients}


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "formulation_history" not in st.session_state:
    st.session_state.formulation_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Inject data-theme attribute for CSS targeting
_dm = st.session_state.dark_mode
st.markdown(
    f'<script>document.documentElement.setAttribute("data-theme","{"dark" if _dm else "light"}");</script>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_report(animal, age, weight, cp_req, energy_req, feed_intake,
                    result_df=None, total_cost=None, prediction=None):
    report = (
        "═" * 50 + "\n"
        "          NECSTECH FEED OPTIMIZER REPORT\n"
        + "═" * 50 + "\n"
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Species: {animal}  |  Age: {age} weeks  |  Weight: {weight} kg\n"
        f"CP Req: {cp_req}%  |  Energy: {energy_req} kcal/kg\n"
    )
    if result_df is not None and total_cost is not None:
        report += f"Total Cost/kg: ₦{total_cost:.2f}  |  Daily Cost: ₦{total_cost * feed_intake:.2f}\n"
        for _, row in result_df.iterrows():
            report += f"  {row['Ingredient']}: {row['Proportion (%)']:.2f}% (₦{row['Cost Contribution (₦)']:.2f})\n"
    if prediction is not None:
        weekly_gain  = prediction * 7
        monthly_gain = prediction * 30
        projected    = weight + (monthly_gain * 3 / 1000)
        fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
        report += (
            f"Daily Gain: {prediction:.1f} g  |  Weekly: {weekly_gain:.0f} g  |  Monthly: {monthly_gain/1000:.2f} kg\n"
            f"90-Day Weight: {projected:.1f} kg  |  FCR: {fcr:.2f}:1\n"
        )
    report += "\nGenerated by Necstech Feed Optimizer v2.0  |  NIAS · FAO · 2026\n"
    return report


# ─────────────────────────────────────────────
#  NAVIGATION BAR
# ─────────────────────────────────────────────
def render_navbar():
    current = st.session_state.page
    cols = st.columns([2, 1, 1, 1, 1, 0.7])
    with cols[0]:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:.6rem;padding:.4rem 0;">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#1a6640,#45c47c);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🌱</div>
            <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:var(--tx);">Necs<span style="color:var(--g600);">tech</span></span>
        </div>
        """, unsafe_allow_html=True)
    nav_items = [
        ("home",          "🏠 Home"),
        ("nutrient_guide","📖 Nutrient Guide"),
        ("breed_database","🐾 Breeds"),
        ("formulator",    "🔬 Formulator"),
    ]
    for idx, (key, label) in enumerate(nav_items):
        with cols[idx + 1]:
            btn_type = "primary" if current == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.page = key
                st.rerun()
    with cols[5]:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(icon, key="nav_theme", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown("<hr style='margin:.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────
def show_home():
    render_navbar()
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">🌍 Built for Nigerian Agriculture · Powered by AI</div>
        <div class="hero-heading">Smarter Feed,<br><em>Healthier Livestock,</em><br>Better Profits</div>
        <div class="hero-body">Necstech Feed Optimizer is an AI-powered precision nutrition platform designed for rabbit, poultry, and cattle farmers across Nigeria. Using advanced linear programming and machine learning trained on 110+ local feeding trials, it generates least-cost feed formulas that meet your animals' exact nutritional requirements.</div>
        <div class="hero-stats">
            <div><div class="hero-stat-num">97+</div><div class="hero-stat-label">Local Ingredients</div></div>
            <div><div class="hero-stat-num">31+</div><div class="hero-stat-label">Breed Profiles</div></div>
            <div><div class="hero-stat-num">3</div><div class="hero-stat-label">Livestock Species</div></div>
            <div><div class="hero-stat-num">110+</div><div class="hero-stat-label">ML Training Trials</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Everything You Need to Optimise Livestock Nutrition</div><div class="section-sub">Four integrated tools in one streamlined platform</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("green",  "💰", "Least-Cost Formulation",  "Linear programming engine automatically blends the cheapest ingredient combination that satisfies all protein, energy, and mineral constraints."),
        ("blue",   "🤖", "AI Growth Prediction",    "Random Forest model forecasts daily weight gain, FCR, and 90-day projections based on Nigerian farm trial data."),
        ("amber",  "🇳🇬", "Nigerian Market Data",   "97 ingredients with verified 2026 local market prices. Edit, add, or remove ingredients to match your region's availability."),
        ("purple", "📊", "Cost & ROI Dashboard",    "Detailed cost breakdowns, herd-level projections, and profit/loss analysis per production cycle to guide investment decisions."),
    ]
    for col, (cls, icon, title, body) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feat-icon {cls}">{icon}</div><div class="feat-title">{title}</div><div class="feat-body">{body}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">How It Works</div><div class="section-sub">From animal parameters to optimised formula in four steps</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-card">
        <div class="steps-row">
            <div class="step-item"><div class="step-num">1</div><div class="step-title">Select Species &amp; Breed</div><div class="step-body">Choose from Rabbit, Poultry, or Cattle — then pick from 31+ breed profiles.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">2</div><div class="step-title">Enter Animal Parameters</div><div class="step-body">Provide age, weight, feed intake and production stage to define exact requirements.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">3</div><div class="step-title">Run the Optimizer</div><div class="step-body">Our LP engine solves for the minimum-cost blend across 97 Nigerian ingredients in seconds.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">4</div><div class="step-title">Analyse &amp; Export</div><div class="step-body">Review cost breakdowns, AI growth predictions, and ROI — then download your formula.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Get Started</div><div class="section-sub">Choose where you\'d like to begin</div>', unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    qs = [
        (qa1, "📖", "Nutrient Guide",    "Browse complete nutritional standards for every livestock species and production stage.", "Open Nutrient Guide →", "home_ng", "nutrient_guide"),
        (qa2, "🐾", "Breed Database",    "Explore 31+ breed profiles with feeding recommendations, growth rates, and market data.",  "Explore Breeds →",      "home_bd", "breed_database"),
        (qa3, "🔬", "Feed Formulator",   "Generate an optimised, least-cost feed formula for your animals right now.",              "Start Formulating →",   "home_ff", "formulator"),
    ]
    for col, icon, title, desc, btn_label, btn_key, target in qs:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;padding:2rem 1.5rem;"><div style="font-size:2.5rem;margin-bottom:.75rem;">{icon}</div><div style="font-weight:600;font-size:1rem;margin-bottom:.4rem;color:var(--tx);">{title}</div><div style="font-size:.85rem;color:var(--tx-3);margin-bottom:1.25rem;">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(btn_label, key=btn_key, type="primary", use_container_width=True):
                st.session_state.page = target
                st.rerun()

    if st.session_state.formulation_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Recent Formulations</div><div class="section-sub">Your last saved optimisation results</div>', unsafe_allow_html=True)
        for history in st.session_state.formulation_history[-3:]:
            with st.expander(f"🐾 {history['animal']}  ·  {history['timestamp']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Age",    f"{history['age']} weeks")
                    st.metric("Weight", f"{history['weight']} kg")
                with c2:
                    st.metric("Protein Req", f"{history['cp_req']}%")
                    st.metric("Energy Req",  f"{history['energy_req']} kcal")
                with c3:
                    if "total_cost" in history:
                        st.metric("Cost/kg",    f"₦{history['total_cost']:.2f}")
                    if "prediction" in history:
                        st.metric("Daily Gain", f"{history['prediction']:.1f} g")


# ─────────────────────────────────────────────
#  BREED DATABASE PAGE
# ─────────────────────────────────────────────
def show_breed_database():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">🐾 Breed Database</div><div class="page-desc">Comprehensive profiles for 31+ livestock breeds suited to Nigerian climate and production systems.</div></div>', unsafe_allow_html=True)
    breed_data  = get_breed_database()
    animal_type = st.selectbox("Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    breeds      = breed_data[animal_type]
    col1, col2  = st.columns([3, 1])
    with col1:
        raw_search = st.text_input("🔍 Search breeds", placeholder="Type breed name…", max_chars=100)
        search = sanitize_text(raw_search)
    with col2:
        if animal_type in ["Rabbit", "Cattle"]:
            type_filter = st.selectbox("Filter by Type", ["All"] + sorted(set(b["Type"] for b in breeds.values())))
        else:
            type_filter = "All"
    st.markdown("---")

    for breed_name, breed_info in breeds.items():
        if search and search.lower() not in breed_name.lower():
            continue
        if type_filter != "All" and breed_info["Type"] != type_filter:
            continue
        st.markdown('<div class="breed-card-wrap">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {breed_name}")
            st.markdown(
                f'<p style="color:var(--tx-2);font-size:.9rem;margin:.2rem 0 .75rem;">'
                f'<strong style="color:var(--tx);">Type:</strong> '
                f'<code style="background:var(--g100);color:var(--g700);padding:.1rem .5rem;border-radius:4px;font-size:.82rem;">{breed_info["Type"]}</code>'
                f'&nbsp;&nbsp;<strong style="color:var(--tx);">Best For:</strong> {breed_info["Best For"]}</p>',
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Mature Weight", breed_info["Mature Weight (kg)"] + " kg")
            with m2: st.metric("Growth Rate",   breed_info["Growth Rate"])
            with m3: st.metric("Feed Efficiency", breed_info["Feed Efficiency"])
        with col2:
            market_key = "Market Age (months)" if animal_type == "Cattle" else "Market Age (weeks)"
            unit = "months" if animal_type == "Cattle" else "weeks"
            st.markdown(
                f'<div style="background:var(--surface-3);border:1px solid var(--border);border-radius:10px;padding:1rem 1rem .75rem;margin-top:.5rem;">'
                f'<div style="font-weight:700;font-size:.75rem;color:var(--tx-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem;">Feeding Guide</div>'
                f'<div style="background:var(--g100);border:1px solid var(--g200);border-radius:8px;padding:.5rem .75rem;margin-bottom:.5rem;">'
                f'<span style="font-size:.75rem;color:var(--tx-3);">Protein</span><br>'
                f'<strong style="color:var(--g700);font-size:.95rem;">{breed_info["Recommended CP (%)"]}%</strong></div>'
                f'<div style="background:#dbeafe;border:1px solid #bfdbfe;border-radius:8px;padding:.5rem .75rem;">'
                f'<span style="font-size:.75rem;color:#1e40af;">Market Age</span><br>'
                f'<strong style="color:#1d4ed8;font-size:.95rem;">{breed_info[market_key]} {unit}</strong></div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Use in Formulator", key=f"breed_{breed_name}"):
                st.session_state.selected_breed = breed_name
                st.session_state.page = "formulator"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Breed Statistics</div>', unsafe_allow_html=True)
    breed_df_stats = pd.DataFrame(breeds).T
    col1, col2 = st.columns(2)
    with col1:
        type_counts = breed_df_stats["Type"].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                     title="Distribution by Production Type",
                     color_discrete_sequence=px.colors.sequential.Greens[::-1])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        growth_counts = breed_df_stats["Growth Rate"].value_counts()
        fig = px.bar(x=growth_counts.index, y=growth_counts.values,
                     title="Breeds by Growth Rate",
                     labels={"x":"Growth Rate","y":"Count"},
                     color=growth_counts.values, color_continuous_scale="Greens")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  NUTRIENT GUIDE PAGE
# ─────────────────────────────────────────────
def show_nutrient_guide():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">📖 Nutrient Requirements Guide</div><div class="page-desc">Science-backed nutritional standards for every livestock type and production stage.</div></div>', unsafe_allow_html=True)
    nutrient_data = get_nutrient_requirements()
    animal_type   = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    requirements  = nutrient_data[animal_type]
    st.markdown("---")

    for stage, nutrients in requirements.items():
        st.markdown(f'<div class="stage-header">🎯 {stage}</div>', unsafe_allow_html=True)
        df_n = pd.DataFrame([nutrients]).T
        df_n.columns = ["Requirement"]
        df_n.index.name = "Nutrient Parameter"
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df_n, use_container_width=True)
        with col2:
            st.markdown("##### Key Values")
            if "Crude Protein (%)" in nutrients:
                st.metric("Crude Protein",    nutrients["Crude Protein (%)"])
            if "Energy (kcal/kg)" in nutrients:
                st.metric("Energy (kcal/kg)", nutrients["Energy (kcal/kg)"])
            if "Crude Fiber (%)" in nutrients:
                st.metric("Crude Fiber",      nutrients["Crude Fiber (%)"])

    st.markdown("---")
    st.markdown("### 📋 Feeding Guidelines")
    if animal_type == "Rabbit":
        st.markdown('<div class="alert-green"><strong>Rabbit Feeding Guidelines:</strong><br>• Provide fresh water at all times (rabbits drink 2–3× their feed weight)<br>• Hay should make up 70–80% of adult rabbit diet<br>• Introduce new feeds gradually over 7–10 days<br>• Monitor body condition score regularly<br>• Higher fiber content prevents digestive issues and hairballs</div>', unsafe_allow_html=True)
    elif animal_type == "Poultry":
        st.markdown('<div class="alert-green"><strong>Poultry Feeding Guidelines:</strong><br>• Layer birds require high calcium (3.5–4%) for strong eggshells<br>• Grit (insoluble granite) aids digestion, especially for whole grains<br>• Feed should be stored in cool, dry, rodent-proof conditions<br>• Sudden feed changes can reduce performance by 10–20%<br>• Water consumption is roughly 2× feed intake</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-green"><strong>Cattle Feeding Guidelines:</strong><br>• TDN = Total Digestible Nutrients (energy measure for ruminants)<br>• Ruminants require 15–20% fiber for proper rumen function<br>• Transition periods are critical — allow 21 days minimum<br>• Fresh, clean water must always be available (50–80 L/day)<br>• Monitor body condition score (BCS 1–9, target: 5–6)</div>', unsafe_allow_html=True)

    st.markdown("---")
    all_stages = []
    for stage, nutrients in requirements.items():
        row = {"Stage": stage}
        row.update(nutrients)
        all_stages.append(row)
    download_df = pd.DataFrame(all_stages)
    csv = download_df.to_csv(index=False)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(label=f"📥 Download {animal_type} Nutrient Guide (CSV)", data=csv,
                           file_name=f"{animal_type.lower()}_nutrient_guide.csv", mime="text/csv", use_container_width=True)
    with col2:
        if st.button("🔬 Proceed to Feed Formulator", type="primary", use_container_width=True):
            st.session_state.page = "formulator"
            st.rerun()


# ─────────────────────────────────────────────
#  FORMULATOR PAGE
# ─────────────────────────────────────────────
def show_formulator():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">🔬 Feed Formulation Centre</div><div class="page-desc">Configure your animal parameters in the sidebar, then use the tabs below to optimise, analyse, and export your custom feed formula.</div></div>', unsafe_allow_html=True)

    animal = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    if animal == "Rabbit":
        df = rabbit_df.copy()
        st.markdown('<div class="alert-green">🐰 <strong>Rabbit Nutrition</strong> — Formulating for herbivores with high fibre needs</div>', unsafe_allow_html=True)
    elif animal == "Poultry":
        df = poultry_df.copy()
        st.markdown('<div class="alert-green">🐔 <strong>Poultry Nutrition</strong> — Optimising for broilers and layers</div>', unsafe_allow_html=True)
    else:
        df = cattle_df.copy()
        st.markdown('<div class="alert-green">🐄 <strong>Cattle Nutrition</strong> — Formulating for ruminants</div>', unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────
    st.sidebar.markdown(
        '<div style="background:linear-gradient(135deg,#0b2518,#1a6640);color:#fff;padding:1.25rem 1rem;border-radius:10px;margin-bottom:1rem;text-align:center;">'
        '<div style="font-size:1.4rem;margin-bottom:.3rem;">⚙️</div>'
        '<div style="font-weight:600;font-size:.95rem;">Animal Parameters</div>'
        '<div style="font-size:.75rem;opacity:.7;margin-top:.2rem;">Configure inputs below</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### 🎨 Appearance")
    current_label = "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"
    if st.sidebar.button(current_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.sidebar.caption(f"Currently: {'Dark mode ON' if st.session_state.dark_mode else 'Light mode ON'}")
    st.sidebar.markdown("---")
    if "selected_breed" in st.session_state:
        st.sidebar.success(f"✓ Breed: {st.session_state.selected_breed}")

    age         = sanitize_int(st.sidebar.slider("Age (weeks)", 1, 120, 8), 1, 120, 8)
    weight      = sanitize_numeric(st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0), 0.1, 600.0, 2.0)
    cp_req      = sanitize_numeric(st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18), 10.0, 30.0, 18.0)
    energy_req  = sanitize_numeric(st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000), 2000.0, 12000.0, 3000.0)
    feed_intake = sanitize_numeric(st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5), 0.05, 30.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Summary")
    st.sidebar.metric("Animal",               animal)
    st.sidebar.metric("Ingredients Available", len(df))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Navigate")
    if st.sidebar.button("📖 Nutrient Guide",  use_container_width=True):
        st.session_state.page = "nutrient_guide"; st.rerun()
    if st.sidebar.button("🐾 Breed Database", use_container_width=True):
        st.session_state.page = "breed_database"; st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Feed Optimizer","📋 Ingredient Database","📈 Growth Prediction","📊 Cost Dashboard"])

    # ── TAB 1: OPTIMIZER ─────────────────────
    with tab1:
        st.header("🔬 Least-Cost Feed Formulation")
        st.markdown("Using **linear programming** to find the cheapest ingredient blend meeting all nutritional requirements.")

        breed_db    = get_breed_database()
        nutrient_db = get_nutrient_requirements()

        st.markdown('<div class="nutrient-panel">', unsafe_allow_html=True)
        st.markdown('<div class="nutrient-panel-title">🐾 Step 1 — Select Breed & Production Stage</div>', unsafe_allow_html=True)
        breed_col, stage_col = st.columns(2)
        with breed_col:
            breed_options = ["— Select a breed (optional) —"] + list(breed_db[animal].keys())
            default_idx   = 0
            if "selected_breed" in st.session_state and st.session_state.selected_breed in breed_db[animal]:
                default_idx = breed_options.index(st.session_state.selected_breed)
            selected_breed = st.selectbox("🐾 Breed", breed_options, index=default_idx, key="opt_breed")
        with stage_col:
            selected_stage = st.selectbox("🎯 Production Stage", list(nutrient_db[animal].keys()), key="opt_stage")

        if selected_breed and selected_breed != "— Select a breed (optional) —":
            binfo = breed_db[animal][selected_breed]
            st.markdown(f'<div class="breed-badge">✓ {selected_breed} · {binfo["Type"]} · Recommended CP: {binfo["Recommended CP (%)"]}</div>', unsafe_allow_html=True)

        stage_data = nutrient_db[animal][selected_stage]
        sd_cols    = st.columns(len(stage_data))
        for idx, (k, v) in enumerate(stage_data.items()):
            short_k = k.replace(" (%)","").replace(" (kcal/kg)","").replace(" (g/day)","").replace(" (kg/day)","")
            with sd_cols[idx]:
                st.markdown(f'<div class="nutrient-chip"><div class="nutrient-chip-label">{short_k}</div><div class="nutrient-chip-value">{v}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        def parse_mid(val_str):
            try:
                parts = str(val_str).split("-")
                return round((float(parts[0]) + float(parts[-1])) / 2, 1)
            except Exception:
                return 0.0

        st.markdown('<div class="nutrient-panel" style="margin-top:.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="nutrient-panel-title">🧪 Step 2 — Set Nutrient Targets</div>', unsafe_allow_html=True)
        st.caption("Auto-filled from the selected stage. Adjust freely before running the optimiser.")

        cp_default     = parse_mid(stage_data.get("Crude Protein (%)", "16-18"))
        energy_default = parse_mid(stage_data.get("Energy (kcal/kg)", "2500-2700"))
        fiber_default  = parse_mid(stage_data.get("Crude Fiber (%)", "12-16"))

        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            cp_req_inp = sanitize_numeric(
                st.number_input("Crude Protein (%)", min_value=8.0, max_value=35.0,
                                value=float(cp_default), step=0.5, key="ni_cp"),
                8.0, 35.0, cp_default)
        with n_col2:
            energy_inp = sanitize_numeric(
                st.number_input("Energy (kcal/kg)", min_value=1500.0, max_value=4500.0,
                                value=float(min(energy_default, 4500)), step=50.0, key="ni_energy"),
                1500.0, 4500.0, energy_default)
        with n_col3:
            intake_inp = sanitize_numeric(
                st.number_input("Daily Feed Intake (kg)", min_value=0.01, max_value=30.0,
                                value=float(st.session_state.get("feed_intake_val", 0.5)), step=0.05, key="ni_intake"),
                0.01, 30.0, 0.5)
        st.session_state["feed_intake_val"] = intake_inp

        n_col4, n_col5, _ = st.columns(3)
        with n_col4:
            use_fiber = st.checkbox("📏 Set Fiber Targets", key="ni_use_fiber")
            if use_fiber:
                min_fiber = sanitize_numeric(
                    st.number_input("Min Fiber (%)", 0.0, 30.0, max(0.0, fiber_default - 2), 0.5, key="ni_fmin"),
                    0.0, 30.0, max(0.0, fiber_default - 2))
                max_fiber = sanitize_numeric(
                    st.number_input("Max Fiber (%)", 0.0, 40.0, fiber_default + 4, 0.5, key="ni_fmax"),
                    0.0, 40.0, fiber_default + 4)
            else:
                min_fiber, max_fiber = 0.0, 40.0
        with n_col5:
            limit_ingredients = st.checkbox("🔢 Limit Ingredient Count", key="ni_limit")
            max_ingredients   = st.slider("Max ingredients", 3, 15, 8, key="ni_max_ingr") if limit_ingredients else 15
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        run_col, _ = st.columns([1, 2])
        with run_col:
            run_btn = st.button("🚀 Optimise Feed Formula", type="primary", use_container_width=True, key="run_opt")

        if run_btn:
            # ── RATE LIMIT CHECK ──
            allowed, msg = check_rate_limit("optimize")
            if not allowed:
                st.warning(msg)
            else:
                with st.spinner("Calculating optimal feed mix…"):
                    try:
                        prob        = LpProblem("FeedMix", LpMinimize)
                        ingredients = df["Ingredient"].tolist()
                        vars_       = LpVariable.dicts("Ingr", ingredients, lowBound=0, upBound=1)
                        prob += lpSum(vars_[i] * df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
                        prob += lpSum(vars_[i] for i in ingredients) == 1
                        prob += lpSum(vars_[i] * df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients) >= cp_req_inp
                        prob += lpSum(vars_[i] * df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients) >= energy_inp
                        if use_fiber and "Fiber" in df.columns:
                            prob += lpSum(vars_[i] * df[df["Ingredient"]==i]["Fiber"].values[0] for i in ingredients) >= min_fiber
                            prob += lpSum(vars_[i] * df[df["Ingredient"]==i]["Fiber"].values[0] for i in ingredients) <= max_fiber
                        prob.solve()

                        if LpStatus[prob.status] == "Optimal":
                            result = {i: vars_[i].value() for i in ingredients if vars_[i].value() > 0.001}
                            if limit_ingredients and len(result) > max_ingredients:
                                st.warning(f"⚠️ Solution uses {len(result)} ingredients (limit: {max_ingredients}).")
                            result_df_out = pd.DataFrame(result.items(), columns=["Ingredient","Proportion"])
                            result_df_out["Proportion (%)"]       = (result_df_out["Proportion"] * 100).round(2)
                            result_df_out["Cost/kg (₦)"]          = result_df_out["Ingredient"].apply(lambda x: df[df["Ingredient"]==x]["Cost"].values[0])
                            result_df_out["Cost Contribution (₦)"] = (result_df_out["Proportion"] * result_df_out["Cost/kg (₦)"]).round(2)
                            result_df_out["CP Contribution"]      = result_df_out["Ingredient"].apply(lambda x: df[df["Ingredient"]==x]["CP"].values[0]) * result_df_out["Proportion"]
                            result_df_out["Energy Contribution"]  = result_df_out["Ingredient"].apply(lambda x: df[df["Ingredient"]==x]["Energy"].values[0]) * result_df_out["Proportion"]
                            total_cp     = result_df_out["CP Contribution"].sum()
                            total_energy = result_df_out["Energy Contribution"].sum()
                            result_df_out = result_df_out.sort_values("Proportion", ascending=False)
                            total_cost   = value(prob.objective)

                            st.session_state["optimization_result"] = result_df_out
                            st.session_state["total_cost"]          = total_cost
                            st.session_state["total_cp"]            = total_cp
                            st.session_state["total_energy"]        = total_energy
                            st.session_state.formulation_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "animal": animal, "age": age, "weight": weight,
                                "cp_req": cp_req_inp, "energy_req": energy_inp, "total_cost": total_cost,
                            })

                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("💰 Feed Cost/kg",   f"₦{total_cost:.2f}")
                            with col2: st.metric("📅 Daily Feed Cost", f"₦{total_cost * intake_inp:.2f}")
                            with col3: st.metric("📦 Ingredients Used", len(result))
                            with col4: st.metric("📆 Monthly Cost",   f"₦{total_cost * intake_inp * 30:.2f}")
                            st.markdown("---")
                            st.subheader("✅ Nutritional Achievement")
                            col1, col2 = st.columns(2)
                            with col1:
                                cp_pct = (total_cp / cp_req_inp * 100) if cp_req_inp > 0 else 0
                                st.metric("Crude Protein", f"{total_cp:.2f}%", delta=f"{cp_pct:.1f}% of requirement")
                            with col2:
                                energy_pct = (total_energy / energy_inp * 100) if energy_inp > 0 else 0
                                st.metric("Energy", f"{total_energy:.0f} kcal/kg", delta=f"{energy_pct:.1f}% of requirement")
                            st.success(f"✅ Optimisation complete! Total cost: ₦{total_cost:.2f}/kg")
                            st.dataframe(result_df_out[["Ingredient","Proportion (%)","Cost/kg (₦)","Cost Contribution (₦)"]],
                                         use_container_width=True, hide_index=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_pie = px.pie(result_df_out, values="Proportion (%)", names="Ingredient",
                                                 title="Feed Composition", color_discrete_sequence=px.colors.sequential.Greens)
                                fig_pie.update_layout(template="plotly_white")
                                st.plotly_chart(fig_pie, use_container_width=True)
                            with col2:
                                fig_bar = px.bar(result_df_out, x="Ingredient", y="Cost Contribution (₦)",
                                                 title="Cost Breakdown by Ingredient",
                                                 color="Cost Contribution (₦)", color_continuous_scale="Greens")
                                fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_white")
                                st.plotly_chart(fig_bar, use_container_width=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_out = result_df_out.to_csv(index=False)
                                st.download_button("📥 Download Formula (CSV)", csv_out,
                                                   f"{animal}_feed_formula_{datetime.now().strftime('%Y%m%d')}.csv",
                                                   "text/csv", use_container_width=True)
                            with col2:
                                # Rate-limit report download generation
                                allowed_r, msg_r = check_rate_limit("report")
                                if allowed_r:
                                    report = generate_report(animal, age, weight, cp_req_inp, energy_inp,
                                                             intake_inp, result_df_out, total_cost)
                                    st.download_button("📄 Download Report (TXT)", report,
                                                       f"{animal}_feed_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                                       "text/plain", use_container_width=True)
                                else:
                                    st.warning(msg_r)
                        else:
                            st.error("❌ No feasible solution found. Try relaxing your nutrient targets or constraints.")
                    except Exception as e:
                        st.error(f"❌ Error during optimisation: {str(e)}")

    # ── TAB 2: INGREDIENT DB ─────────────────
    with tab2:
        st.header("📋 Ingredient Database Manager")
        st.markdown(f"**{len(df)} ingredients** available for {animal} feed formulation.")
        col1, col2 = st.columns(2)
        with col1:
            raw_s = st.text_input("🔍 Search ingredients", placeholder="Type to filter…", max_chars=100)
            search_ingr = sanitize_text(raw_s)
        with col2:
            sort_by = st.selectbox("Sort by", ["Ingredient","CP","Energy","Cost"])
        filtered_df = df[df["Ingredient"].str.contains(search_ingr, case=False, na=False)] if search_ingr else df
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_by == "Ingredient"))
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Ingredients", len(filtered_df))
        with col2: st.metric("Avg Cost/kg",   f"₦{filtered_df['Cost'].mean():.2f}")
        with col3: st.metric("Avg Protein",   f"{filtered_df['CP'].mean():.1f}%")
        with col4: st.metric("Avg Energy",    f"{filtered_df['Energy'].mean():.0f} kcal")
        st.markdown("---")
        edited_df = st.data_editor(
            filtered_df, num_rows="dynamic", use_container_width=True,
            column_config={
                "Ingredient": st.column_config.TextColumn("Ingredient", width="medium"),
                "CP":     st.column_config.NumberColumn("Crude Protein (%)", format="%.1f"),
                "Energy": st.column_config.NumberColumn("Energy (kcal/kg)",  format="%.0f"),
                "Fiber":  st.column_config.NumberColumn("Crude Fiber (%)",   format="%.1f"),
                "Cost":   st.column_config.NumberColumn("Cost (₦/kg)",       format="₦%.2f"),
            },
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes to Database", use_container_width=True):
                allowed_s, msg_s = check_rate_limit("save_db")
                if not allowed_s:
                    st.warning(msg_s)
                else:
                    clean_df = sanitize_df_edit(edited_df)
                    if clean_df.empty:
                        st.error("❌ No valid rows to save after sanitisation.")
                    else:
                        clean_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
                        st.success("✅ Ingredient database updated successfully!")
                        st.cache_data.clear()
        with col2:
            csv_db = sanitize_df_edit(edited_df).to_csv(index=False)
            st.download_button("📥 Download Database (CSV)", csv_db,
                               f"{animal.lower()}_ingredients.csv", "text/csv", use_container_width=True)

    # ── TAB 3: GROWTH PREDICTION ─────────────
    with tab3:
        st.header("📈 AI Weight Gain Prediction")
        st.markdown("**Random Forest ML model** trained on 110+ feeding trials from Nigerian farms.")
        st.markdown("---")
        if st.button("🎯 Calculate Growth Prediction", type="primary"):
            allowed_p, msg_p = check_rate_limit("predict")
            if not allowed_p:
                st.warning(msg_p)
            else:
                with st.spinner("Calculating growth predictions…"):
                    avg_cp     = sanitize_numeric(df["CP"].mean(),     0, 100, 18)
                    avg_energy = sanitize_numeric(df["Energy"].mean(), 0, 10000, 2800)
                    X_input    = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
                    prediction = model.predict(X_input)[0]
                    st.session_state["prediction"] = float(prediction)

        if "prediction" in st.session_state:
            prediction          = st.session_state["prediction"]
            weekly_gain         = prediction * 7
            monthly_gain        = prediction * 30
            projected_weight_90 = weight + (monthly_gain * 3 / 1000)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Weight Gain", f"{prediction:.1f} g/day")
            with col2: st.metric("Weekly Gain",        f"{weekly_gain:.0f} g")
            with col3: st.metric("Monthly Gain",       f"{monthly_gain/1000:.2f} kg")
            with col4: st.metric("90-Day Weight",      f"{projected_weight_90:.1f} kg", delta=f"+{projected_weight_90 - weight:.1f} kg")
            st.subheader("📊 90-Day Weight Projection")
            days              = np.arange(0, 91)
            projected_weights = weight + (prediction * days / 1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=projected_weights, mode="lines", name="Projected Weight",
                                     line=dict(color="#208550", width=3), fill="tozeroy", fillcolor="rgba(32,133,80,.08)"))
            fig.add_trace(go.Scatter(x=[0], y=[weight], mode="markers", name="Current Weight",
                                     marker=dict(size=12, color="#dc2626")))
            fig.update_layout(xaxis_title="Days", yaxis_title="Weight (kg)", hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
                st.metric("Feed Conversion Ratio (FCR)", f"{fcr:.2f}:1")
                st.caption("Feed required to gain 1 kg of body weight")
            with col2:
                if "total_cost" in st.session_state and prediction > 0:
                    cost_per_kg = (st.session_state["total_cost"] * feed_intake * 1000) / prediction
                    st.metric("Cost per kg Gain", f"₦{cost_per_kg:.2f}")
                    st.caption("Feed cost to produce 1 kg of weight gain")
                else:
                    st.info("💡 Run the Feed Optimizer first to see cost metrics")
            st.markdown("---")
            st.subheader("🎯 Growth Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if animal == "Rabbit":
                    perf = "🟢 Excellent" if prediction > 30 else ("🟡 Good" if prediction > 20 else "🔴 Below Average")
                elif animal == "Poultry":
                    perf = "🟢 Excellent" if prediction > 50 else ("🟡 Good" if prediction > 35 else "🔴 Below Average")
                else:
                    perf = "🟢 Excellent" if prediction > 800 else ("🟡 Good" if prediction > 500 else "🔴 Below Average")
                st.metric("Performance Rating", perf)
            with col2:
                target_weight = 2.5 if animal == "Rabbit" else (2.0 if animal == "Poultry" else 300)
                if prediction > 0 and weight < target_weight:
                    days_to_target = int((target_weight - weight) * 1000 / prediction)
                    st.metric("Days to Market Weight", f"{days_to_target} days")
                    st.caption(f"Target: {target_weight} kg")
                else:
                    st.metric("Market Weight", "✅ Achieved")
        else:
            st.info("👆 Click 'Calculate Growth Prediction' above to see results")

    # ── TAB 4: COST DASHBOARD ────────────────
    with tab4:
        st.header("📊 Cost Analysis Dashboard")
        if "optimization_result" not in st.session_state:
            st.markdown('<div class="alert-amber">⚠️ Please run the Feed Optimizer first to unlock the Cost Dashboard</div>', unsafe_allow_html=True)
        else:
            result_df_c = st.session_state["optimization_result"]
            total_cost  = st.session_state["total_cost"]
            st.subheader("💰 Cost Projections")
            daily_cost = total_cost * feed_intake
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Cost",   f"₦{daily_cost:.2f}")
            with col2: st.metric("Weekly Cost",  f"₦{daily_cost * 7:.2f}")
            with col3: st.metric("Monthly Cost", f"₦{daily_cost * 30:.2f}")
            with col4: st.metric("Yearly Cost",  f"₦{daily_cost * 365:,.2f}")
            st.markdown("---")
            st.subheader("🐾 Herd / Flock Cost Calculator")
            col1, col2 = st.columns(2)
            with col1:
                num_animals   = sanitize_int(st.number_input("Number of Animals", 1, 10000, 100), 1, 10000, 100)
            with col2:
                duration_days = sanitize_int(st.slider("Duration (days)", 1, 365, 90), 1, 365, 90)
            total_herd = daily_cost * num_animals * duration_days
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Feed Cost",   f"₦{total_herd:,.2f}")
            with col2: st.metric("Cost per Animal",   f"₦{total_herd / num_animals:,.2f}")
            with col3: st.metric("Daily Herd Cost",   f"₦{daily_cost * num_animals:,.2f}")
            st.markdown("---")
            st.subheader("📊 Cost Breakdown Analysis")
            fig = px.treemap(result_df_c, path=["Ingredient"], values="Cost Contribution (₦)",
                             title="Cost Contribution by Ingredient",
                             color="Cost Contribution (₦)", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                top5 = result_df_c.nlargest(5, "Cost Contribution (₦)")
                fig  = px.bar(top5, x="Ingredient", y="Cost Contribution (₦)",
                              title="Top 5 Cost Contributors",
                              color="Cost Contribution (₦)", color_continuous_scale="Reds")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(result_df_c, x="Proportion (%)", y="Cost/kg (₦)",
                                 size="Cost Contribution (₦)", hover_name="Ingredient",
                                 title="Proportion vs Unit Cost",
                                 color="Cost Contribution (₦)", color_continuous_scale="Viridis")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            if "prediction" in st.session_state:
                st.markdown("---")
                st.subheader("💵 Return on Investment Calculator")
                prediction = st.session_state["prediction"]
                col1, col2 = st.columns(2)
                with col1:
                    default_price = 1500 if animal == "Rabbit" else (1200 if animal == "Poultry" else 2000)
                    price_per_kg  = sanitize_int(
                        st.number_input("Selling Price (₦/kg live weight)", 500, 5000, default_price), 500, 5000, default_price)
                with col2:
                    prod_days = sanitize_int(
                        st.number_input("Production Cycle (days)", 30, 365, 90), 30, 365, 90)
                total_feed_cost = daily_cost * prod_days
                weight_gain_kg  = (prediction * prod_days) / 1000
                final_weight    = weight + weight_gain_kg
                revenue         = final_weight * price_per_kg
                profit          = revenue - total_feed_cost
                roi_pct         = (profit / total_feed_cost * 100) if total_feed_cost > 0 else 0
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Feed Cost", f"₦{total_feed_cost:,.2f}")
                with col2: st.metric("Final Weight",    f"{final_weight:.2f} kg")
                with col3: st.metric("Revenue",         f"₦{revenue:,.2f}")
                with col4: st.metric("Profit",          f"₦{profit:,.2f}", delta=f"{roi_pct:.1f}% ROI")
                roi_data = pd.DataFrame({"Category":["Feed Cost","Profit"],
                                         "Amount":[total_feed_cost, max(profit, 0)]})
                fig = px.pie(roi_data, values="Amount", names="Category",
                             title=f"Cost vs Profit (ROI: {roi_pct:.1f}%)",
                             color_discrete_sequence=["#dc2626","#208550"])
                st.plotly_chart(fig, use_container_width=True)
                if profit > 0:
                    st.success(f"✅ Profitable! Expected profit of ₦{profit:,.2f} per animal over {prod_days} days.")
                else:
                    st.error("⚠️ Loss expected. Adjust feeding programme or selling price.")


# ─────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────
if   st.session_state.page == "home":          show_home()
elif st.session_state.page == "nutrient_guide": show_nutrient_guide()
elif st.session_state.page == "breed_database": show_breed_database()
elif st.session_state.page == "formulator":     show_formulator()

st.markdown("""
<div class="footer-wrap">
    <div>
        <div class="footer-brand">🌱 Necstech Feed Optimizer</div>
        <div class="footer-meta">Optimising African Agriculture · v2.0 · © 2026 Necstech</div>
    </div>
    <div class="footer-badges">
        <div class="badge">🇳🇬 Nigerian Data</div>
        <div class="badge">🤖 ML-Powered</div>
        <div class="badge">📊 NIAS · FAO 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

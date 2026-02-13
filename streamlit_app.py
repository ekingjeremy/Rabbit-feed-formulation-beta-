import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Necstech Feed Optimizer",
    page_icon="\U0001f331",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --green-900: #0d2818;
        --green-800: #154a2e;
        --green-700: #1d6b42;
        --green-600: #228b55;
        --green-500: #2da868;
        --green-400: #46c97f;
        --green-300: #7ddfaa;
        --green-100: #d6f5e5;
        --green-50:  #f0faf5;
        --amber:     #e8a020;
        --amber-light: #fdf0d5;
        --bg-app:        #f1f5f9;
        --bg-card:       #ffffff;
        --bg-card-hover: #f8fafc;
        --bg-input:      #ffffff;
        --bg-sidebar:    #ffffff;
        --border:        rgba(0,0,0,0.07);
        --border-strong: #cbd5e1;
        --text-primary:  #1a2332;
        --text-secondary:#334155;
        --text-muted:    #64748b;
        --text-inverse:  #ffffff;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.06);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.12), 0 4px 12px rgba(0,0,0,0.08);
        --radius:    12px;
        --radius-lg: 20px;
    }
    :root.dark, [data-theme="dark"] {
        --bg-app:        #0f1117;
        --bg-card:       #1a1f2e;
        --bg-card-hover: #1f2640;
        --bg-input:      #242938;
        --bg-sidebar:    #13171f;
        --border:        rgba(255,255,255,0.07);
        --border-strong: rgba(255,255,255,0.12);
        --text-primary:  #e8edf5;
        --text-secondary:#c4cdd8;
        --text-muted:    #8fa3b8;
        --text-inverse:  #0f1117;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.5);
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text-primary); }
    .stApp { background: var(--bg-app) !important; transition: background 0.3s ease; }
    .main .block-container { padding: 0 2rem 3rem 2rem; max-width: 1280px; }
    #MainMenu, footer, header { visibility: hidden; }

    .card { background: var(--bg-card); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--shadow-sm); border: 1px solid var(--border); transition: box-shadow 0.25s, transform 0.25s, background 0.3s; }
    .card:hover { background: var(--bg-card-hover); box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .feature-card { background: var(--bg-card); border-radius: var(--radius); padding: 1.75rem 1.5rem; border: 1px solid var(--border); box-shadow: var(--shadow-sm); transition: box-shadow 0.25s, transform 0.2s, background 0.3s; height: 100%; }
    .feature-card:hover { background: var(--bg-card-hover); box-shadow: var(--shadow-lg); transform: translateY(-3px); }
    .feat-icon { width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.25rem; margin-bottom:1rem; }
    .feat-icon.green  { background: var(--green-100); }
    .feat-icon.amber  { background: var(--amber-light); }
    .feat-icon.blue   { background: #e0f0ff; }
    .feat-icon.purple { background: #ede9ff; }
    :root.dark .feat-icon.green,  [data-theme="dark"] .feat-icon.green  { background: rgba(45,168,104,0.18); }
    :root.dark .feat-icon.amber,  [data-theme="dark"] .feat-icon.amber  { background: rgba(232,160,32,0.18); }
    :root.dark .feat-icon.blue,   [data-theme="dark"] .feat-icon.blue   { background: rgba(59,130,246,0.18); }
    :root.dark .feat-icon.purple, [data-theme="dark"] .feat-icon.purple { background: rgba(139,92,246,0.18); }
    .feat-title { font-weight:600; font-size:0.95rem; color:var(--text-primary); margin-bottom:0.4rem; }
    .feat-body  { font-size:0.85rem; color:var(--text-muted); line-height:1.55; }

    .hero-wrap { background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, var(--green-500) 100%); border-radius: var(--radius-lg); padding: 4rem 3.5rem; margin-bottom: 2.5rem; position: relative; overflow: hidden; }
    .hero-wrap::before { content:''; position:absolute; top:-60px; right:-80px; width:340px; height:340px; background:rgba(255,255,255,0.04); border-radius:50%; }
    .hero-wrap::after  { content:''; position:absolute; bottom:-40px; left:30%; width:220px; height:220px; background:rgba(255,255,255,0.03); border-radius:50%; }
    .hero-tag { display:inline-block; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2); color:var(--green-300); font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; padding:0.3rem 0.9rem; border-radius:20px; margin-bottom:1.2rem; }
    .hero-heading { font-family:'DM Serif Display',serif; font-size:3.2rem; color:#ffffff; line-height:1.15; margin-bottom:1.2rem; max-width:620px; }
    .hero-heading em { font-style:italic; color:var(--green-300); }
    .hero-body { font-size:1.05rem; color:rgba(255,255,255,0.75); line-height:1.7; max-width:560px; margin-bottom:2rem; }
    .hero-stats { display:flex; gap:2.5rem; margin-top:2.5rem; padding-top:2rem; border-top:1px solid rgba(255,255,255,0.12); }
    .hero-stat-num   { font-family:'DM Serif Display',serif; font-size:2rem; color:#ffffff; line-height:1; }
    .hero-stat-label { font-size:0.8rem; color:rgba(255,255,255,0.55); text-transform:uppercase; letter-spacing:0.05em; margin-top:0.25rem; }

    .stButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; border-radius: 8px !important; transition: all 0.2s !important; white-space: normal !important; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--green-700), var(--green-500)) !important; border: none !important; color: #ffffff !important; padding: 0.55rem 1.5rem !important; }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 6px 20px rgba(34,139,85,0.35) !important; transform: translateY(-1px) !important; }
    .stButton > button[kind="secondary"] { background: var(--bg-card) !important; border: 1.5px solid var(--border-strong) !important; color: var(--text-secondary) !important; }
    .stButton > button[kind="secondary"]:hover { border-color: var(--green-500) !important; color: var(--green-700) !important; }

    .section-heading { font-family:'DM Serif Display',serif; font-size:1.6rem; color:var(--text-primary); margin-bottom:0.3rem; }
    .section-sub     { font-size:0.9rem; color:var(--text-muted); margin-bottom:1.5rem; }
    .page-header     { margin-bottom:1.75rem; }
    .page-title      { font-family:'DM Serif Display',serif; font-size:2.2rem; color:var(--text-primary); line-height:1.2; margin-bottom:0.4rem; }
    .page-desc       { font-size:0.95rem; color:var(--text-muted); max-width:600px; }

    .steps-row { display:flex; gap:0; position:relative; }
    .step-item { flex:1; text-align:center; padding:1.5rem 1rem; position:relative; }
    .step-num  { width:36px; height:36px; border-radius:50%; background:var(--green-700); color:#ffffff; font-weight:700; font-size:0.9rem; display:flex; align-items:center; justify-content:center; margin:0 auto 0.75rem; position:relative; z-index:2; }
    .step-title { font-weight:600; font-size:0.9rem; color:var(--text-primary); }
    .step-body  { font-size:0.8rem; color:var(--text-muted); margin-top:0.3rem; line-height:1.5; }
    .step-connector { position:absolute; top:2.3rem; left:calc(50% + 18px); right:calc(-50% + 18px); height:2px; background:var(--green-300); z-index:1; }
    .steps-card { background: var(--bg-card); border-radius: var(--radius); padding: 1.5rem 2rem; border: 1px solid var(--border); box-shadow: var(--shadow-sm); transition: background 0.3s; }

    /* STAGE HEADER — vivid gradient, always readable */
    .stage-header { background: linear-gradient(90deg, var(--green-800), var(--green-600)); color: #ffffff; padding: 0.75rem 1.2rem; border-radius: 8px; margin: 1.2rem 0 0.8rem; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em; border-left: 4px solid var(--green-300); box-shadow: var(--shadow-sm); }

    /* BREED CARDS — explicit color on all text children */
    .breed-card-wrap { background: var(--bg-card); border-radius: var(--radius); border: 1px solid var(--border); border-left: 4px solid var(--green-500); padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: var(--shadow-sm); transition: box-shadow 0.2s, background 0.3s; }
    .breed-card-wrap:hover { background: var(--bg-card-hover); box-shadow: var(--shadow-md); }
    .breed-card-wrap h3, .breed-card-wrap h4 { color: var(--text-primary) !important; }
    .breed-card-wrap p, .breed-card-wrap div { color: var(--text-secondary) !important; }

    .alert-amber { background: var(--amber-light); border-left: 4px solid var(--amber); padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.75rem 0; font-size: 0.9rem; color: #4a3500; }
    .alert-green { background: var(--green-50); border-left: 4px solid var(--green-500); padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.75rem 0; font-size: 0.9rem; color: #1a3a2a; }
    :root.dark .alert-amber, [data-theme="dark"] .alert-amber { background: rgba(232,160,32,0.14); color: #f0d080; border-left-color: #e8a020; }
    :root.dark .alert-green, [data-theme="dark"] .alert-green { background: rgba(45,168,104,0.14); color: #7ddfaa; border-left-color: var(--green-400); }

    hr { border-color:var(--border-strong) !important; margin:1.5rem 0 !important; }

    [data-testid="stSidebar"] { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border-strong) !important; transition: background 0.3s; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size:1.1rem !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 { color: var(--text-primary) !important; }

    [data-testid="metric-container"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 0.9rem 1rem !important; box-shadow: var(--shadow-sm) !important; transition: background 0.3s !important; }
    [data-testid="metric-container"] label, [data-testid="stMetricLabel"] { font-size: 0.78rem !important; font-weight: 600 !important; color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; white-space: normal !important; }
    [data-testid="stMetricValue"] { font-family: 'DM Serif Display', serif !important; font-size: 1.55rem !important; color: var(--text-primary) !important; white-space: normal !important; overflow: visible !important; line-height: 1.2 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; white-space: normal !important; overflow: visible !important; }

    .stTabs [role="tablist"] { background: var(--bg-card); border-radius: 10px; padding: 0.3rem; border: 1px solid var(--border); box-shadow: var(--shadow-sm); gap: 0.2rem; transition: background 0.3s; }
    .stTabs [role="tab"] { border-radius: 7px !important; font-size: 0.85rem !important; font-weight: 500 !important; padding: 0.45rem 1rem !important; color: var(--text-secondary) !important; transition: all 0.2s !important; }
    .stTabs [role="tab"][aria-selected="true"] { background: var(--green-700) !important; color: #ffffff !important; }
    .stTabs [data-baseweb="tab-panel"] { background: var(--bg-card) !important; border-radius: 10px !important; padding: 1.75rem 1.5rem !important; border: 1px solid var(--border) !important; margin-top: 0.5rem !important; box-shadow: var(--shadow-sm) !important; transition: background 0.3s !important; }
    .stTabs [data-baseweb="tab-panel"] p, .stTabs [data-baseweb="tab-panel"] span, .stTabs [data-baseweb="tab-panel"] div, .stTabs [data-baseweb="tab-panel"] label, .stTabs [data-baseweb="tab-panel"] li { color: var(--text-secondary) !important; }
    .stTabs [data-baseweb="tab-panel"] h1, .stTabs [data-baseweb="tab-panel"] h2, .stTabs [data-baseweb="tab-panel"] h3, .stTabs [data-baseweb="tab-panel"] h4 { color: var(--text-primary) !important; }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label, .stCheckbox label { font-size: 0.88rem !important; font-weight: 500 !important; color: var(--text-secondary) !important; }
    [data-baseweb="select"] span, [data-baseweb="input"] input, input[type="number"], input[type="text"] { color: var(--text-primary) !important; }
    [data-baseweb="select"] > div { background: var(--bg-input) !important; border-color: var(--border-strong) !important; color: var(--text-primary) !important; }

    [data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 0.9rem !important; color: var(--green-700) !important; }
    [data-testid="stExpander"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; transition: background 0.3s; }
    :root.dark [data-testid="stExpander"] summary, [data-theme="dark"] [data-testid="stExpander"] summary { color: var(--green-400) !important; }

    [data-testid="stDataFrame"] { border-radius:8px !important; overflow:hidden !important; border:1px solid var(--border-strong) !important; }
    [data-testid="stDataFrame"] th { background:var(--bg-input) !important; color:var(--text-secondary) !important; font-weight:600 !important; font-size:0.82rem !important; text-transform:uppercase !important; letter-spacing:0.04em !important; padding:0.6rem 0.8rem !important; }
    [data-testid="stDataFrame"] td { color:var(--text-primary) !important; font-size:0.88rem !important; padding:0.5rem 0.8rem !important; white-space:normal !important; overflow:visible !important; }

    [data-testid="stSuccess"] { background:var(--green-50) !important; border-left:4px solid var(--green-500) !important; color:var(--text-primary) !important; border-radius:8px !important; }
    [data-testid="stInfo"]    { background:#eff6ff !important; border-left:4px solid #3b82f6 !important; color:var(--text-primary) !important; border-radius:8px !important; }
    [data-testid="stWarning"] { background:var(--amber-light) !important; border-left:4px solid var(--amber) !important; color:var(--text-primary) !important; border-radius:8px !important; }
    [data-testid="stError"]   { background:#fff0f0 !important; border-left:4px solid #e74c3c !important; color:var(--text-primary) !important; border-radius:8px !important; }
    :root.dark [data-testid="stSuccess"], [data-theme="dark"] [data-testid="stSuccess"] { background:rgba(45,168,104,0.14) !important; color:var(--text-primary) !important; }
    :root.dark [data-testid="stInfo"],    [data-theme="dark"] [data-testid="stInfo"]    { background:rgba(59,130,246,0.14) !important; color:var(--text-primary) !important; }
    :root.dark [data-testid="stWarning"], [data-theme="dark"] [data-testid="stWarning"] { background:rgba(232,160,32,0.14) !important; color:var(--text-primary) !important; }
    :root.dark [data-testid="stError"],   [data-theme="dark"] [data-testid="stError"]   { background:rgba(231,76,60,0.14)  !important; color:var(--text-primary) !important; }

    [data-testid="stCaptionContainer"] p { color:var(--text-muted) !important; font-size:0.8rem !important; }

    .footer-wrap { background:var(--green-900); color:rgba(255,255,255,0.6); border-radius:var(--radius-lg); padding:2.5rem 3rem; margin-top:3rem; display:flex; align-items:center; justify-content:space-between; }
    .footer-brand { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#ffffff; }
    .footer-meta  { font-size:0.78rem; }
    .footer-badges { display:flex; gap:0.5rem; }
    .badge { padding:0.25rem 0.75rem; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.1); border-radius:20px; font-size:0.72rem; color:rgba(255,255,255,0.65); }

    /* NUTRIENT PANEL */
    .nutrient-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: var(--shadow-sm); transition: background 0.3s; }
    .nutrient-panel-title { font-weight: 700; font-size: 0.9rem; color: var(--green-700); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.4rem; }
    :root.dark .nutrient-panel-title, [data-theme="dark"] .nutrient-panel-title { color: var(--green-400); }

    /* NUTRIENT CHIPS — styled mini-cards replacing plain text */
    .nutrient-chip { text-align: center; padding: 0.5rem 0.4rem; background: var(--green-50); border: 1px solid rgba(45,168,104,0.2); border-radius: 8px; transition: background 0.3s; }
    :root.dark .nutrient-chip, [data-theme="dark"] .nutrient-chip { background: rgba(45,168,104,0.12); border-color: rgba(45,168,104,0.3); }
    .nutrient-chip-label { font-size: 0.65rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    :root.dark .nutrient-chip-label, [data-theme="dark"] .nutrient-chip-label { color: #8fa3b8; }
    .nutrient-chip-value { font-size: 0.95rem; font-weight: 800; color: var(--green-700); margin-top: 0.15rem; }
    :root.dark .nutrient-chip-value, [data-theme="dark"] .nutrient-chip-value { color: var(--green-400); }

    .breed-badge { display: inline-flex; align-items: center; gap: 0.35rem; background: var(--green-100); color: var(--green-700); font-weight: 700; font-size: 0.82rem; padding: 0.28rem 0.85rem; border-radius: 20px; margin-bottom: 0.75rem; border: 1px solid rgba(29,107,66,0.15); }
    :root.dark .breed-badge, [data-theme="dark"] .breed-badge { background: rgba(45,168,104,0.18); color: var(--green-300); border-color: rgba(45,168,104,0.3); }

    @media (max-width: 768px) { [data-testid="column"] { width:100% !important; flex:1 1 100% !important; } .stButton > button { width:100% !important; font-size:0.88rem !important; } [data-testid="stMetricValue"] { font-size:1.2rem !important; } .hero-heading { font-size:2.2rem; } .hero-wrap { padding:2.5rem 1.5rem; } }
    @media (max-width: 480px) { .main .block-container { padding:0 1rem 1.5rem; } body { font-size:0.9rem; } [data-testid="stMetricValue"] { font-size:1rem !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

@st.cache_resource
def train_model(data):
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal","Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)


def get_breed_database():
    rabbit_breeds = {
        "New Zealand White": {"Type":"Meat","Mature Weight (kg)":"4.5-5.5","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Commercial meat production","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"},
        "Californian": {"Type":"Meat","Mature Weight (kg)":"4.0-5.0","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Meat and show","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"},
        "Flemish Giant": {"Type":"Meat","Mature Weight (kg)":"6.0-10.0","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Large-scale meat production","Recommended CP (%)":"17-19","Market Age (weeks)":"14-16"},
        "Dutch": {"Type":"Pet/Show","Mature Weight (kg)":"2.0-2.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Pets and breeding","Recommended CP (%)":"15-17","Market Age (weeks)":"8-10"},
        "Rex": {"Type":"Meat/Fur","Mature Weight (kg)":"3.5-4.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Fur and meat","Recommended CP (%)":"16-18","Market Age (weeks)":"10-12"}
    }
    poultry_breeds = {
        "Broiler (Cobb 500)": {"Type":"Meat","Mature Weight (kg)":"2.5-3.0","Growth Rate":"Very Fast","Feed Efficiency":"Excellent (FCR 1.6-1.8)","Best For":"Commercial meat production","Recommended CP (%)":"20-22","Market Age (weeks)":"5-6"},
        "Broiler (Ross 308)": {"Type":"Meat","Mature Weight (kg)":"2.3-2.8","Growth Rate":"Very Fast","Feed Efficiency":"Excellent (FCR 1.65-1.85)","Best For":"Commercial meat production","Recommended CP (%)":"20-22","Market Age (weeks)":"5-6"},
        "Layer (Isa Brown)": {"Type":"Eggs","Mature Weight (kg)":"1.8-2.0","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"High egg production (300+ eggs/year)","Recommended CP (%)":"16-18","Market Age (weeks)":"18-20 (point of lay)"},
        "Layer (Lohmann Brown)": {"Type":"Eggs","Mature Weight (kg)":"1.9-2.1","Growth Rate":"Moderate","Feed Efficiency":"Excellent","Best For":"Egg production (320+ eggs/year)","Recommended CP (%)":"16-18","Market Age (weeks)":"18-20 (point of lay)"},
        "Noiler": {"Type":"Dual Purpose","Mature Weight (kg)":"2.0-2.5","Growth Rate":"Fast","Feed Efficiency":"Good","Best For":"Meat and eggs (Nigerian adapted)","Recommended CP (%)":"18-20","Market Age (weeks)":"12-16"},
        "Kuroiler": {"Type":"Dual Purpose","Mature Weight (kg)":"2.5-3.5","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Free-range, dual purpose","Recommended CP (%)":"16-18","Market Age (weeks)":"14-18"},
        "Local Nigerian": {"Type":"Dual Purpose","Mature Weight (kg)":"1.2-1.8","Growth Rate":"Slow","Feed Efficiency":"Moderate","Best For":"Free-range, disease resistant","Recommended CP (%)":"14-16","Market Age (weeks)":"20-24"}
    }
    cattle_breeds = {
        "White Fulani": {"Type":"Beef/Dairy","Mature Weight (kg)":"300-450","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Milk and beef (Nigerian indigenous)","Recommended CP (%)":"14-16","Market Age (months)":"24-30"},
        "Red Bororo": {"Type":"Beef","Mature Weight (kg)":"250-350","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Beef production (heat tolerant)","Recommended CP (%)":"13-15","Market Age (months)":"24-28"},
        "Sokoto Gudali": {"Type":"Beef","Mature Weight (kg)":"350-500","Growth Rate":"Moderate-Fast","Feed Efficiency":"Good","Best For":"Beef (large frame)","Recommended CP (%)":"14-16","Market Age (months)":"24-30"},
        "N'Dama": {"Type":"Beef/Draft","Mature Weight (kg)":"300-400","Growth Rate":"Moderate","Feed Efficiency":"Good","Best For":"Trypanosomiasis resistant","Recommended CP (%)":"12-14","Market Age (months)":"30-36"},
        "Muturu": {"Type":"Beef/Draft","Mature Weight (kg)":"200-300","Growth Rate":"Slow","Feed Efficiency":"Moderate","Best For":"Small-holder, disease resistant","Recommended CP (%)":"12-14","Market Age (months)":"30-36"},
        "Holstein Friesian (Cross)": {"Type":"Dairy","Mature Weight (kg)":"450-650","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"High milk production","Recommended CP (%)":"16-18","Market Age (months)":"24-28"},
        "Brahman Cross": {"Type":"Beef","Mature Weight (kg)":"400-550","Growth Rate":"Fast","Feed Efficiency":"Excellent","Best For":"Beef (heat adapted)","Recommended CP (%)":"14-16","Market Age (months)":"20-24"}
    }
    return {"Rabbit": rabbit_breeds, "Poultry": poultry_breeds, "Cattle": cattle_breeds}


def get_nutrient_requirements():
    rabbit_nutrients = {
        "Grower (4-12 weeks)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"12-16","Calcium (%)":"0.4-0.8","Phosphorus (%)":"0.3-0.5","Lysine (%)":"0.65-0.75","Feed Intake (g/day)":"80-120"},
        "Finisher (12-16 weeks)": {"Crude Protein (%)":"14-16","Energy (kcal/kg)":"2400-2600","Crude Fiber (%)":"14-18","Calcium (%)":"0.4-0.7","Phosphorus (%)":"0.3-0.5","Lysine (%)":"0.55-0.65","Feed Intake (g/day)":"120-180"},
        "Doe (Maintenance)": {"Crude Protein (%)":"15-16","Energy (kcal/kg)":"2500-2600","Crude Fiber (%)":"14-16","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.4-0.5","Lysine (%)":"0.60-0.70","Feed Intake (g/day)":"100-150"},
        "Doe (Pregnant)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2600-2800","Crude Fiber (%)":"12-15","Calcium (%)":"0.8-1.2","Phosphorus (%)":"0.5-0.7","Lysine (%)":"0.70-0.80","Feed Intake (g/day)":"150-200"},
        "Doe (Lactating)": {"Crude Protein (%)":"17-19","Energy (kcal/kg)":"2700-3000","Crude Fiber (%)":"12-14","Calcium (%)":"1.0-1.5","Phosphorus (%)":"0.6-0.8","Lysine (%)":"0.75-0.90","Feed Intake (g/day)":"200-400"},
        "Buck (Breeding)": {"Crude Protein (%)":"15-17","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"14-16","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.4-0.6","Lysine (%)":"0.65-0.75","Feed Intake (g/day)":"120-170"}
    }
    poultry_nutrients = {
        "Broiler Starter (0-3 weeks)": {"Crude Protein (%)":"22-24","Energy (kcal/kg)":"3000-3200","Crude Fiber (%)":"3-4","Calcium (%)":"0.9-1.0","Phosphorus (%)":"0.45-0.50","Lysine (%)":"1.20-1.35","Methionine (%)":"0.50-0.55","Feed Intake (g/day)":"25-35"},
        "Broiler Grower (3-6 weeks)": {"Crude Protein (%)":"20-22","Energy (kcal/kg)":"3100-3300","Crude Fiber (%)":"3-5","Calcium (%)":"0.85-0.95","Phosphorus (%)":"0.40-0.45","Lysine (%)":"1.05-1.20","Methionine (%)":"0.45-0.50","Feed Intake (g/day)":"80-120"},
        "Broiler Finisher (6+ weeks)": {"Crude Protein (%)":"18-20","Energy (kcal/kg)":"3200-3400","Crude Fiber (%)":"3-5","Calcium (%)":"0.80-0.90","Phosphorus (%)":"0.35-0.40","Lysine (%)":"0.95-1.10","Methionine (%)":"0.40-0.45","Feed Intake (g/day)":"140-180"},
        "Layer Starter (0-6 weeks)": {"Crude Protein (%)":"18-20","Energy (kcal/kg)":"2800-3000","Crude Fiber (%)":"3-5","Calcium (%)":"0.9-1.0","Phosphorus (%)":"0.45-0.50","Lysine (%)":"0.95-1.05","Methionine (%)":"0.40-0.45","Feed Intake (g/day)":"20-40"},
        "Layer Grower (6-18 weeks)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2700-2900","Crude Fiber (%)":"4-6","Calcium (%)":"0.8-0.9","Phosphorus (%)":"0.40-0.45","Lysine (%)":"0.75-0.85","Methionine (%)":"0.35-0.40","Feed Intake (g/day)":"60-90"},
        "Layer Production (18+ weeks)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2750-2900","Crude Fiber (%)":"4-6","Calcium (%)":"3.5-4.0","Phosphorus (%)":"0.35-0.40","Lysine (%)":"0.75-0.85","Methionine (%)":"0.38-0.42","Feed Intake (g/day)":"110-130"}
    }
    cattle_nutrients = {
        "Calf Starter (0-3 months)": {"Crude Protein (%)":"18-20","Energy (kcal/kg)":"3000-3200","Crude Fiber (%)":"8-12","Calcium (%)":"0.7-1.0","Phosphorus (%)":"0.4-0.6","TDN (%)":"72-78","Feed Intake (kg/day)":"0.5-1.5"},
        "Calf Grower (3-6 months)": {"Crude Protein (%)":"16-18","Energy (kcal/kg)":"2800-3000","Crude Fiber (%)":"10-15","Calcium (%)":"0.6-0.9","Phosphorus (%)":"0.35-0.50","TDN (%)":"68-74","Feed Intake (kg/day)":"2-4"},
        "Heifer (6-12 months)": {"Crude Protein (%)":"14-16","Energy (kcal/kg)":"2600-2800","Crude Fiber (%)":"12-18","Calcium (%)":"0.5-0.8","Phosphorus (%)":"0.30-0.45","TDN (%)":"65-70","Feed Intake (kg/day)":"4-7"},
        "Bull (Breeding)": {"Crude Protein (%)":"12-14","Energy (kcal/kg)":"2500-2700","Crude Fiber (%)":"15-20","Calcium (%)":"0.4-0.7","Phosphorus (%)":"0.25-0.40","TDN (%)":"62-68","Feed Intake (kg/day)":"8-12"},
        "Cow (Dry)": {"Crude Protein (%)":"10-12","Energy (kcal/kg)":"2400-2600","Crude Fiber (%)":"18-25","Calcium (%)":"0.4-0.6","Phosphorus (%)":"0.25-0.35","TDN (%)":"58-65","Feed Intake (kg/day)":"10-15"},
        "Cow (Lactating)": {"Crude Protein (%)":"14-18","Energy (kcal/kg)":"2700-3000","Crude Fiber (%)":"15-22","Calcium (%)":"0.6-0.9","Phosphorus (%)":"0.35-0.50","TDN (%)":"68-75","Feed Intake (kg/day)":"12-20"},
        "Beef Finisher": {"Crude Protein (%)":"12-14","Energy (kcal/kg)":"2800-3100","Crude Fiber (%)":"8-15","Calcium (%)":"0.5-0.7","Phosphorus (%)":"0.30-0.45","TDN (%)":"70-78","Feed Intake (kg/day)":"8-14"}
    }
    return {"Rabbit": rabbit_nutrients, "Poultry": poultry_nutrients, "Cattle": cattle_nutrients}


if "page" not in st.session_state:
    st.session_state.page = "home"
if "formulation_history" not in st.session_state:
    st.session_state.formulation_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

_dm = st.session_state.dark_mode
st.markdown(
    f"<script>document.documentElement.classList.{'add' if _dm else 'remove'}('dark');</script>",
    unsafe_allow_html=True,
)


def generate_report(animal, age, weight, cp_req, energy_req, feed_intake,
                    result_df=None, total_cost=None, prediction=None):
    report = f"""
    \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
                    NECSTECH FEED OPTIMIZER REPORT
    \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Species: {animal} | Age: {age} weeks | Weight: {weight} kg
    CP Req: {cp_req}% | Energy: {energy_req} kcal/kg
    """
    if result_df is not None and total_cost is not None:
        report += f"Total Cost/kg: \u20a6{total_cost:.2f} | Daily Cost: \u20a6{total_cost * feed_intake:.2f}\n"
        for _, row in result_df.iterrows():
            report += f"  {row['Ingredient']}: {row['Proportion (%)']:.2f}% (\u20a6{row['Cost Contribution (\u20a6)']:.2f})\n"
    if prediction is not None:
        weekly_gain = prediction * 7
        monthly_gain = prediction * 30
        projected_weight = weight + (monthly_gain * 3 / 1000)
        fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
        report += f"Daily Gain: {prediction:.1f} g | Weekly: {weekly_gain:.0f} g | Monthly: {monthly_gain/1000:.2f} kg\n"
        report += f"90-Day Weight: {projected_weight:.1f} kg | FCR: {fcr:.2f}:1\n"
    report += "\nGenerated by Necstech Feed Optimizer v2.0 | NIAS · FAO · 2026\n"
    return report


def render_navbar():
    current = st.session_state.page
    cols = st.columns([2, 1, 1, 1, 1, 0.7])
    with cols[0]:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#1d6b42,#46c97f);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">\U0001f331</div>
            <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:var(--text-primary);">Necs<span style="color:#228b55;">tech</span></span>
        </div>
        """, unsafe_allow_html=True)
    nav_items = [("home","\U0001f3e0 Home"),("nutrient_guide","\U0001f4d6 Nutrient Guide"),("breed_database","\U0001f43e Breeds"),("formulator","\U0001f52c Formulator")]
    for idx, (key, label) in enumerate(nav_items):
        with cols[idx + 1]:
            btn_type = "primary" if current == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.page = key
                st.rerun()
    with cols[5]:
        icon = "\u2600\ufe0f" if st.session_state.dark_mode else "\U0001f319"
        if st.button(icon, key="nav_theme", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)


def show_home():
    render_navbar()
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">\U0001f30d Built for Nigerian Agriculture · Powered by AI</div>
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
        ("green", "\U0001f4b0", "Least-Cost Formulation", "Linear programming engine automatically blends the cheapest ingredient combination that satisfies all protein, energy, and mineral constraints."),
        ("blue", "\U0001f916", "AI Growth Prediction", "Random Forest model forecasts daily weight gain, FCR, and 90-day projections based on Nigerian farm trial data."),
        ("amber", "\U0001f1f3\U0001f1ec", "Nigerian Market Data", "97 ingredients with verified 2026 local market prices. Edit, add, or remove ingredients to match your region's availability."),
        ("purple", "\U0001f4ca", "Cost & ROI Dashboard", "Detailed cost breakdowns, herd-level projections, and profit/loss analysis per production cycle to guide investment decisions."),
    ]
    for col, (icon_class, icon, title, body) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feat-icon {icon_class}">{icon}</div><div class="feat-title">{title}</div><div class="feat-body">{body}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">How It Works</div><div class="section-sub">From animal parameters to optimised formula in four steps</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-card">
        <div class="steps-row">
            <div class="step-item"><div class="step-num">1</div><div class="step-title">Select Species & Breed</div><div class="step-body">Choose from Rabbit, Poultry, or Cattle — then pick from 31+ breed profiles.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">2</div><div class="step-title">Enter Animal Parameters</div><div class="step-body">Provide age, weight, feed intake and production stage to define exact requirements.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">3</div><div class="step-title">Run the Optimizer</div><div class="step-body">Our LP engine solves for the minimum-cost blend across 97 Nigerian ingredients in seconds.</div><div class="step-connector"></div></div>
            <div class="step-item"><div class="step-num">4</div><div class="step-title">Analyse & Export</div><div class="step-body">Review cost breakdowns, AI growth predictions, and ROI — then download your formula.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Get Started</div><div class="section-sub">Choose where you\u2019d like to begin</div>', unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    qs = [
        (qa1, "\U0001f4d6", "Nutrient Guide", "Browse complete nutritional standards for every livestock species and production stage.", "Open Nutrient Guide \u2192", "home_ng", "nutrient_guide"),
        (qa2, "\U0001f43e", "Breed Database", "Explore 31+ breed profiles with feeding recommendations, growth rates, and market data.", "Explore Breeds \u2192", "home_bd", "breed_database"),
        (qa3, "\U0001f52c", "Feed Formulator", "Generate an optimised, least-cost feed formula for your animals right now.", "Start Formulating \u2192", "home_ff", "formulator"),
    ]
    for col, icon, title, desc, btn_label, btn_key, target in qs:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;padding:2rem 1.5rem;"><div style="font-size:2.5rem;margin-bottom:0.75rem;">{icon}</div><div style="font-weight:600;font-size:1rem;margin-bottom:0.4rem;color:var(--text-primary);">{title}</div><div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1.25rem;">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(btn_label, key=btn_key, type="primary", use_container_width=True):
                st.session_state.page = target
                st.rerun()

    if st.session_state.formulation_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Recent Formulations</div><div class="section-sub">Your last saved optimisation results</div>', unsafe_allow_html=True)
        for history in st.session_state.formulation_history[-3:]:
            with st.expander(f"\U0001f43e {history['animal']}  ·  {history['timestamp']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Age", f"{history['age']} weeks")
                    st.metric("Weight", f"{history['weight']} kg")
                with c2:
                    st.metric("Protein Req", f"{history['cp_req']}%")
                    st.metric("Energy Req", f"{history['energy_req']} kcal")
                with c3:
                    if "total_cost" in history:
                        st.metric("Cost/kg", f"\u20a6{history['total_cost']:.2f}")
                    if "prediction" in history:
                        st.metric("Daily Gain", f"{history['prediction']:.1f} g")


def show_breed_database():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">\U0001f43e Breed Database</div><div class="page-desc">Comprehensive profiles for 31+ livestock breeds suited to Nigerian climate and production systems.</div></div>', unsafe_allow_html=True)
    breed_data = get_breed_database()
    animal_type = st.selectbox("Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    breeds = breed_data[animal_type]
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("\U0001f50d Search breeds", placeholder="Type breed name\u2026")
    with col2:
        if animal_type in ["Rabbit", "Cattle"]:
            type_filter = st.selectbox("Filter by Type", ["All"] + list(set([b["Type"] for b in breeds.values()])))
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
                f'<p style="color:var(--text-secondary);font-size:0.9rem;margin:0.2rem 0 0.75rem;">' +
                f'<strong style="color:var(--text-primary);">Type:</strong> ' +
                f'<code style="background:rgba(45,168,104,0.12);color:var(--green-700);padding:0.1rem 0.5rem;border-radius:4px;font-size:0.82rem;">{breed_info["Type"]}</code>' +
                f'&nbsp;&nbsp;<strong style="color:var(--text-primary);">Best For:</strong> {breed_info["Best For"]}</p>',
                unsafe_allow_html=True
            )
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Mature Weight", breed_info["Mature Weight (kg)"] + " kg")
            with m2: st.metric("Growth Rate", breed_info["Growth Rate"])
            with m3: st.metric("Feed Efficiency", breed_info["Feed Efficiency"])
        with col2:
            market_key = "Market Age (months)" if animal_type == "Cattle" else "Market Age (weeks)"
            unit = "months" if animal_type == "Cattle" else "weeks"
            st.markdown(
                f'<div style="background:var(--bg-app);border:1px solid var(--border);border-radius:10px;padding:1rem 1rem 0.75rem;margin-top:0.5rem;">' +
                f'<div style="font-weight:700;font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">Feeding Guide</div>' +
                f'<div style="background:rgba(45,168,104,0.12);border:1px solid rgba(45,168,104,0.25);border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.5rem;">' +
                f'<span style="font-size:0.75rem;color:var(--text-muted);">Protein</span><br>' +
                f'<strong style="color:var(--green-700);font-size:0.95rem;">{breed_info["Recommended CP (%)"]}%</strong></div>' +
                f'<div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:8px;padding:0.5rem 0.75rem;">' +
                f'<span style="font-size:0.75rem;color:var(--text-muted);">Market Age</span><br>' +
                f'<strong style="color:#2563eb;font-size:0.95rem;">{breed_info[market_key]} {unit}</strong></div></div>',
                unsafe_allow_html=True
            )
            if st.button("Use in Formulator", key=f"breed_{breed_name}"):
                st.session_state.selected_breed = breed_name
                st.session_state.page = "formulator"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Breed Statistics</div>', unsafe_allow_html=True)
    breed_df = pd.DataFrame(breeds).T
    col1, col2 = st.columns(2)
    with col1:
        type_counts = breed_df["Type"].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Distribution by Production Type", color_discrete_sequence=px.colors.sequential.Greens[::-1])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        growth_counts = breed_df["Growth Rate"].value_counts()
        fig = px.bar(x=growth_counts.index, y=growth_counts.values, title="Breeds by Growth Rate", labels={"x":"Growth Rate","y":"Count"}, color=growth_counts.values, color_continuous_scale="Greens")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_nutrient_guide():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">\U0001f4d6 Nutrient Requirements Guide</div><div class="page-desc">Science-backed nutritional standards for every livestock type and production stage.</div></div>', unsafe_allow_html=True)
    nutrient_data = get_nutrient_requirements()
    animal_type = st.selectbox("\U0001f43e Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    requirements = nutrient_data[animal_type]
    st.markdown("---")
    for stage, nutrients in requirements.items():
        st.markdown(f'<div class="stage-header">\U0001f3af {stage}</div>', unsafe_allow_html=True)
        df_nutrients = pd.DataFrame([nutrients]).T
        df_nutrients.columns = ["Requirement"]
        df_nutrients.index.name = "Nutrient Parameter"
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df_nutrients, use_container_width=True)
        with col2:
            st.markdown("##### Key Values")
            if "Crude Protein (%)" in nutrients:
                st.metric("Crude Protein", nutrients["Crude Protein (%)"])
            if "Energy (kcal/kg)" in nutrients:
                st.metric("Energy (kcal/kg)", nutrients["Energy (kcal/kg)"])
            if "Crude Fiber (%)" in nutrients:
                st.metric("Crude Fiber", nutrients["Crude Fiber (%)"])

    st.markdown("---")
    st.markdown("### \U0001f4cb Feeding Guidelines")
    if animal_type == "Rabbit":
        st.markdown('<div class="alert-green"><strong>Rabbit Feeding Guidelines:</strong><br>\u2022 Provide fresh water at all times (rabbits drink 2\u20133\u00d7 their feed weight)<br>\u2022 Hay should make up 70\u201380% of adult rabbit diet<br>\u2022 Introduce new feeds gradually over 7\u201310 days<br>\u2022 Monitor body condition score regularly<br>\u2022 Higher fiber content prevents digestive issues and hairballs</div>', unsafe_allow_html=True)
    elif animal_type == "Poultry":
        st.markdown('<div class="alert-green"><strong>Poultry Feeding Guidelines:</strong><br>\u2022 Layer birds require high calcium (3.5\u20134%) for strong eggshells<br>\u2022 Grit (insoluble granite) aids digestion, especially for whole grains<br>\u2022 Feed should be stored in cool, dry, rodent-proof conditions<br>\u2022 Sudden feed changes can reduce performance by 10\u201320%<br>\u2022 Water consumption is roughly 2\u00d7 feed intake</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-green"><strong>Cattle Feeding Guidelines:</strong><br>\u2022 TDN = Total Digestible Nutrients (energy measure for ruminants)<br>\u2022 Ruminants require 15\u201320% fiber for proper rumen function<br>\u2022 Transition periods are critical \u2014 allow 21 days minimum<br>\u2022 Fresh, clean water must always be available (50\u201380 L/day)<br>\u2022 Monitor body condition score (BCS 1\u20139, target: 5\u20136)</div>', unsafe_allow_html=True)

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
        st.download_button(label=f"\U0001f4e5 Download {animal_type} Nutrient Guide (CSV)", data=csv, file_name=f"{animal_type.lower()}_nutrient_guide.csv", mime="text/csv", use_container_width=True)
    with col2:
        if st.button("\U0001f52c Proceed to Feed Formulator", type="primary", use_container_width=True):
            st.session_state.page = "formulator"
            st.rerun()


def show_formulator():
    render_navbar()
    st.markdown('<div class="page-header"><div class="page-title">\U0001f52c Feed Formulation Centre</div><div class="page-desc">Configure your animal parameters in the sidebar, then use the tabs below to optimise, analyse, and export your custom feed formula.</div></div>', unsafe_allow_html=True)

    animal = st.selectbox("\U0001f43e Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    if animal == "Rabbit":
        df = rabbit_df.copy()
        st.markdown('<div class="alert-green">\U0001f430 <strong>Rabbit Nutrition</strong> \u2014 Formulating for herbivores with high fibre needs</div>', unsafe_allow_html=True)
    elif animal == "Poultry":
        df = poultry_df.copy()
        st.markdown('<div class="alert-green">\U0001f414 <strong>Poultry Nutrition</strong> \u2014 Optimising for broilers and layers</div>', unsafe_allow_html=True)
    else:
        df = cattle_df.copy()
        st.markdown('<div class="alert-green">\U0001f404 <strong>Cattle Nutrition</strong> \u2014 Formulating for ruminants</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div style="background:linear-gradient(135deg,#0d2818,#1d6b42);color:white;padding:1.25rem 1rem;border-radius:10px;margin-bottom:1rem;text-align:center;"><div style="font-size:1.4rem;margin-bottom:0.3rem;">\u2699\ufe0f</div><div style="font-weight:600;font-size:0.95rem;">Animal Parameters</div><div style="font-size:0.75rem;opacity:0.7;margin-top:0.2rem;">Configure inputs below</div></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### \U0001f3a8 Appearance")
    current_mode_label = "\U0001f319 Dark Mode" if not st.session_state.dark_mode else "\u2600\ufe0f Light Mode"
    if st.sidebar.button(current_mode_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    mode_status = "Dark mode ON" if st.session_state.dark_mode else "Light mode ON"
    st.sidebar.caption(f"Currently: {mode_status}")
    st.sidebar.markdown("---")
    if "selected_breed" in st.session_state:
        st.sidebar.success(f"\u2713 Breed: {st.session_state.selected_breed}")
    age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
    weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
    cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
    energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
    feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f4ca Summary")
    st.sidebar.metric("Animal", animal)
    st.sidebar.metric("Ingredients Available", len(df))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f517 Navigate")
    if st.sidebar.button("\U0001f4d6 Nutrient Guide", use_container_width=True):
        st.session_state.page = "nutrient_guide"
        st.rerun()
    if st.sidebar.button("\U0001f43e Breed Database", use_container_width=True):
        st.session_state.page = "breed_database"
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["\U0001f52c Feed Optimizer","\U0001f4cb Ingredient Database","\U0001f4c8 Growth Prediction","\U0001f4ca Cost Dashboard"])

    with tab1:
        st.header("\U0001f52c Least-Cost Feed Formulation")
        st.markdown("Using **linear programming** to find the cheapest ingredient blend meeting all nutritional requirements.")
        st.markdown('<div class="nutrient-panel">', unsafe_allow_html=True)
        st.markdown('<div class="nutrient-panel-title">\U0001f43e Step 1 \u2014 Select Breed & Production Stage</div>', unsafe_allow_html=True)
        breed_db = get_breed_database()
        nutrient_db = get_nutrient_requirements()
        breed_col, stage_col = st.columns(2)
        with breed_col:
            breed_options = ["\u2014 Select a breed (optional) \u2014"] + list(breed_db[animal].keys())
            default_breed_idx = 0
            if "selected_breed" in st.session_state and st.session_state.selected_breed in breed_db[animal]:
                default_breed_idx = breed_options.index(st.session_state.selected_breed)
            selected_breed = st.selectbox("\U0001f43e Breed", breed_options, index=default_breed_idx, key="opt_breed")
        with stage_col:
            stage_options = list(nutrient_db[animal].keys())
            selected_stage = st.selectbox("\U0001f3af Production Stage", stage_options, key="opt_stage")
        if selected_breed and selected_breed != "\u2014 Select a breed (optional) \u2014":
            binfo = breed_db[animal][selected_breed]
            st.markdown(f'<div class="breed-badge">\u2713 {selected_breed} &nbsp;\u00b7&nbsp; {binfo["Type"]} &nbsp;\u00b7&nbsp; Recommended CP: {binfo["Recommended CP (%)"]}</div>', unsafe_allow_html=True)
        stage_data = nutrient_db[animal][selected_stage]
        sd_cols = st.columns(len(stage_data))
        for idx, (k, v) in enumerate(stage_data.items()):
            short_k = k.replace(" (%)","").replace(" (kcal/kg)","").replace(" (g/day)","").replace(" (kg/day)","")
            with sd_cols[idx]:
                st.markdown(f'<div class="nutrient-chip"><div class="nutrient-chip-label">{short_k}</div><div class="nutrient-chip-value">{v}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="nutrient-panel" style="margin-top:0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="nutrient-panel-title">\U0001f9ea Step 2 \u2014 Set Nutrient Targets</div>', unsafe_allow_html=True)
        st.caption("Auto-filled from the selected stage. Adjust freely before running the optimiser.")

        def parse_mid(val_str):
            try:
                parts = str(val_str).split("-")
                return round((float(parts[0]) + float(parts[-1])) / 2, 1)
            except Exception:
                return 0.0

        cp_default = parse_mid(stage_data.get("Crude Protein (%)", "16-18"))
        energy_default = parse_mid(stage_data.get("Energy (kcal/kg)", "2500-2700"))
        fiber_default = parse_mid(stage_data.get("Crude Fiber (%)", "12-16"))
        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            cp_req = st.number_input("Crude Protein (%)", min_value=8.0, max_value=35.0, value=float(cp_default), step=0.5, key="ni_cp")
        with n_col2:
            energy_req = st.number_input("Energy (kcal/kg)", min_value=1500.0, max_value=4500.0, value=float(min(energy_default, 4500)), step=50.0, key="ni_energy")
        with n_col3:
            feed_intake = st.number_input("Daily Feed Intake (kg)", min_value=0.01, max_value=30.0, value=float(st.session_state.get("feed_intake_val", 0.5)), step=0.05, key="ni_intake")
        n_col4, n_col5, n_col6 = st.columns(3)
        with n_col4:
            use_fiber = st.checkbox("\U0001f4cf Set Fiber Targets", key="ni_use_fiber")
            if use_fiber:
                min_fiber = st.number_input("Min Fiber (%)", 0.0, 30.0, max(0.0, fiber_default - 2), 0.5, key="ni_fmin")
                max_fiber = st.number_input("Max Fiber (%)", 0.0, 40.0, fiber_default + 4, 0.5, key="ni_fmax")
            else:
                min_fiber, max_fiber = 0.0, 40.0
        with n_col5:
            limit_ingredients = st.checkbox("\U0001f522 Limit Ingredient Count", key="ni_limit")
            if limit_ingredients:
                max_ingredients = st.slider("Max ingredients", 3, 15, 8, key="ni_max_ingr")
            else:
                max_ingredients = 15
        with n_col6:
            st.markdown("&nbsp;")
        cp_req_final = cp_req
        energy_req_final = energy_req
        feed_intake_final = feed_intake
        st.session_state["feed_intake_val"] = feed_intake
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 2])
        with run_col:
            run_btn = st.button("\U0001f680 Optimise Feed Formula", type="primary", use_container_width=True, key="run_opt")

        if run_btn:
            with st.spinner("Calculating optimal feed mix\u2026"):
                try:
                    prob = LpProblem("FeedMix", LpMinimize)
                    ingredients = df["Ingredient"].tolist()
                    vars = LpVariable.dicts("Ingr", ingredients, lowBound=0, upBound=1)
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
                    prob += lpSum(vars[i] for i in ingredients) == 1
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req_final
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req_final
                    if use_fiber and "Fiber" in df.columns:
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) >= min_fiber
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) <= max_fiber
                    prob.solve()
                    if LpStatus[prob.status] == "Optimal":
                        result = {i: vars[i].value() for i in ingredients if vars[i].value() > 0.001}
                        if limit_ingredients and len(result) > max_ingredients:
                            st.warning(f"\u26a0\ufe0f Solution uses {len(result)} ingredients (limit: {max_ingredients}).")
                        result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
                        result_df["Proportion (%)"] = (result_df["Proportion"] * 100).round(2)
                        result_df["Cost/kg (\u20a6)"] = result_df["Ingredient"].apply(lambda x: df[df["Ingredient"] == x]["Cost"].values[0])
                        result_df["Cost Contribution (\u20a6)"] = (result_df["Proportion"] * result_df["Cost/kg (\u20a6)"]).round(2)
                        result_df["CP Contribution"] = result_df["Ingredient"].apply(lambda x: df[df["Ingredient"] == x]["CP"].values[0]) * result_df["Proportion"]
                        result_df["Energy Contribution"] = result_df["Ingredient"].apply(lambda x: df[df["Ingredient"] == x]["Energy"].values[0]) * result_df["Proportion"]
                        total_cp = result_df["CP Contribution"].sum()
                        total_energy = result_df["Energy Contribution"].sum()
                        result_df = result_df.sort_values("Proportion", ascending=False)
                        total_cost = value(prob.objective)
                        st.session_state["optimization_result"] = result_df
                        st.session_state["total_cost"] = total_cost
                        st.session_state["total_cp"] = total_cp
                        st.session_state["total_energy"] = total_energy
                        st.session_state.formulation_history.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "animal": animal, "age": age, "weight": weight, "cp_req": cp_req_final, "energy_req": energy_req_final, "total_cost": total_cost})
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("\U0001f4b0 Feed Cost/kg", f"\u20a6{total_cost:.2f}")
                        with col2: st.metric("\U0001f4c5 Daily Feed Cost", f"\u20a6{total_cost * feed_intake_final:.2f}")
                        with col3: st.metric("\U0001f4e6 Ingredients Used", len(result))
                        with col4: st.metric("\U0001f4c6 Monthly Cost", f"\u20a6{total_cost * feed_intake_final * 30:.2f}")
                        st.markdown("---")
                        st.subheader("\u2705 Nutritional Achievement")
                        col1, col2 = st.columns(2)
                        with col1:
                            cp_pct = (total_cp / cp_req_final) * 100 if cp_req_final > 0 else 0
                            st.metric("Crude Protein", f"{total_cp:.2f}%", delta=f"{cp_pct:.1f}% of requirement")
                        with col2:
                            energy_pct = (total_energy / energy_req_final) * 100 if energy_req_final > 0 else 0
                            st.metric("Energy", f"{total_energy:.0f} kcal/kg", delta=f"{energy_pct:.1f}% of requirement")
                        st.success(f"\u2705 Optimisation complete! Total cost: \u20a6{total_cost:.2f}/kg")
                        st.dataframe(result_df[["Ingredient","Proportion (%)","Cost/kg (\u20a6)","Cost Contribution (\u20a6)"]], use_container_width=True, hide_index=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_pie = px.pie(result_df, values="Proportion (%)", names="Ingredient", title="Feed Composition", color_discrete_sequence=px.colors.sequential.Greens)
                            fig_pie.update_layout(template="plotly_white")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            fig_bar = px.bar(result_df, x="Ingredient", y="Cost Contribution (\u20a6)", title="Cost Breakdown by Ingredient", color="Cost Contribution (\u20a6)", color_continuous_scale="Greens")
                            fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_white")
                            st.plotly_chart(fig_bar, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = result_df.to_csv(index=False)
                            st.download_button(label="\U0001f4e5 Download Formula (CSV)", data=csv, file_name=f"{animal}_feed_formula_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
                        with col2:
                            report = generate_report(animal, age, weight, cp_req_final, energy_req_final, feed_intake_final, result_df, total_cost)
                            st.download_button(label="\U0001f4c4 Download Report (TXT)", data=report, file_name=f"{animal}_feed_report_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", use_container_width=True)
                    else:
                        st.error("\u274c No feasible solution found. Try relaxing your nutrient targets or constraints.")
                except Exception as e:
                    st.error(f"\u274c Error during optimisation: {str(e)}")

    with tab2:
        st.header("\U0001f4cb Ingredient Database Manager")
        st.markdown(f"**{len(df)} ingredients** available for {animal} feed formulation.")
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("\U0001f50d Search ingredients", placeholder="Type to filter\u2026")
        with col2:
            sort_by = st.selectbox("Sort by", ["Ingredient","CP","Energy","Cost"])
        filtered_df = df[df["Ingredient"].str.contains(search, case=False, na=False)] if search else df
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_by == "Ingredient"))
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Ingredients", len(filtered_df))
        with col2: st.metric("Avg Cost/kg", f"\u20a6{filtered_df['Cost'].mean():.2f}")
        with col3: st.metric("Avg Protein", f"{filtered_df['CP'].mean():.1f}%")
        with col4: st.metric("Avg Energy", f"{filtered_df['Energy'].mean():.0f} kcal")
        st.markdown("---")
        edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True,
            column_config={
                "Ingredient": st.column_config.TextColumn("Ingredient", width="medium"),
                "CP": st.column_config.NumberColumn("Crude Protein (%)", format="%.1f"),
                "Energy": st.column_config.NumberColumn("Energy (kcal/kg)", format="%.0f"),
                "Fiber": st.column_config.NumberColumn("Crude Fiber (%)", format="%.1f"),
                "Cost": st.column_config.NumberColumn("Cost (\u20a6/kg)", format="\u20a6%.2f"),
            })
        col1, col2 = st.columns(2)
        with col1:
            if st.button("\U0001f4be Save Changes to Database", use_container_width=True):
                edited_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
                st.success("\u2705 Ingredient database updated successfully!")
                st.cache_data.clear()
        with col2:
            csv = edited_df.to_csv(index=False)
            st.download_button("\U0001f4e5 Download Database (CSV)", csv, f"{animal.lower()}_ingredients.csv", "text/csv", use_container_width=True)

    with tab3:
        st.header("\U0001f4c8 AI Weight Gain Prediction")
        st.markdown("**Random Forest ML model** trained on 110+ feeding trials from Nigerian farms.")
        st.markdown("---")
        if st.button("\U0001f3af Calculate Growth Prediction", type="primary"):
            with st.spinner("Calculating growth predictions\u2026"):
                avg_cp = df["CP"].mean()
                avg_energy = df["Energy"].mean()
                X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
                prediction = model.predict(X_input)[0]
                st.session_state["prediction"] = prediction
        if "prediction" in st.session_state:
            prediction = st.session_state["prediction"]
            weekly_gain = prediction * 7
            monthly_gain = prediction * 30
            projected_weight_90d = weight + (monthly_gain * 3 / 1000)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Weight Gain", f"{prediction:.1f} g/day")
            with col2: st.metric("Weekly Gain", f"{weekly_gain:.0f} g")
            with col3: st.metric("Monthly Gain", f"{monthly_gain / 1000:.2f} kg")
            with col4: st.metric("90-Day Weight", f"{projected_weight_90d:.1f} kg", delta=f"+{projected_weight_90d - weight:.1f} kg")
            st.subheader("\U0001f4ca 90-Day Weight Projection")
            days = np.arange(0, 91)
            projected_weights = weight + (prediction * days / 1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=projected_weights, mode="lines", name="Projected Weight", line=dict(color="#228b55", width=3), fill="tozeroy", fillcolor="rgba(34,139,85,0.08)"))
            fig.add_trace(go.Scatter(x=[0], y=[weight], mode="markers", name="Current Weight", marker=dict(size=12, color="#e74c3c")))
            fig.update_layout(xaxis_title="Days", yaxis_title="Weight (kg)", hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("\U0001f4ca Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
                st.metric("Feed Conversion Ratio (FCR)", f"{fcr:.2f}:1")
                st.caption("Feed required to gain 1 kg of body weight")
            with col2:
                if "total_cost" in st.session_state and prediction > 0:
                    cost_per_kg_gain = (st.session_state["total_cost"] * feed_intake * 1000) / prediction
                    st.metric("Cost per kg Gain", f"\u20a6{cost_per_kg_gain:.2f}")
                    st.caption("Feed cost to produce 1 kg of weight gain")
                else:
                    st.info("\U0001f4a1 Run the Feed Optimizer first to see cost metrics")
            st.markdown("---")
            st.subheader("\U0001f3af Growth Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if animal == "Rabbit":
                    performance = "\U0001f7e2 Excellent" if prediction > 30 else ("\U0001f7e1 Good" if prediction > 20 else "\U0001f534 Below Average")
                elif animal == "Poultry":
                    performance = "\U0001f7e2 Excellent" if prediction > 50 else ("\U0001f7e1 Good" if prediction > 35 else "\U0001f534 Below Average")
                else:
                    performance = "\U0001f7e2 Excellent" if prediction > 800 else ("\U0001f7e1 Good" if prediction > 500 else "\U0001f534 Below Average")
                st.metric("Performance Rating", performance)
            with col2:
                target_weight = 2.5 if animal == "Rabbit" else (2.0 if animal == "Poultry" else 300)
                if prediction > 0 and weight < target_weight:
                    days_to_target = int((target_weight - weight) * 1000 / prediction)
                    st.metric("Days to Market Weight", f"{days_to_target} days")
                    st.caption(f"Target: {target_weight} kg")
                else:
                    st.metric("Market Weight", "\u2705 Achieved")
        else:
            st.info("\U0001f446 Click 'Calculate Growth Prediction' above to see results")

    with tab4:
        st.header("\U0001f4ca Cost Analysis Dashboard")
        if "optimization_result" not in st.session_state:
            st.markdown('<div class="alert-amber">\u26a0\ufe0f Please run the Feed Optimizer first to unlock the Cost Dashboard</div>', unsafe_allow_html=True)
        else:
            result_df = st.session_state["optimization_result"]
            total_cost = st.session_state["total_cost"]
            st.subheader("\U0001f4b0 Cost Projections")
            daily_cost = total_cost * feed_intake
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Cost", f"\u20a6{daily_cost:.2f}")
            with col2: st.metric("Weekly Cost", f"\u20a6{daily_cost * 7:.2f}")
            with col3: st.metric("Monthly Cost", f"\u20a6{daily_cost * 30:.2f}")
            with col4: st.metric("Yearly Cost", f"\u20a6{daily_cost * 365:,.2f}")
            st.markdown("---")
            st.subheader("\U0001f43e Herd / Flock Cost Calculator")
            col1, col2 = st.columns(2)
            with col1: num_animals = st.number_input("Number of Animals", min_value=1, max_value=10000, value=100)
            with col2: duration_days = st.slider("Duration (days)", 1, 365, 90)
            total_herd_cost = daily_cost * num_animals * duration_days
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Feed Cost", f"\u20a6{total_herd_cost:,.2f}")
            with col2: st.metric("Cost per Animal", f"\u20a6{total_herd_cost / num_animals:,.2f}")
            with col3: st.metric("Daily Herd Cost", f"\u20a6{daily_cost * num_animals:,.2f}")
            st.markdown("---")
            st.subheader("\U0001f4ca Cost Breakdown Analysis")
            fig = px.treemap(result_df, path=["Ingredient"], values="Cost Contribution (\u20a6)", title="Cost Contribution by Ingredient", color="Cost Contribution (\u20a6)", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                top_5 = result_df.nlargest(5, "Cost Contribution (\u20a6)")
                fig = px.bar(top_5, x="Ingredient", y="Cost Contribution (\u20a6)", title="Top 5 Cost Contributors", color="Cost Contribution (\u20a6)", color_continuous_scale="Reds")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(result_df, x="Proportion (%)", y="Cost/kg (\u20a6)", size="Cost Contribution (\u20a6)", hover_name="Ingredient", title="Proportion vs Unit Cost", color="Cost Contribution (\u20a6)", color_continuous_scale="Viridis")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            if "prediction" in st.session_state:
                st.markdown("---")
                st.subheader("\U0001f4b5 Return on Investment Calculator")
                prediction = st.session_state["prediction"]
                col1, col2 = st.columns(2)
                with col1:
                    default_price = 1500 if animal == "Rabbit" else (1200 if animal == "Poultry" else 2000)
                    price_per_kg = st.number_input("Selling Price (\u20a6/kg live weight)", min_value=500, max_value=5000, value=default_price)
                with col2:
                    production_days = st.number_input("Production Cycle (days)", min_value=30, max_value=365, value=90)
                total_feed_cost = daily_cost * production_days
                weight_gain_kg = (prediction * production_days) / 1000
                final_weight = weight + weight_gain_kg
                revenue = final_weight * price_per_kg
                profit = revenue - total_feed_cost
                roi_percent = (profit / total_feed_cost * 100) if total_feed_cost > 0 else 0
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Feed Cost", f"\u20a6{total_feed_cost:,.2f}")
                with col2: st.metric("Final Weight", f"{final_weight:.2f} kg")
                with col3: st.metric("Revenue", f"\u20a6{revenue:,.2f}")
                with col4: st.metric("Profit", f"\u20a6{profit:,.2f}", delta=f"{roi_percent:.1f}% ROI")
                roi_data = pd.DataFrame({"Category":["Feed Cost","Profit"], "Amount":[total_feed_cost, profit if profit > 0 else 0]})
                fig = px.pie(roi_data, values="Amount", names="Category", title=f"Cost vs Profit (ROI: {roi_percent:.1f}%)", color_discrete_sequence=["#e74c3c","#228b55"])
                st.plotly_chart(fig, use_container_width=True)
                if profit > 0:
                    st.success(f"\u2705 Profitable! Expected profit of \u20a6{profit:,.2f} per animal over {production_days} days.")
                else:
                    st.error("\u26a0\ufe0f Loss expected. Adjust feeding programme or selling price.")


if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "nutrient_guide":
    show_nutrient_guide()
elif st.session_state.page == "breed_database":
    show_breed_database()
elif st.session_state.page == "formulator":
    show_formulator()

st.markdown("""
<div class="footer-wrap">
    <div>
        <div class="footer-brand">\U0001f331 Necstech Feed Optimizer</div>
        <div class="footer-meta" style="margin-top:0.3rem;">Optimising African Agriculture · v2.0 · \u00a9 2026 Necstech</div>
    </div>
    <div class="footer-badges">
        <div class="badge">\U0001f1f3\U0001f1ec Nigerian Data</div>
        <div class="badge">\U0001f916 ML-Powered</div>
        <div class="badge">\U0001f4ca NIAS · FAO 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

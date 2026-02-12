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
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
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
        --slate-900: #1a2332;
        --slate-700: #334155;
        --slate-500: #64748b;
        --slate-300: #cbd5e1;
        --slate-100: #f1f5f9;
        --white:     #ffffff;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.06);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.12), 0 4px 12px rgba(0,0,0,0.08);
        --radius:    12px;
        --radius-lg: 20px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--slate-900);
    }

    .stApp { background: var(--slate-100); }

    .main .block-container {
        padding: 0 2rem 3rem 2rem;
        max-width: 1280px;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .card {
        background: var(--white);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(0,0,0,0.04);
        transition: box-shadow 0.25s, transform 0.25s;
    }
    .card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .feature-card {
        background: var(--white);
        border-radius: var(--radius);
        padding: 1.75rem 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.25s, transform 0.2s;
        height: 100%;
    }
    .feature-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-3px);
    }
    .feat-icon {
        width: 44px; height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    .feat-icon.green  { background: var(--green-100); }
    .feat-icon.amber  { background: var(--amber-light); }
    .feat-icon.blue   { background: #e0f0ff; }
    .feat-icon.purple { background: #ede9ff; }
    .feat-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--slate-900);
        margin-bottom: 0.4rem;
    }
    .feat-body {
        font-size: 0.85rem;
        color: var(--slate-500);
        line-height: 1.55;
    }

    .hero-wrap {
        background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, var(--green-500) 100%);
        border-radius: var(--radius-lg);
        padding: 4rem 3.5rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -60px; right: -80px;
        width: 340px; height: 340px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-wrap::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: var(--green-300);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        margin-bottom: 1.2rem;
    }
    .hero-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        color: var(--white);
        line-height: 1.15;
        margin-bottom: 1.2rem;
        max-width: 620px;
    }
    .hero-heading em {
        font-style: italic;
        color: var(--green-300);
    }
    .hero-body {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.75);
        line-height: 1.7;
        max-width: 560px;
        margin-bottom: 2rem;
    }
    .hero-stats {
        display: flex;
        gap: 2.5rem;
        margin-top: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.12);
    }
    .hero-stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: var(--white);
        line-height: 1;
    }
    .hero-stat-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--green-700), var(--green-500)) !important;
        border: none !important;
        color: var(--white) !important;
        padding: 0.55rem 1.5rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(34, 139, 85, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--white) !important;
        border: 1.5px solid var(--slate-300) !important;
        color: var(--slate-700) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--green-500) !important;
        color: var(--green-700) !important;
    }

    .section-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: var(--slate-900);
        margin-bottom: 0.3rem;
    }
    .section-sub {
        font-size: 0.9rem;
        color: var(--slate-500);
        margin-bottom: 1.5rem;
    }

    .steps-row {
        display: flex;
        gap: 0;
        position: relative;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 1.5rem 1rem;
        position: relative;
    }
    .step-num {
        width: 36px; height: 36px;
        border-radius: 50%;
        background: var(--green-700);
        color: var(--white);
        font-weight: 700;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.75rem;
        position: relative;
        z-index: 2;
    }
    .step-title { font-weight: 600; font-size: 0.9rem; color: var(--slate-900); }
    .step-body  { font-size: 0.8rem; color: var(--slate-500); margin-top: 0.3rem; line-height: 1.5; }
    .step-connector {
        position: absolute;
        top: 2.3rem;
        left: calc(50% + 18px);
        right: calc(-50% + 18px);
        height: 2px;
        background: var(--green-300);
        z-index: 1;
    }

    .page-header { margin-bottom: 1.75rem; }
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: var(--slate-900);
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }
    .page-desc {
        font-size: 0.95rem;
        color: var(--slate-500);
        max-width: 600px;
    }

    .stage-header {
        background: linear-gradient(90deg, var(--green-700), var(--green-500));
        color: var(--white);
        padding: 0.7rem 1.1rem;
        border-radius: 8px;
        margin: 1.2rem 0 0.8rem;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }

    .breed-card-wrap {
        background: var(--white);
        border-radius: var(--radius);
        border: 1px solid rgba(0,0,0,0.06);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--green-500);
        transition: box-shadow 0.2s;
    }
    .breed-card-wrap:hover { box-shadow: var(--shadow-md); }

    .alert-amber {
        background: var(--amber-light);
        border-left: 4px solid var(--amber);
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: var(--slate-700);
    }
    .alert-green {
        background: var(--green-50);
        border-left: 4px solid var(--green-500);
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: var(--slate-700);
    }

    hr { border-color: var(--slate-300) !important; margin: 1.5rem 0 !important; }

    [data-testid="stSidebar"] {
        background: var(--white) !important;
        border-right: 1px solid var(--slate-300) !important;
    }

    [data-testid="metric-container"] {
        background: var(--white);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: var(--radius);
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: var(--slate-500) !important;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.6rem !important;
        color: var(--slate-900) !important;
    }

    .stTabs [role="tablist"] {
        background: var(--white);
        border-radius: 10px;
        padding: 0.3rem;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: var(--shadow-sm);
        gap: 0.2rem;
    }
    .stTabs [role="tab"] {
        border-radius: 7px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
        color: var(--slate-600) !important;
        transition: all 0.2s !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--green-700) !important;
        color: var(--white) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--white);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
        margin-top: 0.5rem;
        box-shadow: var(--shadow-sm);
    }

    .footer-wrap {
        background: var(--slate-900);
        color: rgba(255,255,255,0.6);
        border-radius: var(--radius-lg);
        padding: 2.5rem 3rem;
        margin-top: 3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .footer-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: var(--white);
    }
    .footer-meta { font-size: 0.78rem; }
    .footer-badges { display: flex; gap: 0.5rem; }
    .badge {
        padding: 0.25rem 0.75rem;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.65);
    }
</style>
""", unsafe_allow_html=True)


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
    X = data[[
        "Age_Weeks", "Body_Weight_kg",
        "CP_Requirement_%", "Energy_Requirement_Kcal",
        "Feed_Intake_kg", "Ingredient_CP_%", "Ingredient_Energy"
    ]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)


def get_breed_database():
    rabbit_breeds = {
        "New Zealand White": {"Type": "Meat", "Mature Weight (kg)": "4.5-5.5", "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent", "Best For": "Commercial meat production",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "10-12"},
        "Californian": {"Type": "Meat", "Mature Weight (kg)": "4.0-5.0", "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent", "Best For": "Meat and show",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "10-12"},
        "Flemish Giant": {"Type": "Meat", "Mature Weight (kg)": "6.0-10.0", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Large-scale meat production",
            "Recommended CP (%)": "17-19", "Market Age (weeks)": "14-16"},
        "Dutch": {"Type": "Pet/Show", "Mature Weight (kg)": "2.0-2.5", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Pets and breeding",
            "Recommended CP (%)": "15-17", "Market Age (weeks)": "8-10"},
        "Rex": {"Type": "Meat/Fur", "Mature Weight (kg)": "3.5-4.5", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Fur and meat",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "10-12"}
    }
    poultry_breeds = {
        "Broiler (Cobb 500)": {"Type": "Meat", "Mature Weight (kg)": "2.5-3.0", "Growth Rate": "Very Fast",
            "Feed Efficiency": "Excellent (FCR 1.6-1.8)", "Best For": "Commercial meat production",
            "Recommended CP (%)": "20-22", "Market Age (weeks)": "5-6"},
        "Broiler (Ross 308)": {"Type": "Meat", "Mature Weight (kg)": "2.3-2.8", "Growth Rate": "Very Fast",
            "Feed Efficiency": "Excellent (FCR 1.65-1.85)", "Best For": "Commercial meat production",
            "Recommended CP (%)": "20-22", "Market Age (weeks)": "5-6"},
        "Layer (Isa Brown)": {"Type": "Eggs", "Mature Weight (kg)": "1.8-2.0", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "High egg production (300+ eggs/year)",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "18-20 (point of lay)"},
        "Layer (Lohmann Brown)": {"Type": "Eggs", "Mature Weight (kg)": "1.9-2.1", "Growth Rate": "Moderate",
            "Feed Efficiency": "Excellent", "Best For": "Egg production (320+ eggs/year)",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "18-20 (point of lay)"},
        "Noiler": {"Type": "Dual Purpose", "Mature Weight (kg)": "2.0-2.5", "Growth Rate": "Fast",
            "Feed Efficiency": "Good", "Best For": "Meat and eggs (Nigerian adapted)",
            "Recommended CP (%)": "18-20", "Market Age (weeks)": "12-16"},
        "Kuroiler": {"Type": "Dual Purpose", "Mature Weight (kg)": "2.5-3.5", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Free-range, dual purpose",
            "Recommended CP (%)": "16-18", "Market Age (weeks)": "14-18"},
        "Local Nigerian": {"Type": "Dual Purpose", "Mature Weight (kg)": "1.2-1.8", "Growth Rate": "Slow",
            "Feed Efficiency": "Moderate", "Best For": "Free-range, disease resistant",
            "Recommended CP (%)": "14-16", "Market Age (weeks)": "20-24"}
    }
    cattle_breeds = {
        "White Fulani": {"Type": "Beef/Dairy", "Mature Weight (kg)": "300-450", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Milk and beef (Nigerian indigenous)",
            "Recommended CP (%)": "14-16", "Market Age (months)": "24-30"},
        "Red Bororo": {"Type": "Beef", "Mature Weight (kg)": "250-350", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Beef production (heat tolerant)",
            "Recommended CP (%)": "13-15", "Market Age (months)": "24-28"},
        "Sokoto Gudali": {"Type": "Beef", "Mature Weight (kg)": "350-500", "Growth Rate": "Moderate-Fast",
            "Feed Efficiency": "Good", "Best For": "Beef (large frame)",
            "Recommended CP (%)": "14-16", "Market Age (months)": "24-30"},
        "N'Dama": {"Type": "Beef/Draft", "Mature Weight (kg)": "300-400", "Growth Rate": "Moderate",
            "Feed Efficiency": "Good", "Best For": "Trypanosomiasis resistant",
            "Recommended CP (%)": "12-14", "Market Age (months)": "30-36"},
        "Muturu": {"Type": "Beef/Draft", "Mature Weight (kg)": "200-300", "Growth Rate": "Slow",
            "Feed Efficiency": "Moderate", "Best For": "Small-holder, disease resistant",
            "Recommended CP (%)": "12-14", "Market Age (months)": "30-36"},
        "Holstein Friesian (Cross)": {"Type": "Dairy", "Mature Weight (kg)": "450-650", "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent", "Best For": "High milk production",
            "Recommended CP (%)": "16-18", "Market Age (months)": "24-28"},
        "Brahman Cross": {"Type": "Beef", "Mature Weight (kg)": "400-550", "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent", "Best For": "Beef (heat adapted)",
            "Recommended CP (%)": "14-16", "Market Age (months)": "20-24"}
    }
    return {"Rabbit": rabbit_breeds, "Poultry": poultry_breeds, "Cattle": cattle_breeds}


def get_nutrient_requirements():
    rabbit_nutrients = {
        "Grower (4-12 weeks)": {"Crude Protein (%)": "16-18", "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "12-16", "Calcium (%)": "0.4-0.8", "Phosphorus (%)": "0.3-0.5",
            "Lysine (%)": "0.65-0.75", "Feed Intake (g/day)": "80-120"},
        "Finisher (12-16 weeks)": {"Crude Protein (%)": "14-16", "Energy (kcal/kg)": "2400-2600",
            "Crude Fiber (%)": "14-18", "Calcium (%)": "0.4-0.7", "Phosphorus (%)": "0.3-0.5",
            "Lysine (%)": "0.55-0.65", "Feed Intake (g/day)": "120-180"},
        "Doe (Maintenance)": {"Crude Protein (%)": "15-16", "Energy (kcal/kg)": "2500-2600",
            "Crude Fiber (%)": "14-16", "Calcium (%)": "0.5-0.8", "Phosphorus (%)": "0.4-0.5",
            "Lysine (%)": "0.60-0.70", "Feed Intake (g/day)": "100-150"},
        "Doe (Pregnant)": {"Crude Protein (%)": "16-18", "Energy (kcal/kg)": "2600-2800",
            "Crude Fiber (%)": "12-15", "Calcium (%)": "0.8-1.2", "Phosphorus (%)": "0.5-0.7",
            "Lysine (%)": "0.70-0.80", "Feed Intake (g/day)": "150-200"},
        "Doe (Lactating)": {"Crude Protein (%)": "17-19", "Energy (kcal/kg)": "2700-3000",
            "Crude Fiber (%)": "12-14", "Calcium (%)": "1.0-1.5", "Phosphorus (%)": "0.6-0.8",
            "Lysine (%)": "0.75-0.90", "Feed Intake (g/day)": "200-400"},
        "Buck (Breeding)": {"Crude Protein (%)": "15-17", "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "14-16", "Calcium (%)": "0.5-0.8", "Phosphorus (%)": "0.4-0.6",
            "Lysine (%)": "0.65-0.75", "Feed Intake (g/day)": "120-170"}
    }
    poultry_nutrients = {
        "Broiler Starter (0-3 weeks)": {"Crude Protein (%)": "22-24", "Energy (kcal/kg)": "3000-3200",
            "Crude Fiber (%)": "3-4", "Calcium (%)": "0.9-1.0", "Phosphorus (%)": "0.45-0.50",
            "Lysine (%)": "1.20-1.35", "Methionine (%)": "0.50-0.55", "Feed Intake (g/day)": "25-35"},
        "Broiler Grower (3-6 weeks)": {"Crude Protein (%)": "20-22", "Energy (kcal/kg)": "3100-3300",
            "Crude Fiber (%)": "3-5", "Calcium (%)": "0.85-0.95", "Phosphorus (%)": "0.40-0.45",
            "Lysine (%)": "1.05-1.20", "Methionine (%)": "0.45-0.50", "Feed Intake (g/day)": "80-120"},
        "Broiler Finisher (6+ weeks)": {"Crude Protein (%)": "18-20", "Energy (kcal/kg)": "3200-3400",
            "Crude Fiber (%)": "3-5", "Calcium (%)": "0.80-0.90", "Phosphorus (%)": "0.35-0.40",
            "Lysine (%)": "0.95-1.10", "Methionine (%)": "0.40-0.45", "Feed Intake (g/day)": "140-180"},
        "Layer Starter (0-6 weeks)": {"Crude Protein (%)": "18-20", "Energy (kcal/kg)": "2800-3000",
            "Crude Fiber (%)": "3-5", "Calcium (%)": "0.9-1.0", "Phosphorus (%)": "0.45-0.50",
            "Lysine (%)": "0.95-1.05", "Methionine (%)": "0.40-0.45", "Feed Intake (g/day)": "20-40"},
        "Layer Grower (6-18 weeks)": {"Crude Protein (%)": "16-18", "Energy (kcal/kg)": "2700-2900",
            "Crude Fiber (%)": "4-6", "Calcium (%)": "0.8-0.9", "Phosphorus (%)": "0.40-0.45",
            "Lysine (%)": "0.75-0.85", "Methionine (%)": "0.35-0.40", "Feed Intake (g/day)": "60-90"},
        "Layer Production (18+ weeks)": {"Crude Protein (%)": "16-18", "Energy (kcal/kg)": "2750-2900",
            "Crude Fiber (%)": "4-6", "Calcium (%)": "3.5-4.0", "Phosphorus (%)": "0.35-0.40",
            "Lysine (%)": "0.75-0.85", "Methionine (%)": "0.38-0.42", "Feed Intake (g/day)": "110-130"}
    }
    cattle_nutrients = {
        "Calf Starter (0-3 months)": {"Crude Protein (%)": "18-20", "Energy (kcal/kg)": "3000-3200",
            "Crude Fiber (%)": "8-12", "Calcium (%)": "0.7-1.0", "Phosphorus (%)": "0.4-0.6",
            "TDN (%)": "72-78", "Feed Intake (kg/day)": "0.5-1.5"},
        "Calf Grower (3-6 months)": {"Crude Protein (%)": "16-18", "Energy (kcal/kg)": "2800-3000",
            "Crude Fiber (%)": "10-15", "Calcium (%)": "0.6-0.9", "Phosphorus (%)": "0.35-0.50",
            "TDN (%)": "68-74", "Feed Intake (kg/day)": "2-4"},
        "Heifer (6-12 months)": {"Crude Protein (%)": "14-16", "Energy (kcal/kg)": "2600-2800",
            "Crude Fiber (%)": "12-18", "Calcium (%)": "0.5-0.8", "Phosphorus (%)": "0.30-0.45",
            "TDN (%)": "65-70", "Feed Intake (kg/day)": "4-7"},
        "Bull (Breeding)": {"Crude Protein (%)": "12-14", "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "15-20", "Calcium (%)": "0.4-0.7", "Phosphorus (%)": "0.25-0.40",
            "TDN (%)": "62-68", "Feed Intake (kg/day)": "8-12"},
        "Cow (Dry)": {"Crude Protein (%)": "10-12", "Energy (kcal/kg)": "2400-2600",
            "Crude Fiber (%)": "18-25", "Calcium (%)": "0.4-0.6", "Phosphorus (%)": "0.25-0.35",
            "TDN (%)": "58-65", "Feed Intake (kg/day)": "10-15"},
        "Cow (Lactating)": {"Crude Protein (%)": "14-18", "Energy (kcal/kg)": "2700-3000",
            "Crude Fiber (%)": "15-22", "Calcium (%)": "0.6-0.9", "Phosphorus (%)": "0.35-0.50",
            "TDN (%)": "68-75", "Feed Intake (kg/day)": "12-20"},
        "Beef Finisher": {"Crude Protein (%)": "12-14", "Energy (kcal/kg)": "2800-3100",
            "Crude Fiber (%)": "8-15", "Calcium (%)": "0.5-0.7", "Phosphorus (%)": "0.30-0.45",
            "TDN (%)": "70-78", "Feed Intake (kg/day)": "8-14"}
    }
    return {"Rabbit": rabbit_nutrients, "Poultry": poultry_nutrients, "Cattle": cattle_nutrients}


if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'formulation_history' not in st.session_state:
    st.session_state.formulation_history = []


def generate_report(animal, age, weight, cp_req, energy_req, feed_intake,
                    result_df=None, total_cost=None, prediction=None):
    report = f"""
    ═══════════════════════════════════════════════════════════════
                    NECSTECH FEED OPTIMIZER REPORT
                         Livestock Feed Formulation
    ═══════════════════════════════════════════════════════════════
    Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ANIMAL INFORMATION
    ─────────────────────────────────────────────────────────────
    Species: {animal}
    Age: {age} weeks
    Body Weight: {weight} kg
    Daily Feed Intake: {feed_intake} kg

    NUTRITIONAL REQUIREMENTS
    ─────────────────────────────────────────────────────────────
    Crude Protein: {cp_req}%
    Energy: {energy_req} kcal/kg
    """
    if result_df is not None and total_cost is not None:
        report += f"""
    OPTIMIZED FEED FORMULA
    ─────────────────────────────────────────────────────────────
    Total Cost per kg: ₦{total_cost:.2f}
    Daily Feed Cost: ₦{total_cost * feed_intake:.2f}
    Number of Ingredients: {len(result_df)}

    INGREDIENT COMPOSITION
    ─────────────────────────────────────────────────────────────
    """
        for _, row in result_df.iterrows():
            report += f"    {row['Ingredient']}: {row['Proportion (%)']:.2f}% (₦{row['Cost Contribution (₦)']:.2f})\n"
    if prediction is not None:
        weekly_gain = prediction * 7
        monthly_gain = prediction * 30
        projected_weight = weight + (monthly_gain * 3 / 1000)
        fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
        report += f"""
    GROWTH PREDICTIONS
    ─────────────────────────────────────────────────────────────
    Daily Weight Gain: {prediction:.1f} g/day
    Weekly Gain: {weekly_gain:.0f} g
    Monthly Gain: {monthly_gain/1000:.2f} kg
    90-Day Projected Weight: {projected_weight:.1f} kg
    Feed Conversion Ratio: {fcr:.2f}:1
    """
        if total_cost is not None:
            cost_per_kg_gain = (total_cost * feed_intake * 1000) / prediction
            report += f"    Cost per kg Gain: ₦{cost_per_kg_gain:.2f}\n"
    report += """
    ═══════════════════════════════════════════════════════════════
    Generated by Necstech Feed Optimizer v2.0
    Powered by Nigerian Agricultural Data · NIAS · FAO · 2026
    ═══════════════════════════════════════════════════════════════
    """
    return report


def render_navbar():
    current = st.session_state.page
    cols = st.columns([2, 1, 1, 1, 1])
    with cols[0]:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#1d6b42,#46c97f);
                border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
                🌱
            </div>
            <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1a2332;">
                Necs<span style="color:#228b55;">tech</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    nav_items = [
        ('home', '🏠 Home'),
        ('nutrient_guide', '📖 Nutrient Guide'),
        ('breed_database', '🐾 Breeds'),
        ('formulator', '🔬 Formulator')
    ]
    for idx, (key, label) in enumerate(nav_items):
        with cols[idx + 1]:
            btn_type = "primary" if current == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.page = key
                st.rerun()
    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)


def show_home():
    render_navbar()

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">🌍 Built for Nigerian Agriculture · Powered by AI</div>
        <div class="hero-heading">
            Smarter Feed,<br><em>Healthier Livestock,</em><br>Better Profits
        </div>
        <div class="hero-body">
            Necstech Feed Optimizer is an AI-powered precision nutrition platform designed for
            rabbit, poultry, and cattle farmers across Nigeria. Using advanced linear programming
            and machine learning trained on 110+ local feeding trials, it generates least-cost
            feed formulas that meet your animals' exact nutritional requirements — helping you cut
            feed waste, improve growth rates, and maximise return on every naira invested.
        </div>
        <div class="hero-stats">
            <div><div class="hero-stat-num">97+</div><div class="hero-stat-label">Local Ingredients</div></div>
            <div><div class="hero-stat-num">31+</div><div class="hero-stat-label">Breed Profiles</div></div>
            <div><div class="hero-stat-num">3</div><div class="hero-stat-label">Livestock Species</div></div>
            <div><div class="hero-stat-num">110+</div><div class="hero-stat-label">ML Training Trials</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Everything You Need to Optimise Livestock Nutrition</div>
    <div class="section-sub">Four integrated tools in one streamlined platform</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feat-icon green">💰</div>
            <div class="feat-title">Least-Cost Formulation</div>
            <div class="feat-body">Linear programming engine automatically blends the cheapest ingredient
            combination that satisfies all protein, energy, and mineral constraints.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feat-icon blue">🤖</div>
            <div class="feat-title">AI Growth Prediction</div>
            <div class="feat-body">Random Forest model forecasts daily weight gain, FCR, and 90-day
            projections based on Nigerian farm trial data.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feat-icon amber">🇳🇬</div>
            <div class="feat-title">Nigerian Market Data</div>
            <div class="feat-body">97 ingredients with verified 2026 local market prices. Edit, add,
            or remove ingredients to match your region's availability.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-card">
            <div class="feat-icon purple">📊</div>
            <div class="feat-title">Cost & ROI Dashboard</div>
            <div class="feat-body">Detailed cost breakdowns, herd-level projections, and profit/loss
            analysis per production cycle to guide investment decisions.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">How It Works</div>
    <div class="section-sub">From animal parameters to optimised formula in four steps</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:12px;padding:1.5rem 2rem;
                border:1px solid rgba(0,0,0,0.06);box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div class="steps-row">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">Select Species & Breed</div>
                <div class="step-body">Choose from Rabbit, Poultry, or Cattle — then pick from 31+ breed profiles for accurate baselines.</div>
                <div class="step-connector"></div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">Enter Animal Parameters</div>
                <div class="step-body">Provide age, weight, feed intake and production stage to define exact nutritional requirements.</div>
                <div class="step-connector"></div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">Run the Optimizer</div>
                <div class="step-body">Our LP engine solves for the minimum-cost blend across 97 Nigerian ingredients in seconds.</div>
                <div class="step-connector"></div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-title">Analyse & Export</div>
                <div class="step-body">Review cost breakdowns, AI growth predictions, and ROI — then download your formula as CSV or report.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Get Started</div>
    <div class="section-sub">Choose where you'd like to begin</div>
    """, unsafe_allow_html=True)

    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">📖</div>
            <div style="font-weight:600;font-size:1rem;margin-bottom:0.4rem;">Nutrient Guide</div>
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:1.25rem;">
                Browse complete nutritional standards for every livestock species and production stage.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Nutrient Guide →", key="home_ng", type="primary", use_container_width=True):
            st.session_state.page = 'nutrient_guide'
            st.rerun()
    with qa2:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🐾</div>
            <div style="font-weight:600;font-size:1rem;margin-bottom:0.4rem;">Breed Database</div>
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:1.25rem;">
                Explore 31+ breed profiles with feeding recommendations, growth rates, and market data.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Breeds →", key="home_bd", type="primary", use_container_width=True):
            st.session_state.page = 'breed_database'
            st.rerun()
    with qa3:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem 1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔬</div>
            <div style="font-weight:600;font-size:1rem;margin-bottom:0.4rem;">Feed Formulator</div>
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:1.25rem;">
                Generate an optimised, least-cost feed formula for your animals right now.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Formulating →", key="home_ff", type="primary", use_container_width=True):
            st.session_state.page = 'formulator'
            st.rerun()

    if st.session_state.formulation_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-heading">Recent Formulations</div>
        <div class="section-sub">Your last saved optimisation results</div>
        """, unsafe_allow_html=True)
        for history in st.session_state.formulation_history[-3:]:
            with st.expander(f"🐾 {history['animal']}  ·  {history['timestamp']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Age", f"{history['age']} weeks")
                    st.metric("Weight", f"{history['weight']} kg")
                with c2:
                    st.metric("Protein Req", f"{history['cp_req']}%")
                    st.metric("Energy Req", f"{history['energy_req']} kcal")
                with c3:
                    if 'total_cost' in history:
                        st.metric("Cost/kg", f"₦{history['total_cost']:.2f}")
                    if 'prediction' in history:
                        st.metric("Daily Gain", f"{history['prediction']:.1f} g")


def show_breed_database():
    render_navbar()
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🐾 Breed Database</div>
        <div class="page-desc">Comprehensive profiles for 31+ livestock breeds suited to Nigerian
        climate and production systems — including feeding recommendations and market targets.</div>
    </div>
    """, unsafe_allow_html=True)

    breed_data = get_breed_database()
    animal_type = st.selectbox("Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    breeds = breed_data[animal_type]

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search breeds", placeholder="Type breed name…")
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
            st.markdown(f"**Type:** `{breed_info['Type']}`  &nbsp;·&nbsp;  **Best For:** {breed_info['Best For']}")
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Mature Weight", breed_info["Mature Weight (kg)"] + " kg")
            with m2: st.metric("Growth Rate", breed_info["Growth Rate"])
            with m3: st.metric("Feed Efficiency", breed_info["Feed Efficiency"])
        with col2:
            st.markdown("**Feeding Recommendation**")
            st.info(f"Protein: **{breed_info['Recommended CP (%)']}%**")
            market_key = 'Market Age (months)' if animal_type == "Cattle" else 'Market Age (weeks)'
            unit = "months" if animal_type == "Cattle" else "weeks"
            st.info(f"Market: **{breed_info[market_key]} {unit}**")
            if st.button(f"Use in Formulator", key=f"breed_{breed_name}"):
                st.session_state.selected_breed = breed_name
                st.session_state.page = 'formulator'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Breed Statistics</div>', unsafe_allow_html=True)
    breed_df = pd.DataFrame(breeds).T
    col1, col2 = st.columns(2)
    with col1:
        type_counts = breed_df['Type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                     title="Distribution by Production Type",
                     color_discrete_sequence=px.colors.sequential.Greens[::-1])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        growth_counts = breed_df['Growth Rate'].value_counts()
        fig = px.bar(x=growth_counts.index, y=growth_counts.values,
                     title="Breeds by Growth Rate",
                     labels={'x': 'Growth Rate', 'y': 'Count'},
                     color=growth_counts.values, color_continuous_scale='Greens')
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_nutrient_guide():
    render_navbar()
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📖 Nutrient Requirements Guide</div>
        <div class="page-desc">Science-backed nutritional standards for every livestock type and
        production stage — from starter to finisher, maintenance to lactation.</div>
    </div>
    """, unsafe_allow_html=True)

    nutrient_data = get_nutrient_requirements()
    animal_type = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    requirements = nutrient_data[animal_type]
    st.markdown("---")

    for stage, nutrients in requirements.items():
        st.markdown(f'<div class="stage-header">🎯 {stage}</div>', unsafe_allow_html=True)
        df_nutrients = pd.DataFrame([nutrients]).T
        df_nutrients.columns = ['Requirement']
        df_nutrients.index.name = 'Nutrient Parameter'
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
    st.markdown("### 📋 Feeding Guidelines")
    if animal_type == "Rabbit":
        st.markdown("""<div class="alert-green">
        <strong>Rabbit Feeding Guidelines:</strong><br>
        • Provide fresh water at all times (rabbits drink 2–3× their feed weight)<br>
        • Hay should make up 70–80% of adult rabbit diet<br>
        • Introduce new feeds gradually over 7–10 days<br>
        • Monitor body condition score regularly (ideal: ribs barely palpable)<br>
        • Higher fiber content prevents digestive issues and hairballs<br>
        • Avoid sudden diet changes which can cause enteritis
        </div>""", unsafe_allow_html=True)
    elif animal_type == "Poultry":
        st.markdown("""<div class="alert-green">
        <strong>Poultry Feeding Guidelines:</strong><br>
        • Layer birds require high calcium (3.5–4%) for strong eggshells<br>
        • Grit (insoluble granite) aids digestion, especially for whole grains<br>
        • Feed should be stored in cool, dry, rodent-proof conditions<br>
        • Sudden feed changes can reduce performance by 10–20%<br>
        • Water consumption is roughly 2× feed intake (more in hot weather)<br>
        • Use feeders that minimise waste (adjust to bird back height)
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-green">
        <strong>Cattle Feeding Guidelines:</strong><br>
        • TDN = Total Digestible Nutrients (energy measure for ruminants)<br>
        • Ruminants require 15–20% fiber for proper rumen function<br>
        • Transition periods are critical — allow 21 days minimum<br>
        • Fresh, clean water must always be available (50–80 L/day for lactating cows)<br>
        • Monitor body condition score (BCS 1–9, target: 5–6)<br>
        • Avoid over-feeding grain (acidosis risk) — max 60% of diet
        </div>""", unsafe_allow_html=True)

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
        st.download_button(
            label=f"📥 Download {animal_type} Nutrient Guide (CSV)",
            data=csv, file_name=f"{animal_type.lower()}_nutrient_guide.csv",
            mime="text/csv", use_container_width=True)
    with col2:
        if st.button("🔬 Proceed to Feed Formulator", type="primary", use_container_width=True):
            st.session_state.page = 'formulator'
            st.rerun()


def show_formulator():
    render_navbar()
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔬 Feed Formulation Centre</div>
        <div class="page-desc">Configure your animal parameters in the sidebar, then use the tabs
        below to optimise, analyse, and export your custom feed formula.</div>
    </div>
    """, unsafe_allow_html=True)

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

    st.sidebar.markdown("""
    <div style="background:linear-gradient(135deg,#0d2818,#1d6b42);color:white;padding:1.25rem 1rem;
         border-radius:10px;margin-bottom:1rem;text-align:center;">
        <div style="font-size:1.4rem;margin-bottom:0.3rem;">⚙️</div>
        <div style="font-weight:600;font-size:0.95rem;">Animal Parameters</div>
        <div style="font-size:0.75rem;opacity:0.7;margin-top:0.2rem;">Configure inputs below</div>
    </div>
    """, unsafe_allow_html=True)

    if 'selected_breed' in st.session_state:
        st.sidebar.success(f"✓ Breed: {st.session_state.selected_breed}")

    age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
    weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
    cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
    energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
    feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Summary")
    st.sidebar.metric("Animal", animal)
    st.sidebar.metric("Ingredients Available", len(df))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Navigate")
    if st.sidebar.button("📖 Nutrient Guide", use_container_width=True):
        st.session_state.page = 'nutrient_guide'
        st.rerun()
    if st.sidebar.button("🐾 Breed Database", use_container_width=True):
        st.session_state.page = 'breed_database'
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Feed Optimizer",
        "📋 Ingredient Database",
        "📈 Growth Prediction",
        "📊 Cost Dashboard"
    ])

    with tab1:
        st.header("🔬 Least-Cost Feed Formulation")
        st.markdown("Using **linear programming** to find the cheapest ingredient blend meeting all nutritional requirements.")

        with st.expander("⚙️ Advanced Constraints (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                use_fiber_constraint = st.checkbox("Add Fiber Constraint")
                if use_fiber_constraint:
                    min_fiber = st.slider("Minimum Fiber (%)", 0, 30, 12)
                    max_fiber = st.slider("Maximum Fiber (%)", 0, 40, 20)
                else:
                    min_fiber, max_fiber = 12, 20
            with col2:
                limit_ingredients = st.checkbox("Limit Number of Ingredients")
                if limit_ingredients:
                    max_ingredients = st.slider("Maximum Ingredients", 3, 15, 8)
                else:
                    max_ingredients = 15

        if st.button("🚀 Optimise Feed Formula", type="primary"):
            with st.spinner("Calculating optimal feed mix…"):
                try:
                    prob = LpProblem("FeedMix", LpMinimize)
                    ingredients = df["Ingredient"].tolist()
                    vars = LpVariable.dicts("Ingr", ingredients, lowBound=0, upBound=1)
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
                    prob += lpSum(vars[i] for i in ingredients) == 1
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req
                    if use_fiber_constraint and 'Fiber' in df.columns:
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) >= min_fiber
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) <= max_fiber
                    prob.solve()

                    if LpStatus[prob.status] == "Optimal":
                        result = {i: vars[i].value() for i in ingredients if vars[i].value() > 0.001}
                        if limit_ingredients and len(result) > max_ingredients:
                            st.warning(f"⚠️ Solution uses {len(result)} ingredients (limit: {max_ingredients}).")

                        result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
                        result_df["Proportion (%)"] = (result_df["Proportion"] * 100).round(2)
                        result_df["Cost/kg (₦)"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["Cost"].values[0])
                        result_df["Cost Contribution (₦)"] = (result_df["Proportion"] * result_df["Cost/kg (₦)"]).round(2)
                        result_df["CP Contribution"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["CP"].values[0]) * result_df["Proportion"]
                        result_df["Energy Contribution"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["Energy"].values[0]) * result_df["Proportion"]
                        total_cp = result_df["CP Contribution"].sum()
                        total_energy = result_df["Energy Contribution"].sum()
                        result_df = result_df.sort_values("Proportion", ascending=False)
                        total_cost = value(prob.objective)

                        st.session_state['optimization_result'] = result_df
                        st.session_state['total_cost'] = total_cost
                        st.session_state['total_cp'] = total_cp
                        st.session_state['total_energy'] = total_energy
                        st.session_state.formulation_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'animal': animal, 'age': age, 'weight': weight,
                            'cp_req': cp_req, 'energy_req': energy_req, 'total_cost': total_cost
                        })

                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("💰 Feed Cost/kg", f"₦{total_cost:.2f}")
                        with col2: st.metric("📅 Daily Feed Cost", f"₦{total_cost * feed_intake:.2f}")
                        with col3: st.metric("📦 Ingredients Used", len(result))
                        with col4: st.metric("📆 Monthly Cost", f"₦{total_cost * feed_intake * 30:.2f}")

                        st.markdown("---")
                        st.subheader("✅ Nutritional Achievement")
                        col1, col2 = st.columns(2)
                        with col1:
                            cp_pct = (total_cp / cp_req) * 100 if cp_req > 0 else 0
                            st.metric("Crude Protein", f"{total_cp:.2f}%", delta=f"{cp_pct:.1f}% of requirement")
                        with col2:
                            energy_pct = (total_energy / energy_req) * 100 if energy_req > 0 else 0
                            st.metric("Energy", f"{total_energy:.0f} kcal/kg", delta=f"{energy_pct:.1f}% of requirement")

                        st.success(f"✅ Optimisation complete! Total cost: ₦{total_cost:.2f}/kg")
                        st.dataframe(
                            result_df[["Ingredient", "Proportion (%)", "Cost/kg (₦)", "Cost Contribution (₦)"]],
                            use_container_width=True, hide_index=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_pie = px.pie(result_df, values="Proportion (%)", names="Ingredient",
                                             title="Feed Composition",
                                             color_discrete_sequence=px.colors.sequential.Greens)
                            fig_pie.update_layout(template="plotly_white")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            fig_bar = px.bar(result_df, x="Ingredient", y="Cost Contribution (₦)",
                                             title="Cost Breakdown by Ingredient",
                                             color="Cost Contribution (₦)", color_continuous_scale="Greens")
                            fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_white")
                            st.plotly_chart(fig_bar, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            csv = result_df.to_csv(index=False)
                            st.download_button(label="📥 Download Formula (CSV)", data=csv,
                                               file_name=f"{animal}_feed_formula_{datetime.now().strftime('%Y%m%d')}.csv",
                                               mime="text/csv", use_container_width=True)
                        with col2:
                            report = generate_report(animal, age, weight, cp_req, energy_req,
                                                     feed_intake, result_df, total_cost)
                            st.download_button(label="📄 Download Report (TXT)", data=report,
                                               file_name=f"{animal}_feed_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                               mime="text/plain", use_container_width=True)
                    else:
                        st.error("❌ No feasible solution found. Try adjusting your requirements or constraints.")
                except Exception as e:
                    st.error(f"❌ Error during optimisation: {str(e)}")

    with tab2:
        st.header("📋 Ingredient Database Manager")
        st.markdown(f"**{len(df)} ingredients** available for {animal} feed formulation.")
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Search ingredients", placeholder="Type to filter…")
        with col2:
            sort_by = st.selectbox("Sort by", ["Ingredient", "CP", "Energy", "Cost"])

        filtered_df = df[df["Ingredient"].str.contains(search, case=False, na=False)] if search else df
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_by == "Ingredient"))

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Ingredients", len(filtered_df))
        with col2: st.metric("Avg Cost/kg", f"₦{filtered_df['Cost'].mean():.2f}")
        with col3: st.metric("Avg Protein", f"{filtered_df['CP'].mean():.1f}%")
        with col4: st.metric("Avg Energy", f"{filtered_df['Energy'].mean():.0f} kcal")

        st.markdown("---")
        edited_df = st.data_editor(
            filtered_df, num_rows="dynamic", use_container_width=True,
            column_config={
                "Ingredient": st.column_config.TextColumn("Ingredient", width="medium"),
                "CP": st.column_config.NumberColumn("Crude Protein (%)", format="%.1f"),
                "Energy": st.column_config.NumberColumn("Energy (kcal/kg)", format="%.0f"),
                "Fiber": st.column_config.NumberColumn("Crude Fiber (%)", format="%.1f"),
                "Cost": st.column_config.NumberColumn("Cost (₦/kg)", format="₦%.2f"),
            })

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes to Database", use_container_width=True):
                edited_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
                st.success("✅ Ingredient database updated successfully!")
                st.cache_data.clear()
        with col2:
            csv = edited_df.to_csv(index=False)
            st.download_button("📥 Download Database (CSV)", csv,
                               f"{animal.lower()}_ingredients.csv", "text/csv", use_container_width=True)

    with tab3:
        st.header("📈 AI Weight Gain Prediction")
        st.markdown("**Random Forest ML model** trained on 110+ feeding trials from Nigerian farms.")
        st.markdown("---")

        if st.button("🎯 Calculate Growth Prediction", type="primary"):
            with st.spinner("Calculating growth predictions…"):
                avg_cp = df["CP"].mean()
                avg_energy = df["Energy"].mean()
                X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
                prediction = model.predict(X_input)[0]
                st.session_state['prediction'] = prediction

        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            weekly_gain = prediction * 7
            monthly_gain = prediction * 30
            projected_weight_90d = weight + (monthly_gain * 3 / 1000)

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Weight Gain", f"{prediction:.1f} g/day")
            with col2: st.metric("Weekly Gain", f"{weekly_gain:.0f} g")
            with col3: st.metric("Monthly Gain", f"{monthly_gain / 1000:.2f} kg")
            with col4: st.metric("90-Day Weight", f"{projected_weight_90d:.1f} kg",
                                  delta=f"+{projected_weight_90d - weight:.1f} kg")

            st.subheader("📊 90-Day Weight Projection")
            days = np.arange(0, 91)
            projected_weights = weight + (prediction * days / 1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=projected_weights, mode='lines',
                                     name='Projected Weight', line=dict(color='#228b55', width=3),
                                     fill='tozeroy', fillcolor='rgba(34,139,85,0.08)'))
            fig.add_trace(go.Scatter(x=[0], y=[weight], mode='markers',
                                     name='Current Weight', marker=dict(size=12, color='#e74c3c')))
            fig.update_layout(xaxis_title="Days", yaxis_title="Weight (kg)",
                              hovermode='x unified', template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
                st.metric("Feed Conversion Ratio (FCR)", f"{fcr:.2f}:1")
                st.caption("Feed required to gain 1 kg of body weight")
            with col2:
                if 'total_cost' in st.session_state and prediction > 0:
                    cost_per_kg_gain = (st.session_state['total_cost'] * feed_intake * 1000) / prediction
                    st.metric("Cost per kg Gain", f"₦{cost_per_kg_gain:.2f}")
                    st.caption("Feed cost to produce 1 kg of weight gain")
                else:
                    st.info("💡 Run the Feed Optimizer first to see cost metrics")

            st.markdown("---")
            st.subheader("🎯 Growth Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if animal == "Rabbit":
                    performance = "🟢 Excellent" if prediction > 30 else ("🟡 Good" if prediction > 20 else "🔴 Below Average")
                elif animal == "Poultry":
                    performance = "🟢 Excellent" if prediction > 50 else ("🟡 Good" if prediction > 35 else "🔴 Below Average")
                else:
                    performance = "🟢 Excellent" if prediction > 800 else ("🟡 Good" if prediction > 500 else "🔴 Below Average")
                st.metric("Performance Rating", performance)
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

    with tab4:
        st.header("📊 Cost Analysis Dashboard")
        if 'optimization_result' not in st.session_state:
            st.markdown('<div class="alert-amber">⚠️ Please run the Feed Optimizer first to unlock the Cost Dashboard</div>', unsafe_allow_html=True)
        else:
            result_df = st.session_state['optimization_result']
            total_cost = st.session_state['total_cost']

            st.subheader("💰 Cost Projections")
            daily_cost = total_cost * feed_intake
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Daily Cost", f"₦{daily_cost:.2f}")
            with col2: st.metric("Weekly Cost", f"₦{daily_cost * 7:.2f}")
            with col3: st.metric("Monthly Cost", f"₦{daily_cost * 30:.2f}")
            with col4: st.metric("Yearly Cost", f"₦{daily_cost * 365:,.2f}")

            st.markdown("---")
            st.subheader("🐾 Herd / Flock Cost Calculator")
            col1, col2 = st.columns(2)
            with col1: num_animals = st.number_input("Number of Animals", min_value=1, max_value=10000, value=100)
            with col2: duration_days = st.slider("Duration (days)", 1, 365, 90)

            total_herd_cost = daily_cost * num_animals * duration_days
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Feed Cost", f"₦{total_herd_cost:,.2f}")
            with col2: st.metric("Cost per Animal", f"₦{total_herd_cost / num_animals:,.2f}")
            with col3: st.metric("Daily Herd Cost", f"₦{daily_cost * num_animals:,.2f}")

            st.markdown("---")
            st.subheader("📊 Cost Breakdown Analysis")
            fig = px.treemap(result_df, path=['Ingredient'], values='Cost Contribution (₦)',
                             title='Cost Contribution by Ingredient', color='Cost Contribution (₦)',
                             color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                top_5 = result_df.nlargest(5, 'Cost Contribution (₦)')
                fig = px.bar(top_5, x='Ingredient', y='Cost Contribution (₦)',
                             title='Top 5 Cost Contributors', color='Cost Contribution (₦)',
                             color_continuous_scale='Reds')
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(result_df, x='Proportion (%)', y='Cost/kg (₦)',
                                 size='Cost Contribution (₦)', hover_name='Ingredient',
                                 title='Proportion vs Unit Cost', color='Cost Contribution (₦)',
                                 color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            if 'prediction' in st.session_state:
                st.markdown("---")
                st.subheader("💵 Return on Investment Calculator")
                prediction = st.session_state['prediction']
                col1, col2 = st.columns(2)
                with col1:
                    default_price = 1500 if animal == "Rabbit" else (1200 if animal == "Poultry" else 2000)
                    price_per_kg = st.number_input("Selling Price (₦/kg live weight)",
                                                   min_value=500, max_value=5000, value=default_price)
                with col2:
                    production_days = st.number_input("Production Cycle (days)", min_value=30, max_value=365, value=90)

                total_feed_cost = daily_cost * production_days
                weight_gain_kg = (prediction * production_days) / 1000
                final_weight = weight + weight_gain_kg
                revenue = final_weight * price_per_kg
                profit = revenue - total_feed_cost
                roi_percent = (profit / total_feed_cost * 100) if total_feed_cost > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Feed Cost", f"₦{total_feed_cost:,.2f}")
                with col2: st.metric("Final Weight", f"{final_weight:.2f} kg")
                with col3: st.metric("Revenue", f"₦{revenue:,.2f}")
                with col4: st.metric("Profit", f"₦{profit:,.2f}", delta=f"{roi_percent:.1f}% ROI")

                roi_data = pd.DataFrame({'Category': ['Feed Cost', 'Profit'],
                                         'Amount': [total_feed_cost, profit if profit > 0 else 0]})
                fig = px.pie(roi_data, values='Amount', names='Category',
                             title=f'Cost vs Profit (ROI: {roi_percent:.1f}%)',
                             color_discrete_sequence=['#e74c3c', '#228b55'])
                st.plotly_chart(fig, use_container_width=True)

                if profit > 0:
                    st.success(f"✅ Profitable! Expected profit of ₦{profit:,.2f} per animal over {production_days} days.")
                else:
                    st.error("⚠️ Loss expected. Adjust feeding programme or selling price.")


# ── Routing ──
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'nutrient_guide':
    show_nutrient_guide()
elif st.session_state.page == 'breed_database':
    show_breed_database()
elif st.session_state.page == 'formulator':
    show_formulator()

st.markdown("""
<style>

/* ===============================
   MOBILE RESPONSIVENESS
   =============================== */

@media (max-width: 1024px) {
  .main .block-container {
    padding: 0 1.25rem 2rem;
  }
}

@media (max-width: 768px) {
  [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }

  .stButton > button {
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.9rem !important;
  }
}

@media (max-width: 480px) {
  .main .block-container {
    padding: 0 1rem 1.5rem;
  }
}

/* ===============================
   COLOR BLENDING
   =============================== */

.stApp {
  background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1f7a4a, #2fbf71) !important;
  border: none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   SOFT AGRI-TECH COLOR PALETTE
   =============================== */

:root {
  --primary-green: #2f6f55;
  --accent-green: #6fae92;
  --background-main: #fafafa;
  --background-soft: #f3f6f4;
  --text-main: #2b2b2b;
  --text-muted: #6b7280;
}

/* App background */
.stApp {
  background-color: var(--background-soft);
  color: var(--text-main);
}

/* Main content card feel */
.main .block-container {
  background-color: var(--background-main);
  border-radius: 12px;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
  background-color: var(--primary-green) !important;
  border: none !important;
  color: #ffffff !important;
}

/* Hover state */
.stButton > button[kind="primary"]:hover {
  background-color: #285e48 !important;
}

/* Tabs */
.stTabs [role="tab"][aria-selected="true"] {
  background-color: var(--primary-green) !important;
  color: #ffffff !important;
  border-radius: 8px;
}

/* Secondary text */
small, .stMarkdown p {
  color: var(--text-muted);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   FIX METRIC NUMBERS CLIPPING
   =============================== */

/* Allow metric values to wrap instead of overflow */
[data-testid="stMetricValue"] {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
  font-size: 1.1rem;
}

/* Adjust metric container spacing (mobile-safe) */
[data-testid="metric-container"] {
  padding: 0.6rem 0.8rem !important;
}

/* Smaller screens */
@media (max-width: 480px) {
  [data-testid="stMetricValue"] {
    font-size: 1rem !important;
  }

  [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
  }
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* ===============================
   UNIVERSAL TEXT & NUMBER VISIBILITY FIX
   =============================== */

/* Allow wrapping everywhere */
* {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
  word-break: break-word;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  line-height: 1.25;
}

/* Paragraphs & markdown */
p, span, div, label {
  line-height: 1.4;
}

/* METRICS (numbers visibility) */
[data-testid="stMetricValue"] {
  white-space: normal !important;
  overflow: visible !important;
  font-size: 1.1rem;
}

[data-testid="stMetricLabel"] {
  white-space: normal !important;
  overflow: visible !important;
}

/* Tables & dataframes */
[data-testid="stDataFrame"] * {
  white-space: normal !important;
  overflow: visible !important;
}

/* Inputs & sliders */
input, textarea, select {
  white-space: normal !important;
  overflow: visible !important;
}

/* Buttons */
.stButton > button {
  white-space: normal !important;
  overflow: visible !important;
}

/* Charts (numbers & labels) */
svg text {
  white-space: normal !important;
}

/* MOBILE SAFETY */
@media (max-width: 480px) {

  body {
    font-size: 0.9rem;
  }

  [data-testid="stMetricValue"] {
    font-size: 1rem !important;
  }

  table {
    font-size: 0.85rem;
  }
}

</style>
""", unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div class="footer-wrap">
    <div>
        <div class="footer-brand">🌱 Necstech Feed Optimizer</div>
        <div class="footer-meta" style="margin-top:0.3rem;">
            Optimising African Agriculture · v2.0 · © 2026 Necstech
        </div>
    </div>
    <div class="footer-badges">
        <div class="badge">🇳🇬 Nigerian Data</div>
        <div class="badge">🤖 ML-Powered</div>
        <div class="badge">📊 NIAS · FAO 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

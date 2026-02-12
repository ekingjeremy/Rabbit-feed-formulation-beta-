import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Necstech Feed Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# ORIGINAL DESIGN SYSTEM (UNCHANGED)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-900:#0d2818;
    --green-800:#154a2e;
    --green-700:#1d6b42;
    --green-600:#228b55;
    --green-500:#2da868;
    --green-300:#7ddfaa;
    --green-100:#d6f5e5;
    --green-50:#f0faf5;

    --amber:#e8a020;
    --amber-light:#fdf0d5;

    --slate-900:#1a2332;
    --slate-700:#334155;
    --slate-500:#64748b;
    --slate-300:#cbd5e1;
    --slate-100:#f1f5f9;

    --white:#ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--slate-900);
}

.stApp {
    background: var(--slate-100);
}

.main .block-container {
    max-width: 1280px;
    padding: 0 2rem 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# RESPONSIVE + COLOR REFINEMENT (NEW)
# --------------------------------------------------
st.markdown("""
<style>

/* ---------- GLOBAL BACKGROUND ---------- */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

/* ---------- TABLET ---------- */
@media (max-width: 1024px) {
    .main .block-container {
        padding: 0 1.25rem 2rem;
    }
}

/* ---------- MOBILE ---------- */
@media (max-width: 768px) {

    /* Force columns to stack */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Headings */
    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.2rem !important; }

    /* Buttons */
    .stButton > button {
        width: 100% !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.9rem !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        padding: 0.8rem;
    }

    /* Reduce card padding */
    .card, .feature-card {
        padding: 1.1rem !important;
    }
}

/* ---------- SMALL PHONES ---------- */
@media (max-width: 480px) {

    .main .block-container {
        padding: 0 1rem 1.5rem;
    }

    h1 { font-size: 1.7rem !important; }

    .stTabs [role="tab"] {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.7rem !important;
    }
}

/* ---------- COLOR HARMONY ---------- */
:root {
    --green-700:#1f7a4a;
    --green-500:#2fbf71;
    --slate-900:#16202e;
    --slate-500:#6b7280;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--green-700), var(--green-500)) !important;
    border: none !important;
}

.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--green-700), var(--green-500)) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

# --------------------------------------------------
# ML MODEL
# --------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[
        ["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal",
         "Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]
    ]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "formulation_history" not in st.session_state:
    st.session_state.formulation_history = []

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
def render_navbar():
    cols = st.columns([2,1,1,1,1])
    with cols[0]:
        st.markdown("### 🌱 **Necstech Feed Optimizer**")

    pages = {
        "home":"🏠 Home",
        "nutrient":"📖 Nutrients",
        "breed":"🐾 Breeds",
        "formulator":"🔬 Formulator"
    }

    for i,(k,v) in enumerate(pages.items()):
        with cols[i+1]:
            if st.button(v, use_container_width=True):
                st.session_state.page = k
                st.rerun()

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
def show_home():
    render_navbar()
    st.title("Smarter Feed, Healthier Livestock 🌱")
    st.write(
        "AI-powered feed optimisation for **Rabbit, Poultry, and Cattle**, "
        "built for Nigerian farmers and researchers."
    )

# --------------------------------------------------
# FORMULATOR PAGE (UNCHANGED LOGIC)
# --------------------------------------------------
def show_formulator():
    render_navbar()
    st.title("🔬 Feed Formulation Centre")

    animal = st.selectbox("Select Animal", ["Rabbit","Poultry","Cattle"])
    df = rabbit_df if animal=="Rabbit" else poultry_df if animal=="Poultry" else cattle_df

    age = st.slider("Age (weeks)",1,120,8)
    weight = st.slider("Body Weight (kg)",0.1,600.0,2.0)
    cp_req = st.slider("Crude Protein (%)",10,30,18)
    energy_req = st.slider("Energy (kcal/kg)",2000,12000,3000)
    feed_intake = st.slider("Feed Intake (kg/day)",0.05,30.0,0.5)

    if st.button("🚀 Optimise Feed", type="primary"):
        prob = LpProblem("FeedMix", LpMinimize)
        ingredients = df["Ingredient"].tolist()
        vars = LpVariable.dicts("Ingr", ingredients, 0)

        prob += lpSum(vars[i]*df.loc[df["Ingredient"]==i,"Cost"].values[0] for i in ingredients)
        prob += lpSum(vars[i] for i in ingredients) == 1
        prob += lpSum(vars[i]*df.loc[df["Ingredient"]==i,"CP"].values[0] for i in ingredients) >= cp_req
        prob += lpSum(vars[i]*df.loc[df["Ingredient"]==i,"Energy"].values[0] for i in ingredients) >= energy_req

        prob.solve()

        if LpStatus[prob.status]=="Optimal":
            res = {i:vars[i].value() for i in ingredients if vars[i].value()>0}
            st.success("Optimisation Successful ✅")
            st.dataframe(pd.DataFrame(res.items(),columns=["Ingredient","Proportion"]))

# --------------------------------------------------
# PAGE ROUTER
# --------------------------------------------------
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "formulator":
    show_formulator()

import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🌍 AI Livestock Feed Formulator", layout="wide")

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
.main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
h1,h2,h3 {font-family:'Poppins',sans-serif;color:#2c3e50;}
.big-font {font-size:3rem;font-weight:700;color:#27ae60;text-align:center;}
.subtitle {font-size:1.2rem;color:#7f8c8d;text-align:center;margin-bottom:2rem;}
.feature-box {background:white;padding:1.5rem;border-radius:10px;
box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:1rem 0;}
.stButton>button {background:linear-gradient(135deg,#667eea,#764ba2);
color:white;border-radius:25px;padding:0.5rem 2rem;border:none;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

# =====================================================
# TRAIN AI MODEL
# =====================================================
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

# =====================================================
# LANDING PAGE
# =====================================================
st.markdown('<p class="big-font">🌍 Intelligent Livestock Feed Formulator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered nutrition platform for Rabbits, Poultry, and Cattle</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="feature-box"><h3>✔ Least-Cost</h3></div>', unsafe_allow_html=True)
col2.markdown('<div class="feature-box"><h3>✔ AI Prediction</h3></div>', unsafe_allow_html=True)
col3.markdown('<div class="feature-box"><h3>✔ Nigerian Data</h3></div>', unsafe_allow_html=True)
col4.markdown('<div class="feature-box"><h3>✔ 31 Breeds</h3></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# ANIMAL SELECTION
# =====================================================
animal = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
df = rabbit_df if animal == "Rabbit" else poultry_df if animal == "Poultry" else cattle_df

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🎯 Animal Parameters")
age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔬 Feed Optimizer", "📈 AI Growth Prediction", "📋 Ingredient Database"])

# =====================================================
# FEED OPTIMIZER
# =====================================================
with tab1:
    if st.button("🚀 Optimize Feed Formula"):
        prob = LpProblem("FeedMix", LpMinimize)
        ingredients = df["Ingredient"].tolist()
        vars = LpVariable.dicts("Ingr", ingredients, lowBound=0)

        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
        prob += lpSum(vars[i] for i in ingredients) == 1
        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req
        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req
        prob.solve()

        if LpStatus[prob.status] == "Optimal":
            result = {i: vars[i].value() for i in ingredients if vars[i].value() > 0.001}
            result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
            st.dataframe(result_df)
            st.plotly_chart(px.pie(result_df, values="Proportion", names="Ingredient"))

# =====================================================
# AI PREDICTION
# =====================================================
with tab2:
    avg_cp = df["CP"].mean()
    avg_energy = df["Energy"].mean()
    X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
    prediction = model.predict(X_input)[0]
    st.metric("Daily Gain", f"{prediction:.1f} g")

# =====================================================
# INGREDIENT DB
# =====================================================
with tab3:
    st.dataframe(df)

st.markdown("---")
st.markdown("🌾 Powered by AI • Built for Nigerian Farmers")

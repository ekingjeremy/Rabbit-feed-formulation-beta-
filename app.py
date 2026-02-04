import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🌍 Necstech Nigerian Livestock Feed Formulator", layout="wide")

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
.main {background: linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);}
.big-font {font-size:3rem;font-weight:700;color:#27ae60;text-align:center;}
.subtitle {text-align:center;color:#7f8c8d;margin-bottom:2rem;}
.feature-box {background:white;padding:1.5rem;border-radius:10px;
box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:1rem 0;}
.stButton>button {background:linear-gradient(135deg,#667eea,#764ba2);
color:white;border-radius:25px;padding:0.5rem 2rem;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA (ONLY ONCE)
# =====================================================
def categorize_ingredient(cp, fiber, energy):
    if cp > 30: return 'Protein Sources'
    if fiber > 20: return 'Fiber Sources'
    if energy > 2800 and cp < 15: return 'Energy Sources'
    return 'Protein Concentrates'

@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    for df in [rabbit, poultry, cattle]:
        df["Category"] = df.apply(lambda r: categorize_ingredient(r['CP'], r['Fiber'], r['Energy']), axis=1)
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

# =====================================================
# TRAIN AI MODEL
# =====================================================
@st.cache_resource
def train_model(data):
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal",
              "Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)

# =====================================================
# LANDING HEADER
# =====================================================
st.markdown('<p class="big-font">🌍 Necstech Nigerian Livestock Feed Formulator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered nutrition platform for Rabbits, Poultry & Cattle</p>', unsafe_allow_html=True)

# Feature Boxes
c1,c2,c3,c4 = st.columns(4)
c1.markdown('<div class="feature-box"><b>✔ Least-Cost Formulation</b></div>', unsafe_allow_html=True)
c2.markdown('<div class="feature-box"><b>✔ AI Growth Prediction</b></div>', unsafe_allow_html=True)
c3.markdown('<div class="feature-box"><b>✔ Nigerian Ingredients</b></div>', unsafe_allow_html=True)
c4.markdown('<div class="feature-box"><b>✔ Multi-Species Support</b></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# ANIMAL SELECTION
# =====================================================
animal = st.selectbox("🐾 Select Animal Type", ["Rabbit","Poultry","Cattle"])
df = rabbit_df if animal=="Rabbit" else poultry_df if animal=="Poultry" else cattle_df

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🎯 Animal Parameters")
age = st.sidebar.slider("Age (weeks)",1,120,8)
weight = st.sidebar.slider("Body Weight (kg)",0.1,600.0,2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)",8,30,18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)",1800,12000,3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)",0.05,30.0,0.5)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Nutrient Guide",
    "🔬 Feed Optimizer",
    "📋 Ingredient Database",
    "📈 Weight Gain Prediction"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.metric("Crude Protein Requirement", f"{cp_req}%")
    st.metric("Energy Requirement", f"{energy_req} kcal/kg")
    st.metric("Feed Intake", f"{feed_intake} kg/day")

# ---------------- TAB 2 ----------------
with tab2:
    if st.button("🚀 Optimize Feed Formula"):
        prob = LpProblem("FeedMix", LpMinimize)
        ingredients = df["Ingredient"].tolist()
        vars_dict = LpVariable.dicts("Ingr", ingredients, lowBound=0)

        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
        prob += lpSum(vars_dict[i] for i in ingredients) == 1
        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients) >= cp_req
        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients) >= energy_req
        prob.solve()

        if LpStatus[prob.status] == "Optimal":
            result = {i:vars_dict[i].value() for i in ingredients if vars_dict[i].value()>0.001}
            result_df = pd.DataFrame(result.items(), columns=["Ingredient","Proportion"])
            st.dataframe(result_df)
            st.plotly_chart(px.pie(result_df, values="Proportion", names="Ingredient"))

# ---------------- TAB 3 ----------------
with tab3:
    search = st.text_input("🔍 Search ingredients")
    st.dataframe(df[df["Ingredient"].str.contains(search, case=False)] if search else df)

# ---------------- TAB 4 ----------------
with tab4:
    avg_cp = df["CP"].mean()
    avg_energy = df["Energy"].mean()
    X_input = np.array([[age,weight,cp_req,energy_req,feed_intake,avg_cp,avg_energy]])
    prediction = model.predict(X_input)[0]

    st.metric("Daily Gain", f"{prediction:.1f} g")

    days = np.arange(0,91)
    weights = weight + (prediction*days/1000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=weights, mode='lines', fill='tozeroy'))
    fig.update_layout(template="plotly_white", xaxis_title="Days", yaxis_title="Weight (kg)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("🌾 Powered by AI • Built for Nigerian Farmers")

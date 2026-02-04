import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="🌍 Necstech Nigerian Livestock Feed Formulator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Necstech Nigerian Livestock Feed Formulator")
st.markdown("### AI-Powered Precision Nutrition for Rabbits, Poultry & Cattle")

# ---------------- HELPER ----------------
def categorize_ingredient(name, cp, fiber, energy):
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
    for df in [rabbit,poultry,cattle]:
        df["Category"] = df.apply(lambda r: categorize_ingredient(r['Ingredient'],r['CP'],r['Fiber'],r['Energy']),axis=1)
    return rabbit,poultry,cattle,ml_data

rabbit_df,poultry_df,cattle_df,ml_df = load_data()

@st.cache_resource
def train_model(data):
    X=data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal",
            "Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y=data["Expected_Daily_Gain_g"]
    model=RandomForestRegressor(n_estimators=200,random_state=42)
    model.fit(X,y)
    return model

model=train_model(ml_df)

# ---------------- ANIMAL SELECT ----------------
animal = st.selectbox("🐾 Select Animal Type", ["Rabbit","Poultry","Cattle"])
df = rabbit_df.copy() if animal=="Rabbit" else poultry_df.copy() if animal=="Poultry" else cattle_df.copy()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎯 Animal Parameters")
age = st.sidebar.slider("Age (weeks)",1,120,8)
weight = st.sidebar.slider("Body Weight (kg)",0.1,600.0,2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)",8,30,18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)",1800,12000,3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)",0.05,30.0,0.5)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Nutrient Guide",
    "🔬 Feed Optimizer",
    "📋 Ingredient Optimizer",
    "📈 Weight Gain Prediction"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.metric("Crude Protein Requirement", f"{cp_req}%")
    st.metric("Energy Requirement", f"{energy_req} kcal/kg")
    st.metric("Feed Intake", f"{feed_intake} kg/day")

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Least Cost Feed Formulation")
    if st.button("🚀 Optimize Feed Formula"):
        prob=LpProblem("FeedMix",LpMinimize)
        ingredients=df["Ingredient"].tolist()
        vars_dict=LpVariable.dicts("Ingr",ingredients,lowBound=0)

        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
        prob+=lpSum(vars_dict[i] for i in ingredients)==1
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients)>=cp_req
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients)>=energy_req
        prob.solve()

        if LpStatus[prob.status]=="Optimal":
            result={i:vars_dict[i].value() for i in ingredients if vars_dict[i].value()>0.001}
            result_df=pd.DataFrame(result.items(),columns=["Ingredient","Proportion"])
            st.dataframe(result_df)
            st.plotly_chart(px.pie(result_df,values="Proportion",names="Ingredient"))

# ---------------- TAB 3 ----------------
with tab3:
    st.header("Ingredient Database")
    search=st.text_input("🔍 Search ingredients")
    filtered_df=df[df["Ingredient"].str.contains(search,case=False)] if search else df
    st.dataframe(filtered_df)

# ---------------- TAB 4 — AUTO AI ----------------
with tab4:
    st.header("📈 AI Growth Prediction (Live)")

    avg_cp=df["CP"].mean()
    avg_energy=df["Energy"].mean()

    # 🔥 AUTO prediction — runs instantly when sliders change
    X_input=np.array([[age,weight,cp_req,energy_req,feed_intake,avg_cp,avg_energy]])
    prediction=model.predict(X_input)[0]

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Daily Gain",f"{prediction:.1f} g")
    col2.metric("Weekly Gain",f"{prediction*7:.0f} g")
    col3.metric("Monthly Gain",f"{prediction*30/1000:.2f} kg")
    col4.metric("90-Day Weight",f"{weight+(prediction*90/1000):.1f} kg")

    days=np.arange(0,91)
    projected_weights=weight+(prediction*days/1000)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=days,y=projected_weights,mode='lines',fill='tozeroy'))
    fig.update_layout(template="plotly_white",xaxis_title="Days",yaxis_title="Weight (kg)")
    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")
st.markdown("🌾 Powered by AI • Built for Nigerian Farmers")

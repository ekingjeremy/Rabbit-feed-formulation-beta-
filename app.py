import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🌍 AI Livestock Feed Formulator", layout="wide")

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
st.title("🌍 Intelligent Livestock Feed Formulator")
st.markdown("""
AI-powered nutrition platform for **Rabbits, Poultry, and Cattle**.

✔ Least-cost feed formulation  
✔ Ingredient database  
✔ AI growth prediction  
✔ Research & farm use  
✔ Works for Nigeria & globally  
""")

# =====================================================
# ANIMAL SELECTION
# =====================================================
animal = st.selectbox("Select Animal Type", ["Rabbit", "Poultry", "Cattle"])

if animal == "Rabbit":
    df = rabbit_df.copy()
elif animal == "Poultry":
    df = poultry_df.copy()
else:
    df = cattle_df.copy()

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("Animal Parameters")

age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔬 Optimizer", "📈 AI Prediction", "📋 Ingredients"])

# =====================================================
# FEED OPTIMIZER
# =====================================================
with tab1:
    st.header("Least Cost Feed Formulation")

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
        result_df["Cost Contribution"] = result_df["Ingredient"].apply(
            lambda x: df[df["Ingredient"] == x]["Cost"].values[0]
        ) * result_df["Proportion"]

        st.dataframe(result_df)
        st.success(f"Total Feed Cost per kg: ₦{value(prob.objective):.2f}")
        st.plotly_chart(px.pie(result_df, values="Proportion", names="Ingredient"))
    else:
        st.error("No feasible solution found.")

# =====================================================
# AI GROWTH PREDICTION
# =====================================================
with tab2:
    st.header("AI Growth Prediction")

    avg_cp = df["CP"].mean()
    avg_energy = df["Energy"].mean()

    X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
    prediction = model.predict(X_input)[0]

    st.metric("Predicted Daily Weight Gain", f"{prediction:.1f} g/day")

# =====================================================
# INGREDIENT MANAGER
# =====================================================
with tab3:
    st.header("Ingredient Database Manager")

    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    if st.button("Save Changes"):
        edited_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
        st.success("Ingredient database updated successfully.")

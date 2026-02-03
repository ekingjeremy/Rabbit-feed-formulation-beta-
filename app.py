import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🌍 Livestock Feed AI Formulator", layout="wide")

# ---------------- LOAD DATA ----------------
rabbit_df = pd.read_csv("nigeria_rabbit_breeds.csv")
poultry_df = pd.read_csv("nigeria_poultry_breeds.csv")
cattle_df = pd.read_csv("nigeria_cattle_breeds.csv")
ingredients_df = pd.read_csv("nigeria_feed_ingredients.csv")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🌍 Smart Livestock Feed Formulator")

    animal_type = st.selectbox("Select Animal", ["Rabbit", "Poultry", "Cattle"])

    if animal_type == "Rabbit":
        breed_data = rabbit_df
    elif animal_type == "Poultry":
        breed_data = poultry_df
    else:
        breed_data = cattle_df

    selected_breed = st.selectbox("Select Breed", breed_data["Breed"].unique())
    breed_info = breed_data[breed_data["Breed"] == selected_breed].iloc[0]

    age_weeks = st.slider("Age (weeks)", 1, 104, 12)

    st.markdown("---")
    st.subheader("📋 Nutrient Requirements")
    cp_req = st.slider("Crude Protein (%)", 10, 30, int(breed_info["CP_%"]))
    energy_req = st.slider("Energy (Kcal/kg)", 1500, 3500, 2500)
    fibre_req = st.slider("Fibre (%)", 5, 40, 12)
    calcium_req = st.slider("Calcium (%)", 0.1, 5.0, 0.5)

# ---------------- FILTER INGREDIENTS ----------------
ingredients = ingredients_df[
    ingredients_df["Animal_Use"].str.contains(animal_type, case=False) |
    ingredients_df["Animal_Use"].str.contains("All", case=False)
]

ingredients = ingredients.set_index("Ingredient")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔬 Optimizer", "📋 Ingredients", "📈 Prediction"])

# ---------------- OPTIMIZER ----------------
with tab1:
    st.header("🔬 Feed Mix Optimizer")

    model = LpProblem("Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    model += lpSum(vars[i] for i in ingredients.index) == 1
    model += lpSum(vars[i] * ingredients.loc[i, 'CP_%'] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Energy_Kcal/kg'] for i in ingredients.index) >= energy_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Fibre_%'] for i in ingredients.index) >= fibre_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Calcium_%'] for i in ingredients.index) >= calcium_req

    model.solve()

    if LpStatus[model.status] == "Optimal":
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion'])
        st.dataframe(result_df)
        st.plotly_chart(px.pie(result_df, values='Proportion', names=result_df.index))
    else:
        st.error("No feasible solution found.")

# ---------------- INGREDIENT TAB ----------------
with tab2:
    st.header("📋 Ingredient Database")
    st.dataframe(ingredients_df)

# ---------------- GROWTH PREDICTION ----------------
with tab3:
    st.header("📈 Growth Prediction")

    if LpStatus[model.status] == "Optimal":
        proportions = np.array([vars[i].varValue for i in ingredients.index])
        cp_vals = np.array([ingredients.loc[i, "CP_%"] for i in ingredients.index])
        energy_vals = np.array([ingredients.loc[i, "Energy_Kcal/kg"] for i in ingredients.index])

        feed_cp = np.dot(proportions, cp_vals)
        feed_energy = np.dot(proportions, energy_vals)

        base_growth = breed_info["Growth_g_per_day"]
        weight_gain = base_growth * (feed_cp / cp_req) * (feed_energy / energy_req)

        if animal_type == "Poultry":
            adult_weight = breed_info["Mature_Weight_kg"]
        else:
            adult_weight = breed_info["Adult_Weight_kg"]

        expected_weight = adult_weight * (1 - np.exp(-0.05 * age_weeks))

        st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
        st.metric("⚖️ Expected Body Weight (kg)", f"{expected_weight:.2f}")

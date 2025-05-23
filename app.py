import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer")

# -----------------------------
# Default ingredient list
# -----------------------------
def default_ingredients():
    data = [
        # Concentrates
        ["Maize", "Concentrate", 9, 3400, 2, 0.02, 120],
        ["Soybean Meal", "Concentrate", 44, 3200, 7, 0.3, 150],
        ["Groundnut Cake", "Concentrate", 45, 3000, 6, 0.25, 130],
        ["Wheat Offal", "Concentrate", 17, 2100, 10, 0.2, 80],
        ["Palm Kernel Cake", "Concentrate", 18, 2300, 12, 0.25, 85],

        # Fodders
        ["Alfalfa", "Fodder", 18, 2300, 25, 1.5, 80],
        ["Guinea Grass", "Fodder", 9, 1800, 30, 0.35, 40],
        ["Elephant Grass", "Fodder", 8, 1900, 35, 0.3, 35],
        ["Stylosanthes", "Fodder", 16, 2200, 28, 0.5, 45],
        ["Centrosema", "Fodder", 17, 2150, 26, 0.55, 50]
    ]
    return pd.DataFrame(data, columns=["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).set_index("Ingredient")

# Load data from session or default
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = default_ingredients()
df = st.session_state.ingredient_data

# -----------------------------
# Sidebar - Nutrient Inputs
# -----------------------------
st.sidebar.header("Nutrient Requirements")
cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

# -----------------------------
# Ration Type Selection
# -----------------------------
st.sidebar.markdown("---")
ration_type = st.sidebar.selectbox("Select Ration Type", ["Mixed (All)", "Concentrate Only", "Fodder Only"])
if ration_type == "Concentrate Only":
    df_filtered = df[df["Category"] == "Concentrate"]
elif ration_type == "Fodder Only":
    df_filtered = df[df["Category"] == "Fodder"]
else:
    df_filtered = df.copy()

# -----------------------------
# Feed Optimization
# -----------------------------
st.header("🧪 Optimizer")
model = LpProblem("Rabbit_Feed", LpMinimize)
vars = {i: LpVariable(i, lowBound=0) for i in df_filtered.index}
model += lpSum([vars[i] * df_filtered.loc[i, "Cost"] for i in df_filtered.index])
model += lpSum([vars[i] * df_filtered.loc[i, "CP"] for i in df_filtered.index]) >= cp
model += lpSum([vars[i] * df_filtered.loc[i, "Energy"] for i in df_filtered.index]) >= energy
model += lpSum([vars[i] * df_filtered.loc[i, "Fibre"] for i in df_filtered.index]) >= fibre
model += lpSum([vars[i] * df_filtered.loc[i, "Calcium"] for i in df_filtered.index]) >= calcium
model += lpSum([vars[i] for i in df_filtered.index]) == 1

model.solve()

if LpStatus[model.status] == "Optimal":
    st.success("Optimal ration found.")
    result = {i: vars[i].varValue for i in df_filtered.index if vars[i].varValue > 0}
    result_df = pd.DataFrame.from_dict(result, orient="index", columns=["Proportion"])
    result_df["Cost (₦)"] = result_df["Proportion"] * df_filtered.loc[result_df.index, "Cost"]
    st.dataframe(result_df.style.format("{:.2f}"))

    # Pie Chart
    fig = px.pie(result_df, values='Proportion', names=result_df.index, title='Feed Ingredient Breakdown')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Mock Performance Predictor
    # -----------------------------
    st.subheader("📈 Performance Predictor")
    total_cp = sum([vars[i].varValue * df_filtered.loc[i, "CP"] for i in df_filtered.index])
    total_energy = sum([vars[i].varValue * df_filtered.loc[i, "Energy"] for i in df_filtered.index])
    # Mock regression
    gain = 10 + 0.015 * total_cp + 0.002 * total_energy
    st.metric("Estimated Weight Gain (g/day)", f"{gain:.1f}")
else:
    st.error("⚠️ No feasible solution. Try relaxing constraints or include more ingredients.")

# -----------------------------
# Edit Ingredients
# -----------------------------
st.header("✍️ Edit Ingredients")
edited_df = st.data_editor(df.reset_index(), num_rows="dynamic", use_container_width=True)

if st.button("💾 Save Changes"):
    if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
        st.session_state.ingredient_data = edited_df.set_index("Ingredient")
        st.success("Ingredient list updated successfully.")
    else:
        st.error("Ingredient names must be unique and not empty.")

# -----------------------------
# Upload New Ingredients
# -----------------------------
st.header("📤 Upload Ingredients")
file = st.file_uploader("Upload a CSV with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost")
if file:
    uploaded_df = pd.read_csv(file)
    if set(["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(uploaded_df.columns):
        uploaded_df.set_index("Ingredient", inplace=True)
        st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, uploaded_df[~uploaded_df.index.isin(st.session_state.ingredient_data.index)]])
        st.success("Uploaded ingredients added.")
    else:
        st.error("Missing required columns in uploaded file.")

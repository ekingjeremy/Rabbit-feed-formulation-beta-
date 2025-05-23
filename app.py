import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Formulator", layout="wide")
st.title("🐰 Rabbit Feed Formulation & Growth Predictor")

# Initial Ingredient Data (Fodder + Concentrates in Nigeria)
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame([
        # Concentrates
        ["Maize", 9.0, 3400, 2.0, 0.02, 120, "Concentrate"],
        ["Soybean Meal", 44.0, 3200, 7.0, 0.30, 150, "Concentrate"],
        ["Groundnut Cake", 45.0, 3000, 6.0, 0.25, 130, "Concentrate"],
        ["Wheat Offal", 15.0, 1800, 10.0, 0.10, 90, "Concentrate"],
        ["Palm Kernel Cake", 18.0, 2500, 15.0, 0.20, 110, "Concentrate"],
        ["Cottonseed Cake", 23.0, 2600, 10.0, 0.25, 125, "Concentrate"],
        ["Cassava Peel", 3.0, 2800, 12.0, 0.03, 50, "Concentrate"],

        # Fodders
        ["Alfalfa", 18.0, 2300, 25.0, 1.5, 80, "Fodder"],
        ["Guinea Grass", 8.0, 2000, 30.0, 0.6, 50, "Fodder"],
        ["Elephant Grass", 9.0, 2200, 28.0, 0.5, 60, "Fodder"],
        ["Stylosanthes", 16.0, 2400, 22.0, 1.2, 90, "Fodder"],
        ["Centrosema", 14.0, 2300, 24.0, 1.0, 85, "Fodder"],
        ["Leucaena", 25.0, 2600, 20.0, 1.3, 95, "Fodder"]
    ], columns=["Ingredient", "CP", "Energy", "Fibre", "Calcium", "Cost", "Category"])

# Upload feature
st.sidebar.header("📤 Upload Custom Ingredient List")
uploaded_file = st.sidebar.file_uploader("Upload CSV with columns: Ingredient, CP, Energy, Fibre, Calcium, Cost, Category")
if uploaded_file:
    try:
        custom_df = pd.read_csv(uploaded_file)
        if all(col in custom_df.columns for col in ["Ingredient", "CP", "Energy", "Fibre", "Calcium", "Cost", "Category"]):
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, custom_df], ignore_index=True).drop_duplicates("Ingredient")
            st.sidebar.success("Custom ingredients added successfully!")
        else:
            st.sidebar.error("Incorrect columns in uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

df = st.session_state.ingredient_data.set_index("Ingredient")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Predictor"])

# --- Tab 1: Optimizer ---
with tab1:
    st.sidebar.header("Set Nutrient Requirements")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 30, 18)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 35, 15)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 2.0, 0.7)

    model = LpProblem("Rabbit_Feed", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in df.index}
    model += lpSum(vars[i] * df.loc[i, "Cost"] for i in df.index)
    model += lpSum(vars[i] * df.loc[i, "CP"] for i in df.index) >= cp
    model += lpSum(vars[i] * df.loc[i, "Energy"] for i in df.index) >= energy
    model += lpSum(vars[i] * df.loc[i, "Fibre"] for i in df.index) >= fibre
    model += lpSum(vars[i] * df.loc[i, "Calcium"] for i in df.index) >= calcium
    model += lpSum(vars[i] for i in df.index) == 1
    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.subheader("✅ Optimized Feed Mix")
        results = {i: vars[i].varValue for i in df.index if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient="index", columns=["Proportion (kg)"])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, "Cost"]
        result_df["Category"] = df.loc[result_df.index, "Category"]
        st.dataframe(result_df)
        st.metric("💰 Total Cost per kg Feed", f"₦{value(model.objective):.2f}")

        fig = px.pie(result_df, values="Proportion (kg)", names=result_df.index, color="Category", title="Ingredient Distribution")
        st.plotly_chart(fig)
    else:
        st.error("❌ No feasible formulation found.")

# --- Tab 2: Editor ---
with tab2:
    st.subheader("📝 Edit Ingredients")
    editable_df = df.reset_index()
    edited = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        st.session_state.ingredient_data = edited
        st.success("Saved successfully!")

# --- Tab 3: Predictor ---
with tab3:
    st.subheader("📈 Predict Weight Gain")

    if LpStatus[model.status] == "Optimal":
        # Sample ML model (simulated training for illustration)
        X = st.session_state.ingredient_data[["CP", "Energy"]]
        y = 0.015 * X["CP"] + 0.002 * X["Energy"] + 5 + np.random.normal(0, 1, size=len(X))
        reg = LinearRegression().fit(X, y)

        mix = pd.DataFrame({
            "CP": [sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)],
            "Energy": [sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)]
        })
        gain_pred = reg.predict(mix)[0]

        st.metric("📊 Predicted Weight Gain", f"{gain_pred:.2f} g/day")
        st.caption("Model simulated with sample data. Improve with real-world growth records.")
    else:
        st.warning("⚠️ Please run the optimizer first to enable predictions.")

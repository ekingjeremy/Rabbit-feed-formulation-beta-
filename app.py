import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px
import pickle
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation & Performance Predictor")

# Load default ingredient data (both concentrates and fodders)
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "CP": [18, 9, 44, 15, 45, 12, 8],
        "Energy": [2300, 3400, 3200, 1800, 3000, 2200, 2100],
        "Fibre": [25, 2, 7, 10, 6, 28, 32],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25, 1.8, 2.2],
        "Cost": [80, 120, 150, 90, 130, 70, 60],
        "Type": ["Fodder", "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Fodder", "Fodder"]
    }, index=["Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake", "Napier Grass", "Guinea Grass"])

df = st.session_state.ingredient_data

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 AI Predictor"])

# ------------------ OPTIMIZER ------------------
with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 2.0, 0.8)

    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in df.index}

    model += lpSum([vars[i] * df.loc[i, 'Cost'] for i in df.index])
    model += lpSum([vars[i] * df.loc[i, 'CP'] for i in df.index]) >= cp
    model += lpSum([vars[i] * df.loc[i, 'Energy'] for i in df.index]) >= energy
    model += lpSum([vars[i] * df.loc[i, 'Fibre'] for i in df.index]) >= fibre
    model += lpSum([vars[i] * df.loc[i, 'Calcium'] for i in df.index]) >= calcium
    model += lpSum([vars[i] for i in df.index]) == 1

    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.subheader("📊 Optimized Feed Mix")
        results = {i: vars[i].varValue for i in df.index if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, 'Cost']
        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

# ------------------ EDIT INGREDIENTS ------------------
with tab2:
    st.subheader("✍️ Modify Ingredients Table")
    editable_df = df.reset_index().rename(columns={"index": "Ingredient"})
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            edited_df = edited_df.dropna(subset=["Ingredient"])
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Please ensure all ingredients have unique names and no missing values.")

    if LpStatus[model.status] == "Optimal" and st.button("🧹 Remove unused ingredients"):
        used = [i for i in df.index if vars[i].varValue > 0]
        st.session_state.ingredient_data = df.loc[used]
        st.success("Unused ingredients removed.")

# ------------------ AI PREDICTOR ------------------
with tab3:
    st.subheader("🧠 AI Weight Gain Predictor")

    try:
        with open("model.pkl", "rb") as f:
            model_rf = pickle.load(f)
        
        # Extract values
        protein = sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)
        energy_val = sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)
        fibre_val = sum(vars[i].varValue * df.loc[i, "Fibre"] for i in df.index)

        input_features = pd.DataFrame([[protein, energy_val, fibre_val]], columns=["CP", "Energy", "Fibre"])
        predicted_gain = model_rf.predict(input_features)[0]
        
        st.metric("📈 Predicted Daily Weight Gain", f"{predicted_gain:.1f} g/day")
    except FileNotFoundError:
        st.warning("⚠️ AI model not found. Train a model and save it as 'model.pkl' to enable predictions.")

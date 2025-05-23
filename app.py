import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px

st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# Load ingredient data or create default
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "CP": [18, 9, 44, 15, 45, 12, 10],
        "Energy": [2300, 3400, 3200, 1800, 3000, 1900, 1600],
        "Fibre": [25, 2, 7, 10, 6, 30, 35],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25, 0.6, 0.4],
        "Cost": [80, 120, 150, 90, 130, 50, 40]
    }, index=[
        "Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake", "Elephant Grass", "Sweet Potato Vines"
    ])

df = st.session_state.ingredient_data

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Performance Predictor"])

# Optimizer Tab
with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

    # LP Model
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

        # Pie chart
        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

# Edit Ingredients Tab
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
            st.error("❌ Please make sure all ingredients are uniquely named and not empty.")

    if LpStatus[model.status] == "Optimal" and st.button("🧹 Remove unused ingredients"):
        used = [i for i in df.index if vars[i].varValue > 0]
        st.session_state.ingredient_data = df.loc[used]
        st.success("Unused ingredients removed.")

# Performance Predictor Tab
with tab3:
    st.subheader("🚀 Performance Predictor")

    if LpStatus[model.status] == "Optimal":
        # Prepare model input
        protein = lpSum([vars[i].varValue * df.loc[i, "CP"] for i in df.index]).value()
        energy_val = lpSum([vars[i].varValue * df.loc[i, "Energy"] for i in df.index]).value()
        fibre_val = lpSum([vars[i].varValue * df.loc[i, "Fibre"] for i in df.index]).value()
        calcium_val = lpSum([vars[i].varValue * df.loc[i, "Calcium"] for i in df.index]).value()

        X_input = pd.DataFrame([[protein, energy_val, fibre_val, calcium_val]],
                               columns=["CP", "Energy", "Fibre", "Calcium"])

        try:
            with open("model.pkl", "rb") as f:
                model_rf = pickle.load(f)
                predicted_gain = model_rf.predict(X_input)[0]
                st.metric("📈 Predicted Weight Gain", f"{predicted_gain:.1f} g/day")
        except Exception as e:
            st.warning("⚠️ Could not load model. Showing simulated prediction instead.")
            gain = 10 + 0.015 * protein + 0.002 * energy_val
            st.metric("📈 Simulated Weight Gain", f"{gain:.1f} g/day")
            st.info("This is a simulated estimate. For real predictions, train and upload a model.")
    else:
        st.warning("⚠️ Prediction unavailable. Run a successful optimization first.")

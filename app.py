import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# --- Ingredient data with category ---
if "ingredient_data" not in st.session_state:
    data = {
        "Ingredient": [
            # Concentrates
            "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Bran", "Cottonseed Cake",
            # Fodders
            "Alfalfa", "Napier Grass", "Clover", "Moringa Leaves", "Sweet Potato Leaves"
        ],
        "Category": [
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder"
        ],
        "CP": [9, 44, 45, 15, 40, 18, 10, 22, 24, 20],
        "Energy": [3400, 3200, 3000, 1800, 3100, 2300, 2200, 2100, 2500, 2400],
        "Fibre": [2, 7, 6, 10, 8, 25, 28, 30, 20, 22],
        "Calcium": [0.02, 0.3, 0.25, 0.1, 0.35, 1.5, 1.2, 1.3, 1.4, 1.3],
        "Cost": [120, 150, 130, 90, 140, 80, 75, 70, 65, 68]
    }
    df = pd.DataFrame(data).set_index("Ingredient")
    st.session_state.ingredient_data = df.copy()

df = st.session_state.ingredient_data

# --- Option to select feed type ---
feed_type = st.sidebar.radio("Select Feed Type for Optimization:", 
                             ("Both (Concentrate + Fodder)", "Only Concentrates", "Only Fodders"))

if feed_type == "Only Concentrates":
    df_filtered = df[df["Category"] == "Concentrate"]
elif feed_type == "Only Fodders":
    df_filtered = df[df["Category"] == "Fodder"]
else:
    df_filtered = df.copy()

# --- Nutrient requirements ---
st.sidebar.header("Nutrient Requirements (per kg feed)")
cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

# --- Optimization ---
model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
vars = {i: LpVariable(i, lowBound=0) for i in df_filtered.index}

# Objective: minimize cost
model += lpSum([vars[i] * df_filtered.loc[i, 'Cost'] for i in df_filtered.index])

# Constraints
model += lpSum([vars[i] * df_filtered.loc[i, 'CP'] for i in df_filtered.index]) >= cp
model += lpSum([vars[i] * df_filtered.loc[i, 'Energy'] for i in df_filtered.index]) >= energy
model += lpSum([vars[i] * df_filtered.loc[i, 'Fibre'] for i in df_filtered.index]) >= fibre
model += lpSum([vars[i] * df_filtered.loc[i, 'Calcium'] for i in df_filtered.index]) >= calcium
model += lpSum([vars[i] for i in df_filtered.index]) == 1  # total proportion sums to 1kg

model.solve()

if LpStatus[model.status] == "Optimal":
    st.subheader("📊 Optimized Feed Mix")
    results = {i: vars[i].varValue for i in df_filtered.index if vars[i].varValue > 0}
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
    result_df["Category"] = df_filtered.loc[result_df.index, "Category"]
    result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df_filtered.loc[result_df.index, 'Cost']
    st.dataframe(result_df)
    st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

    fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
    st.plotly_chart(fig)

    # --- Performance Prediction ---
    st.subheader("🚀 Predicted Weight Gain")
    # Simple linear regression model trained on dummy data for demo:
    # Inputs: weighted CP and Energy of the mix
    protein_val = sum(vars[i].varValue * df_filtered.loc[i, "CP"] for i in df_filtered.index)
    energy_val = sum(vars[i].varValue * df_filtered.loc[i, "Energy"] for i in df_filtered.index)

    # Dummy linear regression coefficients (replace with trained model coefficients)
    gain = 10 + 0.015 * protein_val + 0.002 * energy_val

    st.metric("Expected Weight Gain (g/day)", f"{gain:.1f}")
    st.info("This prediction is a simulated estimate. Use real rabbit growth data to train a better model.")
else:
    st.error("⚠️ No feasible solution found with current nutrient settings.")

# --- Edit Ingredients Tab ---
tab2, tab3 = st.tabs(["📝 Edit Ingredients", "📈 Upload Ingredients"])

with tab2:
    st.subheader("✍️ Modify Ingredients Table")
    editable_df = df.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            edited_df = edited_df.dropna(subset=["Ingredient"])
            edited_df = edited_df.set_index("Ingredient")
            st.session_state.ingredient_data = edited_df
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Please ensure all ingredients are uniquely named and not empty.")

with tab3:
    st.subheader("⬆️ Upload New Ingredients (CSV)")
    uploaded_file = st.file_uploader("Upload CSV with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost")
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
            if not required_cols.issubset(new_df.columns):
                st.error(f"CSV must contain columns: {required_cols}")
            else:
                new_df = new_df.set_index("Ingredient")
                combined_df = pd.concat([st.session_state.ingredient_data, new_df])
                st.session_state.ingredient_data = combined_df[~combined_df.index.duplicated(keep='last')]
                st.success("New ingredients uploaded and merged successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

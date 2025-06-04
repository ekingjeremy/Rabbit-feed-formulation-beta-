import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🐰 Rabbit Feed Formulation Optimizer", layout="wide")
st.title("🐰 Feed My Rabbit")

# --- Initialize or load data ---
if "ingredient_data" not in st.session_state:
    data = {
        "Ingredient": [
            # Fodders
            "Alfalfa", "Elephant Grass", "Gamba Grass", "Guinea Grass", "Centrosema",
            "Stylosanthes", "Leucaena", "Gliricidia", "Calliandra calothyrsus",
            "Cowpea Fodder", "Sorghum Fodder", "Cassava Leaves", "Napier Grass",
            "Teff Grass", "Faidherbia albida Pods",
            # Concentrates
            "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Offal", "Palm Kernel Cake",
            "Brewer's Dry Grains", "Cassava Peel", "Maize Bran", "Rice Bran",
            "Cottonseed Cake", "Fish Meal", "Blood Meal", "Feather Meal", "Bone Meal",
            # Minerals & Additives
            "Limestone", "Salt", "Methionine", "Lysine", "Vitamin Premix"
        ],
        "Category": [
            "Fodder"] * 15 + ["Concentrate"] * 14 + ["Mineral"] * 2 + ["Additive"] * 3,
        "CP": [18, 8, 7, 10, 17, 14, 25, 24, 22, 20, 8, 18, 12, 10, 14,
                9, 44, 45, 15, 20, 18, 5, 7, 14, 36, 60, 80, 55, 20,
                0, 0, 0, 0, 0],
        "Energy": [2300, 2200, 2100, 2300, 2000, 1900, 2200, 2300, 2100,
                   2200, 2000, 2100, 2200, 2000, 1900,
                   3400, 3200, 3000, 1800, 2200, 2100, 1900, 2000, 2200,
                   2500, 3000, 2800, 2700, 2000,
                   0, 0, 0, 0, 0],
        "Fibre": [25, 32, 30, 28, 18, 22, 15, 16, 20,
                  18, 30, 20, 25, 28, 22,
                  2, 7, 6, 10, 12, 10, 14, 12, 13,
                  12, 1, 1, 3, 2,
                  0, 0, 0, 0, 0],
        "Calcium": [1.5, 0.5, 0.45, 0.6, 1.2, 1.0, 1.8, 1.7, 1.5,
                    1.2, 0.4, 1.0, 0.6, 0.5, 1.3,
                    0.02, 0.3, 0.25, 0.1, 0.2, 0.15, 0.1, 0.1, 0.2,
                    0.3, 5.0, 0.5, 0.4, 25.0,
                    38.0, 0, 0, 0, 0],
        "Cost": [80, 50, 45, 55, 70, 65, 90, 85, 88,
                 75, 40, 60, 58, 50, 60,
                 120, 150, 130, 90, 100, 110, 45, 55, 65,
                 140, 200, 170, 180, 160,
                 50, 30, 500, 500, 400],
    }
    df = pd.DataFrame(data).set_index("Ingredient")
    st.session_state.ingredient_data = df.copy()
else:
    df = st.session_state.ingredient_data

# --- Sidebar ---
st.sidebar.header("🐰 Nutrient Requirements")
cp_req = st.sidebar.slider("Crude Protein (%)", 10, 50, 16)
energy_req = st.sidebar.slider("Energy (Kcal/kg)", 1500, 3500, 2500)
fibre_req = st.sidebar.slider("Fibre (%)", 5, 40, 12)
calcium_req = st.sidebar.slider("Calcium (%)", 0.1, 5.0, 0.5)

st.sidebar.header("🐰 Ration Type")
ration_type = st.sidebar.selectbox(
    "Select feed ration type:",
    ["Mixed (Fodder + Concentrate)", "Concentrate only", "Fodder only"]
)

# --- Filter ingredients ---
if ration_type == "Concentrate only":
    ingredients = df[df['Category'] == "Concentrate"]
elif ration_type == "Fodder only":
    ingredients = df[df['Category'] == "Fodder"]
else:
    ingredients = df[df['Category'].isin(["Fodder", "Concentrate"])]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Ingredients", "📈 Performance"])

# --- Optimizer ---
with tab1:
    st.header("🧪 Feed Mix Optimization")
    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}
    model += lpSum([vars[i] * ingredients.loc[i, 'Cost'] for i in ingredients.index])
    model += lpSum([vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index]) >= cp_req
    model += lpSum([vars[i] * ingredients.loc[i, 'Energy'] for i in ingredients.index]) >= energy_req
    model += lpSum([vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index]) >= fibre_req
    model += lpSum([vars[i] * ingredients.loc[i, 'Calcium'] for i in ingredients.index]) >= calcium_req
    model += lpSum([vars[i] for i in ingredients.index]) == 1

    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.success("Optimal feed formulation found!")
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.0001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * ingredients.loc[result_df.index, 'Cost']
        st.dataframe(result_df.style.format({"Proportion (kg)": "{:.3f}", "Cost (₦)": "₦{:.2f}"}))
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
        st.plotly_chart(px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Ingredient Distribution'))
    else:
        st.error("⚠️ No feasible solution found.")

# --- Ingredients ---
with tab2:
    st.header("📝 Edit Ingredients")
    editable_df = df.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    st.subheader("📤 Upload New Ingredients CSV")
    uploaded_file = st.file_uploader("Upload CSV with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost", type=["csv"])
    if uploaded_file:
        new_ingredients = pd.read_csv(uploaded_file)
        required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
        if required_cols.issubset(new_ingredients.columns):
            new_ingredients = new_ingredients.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, new_ingredients])
            df = st.session_state.ingredient_data.copy()
            st.success(f"Successfully added {len(new_ingredients)} new ingredients.")
        else:
            st.error("CSV must contain the correct columns.")

    if st.button("💾 Save Changes"):
        if edited_df["Ingredient"].is_unique and edited_df["Ingredient"].notnull().all():
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            df = st.session_state.ingredient_data.copy()
            st.success("Ingredients updated successfully!")
        else:
            st.error("All ingredient names must be unique and non-empty.")

# --- Predictor ---
with tab3:
    st.header("📈 Performance Prediction")
    if LpStatus[model.status] == "Optimal":
        proportions = np.array([vars[i].varValue for i in ingredients.index])
        cp_vals = np.array([ingredients.loc[i, "CP"] for i in ingredients.index])
        energy_vals = np.array([ingredients.loc[i, "Energy"] for i in ingredients.index])
        feed_cp = np.dot(proportions, cp_vals)
        feed_energy = np.dot(proportions, energy_vals)
        weight_gain = 8 + 0.02 * feed_cp + 0.0015 * feed_energy
        st.metric("Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
        st.info("Note: Prediction is simulated. Train model with real data for accuracy.")
    else:
        st.warning("Run the optimizer to get predictions.")

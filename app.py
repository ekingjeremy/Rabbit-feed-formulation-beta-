import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed AI", layout="wide")
st.title("🐰 Rabbit Feed Formulation + AI-Powered Performance Predictor")

# Default Ingredient Data
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame([
        # Concentrates
        ["Maize", "Concentrate", 9, 3400, 2, 0.02, 120],
        ["Soybean Meal", "Concentrate", 44, 3200, 7, 0.3, 150],
        ["Groundnut Cake", "Concentrate", 45, 3000, 6, 0.25, 130],
        ["Wheat Bran", "Concentrate", 15, 1800, 10, 0.1, 90],
        ["Fish Meal", "Concentrate", 60, 2900, 1, 5.0, 300],

        # Fodders
        ["Alfalfa", "Fodder", 18, 2300, 25, 1.5, 80],
        ["Elephant Grass", "Fodder", 8, 2100, 32, 0.6, 60],
        ["Sweet Potato Vine", "Fodder", 14, 2000, 20, 0.8, 55],
        ["Moringa Leaves", "Fodder", 27, 2600, 17, 2.0, 90],
        ["Pumpkin Leaves", "Fodder", 13, 1800, 15, 1.0, 70],
    ], columns=["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"])

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Manage Ingredients", "📈 AI Predictor"])

df = st.session_state.ingredient_data.set_index("Ingredient")

# Optimizer Tab
with tab1:
    st.sidebar.header("Set Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 30, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 35, 15)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 3.0, 0.7)

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
        result_df["Category"] = df.loc[result_df.index, "Category"]
        st.dataframe(result_df)
        st.success(f"✅ Total Cost per kg Feed: ₦{value(model.objective):.2f}")

        fig = px.pie(result_df.reset_index(), values='Proportion (kg)', names='index', color='Category',
                     title='Feed Ingredient Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No feasible solution found with current nutrient settings.")

# Manage Ingredients Tab
with tab2:
    st.subheader("📋 Ingredient Editor")
    editable_df = st.session_state.ingredient_data.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True, key="edit_ingredient")

    uploaded_file = st.file_uploader("📤 Upload New Ingredients CSV", type="csv")
    if uploaded_file is not None:
        new_ingredients = pd.read_csv(uploaded_file)
        if set(["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(new_ingredients.columns):
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, new_ingredients], ignore_index=True).drop_duplicates("Ingredient")
            st.success("✅ Uploaded ingredients added.")
        else:
            st.error("❌ CSV must contain the correct column headers.")

    if st.button("💾 Save Changes"):
        st.session_state.ingredient_data = edited_df.set_index("Ingredient")
        st.success("✅ Ingredient list updated.")

# Predictor Tab
with tab3:
    st.subheader("📈 Predicted Rabbit Weight Gain")
    if LpStatus[model.status] == "Optimal":
        protein = lpSum([vars[i].varValue * df.loc[i, "CP"] for i in df.index]).value()
        energy_val = lpSum([vars[i].varValue * df.loc[i, "Energy"] for i in df.index]).value()
        fiber_val = lpSum([vars[i].varValue * df.loc[i, "Fibre"] for i in df.index]).value()
        calcium_val = lpSum([vars[i].varValue * df.loc[i, "Calcium"] for i in df.index]).value()

        gain = 10 + (0.02 * protein) + (0.0015 * energy_val) - (0.01 * fiber_val) + (0.05 * calcium_val)

        st.metric("🐇 Estimated Weight Gain", f"{gain:.2f} g/day")
        st.caption("Estimate based on nutrient contribution from optimized feed.")
    else:
        st.warning("⚠️ Please solve a valid feed optimization to get predictions.")

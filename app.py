import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# Default ingredient data with categories
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "Category": ["Fodder", "Concentrate", "Concentrate", "Fodder", "Concentrate"],
        "CP": [18, 9, 44, 15, 45],
        "Energy": [2300, 3400, 3200, 1800, 3000],
        "Fibre": [25, 2, 7, 10, 6],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25],
        "Cost": [80, 120, 150, 90, 130]
    }, index=["Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake"])

st.sidebar.title("Settings")
ration_type = st.sidebar.radio("Ration Type", ["Mixed", "Concentrate Only", "Fodder Only"])

cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

# Filter data based on ration type
df = st.session_state.ingredient_data
if ration_type == "Concentrate Only":
    df = df[df["Category"] == "Concentrate"]
elif ration_type == "Fodder Only":
    df = df[df["Category"] == "Fodder"]

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Performance Predictor"])

# Optimizer
with tab1:
    st.subheader("🧪 Optimized Feed Mix")

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
        results = {i: vars[i].varValue for i in df.index if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, 'Cost']
        st.dataframe(result_df)
        st.success(f"Total Cost/kg Feed: ₦{value(model.objective):.2f}")

        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Ingredient Composition')
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ No feasible solution found with current nutrient settings.")

# Ingredient Editor
with tab2:
    st.subheader("✍️ Ingredient Editor")
    editable_df = df.reset_index().rename(columns={"index": "Ingredient"})
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            edited_df = edited_df.dropna(subset=["Ingredient"])
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Please ensure all ingredients are named uniquely and not empty.")

    if LpStatus[model.status] == "Optimal" and st.button("🧹 Remove unused ingredients"):
        used = [i for i in df.index if vars[i].varValue > 0]
        st.session_state.ingredient_data = df.loc[used]
        st.success("Unused ingredients removed.")

    st.markdown("---")
    st.subheader("📤 Upload New Ingredients")
    upload = st.file_uploader("Upload CSV with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost", type=["csv"])
    if upload:
        new_data = pd.read_csv(upload)
        if set(["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(new_data.columns):
            new_data = new_data.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, new_data])
            st.success("✅ Ingredients uploaded and added.")
        else:
            st.error("❌ Invalid format. Please ensure all required columns are present.")

# Performance Predictor
with tab3:
    st.subheader("📈 Performance Predictor")

    if LpStatus[model.status] == "Optimal":
        protein = sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)
        energy_val = sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)
        gain = 10 + 0.015 * protein + 0.002 * energy_val

        st.metric("📊 Expected Weight Gain", f"{gain:.1f} g/day")
        st.info("This is a simulated estimate. You can train a model on real rabbit performance data for more accuracy.")
    else:
        st.warning("⚠️ Prediction unavailable. Run a successful optimization first.")

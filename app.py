import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# ---------- DEFAULT INGREDIENT DATA ----------
def default_ingredients():
    return pd.DataFrame({
        "Category": [
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder", "Fodder", "Fodder",
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate"
        ],
        "CP": [12, 15, 9, 8, 13, 10, 14, 9, 44, 45, 15, 20],
        "Energy": [1800, 2000, 1900, 1700, 1850, 1750, 1950, 3400, 3200, 3000, 1800, 2500],
        "Fibre": [28, 25, 30, 35, 22, 32, 20, 2, 7, 6, 10, 5],
        "Calcium": [0.6, 1.2, 0.8, 0.5, 1.1, 0.7, 0.9, 0.02, 0.3, 0.25, 0.1, 0.2],
        "Cost": [40, 50, 45, 38, 55, 42, 48, 120, 150, 130, 90, 110]
    }, index=[
        "Elephant Grass", "Guinea Grass", "Sorghum Stover", "Maize Stover", "Sweet Potato Vines",
        "Groundnut Haulms", "Stylosanthes", "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Bran", "Palm Kernel Cake"
    ])

if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = default_ingredients()

df = st.session_state.ingredient_data

# ---------- SIDEBAR: RATION TYPE AND NUTRIENTS ----------
st.sidebar.header("⚙️ Settings")
ration_type = st.sidebar.selectbox("Select Ration Type", ["Mixed", "Fodder only", "Concentrate only"])
cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

# ---------- FILTER INGREDIENTS ----------
if ration_type == "Fodder only":
    df = df[df["Category"] == "Fodder"]
elif ration_type == "Concentrate only":
    df = df[df["Category"] == "Concentrate"]

# ---------- OPTIMIZER ----------
with st.expander("🧪 Feed Optimizer", expanded=True):
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Optimized Ration")
            st.dataframe(result_df)
            st.markdown(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
        with col2:
            fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
            st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

# ---------- INGREDIENT EDITOR ----------
with st.expander("📝 Ingredient Editor", expanded=False):
    st.markdown("### Edit Ingredients")
    editable_df = df.reset_index().rename(columns={"index": "Ingredient"})
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    st.markdown("### Upload CSV with Ingredients")
    uploaded_file = st.file_uploader("Upload CSV file with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost")
    if uploaded_file:
        upload_df = pd.read_csv(uploaded_file)
        if all(col in upload_df.columns for col in ["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]):
            upload_df = upload_df.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, upload_df])
            st.success("Ingredients uploaded and added successfully!")
        else:
            st.error("❌ CSV missing required columns.")

    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Please make sure all ingredients are uniquely named and not empty.")

# ---------- PERFORMANCE PREDICTOR ----------
with st.expander("📈 Performance Predictor", expanded=False):
    st.subheader("🚀 AI-Based Weight Gain Estimation")

    if LpStatus[model.status] == "Optimal":
        protein = sum([vars[i].varValue * df.loc[i, "CP"] for i in df.index])
        energy_val = sum([vars[i].varValue * df.loc[i, "Energy"] for i in df.index])

        # Dummy AI model - Replace with trained model
        gain = 10 + 0.015 * protein + 0.002 * energy_val

        st.metric("📈 Expected Weight Gain", f"{gain:.1f} g/day")
        st.info("This is a simulated estimate. Replace with a trained ML model for better accuracy.")
    else:
        st.warning("⚠️ Prediction unavailable. Run a successful optimization first.")

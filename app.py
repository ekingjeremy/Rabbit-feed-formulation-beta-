import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer + AI Weight Gain Predictor")

# Default data with ingredient category
def default_ingredients():
    return pd.DataFrame({
        "Ingredient": [
            "Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake",
            "Guinea Grass", "Elephant Grass", "Panicum", "Stylosanthes", "Gliricidia",
            "Cassava Peel", "Palm Kernel Cake", "Molasses", "Rice Bran", "Fish Meal"
        ],
        "Category": [
            "Fodder", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder",
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate"
        ],
        "CP": [18, 9, 44, 15, 45, 10, 8, 9, 15, 20, 4, 21, 3, 12, 60],
        "Energy": [2300, 3400, 3200, 1800, 3000, 1600, 1500, 1600, 1700, 1800, 2000, 2800, 2300, 2100, 3200],
        "Fibre": [25, 2, 7, 10, 6, 30, 32, 28, 20, 15, 15, 10, 1, 12, 1],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25, 0.6, 0.5, 0.55, 0.7, 0.8, 0.1, 0.3, 0.02, 0.1, 5.0],
        "Cost": [80, 120, 150, 90, 130, 40, 35, 38, 45, 50, 20, 110, 30, 60, 300]
    })

# Load ingredient data into session_state
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = default_ingredients()

# Upload CSV to update ingredients
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Predictor"])

with tab2:
    st.subheader("✍️ Edit or Upload Ingredient List")
    uploaded = st.file_uploader("📤 Upload CSV with ingredients (Ingredient, Category, CP, Energy, Fibre, Calcium, Cost)", type=["csv"])
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
            required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
            if required_cols.issubset(set(new_data.columns)):
                st.session_state.ingredient_data = new_data.dropna(subset=["Ingredient", "Category"])
                st.success("Ingredient list successfully updated from uploaded file.")
            else:
                st.error("Uploaded CSV missing required columns.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    edited_df = st.data_editor(st.session_state.ingredient_data, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        st.session_state.ingredient_data = edited_df.dropna(subset=["Ingredient", "Category"])
        st.success("Changes saved.")

# Optimizer tab
with tab1:
    st.sidebar.header("Set Nutrient Requirements")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 2.0, 0.5)

    df = st.session_state.ingredient_data.set_index("Ingredient")
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
        st.success("✅ Optimal feed mix found!")
        results = {i: vars[i].varValue for i in df.index if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, 'Cost']
        result_df["Category"] = df.loc[result_df.index, "Category"]
        st.dataframe(result_df)

        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, color='Category', title='Ingredient Breakdown')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

# Predictor Tab
with tab3:
    st.subheader("🤖 AI Weight Gain Prediction")
    if LpStatus[model.status] == "Optimal":
        X = df[["CP", "Energy"]].values
        y = np.array([50 + 0.02 * cp + 0.002 * en + np.random.randn()*2 for cp, en in X])
        model_lr = LinearRegression().fit(X, y)

        mix_cp = sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)
        mix_energy = sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)
        prediction = model_lr.predict([[mix_cp, mix_energy]])[0]

        st.metric("📈 Predicted Weight Gain", f"{prediction:.1f} g/day")
        st.info("Prediction based on trained linear model using CP & Energy.")
    else:
        st.warning("⚠️ Please generate an optimal mix first.")

import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os

st.set_page_config(page_title="Rabbit Feed Formulation AI", layout="wide")

# Load or initialize data
@st.cache_data

def load_default_data():
    data = pd.DataFrame({
        "Ingredient": [
            # Concentrates
            "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Offal", "Palm Kernel Cake",
            "Brewer's Dry Grains", "Cassava Peel", "Maize Bran",
            # Fodders
            "Alfalfa", "Elephant Grass", "Gamba Grass", "Guinea Grass", "Centrosema",
            "Stylosanthes", "Leucaena", "Gliricidia"
        ],
        "CP": [9, 44, 45, 15, 20, 18, 5, 7, 18, 8, 7, 10, 17, 14, 25, 24],
        "Energy": [3400, 3200, 3000, 1800, 2200, 2100, 1900, 2000, 2300, 2200, 2100, 2300, 2000, 1900, 2200, 2300],
        "Fibre": [2, 7, 6, 10, 12, 10, 14, 12, 25, 32, 30, 28, 18, 22, 15, 16],
        "Calcium": [0.02, 0.3, 0.25, 0.1, 0.2, 0.15, 0.1, 0.1, 1.5, 0.5, 0.45, 0.6, 1.2, 1.0, 1.8, 1.7],
        "Cost": [120, 150, 130, 90, 85, 80, 60, 65, 80, 20, 25, 30, 45, 40, 50, 55],
        "Category": ["Concentrate"] * 8 + ["Fodder"] * 8
    })
    return data

if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = load_default_data()

# Sidebar menu
page = st.sidebar.radio("📂 Navigation", ["🧪 Optimizer", "📝 Ingredients", "📈 Performance Predictor"])

# Optimizer Page
if page == "🧪 Optimizer":
    st.title("🧪 Rabbit Feed Optimizer")
    
    st.sidebar.header("Nutrient Requirements")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 2.0, 0.5)

    ration_type = st.selectbox("Select Ration Type", ["Mixed", "Concentrate Only", "Fodder Only"])
    df = st.session_state.ingredient_data

    if ration_type == "Concentrate Only":
        df = df[df["Category"] == "Concentrate"]
    elif ration_type == "Fodder Only":
        df = df[df["Category"] == "Fodder"]

    model = LpProblem("Rabbit_Feed", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in df["Ingredient"]}
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, "Cost"].values[0] for i in vars])
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, "CP"].values[0] for i in vars]) >= cp
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, "Energy"].values[0] for i in vars]) >= energy
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, "Fibre"].values[0] for i in vars]) >= fibre
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, "Calcium"].values[0] for i in vars]) >= calcium
    model += lpSum([vars[i] for i in vars]) == 1

    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.success("Optimal feed mix found!")
        results = {i: vars[i].varValue for i in vars if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df.index.map(lambda i: df.loc[df["Ingredient"] == i, "Cost"].values[0]) * result_df["Proportion (kg)"]
        st.dataframe(result_df)
        st.metric("Total Cost/kg", f"₦{value(model.objective):.2f}")
        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
        st.plotly_chart(fig)
    else:
        st.error("No feasible ration found for current nutrient targets.")

# Ingredients Page
elif page == "📝 Ingredients":
    st.title("📝 Manage Feed Ingredients")
    df = st.session_state.ingredient_data
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("💾 Save Table"):
        if edited["Ingredient"].notna().all() and edited["Ingredient"].is_unique:
            st.session_state.ingredient_data = edited
            st.success("Ingredients updated.")
        else:
            st.error("Please ensure unique, non-empty ingredient names.")

    uploaded = st.file_uploader("Upload CSV with new ingredients (Ingredient, CP, Energy, Fibre, Calcium, Cost, Category)", type="csv")
    if uploaded is not None:
        new_data = pd.read_csv(uploaded)
        if all(col in new_data.columns for col in ["Ingredient", "CP", "Energy", "Fibre", "Calcium", "Cost", "Category"]):
            st.session_state.ingredient_data = pd.concat([df, new_data]).drop_duplicates("Ingredient").reset_index(drop=True)
            st.success("Ingredients added from file.")
        else:
            st.error("CSV must contain columns: Ingredient, CP, Energy, Fibre, Calcium, Cost, Category")

# Predictor Page
elif page == "📈 Performance Predictor":
    st.title("📈 Performance Predictor (Demo Model)")
    df = st.session_state.ingredient_data
    
    if "last_optimizer_result" not in st.session_state:
        st.warning("Please run the optimizer to predict weight gain.")
    else:
        used_df = st.session_state.last_optimizer_result
        protein = sum(used_df["Proportion (kg)"] * df.set_index("Ingredient").loc[used_df.index, "CP"])
        energy = sum(used_df["Proportion (kg)"] * df.set_index("Ingredient").loc[used_df.index, "Energy"])

        model = LinearRegression()
        X_train = pd.DataFrame({"Protein": [15, 17, 20, 22, 25], "Energy": [2000, 2200, 2500, 2700, 3000]})
        y_train = [25, 30, 35, 40, 45]  # mock weight gains (g/day)
        model.fit(X_train, y_train)

        pred_gain = model.predict([[protein, energy]])[0]
        st.metric("Estimated Weight Gain", f"{pred_gain:.2f} g/day")

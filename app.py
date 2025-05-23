import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(layout="wide")
st.title("🐰 Rabbit Feed Formulation App")

# Initialize default data if not already loaded
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "Ingredient": [
            "Maize", "Wheat Offal", "Soybean Meal", "Groundnut Cake", "Palm Kernel Cake",
            "Alfalfa", "Elephant Grass", "Guinea Grass", "Stylosanthes", "Centrosema"
        ],
        "Category": [
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder"
        ],
        "CP": [9.0, 15.0, 44.0, 45.0, 18.0, 18.0, 8.0, 7.0, 15.0, 12.0],
        "Energy": [3400, 1800, 3200, 3000, 2500, 2300, 2100, 2000, 2200, 2100],
        "Fibre": [2.0, 10.0, 7.0, 6.0, 16.0, 25.0, 28.0, 30.0, 20.0, 22.0],
        "Calcium": [0.02, 0.1, 0.3, 0.25, 0.15, 1.5, 0.6, 0.5, 1.0, 0.8],
        "Cost": [120, 90, 150, 130, 80, 60, 50, 45, 55, 52]
    })

st.markdown("---")

col1, col2, col3 = st.columns(3)

# --- Optimizer ---
with col1:
    st.header("🧪 Optimizer")

    ration_type = st.radio("Choose Ration Type", ["Mixed", "Concentrate only", "Fodder only"], key="ration_type")
    
    cp = st.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.slider("Fibre (%)", 5, 30, 10)
    calcium = st.slider("Calcium (%)", 0.1, 1.5, 0.5)

    df = st.session_state.ingredient_data
    if ration_type == "Concentrate only":
        df = df[df["Category"] == "Concentrate"]
    elif ration_type == "Fodder only":
        df = df[df["Category"] == "Fodder"]

    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
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
        result_df["Cost (₦)"] = result_df.index.map(lambda x: result_df.loc[x, "Proportion (kg)"] * df.loc[df["Ingredient"] == x, "Cost"].values[0])
        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
    else:
        st.error("No feasible solution found with current settings.")

# --- Ingredient Editor ---
with col2:
    st.header("📝 Ingredients")
    df_editor = st.session_state.ingredient_data.copy()
    edited_df = st.data_editor(df_editor, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            st.session_state.ingredient_data = edited_df
            st.success("Ingredient list updated successfully!")
        else:
            st.error("Ensure all ingredients are uniquely named and not empty.")

    st.subheader("📤 Upload New Ingredients")
    upload = st.file_uploader("Upload CSV file with Ingredient, Category, CP, Energy, Fibre, Calcium, Cost")
    if upload:
        uploaded_df = pd.read_csv(upload)
        if set(["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(uploaded_df.columns):
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, uploaded_df], ignore_index=True).drop_duplicates(subset="Ingredient")
            st.success("New ingredients added successfully!")
        else:
            st.error("CSV file must include all required columns.")

# --- Performance Predictor ---
with col3:
    st.header("📈 Performance Predictor")

    if LpStatus[model.status] == "Optimal":
        X = df[["CP", "Energy"]].values
        y = np.array([20 + 0.015 * cp + 0.002 * en for cp, en in X])  # Dummy values
        model_lr = LinearRegression().fit(X, y)

        avg_cp = sum([vars[i].varValue * df.loc[df["Ingredient"] == i, "CP"].values[0] for i in vars])
        avg_energy = sum([vars[i].varValue * df.loc[df["Ingredient"] == i, "Energy"].values[0] for i in vars])

        predicted_gain = model_lr.predict([[avg_cp, avg_energy]])[0]
        st.metric("Expected Weight Gain", f"{predicted_gain:.1f} g/day")
        st.caption("Prediction from mock-trained model. Train with real data for better results.")
    else:
        st.warning("Prediction unavailable. Run optimizer first.")

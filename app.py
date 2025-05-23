import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation App")

# Sample dataset with concentrate and fodder
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "Ingredient": [
            "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Offal", "Fish Meal",
            "Alfalfa", "Elephant Grass", "Guinea Grass", "Stylosanthes", "Leucaena"
        ],
        "Category": [
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder"
        ],
        "CP": [9, 44, 45, 15, 60, 18, 12, 10, 20, 25],
        "Energy": [3400, 3200, 3000, 1800, 2800, 2300, 2200, 2000, 2100, 2300],
        "Fibre": [2, 7, 6, 10, 1, 25, 28, 30, 18, 20],
        "Calcium": [0.02, 0.3, 0.25, 0.1, 5, 1.5, 1.2, 1.3, 1.0, 1.1],
        "Cost": [120, 150, 130, 90, 250, 80, 60, 50, 70, 65]
    })

df = st.session_state.ingredient_data

# Layout
optimizer_col, editor_col, predictor_col = st.columns([1, 1, 1])

with optimizer_col:
    st.header("🧪 Optimizer")

    ration_type = st.selectbox("Select Ration Type", ["Mixed", "Concentrate only", "Fodder only"])

    st.markdown("### Nutrient Requirements (per kg feed)")
    cp = st.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.slider("Fibre (%)", 5, 30, 10)
    calcium = st.slider("Calcium (%)", 0.1, 2.0, 0.5)

    # Filter ingredients
    if ration_type == "Concentrate only":
        df_filtered = df[df["Category"] == "Concentrate"]
    elif ration_type == "Fodder only":
        df_filtered = df[df["Category"] == "Fodder"]
    else:
        df_filtered = df

    # LP Model
    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in df_filtered["Ingredient"]}
    model += lpSum([vars[i] * df_filtered.loc[df_filtered["Ingredient"] == i, 'Cost'].values[0] for i in vars])
    model += lpSum([vars[i] * df_filtered.loc[df_filtered["Ingredient"] == i, 'CP'].values[0] for i in vars]) >= cp
    model += lpSum([vars[i] * df_filtered.loc[df_filtered["Ingredient"] == i, 'Energy'].values[0] for i in vars]) >= energy
    model += lpSum([vars[i] * df_filtered.loc[df_filtered["Ingredient"] == i, 'Fibre'].values[0] for i in vars]) >= fibre
    model += lpSum([vars[i] * df_filtered.loc[df_filtered["Ingredient"] == i, 'Calcium'].values[0] for i in vars]) >= calcium
    model += lpSum([vars[i] for i in vars]) == 1

    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.success("✅ Optimal ration found!")
        results = {i: vars[i].varValue for i in vars if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df_filtered.set_index("Ingredient").loc[result_df.index, 'Cost']
        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
    else:
        st.error("⚠️ No feasible solution found with current settings.")

with editor_col:
    st.header("📝 Ingredient Editor")

    editable_df = df.reset_index(drop=True)
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Save Ingredients"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            st.session_state.ingredient_data = edited_df
            st.success("Ingredients updated successfully.")
        else:
            st.error("Each ingredient must have a unique and non-empty name.")

    uploaded_file = st.file_uploader("Upload New Ingredients (CSV)", type="csv")
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            if set(["Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(new_data.columns):
                st.session_state.ingredient_data = pd.concat([df, new_data], ignore_index=True).drop_duplicates("Ingredient")
                st.success("New ingredients uploaded successfully!")
            else:
                st.error("CSV must contain columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with predictor_col:
    st.header("📈 Performance Predictor")
    if LpStatus[model.status] == "Optimal":
        protein_val = lpSum([vars[i].varValue * df_filtered.set_index("Ingredient").loc[i, "CP"] for i in vars]).value()
        energy_val = lpSum([vars[i].varValue * df_filtered.set_index("Ingredient").loc[i, "Energy"] for i in vars]).value()

        # Dummy model (replace with trained model for production)
        X_train = pd.DataFrame({"CP": [16, 18, 20], "Energy": [2400, 2600, 2800]})
        y_train = [20, 25, 30]
        model_lr = LinearRegression().fit(X_train, y_train)
        gain = model_lr.predict([[protein_val, energy_val]])[0]

        st.metric("📊 Predicted Weight Gain", f"{gain:.2f} g/day")
        st.caption("This is an estimate. Train with real data for accurate results.")
    else:
        st.info("Run the optimizer to see predictions.")

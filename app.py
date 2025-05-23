import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from io import StringIO

st.set_page_config(page_title="🐰 Rabbit Feed Formulator", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# --- DEFAULT INGREDIENT DATA ---

DEFAULT_INGREDIENTS = pd.DataFrame({
    "Ingredient": [
        # Concentrates
        "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Bran", "Cottonseed Cake",
        # Fodders
        "Alfalfa", "Guinea Grass", "Napier Grass", "Stylo", "Elephant Grass",
        "Sorghum Stover", "Cassava Leaves", "Leucaena"
    ],
    "Category": [
        "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
        "Fodder", "Fodder", "Fodder", "Fodder", "Fodder",
        "Fodder", "Fodder", "Fodder"
    ],
    "CP": [
        9, 44, 45, 15, 40,
        18, 10, 9, 20, 14,
        7, 25, 23
    ],
    "Energy": [
        3400, 3200, 3000, 1800, 3100,
        2300, 2200, 2100, 2400, 2000,
        1800, 2300, 2100
    ],
    "Fibre": [
        2, 7, 6, 10, 8,
        25, 30, 28, 22, 26,
        30, 20, 24
    ],
    "Calcium": [
        0.02, 0.3, 0.25, 0.1, 0.4,
        1.5, 1.2, 1.1, 1.3, 1.4,
        1.0, 1.2, 1.3
    ],
    "Cost": [
        120, 150, 130, 90, 140,
        80, 70, 65, 85, 75,
        60, 85, 90
    ],
}).set_index("Ingredient")

# Load ingredient data from session state or default
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = DEFAULT_INGREDIENTS.copy()

df = st.session_state.ingredient_data

# --- Train mock regression model on dummy data for performance prediction ---

def train_mock_model():
    # Dummy data simulating feed nutrient content and weight gain (g/day)
    data = pd.DataFrame({
        "CP": [14, 16, 18, 20, 22, 24, 26],
        "Energy": [2100, 2300, 2500, 2700, 2900, 3100, 3300],
        "Fibre": [12, 11, 10, 9, 8, 7, 6],
        "Calcium": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "WeightGain": [25, 30, 35, 38, 40, 42, 45]  # g/day
    })
    X = data[["CP", "Energy", "Fibre", "Calcium"]]
    y = data["WeightGain"]
    model = LinearRegression()
    model.fit(X, y)
    return model

if "perf_model" not in st.session_state:
    st.session_state.perf_model = train_mock_model()

perf_model = st.session_state.perf_model

# --- App tabs ---
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Performance Predictor"])

# --- OPTIMIZER TAB ---
with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp_req = st.sidebar.slider("Crude Protein (%)", 10, 30, 16)
    energy_req = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre_req = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium_req = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

    ration_type = st.selectbox("Select Ration Type", options=["Mixed (Concentrate + Fodder)", "Concentrate Only", "Fodder Only"])

    if ration_type == "Mixed (Concentrate + Fodder)":
        df_opt = df.copy()
    elif ration_type == "Concentrate Only":
        df_opt = df[df["Category"] == "Concentrate"]
    else:
        df_opt = df[df["Category"] == "Fodder"]

    if df_opt.empty:
        st.error("No ingredients available for selected ration type. Please add ingredients.")
    else:
        model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
        vars = {i: LpVariable(i, lowBound=0) for i in df_opt.index}

        # Objective: minimize cost
        model += lpSum([vars[i] * df_opt.loc[i, "Cost"] for i in df_opt.index])

        # Constraints: nutrients must meet or exceed requirements
        model += lpSum([vars[i] * df_opt.loc[i, "CP"] for i in df_opt.index]) >= cp_req
        model += lpSum([vars[i] * df_opt.loc[i, "Energy"] for i in df_opt.index]) >= energy_req
        model += lpSum([vars[i] * df_opt.loc[i, "Fibre"] for i in df_opt.index]) >= fibre_req
        model += lpSum([vars[i] * df_opt.loc[i, "Calcium"] for i in df_opt.index]) >= calcium_req

        # Sum of proportions = 1 (100%)
        model += lpSum([vars[i] for i in df_opt.index]) == 1

        model.solve()

        if LpStatus[model.status] == "Optimal":
            st.subheader("📊 Optimized Feed Mix")
            results = {i: vars[i].varValue for i in df_opt.index if vars[i].varValue > 0}
            result_df = pd.DataFrame.from_dict(results, orient="index", columns=["Proportion (kg)"])
            result_df["Category"] = df_opt.loc[result_df.index, "Category"]
            result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df_opt.loc[result_df.index, "Cost"]
            st.dataframe(result_df.style.format({"Proportion (kg)": "{:.3f}", "Cost (₦)": "₦{:.2f}"}))

            st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

            # Pie chart by category
            fig = px.pie(result_df, values="Proportion (kg)", names=result_df.index,
                         color=result_df["Category"], title="Feed Ingredient Distribution by Ingredient")
            st.plotly_chart(fig)
        else:
            st.error("⚠️ No feasible solution found with current nutrient settings.")

# --- EDIT INGREDIENTS TAB ---
with tab2:
    st.subheader("✍️ Modify Ingredients Table")

    editable_df = df.reset_index()
    uploaded_file = st.file_uploader("📂 Upload CSV with new ingredients", type=["csv"])
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            # Validate required columns
            required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
            if not required_cols.issubset(set(new_data.columns)):
                st.error(f"CSV must contain columns: {required_cols}")
            else:
                # Remove duplicates & append new data
                existing = df.reset_index()
                combined = pd.concat([existing, new_data], ignore_index=True)
                combined.drop_duplicates(subset="Ingredient", keep="last", inplace=True)
                combined.set_index("Ingredient", inplace=True)
                st.session_state.ingredient_data = combined
                st.success("Ingredients uploaded and merged successfully.")
                df = st.session_state.ingredient_data
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    edited_df = st.data_editor(df.reset_index(), num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            edited_df = edited_df.dropna(subset=["Ingredient"])
            edited_df.set_index("Ingredient", inplace=True)
            st.session_state.ingredient_data = edited_df
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Ensure all ingredients have unique, non-empty names.")

    # Remove unused ingredients
    if st.button("🧹 Remove unused ingredients"):
        # Remove ingredients with zero in last optimization? Let's just keep all for safety here
        st.info("Remove unused ingredients feature requires last optimization results; not implemented yet.")

# --- PERFORMANCE PREDICTOR TAB ---
with tab3:
    st.subheader("🚀 Performance Predictor")

    if 'model' in locals() and LpStatus[model.status] == "Optimal":
        # Calculate nutrient levels of optimized feed
        opt_cp = sum(vars[i].varValue * df.loc[i, "CP"] for i in vars)
        opt_energy = sum(vars[i].varValue * df.loc[i, "Energy"] for i in vars)
        opt_fibre = sum(vars[i].varValue * df.loc[i, "Fibre"] for i in vars)
        opt_calcium = sum(vars[i].varValue * df.loc[i, "Calcium"] for i in vars)

        # Predict weight gain in g/day using regression model
        pred_gain = perf_model.predict(np.array([[opt_cp, opt_energy, opt_fibre, opt_calcium]]))[0]

        st.metric("📈 Expected Weight Gain (g/day)", f"{pred_gain:.1f} g/day")

        st.info(
            "Prediction based on a simple regression model trained on dummy data. "
            "For real-life accuracy, train with real growth performance datasets."
        )
    else:
        st.warning("⚠️ Run a successful optimization first to predict performance.")

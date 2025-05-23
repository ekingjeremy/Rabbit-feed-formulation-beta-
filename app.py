
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Rabbit Feed Formulator", layout="wide")
st.title("🐰 Nigerian Rabbit Feed Formulation + Predictor")

# Initial feed data if not already in session state
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "Ingredient": [
            "Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake", "Cassava Peel",
            "Sweet Potato Vine", "Guinea Grass", "Elephant Grass", "Centro", "Stylosanthes", "Leucaena"
        ],
        "Category": [
            "Fodder", "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder", "Fodder"
        ],
        "CP": [18, 9, 44, 15, 45, 3, 13, 9, 8, 17, 14, 26],
        "Energy": [2300, 3400, 3200, 1800, 3000, 2800, 2300, 2400, 2300, 2000, 2200, 2400],
        "Fibre": [25, 2, 7, 10, 6, 15, 20, 30, 35, 28, 25, 22],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25, 0.1, 1.2, 0.5, 0.7, 1.4, 1.2, 1.3],
        "Cost": [80, 120, 150, 90, 130, 50, 60, 30, 35, 40, 38, 45]
    })

df = st.session_state.ingredient_data.copy()

tab1, tab2, tab3, tab4 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📤 Upload Ingredients", "📈 Performance Predictor"])

with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in df["Ingredient"]}
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, 'Cost'].values[0] for i in df["Ingredient"]])
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, 'CP'].values[0] for i in df["Ingredient"]]) >= cp
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, 'Energy'].values[0] for i in df["Ingredient"]]) >= energy
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, 'Fibre'].values[0] for i in df["Ingredient"]]) >= fibre
    model += lpSum([vars[i] * df.loc[df["Ingredient"] == i, 'Calcium'].values[0] for i in df["Ingredient"]]) >= calcium
    model += lpSum([vars[i] for i in df["Ingredient"]]) == 1

    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.subheader("📊 Optimized Feed Mix")
        results = {i: vars[i].varValue for i in df["Ingredient"] if vars[i].varValue > 0}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Category"] = df.set_index("Ingredient").loc[result_df.index, "Category"]
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.set_index("Ingredient").loc[result_df.index, "Cost"]
        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution', color='Category')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

with tab2:
    st.subheader("✍️ Modify Ingredients Table")
    editable_df = df.copy()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        if "Ingredient" in edited_df.columns and edited_df["Ingredient"].notna().all() and edited_df["Ingredient"].is_unique:
            st.session_state.ingredient_data = edited_df
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Ensure all ingredients are uniquely named and not empty.")

with tab3:
    st.subheader("📤 Upload Ingredients CSV (with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
            if required_cols.issubset(upload_df.columns):
                upload_df = upload_df.dropna(subset=["Ingredient"])
                upload_df = upload_df.drop_duplicates(subset=["Ingredient"])
                st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, upload_df.set_index("Ingredient").reset_index()]).drop_duplicates("Ingredient", keep="last")
                st.success("✅ Ingredients uploaded successfully!")
            else:
                st.error("❌ Missing required columns in uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab4:
    st.subheader("🚀 AI Performance Predictor")
    if LpStatus[model.status] == "Optimal":
        # Dummy training data
        X_train = np.array([
            [16, 2500], [18, 2700], [20, 3000], [22, 3200], [24, 3400]
        ])
        y_train = np.array([25, 30, 35, 38, 42])  # Weight gain in g/day

        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)

        actual_cp = sum(vars[i].varValue * df.loc[df["Ingredient"] == i, "CP"].values[0] for i in df["Ingredient"])
        actual_energy = sum(vars[i].varValue * df.loc[df["Ingredient"] == i, "Energy"].values[0] for i in df["Ingredient"])
        prediction = model_lr.predict([[actual_cp, actual_energy]])[0]

        st.metric("📈 Predicted Weight Gain", f"{prediction:.1f} g/day")
    else:
        st.warning("⚠️ Run a successful optimization first.")

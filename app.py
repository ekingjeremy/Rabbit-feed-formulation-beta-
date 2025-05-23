import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Rabbit Feed Optimizer", layout="wide")
st.title("🐰 Rabbit Feed Formulation & Prediction App")

# Ingredient database with real Nigerian feedstuffs
def get_default_ingredients():
    return pd.DataFrame({
        "Category": [
            # Concentrates
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            # Fodders
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder", "Fodder", "Fodder"
        ],
        "CP": [
            # Concentrate CP
            8.5, 44.0, 45.0, 15.0, 16.0, 12.0, 18.0,
            # Fodder CP
            18.0, 10.0, 16.0, 14.0, 12.0, 9.0, 17.0
        ],
        "Energy": [
            3400, 3200, 3000, 1800, 2000, 2200, 2100,
            2300, 2000, 2200, 2100, 2000, 1800, 2500
        ],
        "Fibre": [
            2, 7, 6, 10, 8, 9, 11,
            25, 24, 20, 22, 19, 26, 21
        ],
        "Calcium": [
            0.02, 0.3, 0.25, 0.1, 0.15, 0.18, 0.2,
            1.5, 1.2, 1.3, 1.1, 1.0, 1.4, 1.6
        ],
        "Cost": [
            120, 150, 130, 90, 100, 110, 115,
            80, 70, 85, 75, 65, 60, 95
        ]
    }, index=[
        # Concentrate names
        "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Bran", "Palm Kernel Cake", "Cassava Peel", "Brewer's Dried Grains",
        # Fodder names
        "Alfalfa", "Guinea Grass", "Stylosanthes", "Centrosema", "Panicum", "Pueraria", "Gliricidia"
    ])

# Load or initialize ingredients
df = st.session_state.get("ingredient_data", get_default_ingredients())

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📤 Upload/Modify Ingredients", "📈 Performance Predictor"])

with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 2.0, 0.5)

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
        result_df["Category"] = df.loc[result_df.index, "Category"]
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, 'Cost']

        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

with tab2:
    st.subheader("📥 Upload or Edit Ingredient Table")

    uploaded = st.file_uploader("Upload Ingredient CSV", type="csv")
    if uploaded:
        uploaded_df = pd.read_csv(uploaded)
        if set(["Category", "CP", "Energy", "Fibre", "Calcium", "Cost"]).issubset(uploaded_df.columns):
            uploaded_df.set_index(uploaded_df.columns[0], inplace=True)
            df = pd.concat([df, uploaded_df])
            st.session_state.ingredient_data = df
            st.success("Uploaded ingredients added!")
        else:
            st.error("CSV must contain columns: Category, CP, Energy, Fibre, Calcium, Cost")

    editable_df = st.data_editor(df.reset_index().rename(columns={"index": "Ingredient"}), num_rows="dynamic")
    if st.button("💾 Save Table"):
        st.session_state.ingredient_data = editable_df.set_index("Ingredient")
        st.success("Ingredients updated successfully!")

with tab3:
    st.subheader("🚀 AI Performance Predictor")
    if LpStatus[model.status] == "Optimal":
        train_data = pd.DataFrame({
            "CP": [16, 18, 20, 22],
            "Energy": [2500, 2700, 2900, 3100],
            "Gain": [25, 30, 35, 40]  # g/day
        })

        reg = LinearRegression()
        reg.fit(train_data[["CP", "Energy"]], train_data["Gain"])

        input_cp = sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)
        input_energy = sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)

        predicted_gain = reg.predict([[input_cp, input_energy]])[0]
        st.metric("📈 Predicted Weight Gain", f"{predicted_gain:.1f} g/day")
        st.info("Prediction based on simple linear model. For higher accuracy, use real growth datasets.")
    else:
        st.warning("⚠️ Run the optimizer first to get prediction.")

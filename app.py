# Add ingredient editing and performance prediction (ML placeholder) features to the Streamlit app

# New app.py content
enhanced_app_py = """
import streamlit as st
import pandas as pd
import os
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px

st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# Load ingredient data or create default
file_path = "ingredients.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path).set_index("Ingredient")
else:
    df = pd.DataFrame({
        "CP": [18, 9, 44, 15, 45],
        "Energy": [2300, 3400, 3200, 1800, 3000],
        "Fibre": [25, 2, 7, 10, 6],
        "Calcium": [1.5, 0.02, 0.3, 0.1, 0.25],
        "Cost": [80, 120, 150, 90, 130]
    }, index=["Alfalfa", "Maize", "Soybean Meal", "Wheat Bran", "Groundnut Cake"])

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Performance Predictor"])

with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

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
        st.dataframe(result_df)
        st.write(f"**Total Cost/kg Feed: ₦{value(model.objective):.2f}**")

        # Pie chart
        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index, title='Feed Ingredient Distribution')
        st.plotly_chart(fig)
    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

with tab2:
    st.subheader("✍️ Modify Ingredients Table")
    edited_df = st.data_editor(df.reset_index(), num_rows="dynamic")
    if st.button("💾 Save Changes"):
        edited_df.set_index("Ingredient").to_csv(file_path)
        st.success("Ingredient list updated successfully! Please reload the app.")

with tab3:
    st.subheader("🚀 Performance Predictor (Mock-up)")

    # Simulate predicted performance based on nutrients
    if LpStatus[model.status] == "Optimal":
        protein = lpSum([vars[i].varValue * df.loc[i, "CP"] for i in df.index])
        energy_val = lpSum([vars[i].varValue * df.loc[i, "Energy"] for i in df.index])
        gain = 10 + 0.015 * protein.value() + 0.002 * energy_val.value()  # dummy logic

        st.metric("📈 Expected Weight Gain", f"{gain:.1f} g/day")
        st.info("This is a simulated estimate. For real predictions, train a model on rabbit growth data.")

    else:
        st.warning("⚠️ Prediction unavailable. Run a successful optimization first.")
"""

# Save this enhanced app.py
enhanced_path = "/mnt/data/rabbit_feed_app_enhanced"
os.makedirs(enhanced_path, exist_ok=True)

with open(f"{enhanced_path}/app.py", "w") as f:
    f.write(enhanced_app_py.strip())

with open(f"{enhanced_path}/ingredients.csv", "w") as f:
    f.write("""Ingredient,CP,Energy,Fibre,Calcium,Cost
Alfalfa,18,2300,25,1.5,80
Maize,9,3400,2,0.02,120
Soybean Meal,44,3200,7,0.3,150
Wheat Bran,15,1800,10,0.1,90
Groundnut Cake,45,3000,6,0.25,130
""")

with open(f"{enhanced_path}/requirements.txt", "w") as f:
    f.write("streamlit\npulp\nplotly\npandas")

# Zip it
zip_path = "/mnt/data/rabbit_feed_app_enhanced.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for folder, _, files in os.walk(enhanced_path):
        for file in files:
            file_path = os.path.join(folder, file)
            zipf.write(file_path, os.path.relpath(file_path, enhanced_path))

zip_path

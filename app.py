import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px

# Load ingredient data from root
df = pd.read_csv("ingredients.csv").set_index("Ingredient")

st.title("🐰 Rabbit Feed Formulation Optimizer")

# Input nutrient requirements
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

# Output
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
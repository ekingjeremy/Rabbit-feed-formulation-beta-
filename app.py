import streamlit as st
"CP": [18,8,7,10,17,14,25,24,22,20,8,18,12,10,14,9,44,45,15,20,18,5,7,14,36,60,80,55,20,0,0,0,0,0],
"Energy": [2300,2200,2100,2300,2000,1900,2200,2300,2100,2200,2000,2100,2200,2000,1900,
3400,3200,3000,1800,2200,2100,1900,2000,2200,2500,3000,2800,2700,2000,0,0,0,0,0],
"Fibre": [25,32,30,28,18,22,15,16,20,18,30,20,25,28,22,2,7,6,10,12,10,14,12,13,12,1,1,3,2,0,0,0,0,0],
"Calcium": [1.5,0.5,0.45,0.6,1.2,1.0,1.8,1.7,1.5,1.2,0.4,1.0,0.6,0.5,1.3,
0.02,0.3,0.25,0.1,0.2,0.15,0.1,0.1,0.2,0.3,5.0,0.5,0.4,25.0,38.0,0,0,0,0],
"Cost": [80,50,45,55,70,65,90,85,88,75,40,60,58,50,60,120,150,130,90,100,110,45,55,65,140,200,170,180,160,50,30,500,500,400],
}
df = pd.DataFrame(data).set_index("Ingredient")
st.session_state.ingredient_data = df.copy()
else:
df = st.session_state.ingredient_data


if ration_type == "Concentrate only":
ingredients = df[df['Category'] == "Concentrate"]
elif ration_type == "Fodder only":
ingredients = df[df['Category'] == "Fodder"]
else:
ingredients = df[df['Category'].isin(["Fodder", "Concentrate"])]


# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔬 Optimizer", "📋 Ingredients", "📈 Prediction"])


# ---------------- OPTIMIZER ----------------
with tab1:
st.header("🔬 Feed Mix Optimizer")
model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}


model += lpSum(vars[i] * ingredients.loc[i, 'Cost'] for i in ingredients.index)
model += lpSum(vars[i] for i in ingredients.index) == 1


model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) >= cp_req
model += lpSum(vars[i] * ingredients.loc[i, 'Energy'] for i in ingredients.index) >= energy_req
model += lpSum(vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index) >= fibre_req
model += lpSum(vars[i] * ingredients.loc[i, 'Calcium'] for i in ingredients.index) >= calcium_req


model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) <= cp_req + 4
model += lpSum(vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index) <= fibre_req + 8


for i in ingredients.index:
if ingredients.loc[i, 'Category'] == "Mineral":
model += vars[i] <= 0.05
elif ingredients.loc[i, 'Category'] == "Additive":
model += vars[i] <= 0.02
elif ingredients.loc[i, 'Category'] == "Concentrate":
model += vars[i] <= 0.6


model.solve()


if LpStatus[model.status] == "Optimal":
results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.0001}
result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
result_df["Cost (₦)"] = result_df["Proportion (kg)"] * ingredients.loc[result_df.index, 'Cost']
st.dataframe(result_df)
st.write(f"**💸 Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
st.plotly_chart(px.pie(result_df, values='Proportion (kg)', names=result_df.index))
else:
st.error("No feasible solution found.")


# ---------------- PREDICTION ----------------
with tab3:
st.header("📈 Growth Prediction")
if LpStatus[model.status] == "Optimal":
proportions = np.array([vars[i].varValue for i in ingredients.index])
cp_vals = np.array([ingredients.loc[i, "CP"] for i in ingredients.index])
energy_vals = np.array([ingredients.loc[i, "Energy"] for i in ingredients.index])


feed_cp = np.dot(proportions, cp_vals)
feed_energy = np.dot(proportions, energy_vals)


base_growth = breed_info["growth_rate"]
weight_gain = base_growth * (0.5 * (feed_cp / cp_req)) * (0.3 * (feed_energy / energy_req))
expected_weight = breed_info["adult_weight"] * (1 - np.exp(-0.08 * age_weeks))


st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
st.metric("⚖️ Expected Body Weight (kg)", f"{expected_weight:.2f}")

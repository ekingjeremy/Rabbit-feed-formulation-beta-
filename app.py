import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

st.set_page_config(page_title="Rabbit Feed Formulator", layout="wide")
st.title("🐇 Rabbit Feed Formulation & Growth Prediction")

# ---------------- BREED DATA ----------------
breed_data = {
    "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
    "Californian": {"adult_weight": 4.2, "growth_rate": 32, "cp_need": 16},
    "Nigerian Local": {"adult_weight": 2.8, "growth_rate": 22, "cp_need": 15},
}

# ---------------- INGREDIENT DATABASE ----------------
if "ingredient_data" not in st.session_state:
    data = {
        "Ingredient": [
            # FODDERS
            "Alfalfa","Elephant Grass","Guinea Grass","Leucaena","Gliricidia",
            "Cowpea Fodder","Cassava Leaves","Napier Grass",

            # CONCENTRATES
            "Maize","Soybean Meal","Groundnut Cake","Wheat Offal","Palm Kernel Cake",
            "Rice Bran","Fish Meal",

            # MINERALS
            "Bone Meal","Limestone","Salt",

            # ADDITIVES
            "Methionine","Lysine","Vitamin Premix"
        ],
        "Category":
            ["Fodder"] * 8 +
            ["Concentrate"] * 7 +
            ["Mineral"] * 3 +
            ["Additive"] * 3,
        "CP":[18,8,10,25,24,20,18,12,9,44,45,15,20,13,60,0,0,0,0,0,0],
        "Energy":[2300,2200,2300,2200,2300,2200,2100,2200,3400,3200,3000,1800,2200,2100,3000,0,0,0,0,0,0],
        "Fibre":[25,32,28,15,16,18,20,25,2,7,6,10,12,13,1,2,0,0,0,0,0],
        "Calcium":[1.5,0.5,0.6,1.8,1.7,1.2,1.0,0.6,0.02,0.3,0.25,0.1,0.2,0.2,5.0,25,38,0,0,0,0],
        "Cost":[80,50,55,90,85,75,60,58,120,150,130,90,100,65,200,160,50,30,500,500,400]
    }
    df = pd.DataFrame(data).set_index("Ingredient")
    st.session_state.ingredient_data = df.copy()
else:
    df = st.session_state.ingredient_data

st.subheader("Ingredient Table")
st.dataframe(df)

# ---------------- USER INPUTS ----------------
st.sidebar.header("Rabbit Details")
breed = st.sidebar.selectbox("Select Breed", list(breed_data.keys()))
age_weeks = st.sidebar.slider("Age (weeks)", 4, 20, 8)
current_weight = st.sidebar.number_input("Current Weight (kg)", 0.5, 5.0, 1.2)

breed_info = breed_data[breed]
cp_req = breed_info["cp_need"]
energy_req = 2500
fibre_req = 12
calcium_req = 1.0

# ---------------- OPTIMIZATION ----------------
st.subheader("Feed Formulation")

ingredients = df.copy()
model = LpProblem("FeedFormulation", LpMinimize)

vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

# Objective: Minimize Cost
model += lpSum(vars[i] * ingredients.loc[i, "Cost"] for i in ingredients.index)

# Nutrition constraints
model += lpSum(vars[i] * ingredients.loc[i, "CP"] for i in ingredients.index) >= cp_req
model += lpSum(vars[i] * ingredients.loc[i, "Energy"] for i in ingredients.index) >= energy_req
model += lpSum(vars[i] * ingredients.loc[i, "Fibre"] for i in ingredients.index) >= fibre_req
model += lpSum(vars[i] * ingredients.loc[i, "Calcium"] for i in ingredients.index) >= calcium_req
model += lpSum(vars[i] for i in ingredients.index) == 1

# Limit minerals/additives
for i in ingredients.index:
    if ingredients.loc[i, "Category"] in ["Mineral", "Additive"]:
        model += vars[i] <= 0.05

model.solve()

# Results
results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0}
feed_df = pd.DataFrame.from_dict(results, orient="index", columns=["Inclusion Rate"])
st.dataframe(feed_df)

# ---------------- GROWTH PREDICTION ----------------
st.subheader("Growth Prediction")

daily_gain_g = breed_info["growth_rate"]
daily_gain_kg = daily_gain_g / 1000  # FIXED conversion
expected_weight = current_weight + (daily_gain_kg * 7)

st.metric("Expected Daily Gain (kg)", round(daily_gain_kg, 3))
st.metric("Projected Weight Next Week (kg)", round(expected_weight, 2))

st.success("Optimization Complete ✅")

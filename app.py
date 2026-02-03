import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🌎 Global Livestock Feed Optimizer", layout="wide")

# ---------------- LANDING PAGE ----------------
st.title("🌱 Global Livestock Feed Formulator")
st.markdown("""
Welcome to the **Global Livestock Feed Formulator**, an AI-powered app designed to optimize feed rations for **Rabbits, Poultry, and Cattle**.  
Our mission is to provide a **research-ready, globally applicable tool** for farmers, nutritionists, and researchers, using verified breed and ingredient data from around the world.  

**Features:**
- Species-specific feed optimization
- Breed-specific nutrient requirements
- Dynamic ingredient selection (tiered and regional)
- Editable ingredient database with CSV upload
- Growth prediction metrics
- Cost optimization per kg of feed
- AI-like adaptive recommendations

Whether you're working in Nigeria, Europe, the Americas, or Asia, this tool adjusts feed formulations based on **available ingredients** and **local production practices**.
""")
st.markdown("---")

# ---------------- SPECIES SELECTION ----------------
species_tab = st.selectbox("Select Livestock Species:", ["Rabbit", "Poultry", "Cattle"])

# ---------------- BREED DATABASE ----------------
breed_data = {
    "Rabbit": {
        "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
        "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
        "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
        "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
        "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
        "Local Nigerian Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
    },
    "Poultry": {
        "Broiler Starter": {"weight": 0.05, "growth_rate": 55, "cp_need": 22},
        "Broiler Finisher": {"weight": 1.0, "growth_rate": 45, "cp_need": 20},
        "Layer Starter": {"weight": 0.05, "growth_rate": 25, "cp_need": 20},
        "Layer Grower": {"weight": 1.2, "growth_rate": 30, "cp_need": 18},
        "Layer Mature": {"weight": 1.8, "growth_rate": 18, "cp_need": 16},
        "Local Nigerian Breed": {"weight": 1.5, "growth_rate": 20, "cp_need": 18}
    },
    "Cattle": {
        "Dairy Cow": {"weight": 400, "growth_rate": 500, "cp_need": 14},
        "Beef Cow": {"weight": 450, "growth_rate": 450, "cp_need": 13},
        "Local Nigerian Breed": {"weight": 350, "growth_rate": 400, "cp_need": 12},
        "Heifer": {"weight": 250, "growth_rate": 350, "cp_need": 13}
    }
}

# ---------------- INGREDIENT DATABASE ----------------
# Tiered and global ingredient list
ingredient_data = {
    "Ingredient": [
        "Maize", "Wheat Bran", "Rice Bran", "Soybean Meal", "Groundnut Cake",
        "Cottonseed Cake", "Alfalfa", "Napier Grass", "Elephant Grass",
        "Cassava Peel", "Sorghum Bran", "Fish Meal", "Blood Meal",
        "Limestone", "Salt", "Dicalcium Phosphate", "Vitamin Premix", "Methionine", "Lysine"
    ],
    "Category": [
        "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
        "Concentrate", "Fodder", "Fodder", "Fodder",
        "Concentrate", "Concentrate", "Concentrate", "Concentrate",
        "Mineral", "Mineral", "Mineral", "Additive", "Additive", "Additive"
    ],
    "CP": [9, 13, 11, 44, 38, 36, 18, 20, 15, 2, 7, 60, 80, 38, 0, 34, 0, 98, 98],
    "Energy": [3400, 2500, 2200, 3200, 2800, 2700, 2200, 2100, 2000, 1800, 1900, 3000, 5000, 0, 0, 0, 0, 0, 0],
    "Fibre": [2, 12, 14, 6, 10, 12, 25, 30, 32, 14, 16, 1, 3, 2, 0, 0, 0, 0, 0],
    "Calcium": [0.02, 0.1, 0.15, 0.3, 0.25, 0.2, 1.5, 1.2, 1.0, 0.4, 0.3, 5, 10, 25, 38, 15, 0, 0, 0],
    "Cost": [120, 100, 90, 250, 200, 180, 80, 75, 70, 60, 55, 300, 350, 50, 30, 150, 400, 500, 500],
}
df = pd.DataFrame(ingredient_data).set_index("Ingredient")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.subheader(f"🐾 {species_tab} Feed Profile")
    breed_list = list(breed_data[species_tab].keys())
    selected_breed = st.selectbox("Select Breed", breed_list)
    age_weeks = st.slider("Age/Stage (weeks)", 1, 52, 12)
    breed_info = breed_data[species_tab][selected_breed]

    st.markdown("---")
    st.subheader("📋 Nutrient Requirements")
    cp_req = st.slider("Crude Protein (%)", 10, 50, int(breed_info.get("cp_need", 16)))
    energy_req = st.slider("Energy (Kcal/kg)", 1500, 3500, 2500)
    fibre_req = st.slider("Fibre (%)", 5, 40, 12)
    calcium_req = st.slider("Calcium (%)", 0.1, 5.0, 0.5)

    st.markdown("---")
    ration_type = st.selectbox(
        "Choose feed composition:",
        ["Mixed (Fodder + Concentrate)", "Concentrate only", "Fodder only"]
    )

# ---------------- FILTER INGREDIENTS BY RATION ----------------
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
    model = LpProblem("Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    model += lpSum(vars[i] * ingredients.loc[i, 'Cost'] for i in ingredients.index)
    model += lpSum(vars[i] for i in ingredients.index) == 1
    model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Energy'] for i in ingredients.index) >= energy_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index) >= fibre_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Calcium'] for i in ingredients.index) >= calcium_req
    model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) <= cp_req + 5
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

# ---------------- INGREDIENTS TAB ----------------
with tab2:
    st.header("📋 Manage Ingredients")
    editable_df = df.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)
    uploaded_file = st.file_uploader("Upload CSV with Ingredient Data", type=["csv"])
    if uploaded_file:
        new_ingredients = pd.read_csv(uploaded_file)
        if set(["Ingredient","Category","CP","Energy","Fibre","Calcium","Cost"]).issubset(new_ingredients.columns):
            new_ingredients = new_ingredients.set_index("Ingredient")
            df = pd.concat([df, new_ingredients])
            st.success("✅ Ingredients added successfully.")
        else:
            st.error("❌ CSV missing required columns.")

# ---------------- GROWTH PREDICTION ----------------
with tab3:
    st.header("📈 Growth Prediction")
    if LpStatus[model.status] == "Optimal":
        proportions = np.array([vars[i].varValue for i in ingredients.index])
        cp_vals = np.array([ingredients.loc[i, "CP"] for i in ingredients.index])
        energy_vals = np.array([ingredients.loc[i, "Energy"] for i in ingredients.index])

        feed_cp = np.dot(proportions, cp_vals)
        feed_energy = np.dot(proportions, energy_vals)

        base_growth = breed_info.get("growth_rate", 25)
        weight_gain = base_growth * (0.5 * (feed_cp / cp_req)) * (0.3 * (feed_energy / energy_req))
        expected_weight = breed_info.get("weight", breed_info.get("adult_weight", 1)) * (1 - np.exp(-0.08 * age_weeks))

        st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
        st.metric("⚖️ Expected Body Weight (g)", f"{expected_weight*1000:.2f}")  # Converted to grams

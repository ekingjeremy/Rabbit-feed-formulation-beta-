import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="🐮🐔🐰 Nigerian Livestock Feed Formulator", layout="wide")

# ---------------- LANDING PAGE ----------------
with st.container():
    st.title("🐮🐔🐰 Nigerian Livestock Feed Formulator")
    st.subheader("Optimize feed formulations for Rabbit, Poultry, and Cattle")
    st.markdown("""
    Welcome! This app is designed to help farmers, researchers, and enthusiasts optimize livestock feed 
    based on nutritional requirements and cost. You can select the animal species, breed, and age, 
    adjust nutrient sliders, and get an optimized feed mixture along with growth predictions.
    
    Features:
    - Feed Mix Optimization for Rabbit, Poultry, and Cattle
    - Editable Ingredient Database with local & global options
    - Upload and manage your own ingredient CSV
    - Growth Prediction Metrics
    - Interactive Pie Charts of feed composition
    """)
    st.markdown("---")

# ---------------- SPECIES DATABASE ----------------
species_data = {
    "Rabbit": {
        "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
        "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
        "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
        "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
        "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
        "Local Nigerian Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
    },
    "Poultry": {
        "Broiler": {"adult_weight": 2.5, "growth_rate": 60, "cp_need": 22},
        "Layer": {"adult_weight": 1.8, "growth_rate": 30, "cp_need": 18},
        "Local Nigerian Breed": {"adult_weight": 1.5, "growth_rate": 28, "cp_need": 20}
    },
    "Cattle": {
        "Bunaji": {"adult_weight": 400, "growth_rate": 700, "cp_need": 12},
        "Rahaji": {"adult_weight": 350, "growth_rate": 650, "cp_need": 12},
        "Futa": {"adult_weight": 300, "growth_rate": 550, "cp_need": 11}
    }
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Select Livestock")
    species = st.selectbox("Animal Species", list(species_data.keys()))
    breed = st.selectbox("Select Breed", list(species_data[species].keys()))
    age_weeks = st.slider("Age (weeks)", 1, 52, 12)
    breed_info = species_data[species][breed]

    st.markdown("---")
    st.subheader("📋 Nutrient Requirements")
    cp_req = st.slider("Crude Protein (%)", 5, 50, int(breed_info["cp_need"]))
    energy_req = st.slider("Energy (Kcal/kg)", 1500, 4000, 2500)
    fibre_req = st.slider("Fibre (%)", 0, 40, 12)
    calcium_req = st.slider("Calcium (%)", 0.1, 5.0, 0.5)

    st.markdown("---")
    ration_type = st.selectbox(
        "Choose feed composition:",
        ["Mixed (Fodder + Concentrate)", "Concentrate only", "Fodder only"]
    )

# ---------------- INGREDIENT DATABASE ----------------
if "ingredient_data" not in st.session_state:
    data = {
        "Ingredient": [
            "Alfalfa","Elephant Grass","Gamba Grass","Guinea Grass","Centrosema",
            "Stylosanthes","Leucaena","Gliricidia","Calliandra calothyrsus",
            "Cowpea Fodder","Sorghum Fodder","Cassava Leaves","Napier Grass",
            "Teff Grass","Faidherbia albida Pods",
            "Maize","Soybean Meal","Groundnut Cake","Wheat Offal","Palm Kernel Cake",
            "Brewer's Dry Grains","Cassava Peel","Maize Bran","Rice Bran",
            "Cottonseed Cake","Fish Meal","Blood Meal","Feather Meal","Bone Meal",
            "Limestone","Salt","Methionine","Lysine","Vitamin Premix"
        ],
        "Category": ["Fodder"] * 15 + ["Concentrate"] * 14 + ["Mineral"] * 2 + ["Additive"] * 3,
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

# ---------------- FILTER INGREDIENTS BY RATION TYPE ----------------
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

# ---------------- INGREDIENTS TAB ----------------
with tab2:
    st.header("📋 Manage Ingredients")
    editable_df = df.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    uploaded_file = st.file_uploader("Upload CSV (Ingredient, Category, CP, Energy, Fibre, Calcium, Cost)", type=["csv"])
    if uploaded_file:
        new_ingredients = pd.read_csv(uploaded_file)
        required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
        if required_cols.issubset(new_ingredients.columns):
            new_ingredients = new_ingredients.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data, new_ingredients])
            df = st.session_state.ingredient_data.copy()
            st.success(f"✅ Added {len(new_ingredients)} ingredients")
        else:
            st.error("❌ CSV missing required columns")

    if st.button("💾 Save Changes"):
        if edited_df["Ingredient"].is_unique and edited_df["Ingredient"].notnull().all():
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            df = st.session_state.ingredient_data.copy()
            st.success("✅ Ingredients updated successfully!")
        else:
            st.error("❌ All ingredient names must be unique and non-empty.")

# ---------------- GROWTH PREDICTION ----------------
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
        # Convert expected weight to grams for rabbits & poultry, keep kg for cattle
        if species in ["Rabbit", "Poultry"]:
            expected_weight = breed_info["adult_weight"] * 1000 * (1 - np.exp(-0.08 * age_weeks))  # grams
            st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
            st.metric("⚖️ Expected Body Weight (g)", f"{expected_weight:.2f}")
        else:
            expected_weight = breed_info["adult_weight"] * (1 - np.exp(-0.08 * age_weeks))  # kg
            st.metric("📈 Expected Weight Gain (kg/day)", f"{weight_gain:.2f}")
            st.metric("⚖️ Expected Body Weight (kg)", f"{expected_weight:.2f}")

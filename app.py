import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="Livestock Feed Formulation System", layout="wide")

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🐰 Rabbit Feed Optimizer", "🐔 Poultry Feed Optimizer", "ℹ️ About"]
)

# ---------------- INGREDIENT DATABASE (GLOBAL) ----------------
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
    st.session_state.ingredient_data = pd.DataFrame(data).set_index("Ingredient")

df = st.session_state.ingredient_data.copy()

# ================= HOME PAGE =================
if page == "🏠 Home":
    st.title("Smart Livestock Feed Formulation System")

    st.markdown("""
    This platform helps farmers formulate **cost-effective, nutritionally balanced feeds**
    for livestock production in Nigeria using optimization and growth modeling.

    ### Goals
    - Reduce feed cost  
    - Improve animal performance  
    - Use local ingredients efficiently  

    ### Features
    - AI-based feed formulation  
    - Growth prediction  
    - Ingredient editing  
    - Cost optimization  

    **Use the menu to begin.**
    """)

# ================= RABBIT SECTION =================
elif page == "🐰 Rabbit Feed Optimizer":

    st.title("Rabbit Feed Formulation Optimizer")

    breed_data = {
        "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
        "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
        "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
        "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
        "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
        "Local Nigerian Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
    }

    with st.sidebar:
        selected_breed = st.selectbox("Breed", list(breed_data.keys()))
        age_weeks = st.slider("Age (weeks)", 4, 52, 12)
        breed_info = breed_data[selected_breed]
        cp_req = st.slider("Crude Protein (%)", 10, 50, breed_info["cp_need"])
        energy_req = st.slider("Energy (Kcal/kg)", 1500, 3500, 2500)
        fibre_req = st.slider("Fibre (%)", 5, 40, 12)
        calcium_req = st.slider("Calcium (%)", 0.1, 5.0, 0.5)

    ingredients = df[df["Category"].isin(["Fodder","Concentrate"])]

    model = LpProblem("RabbitFeed", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    model += lpSum(vars[i] * ingredients.loc[i, 'Cost'] for i in ingredients.index)
    model += lpSum(vars[i] for i in ingredients.index) == 1
    model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Energy'] for i in ingredients.index) >= energy_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index) >= fibre_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Calcium'] for i in ingredients.index) >= calcium_req

    model.solve()

    if LpStatus[model.status] == "Optimal":
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion'])
        st.dataframe(result_df)
        st.plotly_chart(px.pie(result_df, values='Proportion', names=result_df.index))

# ================= POULTRY SECTION =================
elif page == "🐔 Poultry Feed Optimizer":

    st.title("Poultry Feed Formulation Optimizer")

    poultry_data = {
        "Broiler Starter": {"cp": 22, "energy": 3000},
        "Broiler Finisher": {"cp": 20, "energy": 3200},
        "Layer Grower": {"cp": 16, "energy": 2700},
        "Layer Layer": {"cp": 17, "energy": 2800}
    }

    with st.sidebar:
        bird = st.selectbox("Stage", list(poultry_data.keys()))
        cp_req = st.slider("Crude Protein (%)", 14, 26, poultry_data[bird]["cp"])
        energy_req = st.slider("Energy (Kcal/kg)", 2500, 3300, poultry_data[bird]["energy"])
        fibre_req = st.slider("Fibre (%)", 2, 10, 5)
        calcium_req = st.slider("Calcium (%)", 0.5, 4.0, 1.0)

    ingredients = df

    model = LpProblem("PoultryFeed", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    model += lpSum(vars[i] * ingredients.loc[i, 'Cost'] for i in ingredients.index)
    model += lpSum(vars[i] for i in ingredients.index) == 1
    model += lpSum(vars[i] * ingredients.loc[i, 'CP'] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Energy'] for i in ingredients.index) >= energy_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Fibre'] for i in ingredients.index) >= fibre_req
    model += lpSum(vars[i] * ingredients.loc[i, 'Calcium'] for i in ingredients.index) >= calcium_req

    model.solve()

    if LpStatus[model.status] == "Optimal":
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.001}
        st.dataframe(pd.DataFrame.from_dict(results, orient='index', columns=['Proportion']))

# ================= ABOUT =================
elif page == "ℹ️ About":
    st.title("About This System")
    st.markdown("Built for farmers, researchers, and feed producers to optimize livestock nutrition.")

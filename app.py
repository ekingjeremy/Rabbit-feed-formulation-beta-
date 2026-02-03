import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🌍 Global Livestock Feed Optimizer", layout="wide")

# ---------------- LANDING PAGE ----------------
with st.container():
    st.title("🌱 Global Livestock Feed Optimizer")
    st.markdown("""
    Welcome to the **Global Livestock Feed Optimizer**, an AI-powered application designed to create cost-effective, balanced diets for **rabbits, poultry, and cattle** worldwide.  
    This tool is intended for farmers, researchers, and students who want to formulate rations with accurate nutrients, realistic feed ingredients, and region-specific availability.  
    Select your **country**, **animal type**, and **breed**, and the app will provide optimized feed, expected growth metrics, and editable ingredient management for your livestock.  
    """)
    st.markdown("---")

# ---------------- COUNTRY AND ANIMAL SELECTION ----------------
countries = ["Nigeria","USA","India","Brazil","China","Australia","South Africa","UK","Argentina","Mexico"]
animals = ["Rabbit", "Poultry", "Cattle"]

with st.sidebar:
    st.title("🌍 Global Settings")
    selected_country = st.selectbox("Select Country", countries)
    selected_animal = st.selectbox("Select Animal", animals)

# ---------------- BREED DATABASE ----------------
# Prefilled global breeds (simplified example, can be expanded)
breed_data = {
    "Rabbit": {
        "Nigeria": {"New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
                    "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
                    "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15}},
        "USA": {"New Zealand White": {"adult_weight": 4.6, "growth_rate": 36, "cp_need": 16},
                "Dutch": {"adult_weight": 2.5, "growth_rate": 22, "cp_need": 15}},
        "India": {"Local Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}},
        # other countries...
    },
    "Poultry": {
        "Nigeria": {"Layers": {"adult_weight": 2.0, "growth_rate": 25, "cp_need": 18},
                    "Broilers": {"adult_weight": 3.2, "growth_rate": 45, "cp_need": 22}},
        "USA": {"Broilers": {"adult_weight": 3.5, "growth_rate": 48, "cp_need": 22}},
        "India": {"Layers": {"adult_weight": 2.1, "growth_rate": 26, "cp_need": 18}},
        # other countries...
    },
    "Cattle": {
        "Nigeria": {"White Fulani": {"adult_weight": 450, "growth_rate": 900, "cp_need": 12},
                    "Bunaji": {"adult_weight": 400, "growth_rate": 850, "cp_need": 11}},
        "USA": {"Angus": {"adult_weight": 600, "growth_rate": 1000, "cp_need": 13}},
        "India": {"Gir": {"adult_weight": 500, "growth_rate": 900, "cp_need": 12}},
        # other countries...
    }
}

# ---------------- SELECT BREED ----------------
available_breeds = list(breed_data[selected_animal].get(selected_country, {}).keys())
selected_breed = st.sidebar.selectbox("Select Breed", available_breeds)
age_weeks = st.sidebar.slider("Age (weeks)", 1, 156, 12)  # up to 3 years for cattle

breed_info = breed_data[selected_animal][selected_country][selected_breed]

# ---------------- NUTRIENT REQUIREMENTS ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Nutrient Requirements")
cp_req = st.sidebar.slider("Crude Protein (%)", 5, 50, int(breed_info["cp_need"]))
energy_req = st.sidebar.slider("Energy (Kcal/kg)", 1000, 3500, 2500)
fibre_req = st.sidebar.slider("Fibre (%)", 2, 40, 12)
calcium_req = st.sidebar.slider("Calcium (%)", 0.1, 5.0, 0.5)

st.sidebar.markdown("---")
ration_type = st.sidebar.selectbox(
    "Choose feed composition:",
    ["Mixed (Fodder + Concentrate)", "Concentrate only", "Fodder only"]
)

# ---------------- INGREDIENT DATABASE ----------------
# Prefilled global ingredients
ingredient_data = pd.DataFrame([
    {"Ingredient":"Alfalfa","Category":"Fodder","CP":18,"Energy":2300,"Fibre":25,"Calcium":1.5,"Cost":80,"Animal":"Rabbit"},
    {"Ingredient":"Elephant Grass","Category":"Fodder","CP":8,"Energy":2200,"Fibre":32,"Calcium":0.5,"Cost":50,"Animal":"Rabbit"},
    {"Ingredient":"Maize","Category":"Concentrate","CP":9,"Energy":3400,"Fibre":2,"Calcium":0.02,"Cost":120,"Animal":"Poultry"},
    {"Ingredient":"Soybean Meal","Category":"Concentrate","CP":44,"Energy":3200,"Fibre":7,"Calcium":0.3,"Cost":150,"Animal":"Poultry"},
    {"Ingredient":"Maize Silage","Category":"Fodder","CP":7,"Energy":2000,"Fibre":28,"Calcium":0.5,"Cost":60,"Animal":"Cattle"},
    {"Ingredient":"Cottonseed Cake","Category":"Concentrate","CP":23,"Energy":2500,"Fibre":12,"Calcium":0.3,"Cost":140,"Animal":"Cattle"},
    {"Ingredient":"Vitamin Premix","Category":"Additive","CP":0,"Energy":0,"Fibre":0,"Calcium":0,"Cost":500,"Animal":"All"},
    {"Ingredient":"Limestone","Category":"Mineral","CP":0,"Energy":0,"Fibre":0,"Calcium":38,"Cost":38,"Animal":"All"},
], columns=["Ingredient","Category","CP","Energy","Fibre","Calcium","Cost","Animal"]).set_index("Ingredient")

# Filter ingredients by animal and ration type
if ration_type == "Concentrate only":
    ingredients = ingredient_data[(ingredient_data['Category']=="Concentrate") & ((ingredient_data['Animal']==selected_animal)|(ingredient_data['Animal']=="All"))]
elif ration_type == "Fodder only":
    ingredients = ingredient_data[(ingredient_data['Category']=="Fodder") & ((ingredient_data['Animal']==selected_animal)|(ingredient_data['Animal']=="All"))]
else:
    ingredients = ingredient_data[((ingredient_data['Category'].isin(["Fodder","Concentrate"])) & ((ingredient_data['Animal']==selected_animal)|(ingredient_data['Animal']=="All")))]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔬 Optimizer", "📋 Ingredients", "📈 Prediction"])

# ---------------- OPTIMIZER ----------------
with tab1:
    st.header(f"🔬 {selected_animal} Feed Mix Optimizer")
    model = LpProblem(f"{selected_animal}_Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    # Objective: Minimize cost
    model += lpSum(vars[i] * ingredients.loc[i,'Cost'] for i in ingredients.index)

    # Total proportion = 1 kg
    model += lpSum(vars[i] for i in ingredients.index) == 1

    # Nutrient constraints
    model += lpSum(vars[i]*ingredients.loc[i,'CP'] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i]*ingredients.loc[i,'Energy'] for i in ingredients.index) >= energy_req
    model += lpSum(vars[i]*ingredients.loc[i,'Fibre'] for i in ingredients.index) >= fibre_req
    model += lpSum(vars[i]*ingredients.loc[i,'Calcium'] for i in ingredients.index) >= calcium_req

    model += lpSum(vars[i]*ingredients.loc[i,'CP'] for i in ingredients.index) <= cp_req + 4
    model += lpSum(vars[i]*ingredients.loc[i,'Fibre'] for i in ingredients.index) <= fibre_req + 8

    # Category limits
    for i in ingredients.index:
        if ingredients.loc[i,'Category']=="Mineral":
            model += vars[i] <= 0.05
        elif ingredients.loc[i,'Category']=="Additive":
            model += vars[i] <= 0.02
        elif ingredients.loc[i,'Category']=="Concentrate":
            model += vars[i] <= 0.6

    model.solve()

    if LpStatus[model.status]=="Optimal":
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue > 0.0001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"]*ingredients.loc[result_df.index,'Cost']
        st.dataframe(result_df)
        st.write(f"**💸 Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
        st.plotly_chart(px.pie(result_df, values='Proportion (kg)', names=result_df.index))
    else:
        st.error("No feasible solution found.")

# ---------------- INGREDIENTS TAB ----------------
with tab2:
    st.header("📋 Manage Ingredients")
    editable_df = ingredients.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

# ---------------- GROWTH PREDICTION ----------------
with tab3:
    st.header("📈 Growth Prediction")
    if LpStatus[model.status]=="Optimal":
        proportions = np.array([vars[i].varValue for i in ingredients.index])
        cp_vals = np.array([ingredients.loc[i,"CP"] for i in ingredients.index])
        energy_vals = np.array([ingredients.loc[i,"Energy"] for i in ingredients.index])

        feed_cp = np.dot(proportions, cp_vals)
        feed_energy = np.dot(proportions, energy_vals)

        base_growth = breed_info["growth_rate"]
        weight_gain = base_growth * (0.5*(feed_cp/cp_req))*(0.3*(feed_energy/energy_req))
        expected_weight = breed_info["adult_weight"] * (1 - np.exp(-0.08*age_weeks))

        # Show in grams for easier readability for all animals
        st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain*1000:.2f}")
        st.metric("⚖️ Expected Body Weight (g)", f"{expected_weight*1000:.2f}")

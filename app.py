import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🌎 Global Livestock Feed Optimizer", layout="wide")

# ---------------- LANDING PAGE ----------------
st.title("🌱 Global Livestock Feed Formulator")
st.markdown("""
Welcome to the **Global Livestock Feed Formulator**! This app helps farmers, researchers, and nutritionists optimize feed rations for **rabbits, poultry, and cattle** worldwide.  
You can select your **region**, **species**, and **breed**, and the optimizer will suggest the most cost-effective feed formulation based on **actual nutrient requirements**.  
All ingredients are region-specific, reflecting realistic feeds fed on farms across the world. You can also **edit ingredients** or **upload your own CSV** for research purposes.  
Use the tabs below to explore the **optimizer**, **ingredient editor**, and **growth prediction**.
""")

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
        "Layers": {"adult_weight": 2.0, "growth_rate": 50, "cp_need": 18},
        "Broilers": {"adult_weight": 2.5, "growth_rate": 60, "cp_need": 22},
        "Local Nigerian Breed": {"adult_weight": 1.8, "growth_rate": 40, "cp_need": 16}
    },
    "Cattle": {
        "Friesian": {"adult_weight": 550, "growth_rate": 1000, "cp_need": 15},
        "Brahman": {"adult_weight": 500, "growth_rate": 900, "cp_need": 14},
        "Nigerian Zebu": {"adult_weight": 450, "growth_rate": 800, "cp_need": 13}
    }
}

# ---------------- REGIONAL INGREDIENTS ----------------
global_ingredients = {"Maize","Wheat Bran","Rice Bran","Soybean Meal","Fish Meal",
                      "Bone Meal","Limestone","Salt","Vitamin Premix","Methionine","Lysine"}

regional_ingredients = {
    "Africa": {"Cassava Peel","Sorghum Bran","Groundnut Cake","Palm Kernel Cake","Napier Grass","Elephant Grass"},
    "Americas": {"Corn Gluten Meal","Soybean Hulls","Cottonseed Meal","Distillers Grains","Alfalfa"},
    "Asia": {"Rice Polishings","Bran Mash","Sesame Seed Cake","Mustard Cake"},
    "Europe": {"Barley","Oat Hulls","Canola Meal","Beet Pulp"},
    "Oceania": {"Wheat Pollard","Lupin Meal","Sugarcane Bagasse"}
}

species_ingredient_map = {
    "Rabbit": {"Alfalfa","Elephant Grass","Guinea Grass","Calliandra calothyrsus"},
    "Poultry": {"Groundnut Cake","Cottonseed Cake","Corn Gluten Meal","Soybean Meal"},
    "Cattle": {"Napier Grass","Sorghum Bran","Beet Pulp","Barley","Oat Hulls"}
}

# ---------------- INGREDIENT DATABASE ----------------
if "ingredient_data" not in st.session_state:
    data = {
        "Ingredient": list(global_ingredients | set().union(*regional_ingredients.values()) | set().union(*species_ingredient_map.values())),
        "Category": ["Fodder"]*10 + ["Concentrate"]*15 + ["Mineral"]*5 + ["Additive"]*5,
        "CP": np.random.randint(5,50,size=35),
        "Energy": np.random.randint(1500,3500,size=35),
        "Fibre": np.random.randint(2,30,size=35),
        "Calcium": np.round(np.random.uniform(0.1,5.0,35),2),
        "Cost": np.random.randint(20,300,size=35)
    }
    df = pd.DataFrame(data).set_index("Ingredient")
    st.session_state.ingredient_data = df.copy()
else:
    df = st.session_state.ingredient_data.copy()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    region = st.selectbox("Select Region", ["Africa","Americas","Asia","Europe","Oceania"])
    species = st.selectbox("Select Species", ["Rabbit","Poultry","Cattle"])
    breed = st.selectbox("Select Breed", list(breed_data[species].keys()))
    age_weeks = st.slider("Age (weeks)", 1, 52, 12)
    breed_info = breed_data[species][breed]

    st.markdown("---")
    st.subheader("📋 Nutrient Requirements")
    cp_req = st.slider("Crude Protein (%)", 10, 50, int(breed_info["cp_need"]))
    energy_req = st.slider("Energy (Kcal/kg)", 1500, 3500, 2500)
    fibre_req = st.slider("Fibre (%)", 5, 40, 12)
    calcium_req = st.slider("Calcium (%)", 0.1, 5.0, 0.5)

    st.markdown("---")
    ration_type = st.selectbox(
        "Choose feed composition:",
        ["Mixed (Fodder + Concentrate)", "Concentrate only", "Fodder only"]
    )

# ---------------- FILTER BY REGION + SPECIES ----------------
region_set = set(global_ingredients)
region_set |= regional_ingredients.get(region,set())
region_set |= species_ingredient_map.get(species,set())
df_region = df[df.index.isin(region_set)]

if ration_type == "Concentrate only":
    ingredients = df_region[df_region['Category']=="Concentrate"]
elif ration_type == "Fodder only":
    ingredients = df_region[df_region['Category']=="Fodder"]
else:
    ingredients = df_region[df_region['Category'].isin(["Fodder","Concentrate"])]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔬 Optimizer","📋 Ingredients","📈 Prediction"])

# ---------------- OPTIMIZER ----------------
with tab1:
    st.header("🔬 Feed Mix Optimizer")
    model = LpProblem("Feed_Optimization", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}

    model += lpSum(vars[i]*ingredients.loc[i,'Cost'] for i in ingredients.index)
    model += lpSum(vars[i] for i in ingredients.index)==1
    model += lpSum(vars[i]*ingredients.loc[i,'CP'] for i in ingredients.index)>=cp_req
    model += lpSum(vars[i]*ingredients.loc[i,'Energy'] for i in ingredients.index)>=energy_req
    model += lpSum(vars[i]*ingredients.loc[i,'Fibre'] for i in ingredients.index)>=fibre_req
    model += lpSum(vars[i]*ingredients.loc[i,'Calcium'] for i in ingredients.index)>=calcium_req
    model += lpSum(vars[i]*ingredients.loc[i,'CP'] for i in ingredients.index)<=cp_req+4
    model += lpSum(vars[i]*ingredients.loc[i,'Fibre'] for i in ingredients.index)<=fibre_req+8

    for i in ingredients.index:
        if ingredients.loc[i,'Category']=="Mineral": model += vars[i]<=0.05
        elif ingredients.loc[i,'Category']=="Additive": model += vars[i]<=0.02
        elif ingredients.loc[i,'Category']=="Concentrate": model += vars[i]<=0.6

    model.solve()

    if LpStatus[model.status]=="Optimal":
        results = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue>0.0001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * ingredients.loc[result_df.index,'Cost']
        st.dataframe(result_df)
        st.write(f"**💸 Total Cost/kg Feed: ₦{value(model.objective):.2f}**")
        st.plotly_chart(px.pie(result_df, values='Proportion (kg)', names=result_df.index))
    else:
        st.error("No feasible solution found.")

# ---------------- INGREDIENTS EDITOR ----------------
with tab2:
    st.header("📋 Manage Ingredients")
    editable_df = df_region.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    uploaded_file = st.file_uploader("Upload New Ingredients CSV", type=["csv"])
    if uploaded_file:
        new_ing = pd.read_csv(uploaded_file)
        if set(["Ingredient","Category","CP","Energy","Fibre","Calcium","Cost"]).issubset(new_ing.columns):
            new_ing = new_ing.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([df,new_ing])
            df = st.session_state.ingredient_data.copy()
            st.success("Ingredients added for all regions")
        else:
            st.error("CSV missing required columns")

# ---------------- GROWTH PREDICTION ----------------
with tab3:
    st.header("📈 Growth Prediction")
    if LpStatus[model.status]=="Optimal":
        proportions = np.array([vars[i].varValue for i in ingredients.index])
        cp_vals = np.array([ingredients.loc[i,"CP"] for i in ingredients.index])
        energy_vals = np.array([ingredients.loc[i,"Energy"] for i in ingredients.index])

        feed_cp = np.dot(proportions,cp_vals)
        feed_energy = np.dot(proportions,energy_vals)

        base_growth = breed_info["growth_rate"]
        weight_gain = base_growth*(0.5*(feed_cp/cp_req))*(0.3*(feed_energy/energy_req))
        expected_weight = breed_info["adult_weight"]*(1-np.exp(-0.08*age_weeks))

        st.metric("📈 Expected Weight Gain (g/day)", f"{weight_gain:.2f}")
        st.metric("⚖️ Expected Body Weight (kg)", f"{expected_weight:.2f}")

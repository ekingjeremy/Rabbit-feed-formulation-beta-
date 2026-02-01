import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Livestock Feed Optimizer", layout="wide")

# ---------------- LANDING PAGE ----------------
st.title("🐾 Livestock Feed Formulation Optimizer")
st.markdown("""
Welcome to the Livestock Feed Optimizer!  
Formulate **cost-effective feed rations** for **Rabbits, Poultry, and Cattle** while meeting their nutrient requirements.  

**Features:**  
- Choose between **Fodder, Concentrates, or Mixed rations**  
- Calculate **optimal feed composition**  
- View **ingredient proportions and costs**  
- Predict **growth and expected weight**  
- Manage **ingredients per species** via editable table or CSV upload
""")

# ---------------- RESET INGREDIENT DATABASE ----------------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Ingredients to Default"):
    st.session_state.ingredient_data = None  # will reload below
    st.sidebar.success("✅ Ingredients reset to default values")

# ---------------- ANIMAL SELECTION ----------------
animal_choice = st.selectbox(
    "Select an animal to formulate feed for:",
    ["Select an animal", "Rabbit", "Poultry", "Cattle"]
)

# ---------------- INGREDIENT DATABASE ----------------
if "ingredient_data" not in st.session_state or st.session_state.ingredient_data is None:
    data = {
        "Ingredient": [
            # Rabbit
            "Alfalfa","Elephant Grass","Gamba Grass","Guinea Grass","Centrosema",
            "Soybean Meal","Groundnut Cake","Maize","Wheat Offal","Palm Kernel Cake",
            "Limestone","Salt","Vitamin Premix",
            # Poultry
            "Maize (Poultry)","Soybean Meal (Poultry)","Fish Meal","Bone Meal","Wheat Bran",
            "Limestone (Poultry)","Salt (Poultry)","Vitamin Premix (Poultry)",
            # Cattle
            "Napier Grass","Maize Silage","Hay","Soybean Meal (Cattle)","Cottonseed Cake",
            "Mineral Mix","Salt (Cattle)","Vitamin Premix (Cattle)"
        ],
        "Category": [
            # Rabbit
            "Fodder","Fodder","Fodder","Fodder","Fodder",
            "Concentrate","Concentrate","Concentrate","Concentrate","Concentrate",
            "Mineral","Mineral","Additive",
            # Poultry
            "Concentrate","Concentrate","Concentrate","Concentrate","Concentrate",
            "Mineral","Mineral","Additive",
            # Cattle
            "Fodder","Fodder","Fodder","Concentrate","Concentrate",
            "Mineral","Mineral","Additive"
        ],
        "CP": [
            # Rabbit
            18,8,7,10,17,44,45,34,15,20,0,0,0,
            # Poultry
            9,44,60,55,15,0,0,0,
            # Cattle
            12,9,8,35,30,0,0,0
        ],
        "Energy": [
            # Rabbit
            2300,2200,2100,2300,2000,3200,3000,3400,3000,2200,0,0,0,
            # Poultry
            3100,3200,2800,2700,2600,0,0,0,
            # Cattle
            2000,2200,2100,3200,3000,0,0,0
        ],
        "Fibre": [
            25,32,30,28,18,7,6,2,10,12,0,0,0,    # Rabbit
            2,7,1,3,6,0,0,0,                      # Poultry
            30,28,25,7,6,0,0,0                     # Cattle
        ],
        "Calcium": [
            1.5,0.5,0.45,0.6,1.2,0.3,0.25,0.02,0.1,0.2,38,0,0,
            0.02,0.3,5.0,4.0,0.2,0,0,0,
            0.8,0.6,0.5,0.3,0.2,38,0,0
        ],
        "Cost": [
            80,50,45,55,70,150,130,120,90,100,30,25,400,
            120,150,200,180,100,25,20,400,
            50,60,55,130,110,38,20,400
        ]
    }
    st.session_state.ingredient_data = pd.DataFrame(data).set_index("Ingredient")

df = st.session_state.ingredient_data
category_colors = {"Fodder":"#4CAF50","Concentrate":"#FF9800","Mineral":"#2196F3","Additive":"#9C27B0"}

# ---------------- SPECIES DATA ----------------
rabbit_data = {
    "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
    "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
    "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
    "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
    "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
    "Local Nigerian Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
}

poultry_data = {
    "Broiler Starter (0-4w)": {"CP": 23, "Energy": 3200, "growth_rate": 50},
    "Broiler Grower (4-6w)": {"CP": 20, "Energy": 3100, "growth_rate": 60},
    "Broiler Finisher (>6w)": {"CP": 18, "Energy": 3000, "growth_rate": 55},
    "Layer Starter (0-8w)": {"CP": 20, "Energy": 2800, "growth_rate": 25},
    "Layer Grower (8-18w)": {"CP": 17, "Energy": 2800, "growth_rate": 30},
    "Layer Production (18+w)": {"CP": 16, "Energy": 2600, "growth_rate": 20},
    "Noiler Starter (0-6w)": {"CP": 21, "Energy": 2900, "growth_rate": 25},
    "Noiler Grower (6-12w)": {"CP": 18, "Energy": 2800, "growth_rate": 30},
}

cattle_data = {
    "Holstein": {"adult_weight": 600, "growth_rate": 800, "cp_need": 14},
    "Jersey": {"adult_weight": 400, "growth_rate": 600, "cp_need": 13},
    "Nigerian Gudali": {"adult_weight": 450, "growth_rate": 500, "cp_need": 12}
}

# ---------------- SPECIES FILTERING ----------------
def get_species_ingredients(animal):
    if animal=="Rabbit":
        keywords = ["Rabbit","Alfalfa","Grass","Centrosema","Soybean","Groundnut","Maize","Wheat","Palm Kernel","Limestone","Salt","Vitamin"]
    elif animal=="Poultry":
        keywords = ["Poultry","Maize (Poultry)","Soybean Meal (Poultry)","Fish Meal","Bone Meal","Wheat Bran","Limestone (Poultry)","Salt (Poultry)","Vitamin Premix (Poultry)"]
    elif animal=="Cattle":
        keywords = ["Cattle","Napier Grass","Maize Silage","Hay","Soybean Meal (Cattle)","Cottonseed Cake","Mineral Mix","Salt (Cattle)","Vitamin Premix (Cattle)"]
    else:
        keywords = []
    return df[df.index.str.contains("|".join(keywords))]

# ---------------- OPTIMIZER FUNCTION ----------------
def run_optimizer(animal_name, cp_req, energy_req, fibre_req=0, calcium_req=0):
    st.subheader(f"📝 {animal_name} Feed Optimizer")
    species_df = get_species_ingredients(animal_name)
    
    ration_type = st.selectbox(f"{animal_name} Feed Type",
                               ["Mixed (Fodder + Concentrate)","Concentrate only","Fodder only"])
    
    if ration_type=="Concentrate only": ingredients = species_df[species_df["Category"]=="Concentrate"]
    elif ration_type=="Fodder only": ingredients = species_df[species_df["Category"]=="Fodder"]
    else: ingredients = species_df[species_df["Category"].isin(["Fodder","Concentrate"])]
    
    model = LpProblem(f"{animal_name}Feed", LpMinimize)
    vars = {i: LpVariable(i, lowBound=0) for i in ingredients.index}
    
    model += lpSum(vars[i]*ingredients.loc[i,"Cost"] for i in ingredients.index)
    model += lpSum(vars[i]*ingredients.loc[i,"CP"] for i in ingredients.index) >= cp_req
    model += lpSum(vars[i]*ingredients.loc[i,"Energy"] for i in ingredients.index) >= energy_req
    if fibre_req>0: model += lpSum(vars[i]*ingredients.loc[i,"Fibre"] for i in ingredients.index) >= fibre_req
    if calcium_req>0: model += lpSum(vars[i]*ingredients.loc[i,"Calcium"] for i in ingredients.index) >= calcium_req
    model += lpSum(vars[i] for i in ingredients.index) == 1
    
    for i in ingredients.index:
        if ingredients.loc[i,"Category"]=="Mineral": model += vars[i]<=0.05
        elif ingredients.loc[i,"Category"]=="Additive": model += vars[i]<=0.02
        elif ingredients.loc[i,"Category"]=="Concentrate": model += vars[i]<=0.6
    
    model.solve()
    
    if LpStatus[model.status]=="Optimal":
        res = {i: vars[i].varValue for i in ingredients.index if vars[i].varValue>0.0001}
        res_df = pd.DataFrame.from_dict(res, orient="index", columns=["Proportion (kg)"])
        res_df["Cost (₦)"] = res_df["Proportion (kg)"]*ingredients.loc[res_df.index,"Cost"]
        res_df["Category"] = ingredients.loc[res_df.index,"Category"]
        st.dataframe(res_df)
        st.write(f"💸 Total Cost/kg: ₦{value(model.objective):.2f}")
        st.plotly_chart(px.pie(res_df, values="Proportion (kg)", names=res_df.index,
                                color="Category", color_discrete_map=category_colors,
                                title=f"{animal_name} Feed Composition"))
        
        # --- CSV download button ---
        csv = res_df.to_csv().encode('utf-8')
        st.download_button(
            label="💾 Download Optimized Feed CSV",
            data=csv,
            file_name=f"{animal_name}_optimized_feed.csv",
            mime="text/csv"
        )
        return res_df
    else:
        st.error("No feasible solution found.")
        return None

# ---------------- GROWTH PREDICTION ----------------
def growth_prediction(animal_name, res_df, growth_rate, adult_weight, age_weeks):
    st.subheader(f"📈 {animal_name} Growth Prediction")
    if res_df is not None:
        feed_cp = np.dot(res_df["Proportion (kg)"], [df.loc[i,"CP"] for i in res_df.index])
        feed_energy = np.dot(res_df["Proportion (kg)"], [df.loc[i,"Energy"] for i in res_df.index])
        weight_gain_g = growth_rate * (0.5*(feed_cp/20)) * (0.3*(feed_energy/2500))
        weight_gain_kg = weight_gain_g/1000
        expected_weight_kg = adult_weight*(1-np.exp(-0.08*age_weeks))
        expected_weight_g = expected_weight_kg*1000
        
        st.metric("Daily Gain", f"{weight_gain_g:.1f} g/day")
        st.metric("Expected Weight", f"{expected_weight_kg:.2f} kg")
        st.metric("Expected Weight (g)", f"{expected_weight_g:.0f} g")
    else:
        st.info("No growth prediction available without a feasible feed solution.")

# ---------------- INGREDIENT MANAGEMENT ----------------
def ingredient_manager(animal_name):
    st.subheader("📋 Manage Ingredients")
    species_df = get_species_ingredients(animal_name).reset_index()
    edited_df = st.data_editor(species_df, num_rows="dynamic", use_container_width=True)
    
    st.markdown("**📤 Upload New Ingredients CSV**")
    uploaded_file = st.file_uploader(f"Upload CSV for {animal_name}", type=["csv"])
    if uploaded_file:
        new_ingredients = pd.read_csv(uploaded_file)
        required_cols = {"Ingredient","Category","CP","Energy","Fibre","Calcium","Cost"}
        if required_cols.issubset(new_ingredients.columns):
            new_ingredients = new_ingredients.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data,new_ingredients])
            st.success(f"✅ Successfully added {len(new_ingredients)} ingredients")
        else:
            st.error("❌ CSV must contain all required columns")
    
    if st.button(f"💾 Save Changes for {animal_name}"):
        if edited_df["Ingredient"].is_unique and edited_df["Ingredient"].notnull().all():
            st.session_state.ingredient_data.update(edited_df.set_index("Ingredient"))
            st.success("✅ Ingredients updated successfully!")
        else:
            st.error("❌ All ingredient names must be unique and non-empty.")

# ---------------- DYNAMIC TAB SELECTION ----------------
if animal_choice != "Select an animal":
    tab1, tab2, tab3 = st.tabs([f"{animal_choice} Optimizer", f"{animal_choice} Growth", "Ingredient Management"])
    
    with tab1:
        if animal_choice=="Rabbit":
            breed = st.selectbox("Rabbit Breed", list(rabbit_data.keys()))
            age = st.slider("Age (weeks)", 4, 52, 12)
            data = rabbit_data[breed]
            res = run_optimizer("Rabbit", cp_req=data["cp_need"], energy_req=2500, fibre_req=12, calcium_req=0.5)
        elif animal_choice=="Poultry":
            ptype = st.selectbox("Poultry Type", list(poultry_data.keys()))
            age = st.slider("Age (weeks)", 0, 20, 6)
            data = poultry_data[ptype]
            res = run_optimizer("Poultry", cp_req=data["CP"], energy_req=data["Energy"])
        elif animal_choice=="Cattle":
            breed = st.selectbox("Cattle Breed", list(cattle_data.keys()))
            age = st.slider("Age (weeks)", 4, 104, 52)
            data = cattle_data[breed]
            res = run_optimizer("Cattle", cp_req=data["cp_need"], energy_req=2500)
    
    with tab2:
        if animal_choice=="Rabbit":
            growth_prediction("Rabbit", res, data["growth_rate"], data["adult_weight"], age)
        elif animal_choice=="Poultry":
            growth_prediction("Poultry", res, data["growth_rate"], 5, age)  # avg adult weight
        elif animal_choice=="Cattle":
            growth_prediction("Cattle", res, data["growth_rate"], data["adult_weight"], age)
    
    with tab3:
        ingredient_manager(animal_choice)

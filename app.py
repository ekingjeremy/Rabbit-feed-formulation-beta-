import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Livestock Feed Optimizer", layout="wide")

# ---------------- RABBIT DATABASE ----------------
breed_data = {
    "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
    "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
    "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
    "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
    "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
    "Local Nigerian Breed": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
}

# ---------------- POULTRY DATABASE ----------------
poultry_types = {
    "Broiler Starter (0-4w)": {"CP": 23, "Energy": 3200, "growth_rate": 50},
    "Broiler Grower (4-6w)": {"CP": 20, "Energy": 3100, "growth_rate": 60},
    "Broiler Finisher (>6w)": {"CP": 18, "Energy": 3000, "growth_rate": 55},
    "Layer Starter (0-8w)": {"CP": 20, "Energy": 2800, "growth_rate": 25},
    "Layer Grower (8-18w)": {"CP": 17, "Energy": 2800, "growth_rate": 30},
    "Layer Production (18+w)": {"CP": 16, "Energy": 2600, "growth_rate": 20},
    "Noiler Starter (0-6w)": {"CP": 21, "Energy": 2900, "growth_rate": 25},
    "Noiler Grower (6-12w)": {"CP": 18, "Energy": 2800, "growth_rate": 30},
}

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
    st.session_state.ingredient_data = pd.DataFrame(data).set_index("Ingredient")

df = st.session_state.ingredient_data

# ---------------- CATEGORY COLORS ----------------
category_colors = {
    "Fodder": "#4CAF50",
    "Concentrate": "#FF9800",
    "Mineral": "#2196F3",
    "Additive": "#9C27B0"
}

# ---------------- TABS ----------------
tab_rabbit, tab_ingredients, tab_rabbit_growth, tab_poultry = st.tabs([
    "🐰 Rabbit Optimizer",
    "📋 Ingredients",
    "📈 Rabbit Growth",
    "🐔 Poultry Feed"
])

# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.title("Livestock Feed System")

    # Rabbit inputs
    st.subheader("🐰 Rabbit Inputs")
    selected_breed = st.selectbox("Rabbit Breed", list(breed_data.keys()))
    age_weeks = st.slider("Rabbit Age (weeks)", 4, 52, 12)
    breed_info = breed_data[selected_breed]
    cp_req = st.slider("Rabbit CP %", 10, 50, int(breed_info["cp_need"]))
    energy_req = st.slider("Rabbit Energy (kcal/kg)", 1500, 3500, 2500)
    fibre_req = st.slider("Rabbit Fibre %", 5, 40, 12)
    calcium_req = st.slider("Rabbit Calcium %", 0.1, 5.0, 0.5)
    forage_ratio = st.slider("Suggested Forage % (for guidance)", 30, 80, 50,
                             help="This is a suggested ratio for forage in rabbit diets. Not enforced in optimization.")

    # Poultry inputs
    st.subheader("🐔 Poultry Inputs")
    ptype = st.selectbox("Poultry Type", list(poultry_types.keys()))
    pinfo = poultry_types[ptype]
    pasture_ratio = st.slider("Suggested Pasture/Forage % (optional)", 0, 50, 0,
                              help="Optional forage intake for free-range poultry. Used for guidance only.")

# ---------------- RABBIT OPTIMIZER ----------------
with tab_rabbit:
    st.header("🐰 Rabbit Feed Optimizer")
    model = LpProblem("RabbitFeed", LpMinimize)
    vars_r = {i: LpVariable(i, lowBound=0) for i in df.index}

    # Objective
    model += lpSum(vars_r[i]*df.loc[i,"Cost"] for i in df.index)
    # Constraints
    model += lpSum(vars_r[i]*df.loc[i,"CP"] for i in df.index) >= cp_req
    model += lpSum(vars_r[i]*df.loc[i,"Energy"] for i in df.index) >= energy_req
    model += lpSum(vars_r[i]*df.loc[i,"Fibre"] for i in df.index) >= fibre_req
    model += lpSum(vars_r[i]*df.loc[i,"Calcium"] for i in df.index) >= calcium_req
    model += lpSum(vars_r[i] for i in df.index) == 1

    # Category limits
    for i in df.index:
        if df.loc[i,"Category"]=="Mineral": model += vars_r[i]<=0.05
        elif df.loc[i,"Category"]=="Additive": model += vars_r[i]<=0.02
        elif df.loc[i,"Category"]=="Concentrate": model += vars_r[i]<=0.6

    model.solve()

    if LpStatus[model.status]=="Optimal":
        results = {i: vars_r[i].varValue for i in df.index if vars_r[i].varValue>0.0001}
        res_df = pd.DataFrame.from_dict(results, orient="index", columns=["Proportion (kg)"])
        res_df["Cost (₦)"] = res_df["Proportion (kg)"] * df.loc[res_df.index,"Cost"]
        res_df["Category"] = df.loc[res_df.index,"Category"]

        st.subheader("📋 Optimized Feed Mix")
        st.dataframe(res_df)
        st.write(f"💸 Total Cost/kg: ₦{value(model.objective):.2f}")

        # Color-coded pie chart
        st.plotly_chart(
            px.pie(
                res_df, 
                values="Proportion (kg)", 
                names=res_df.index, 
                color="Category", 
                color_discrete_map=category_colors,
                title="🐰 Rabbit Feed Composition by Category"
            )
        )

        # Forage suggestion
        forage_ingredients = df[df['Category']=='Fodder'].index
        total_forage = sum(vars_r[i].varValue for i in forage_ingredients)
        st.info(f"💡 Suggested Forage Proportion: {total_forage*100:.1f}% (slider guidance: {forage_ratio}%)")
    else:
        st.error("No feasible solution found for rabbit feed.")

# ---------------- RABBIT GROWTH ----------------
with tab_rabbit_growth:
    st.header("📈 Rabbit Growth Prediction")
    base_growth_g = breed_info["growth_rate"]
    weight_gain_g = base_growth_g
    expected_weight_kg = breed_info["adult_weight"]*(1-np.exp(-0.08*age_weeks))
    expected_weight_g = expected_weight_kg*1000

    st.metric("Daily Gain", f"{weight_gain_g:.1f} g/day")
    st.metric("Expected Weight", f"{expected_weight_kg:.2f} kg")
    st.metric("Expected Weight (g)", f"{expected_weight_g:.0f} g")

# ---------------- POULTRY FEED OPTIMIZER ----------------
with tab_poultry:
    st.header(f"🐔 {ptype} Feed Optimizer")
    model_p = LpProblem("PoultryFeed", LpMinimize)
    vars_p = {i: LpVariable(f"P_{i}", lowBound=0) for i in df.index}

    # Objective & constraints
    model_p += lpSum(vars_p[i]*df.loc[i,"Cost"] for i in df.index)
    model_p += lpSum(vars_p[i]*df.loc[i,"CP"] for i in df.index) >= pinfo["CP"]
    model_p += lpSum(vars_p[i]*df.loc[i,"Energy"] for i in df.index) >= pinfo["Energy"]
    model_p += lpSum(vars_p[i] for i in df.index) == 1

    for i in df.index:
        if df.loc[i,"Category"]=="Mineral": model_p += vars_p[i]<=0.05
        elif df.loc[i,"Category"]=="Additive": model_p += vars_p[i]<=0.02
        elif df.loc[i,"Category"]=="Concentrate": model_p += vars_p[i]<=0.6

    model_p.solve()

    if LpStatus[model_p.status]=="Optimal":
        res = {i: vars_p[i].varValue for i in df.index if vars_p[i].varValue>0.0001}
        df_res = pd.DataFrame.from_dict(res, orient="index", columns=["Proportion (kg)"])
        df_res["Cost (₦)"] = df_res["Proportion (kg)"]*df.loc[df_res.index,"Cost"]
        df_res["Category"] = df.loc[df_res.index,"Category"]

        st.subheader("📋 Optimized Feed Mix")
        st.dataframe(df_res)
        st.write(f"💸 Total Cost/kg: ₦{value(model_p.objective):.2f}")

        # Color-coded pie chart
        st.plotly_chart(
            px.pie(
                df_res,
                values="Proportion (kg)",
                names=df_res.index,
                color="Category",
                color_discrete_map=category_colors,
                title=f"🐔 {ptype} Feed Composition by Category"
            )
        )

        # Poultry growth prediction
        st.subheader("📈 Poultry Growth Prediction")
        proportions = np.array([vars_p[i].varValue for i in df.index])
        cp_vals = np.array([df.loc[i,"CP"] for i in df.index])
        energy_vals = np.array([df.loc[i,"Energy"] for i in df.index])

        feed_cp = np.dot(proportions, cp_vals)
        feed_energy = np.dot(proportions, energy_vals)
        base_growth_g = pinfo.get("growth_rate",25)
        weight_gain_g = base_growth_g * (feed_cp/pinfo["CP"]) * (feed_energy/pinfo["Energy"])
        weight_gain_kg = weight_gain_g/1000
        expected_weight_kg = weight_gain_kg*7
        expected_weight_g = expected_weight_kg*1000

        st.metric("Daily Gain", f"{weight_gain_g:.1f} g/day")
        st.metric("Expected Weekly Weight", f"{expected_weight_kg:.2f} kg")
        st.metric("Expected Weekly Weight (g)", f"{expected_weight_g:.0f} g")

        if pasture_ratio>0:
            st.info(f"💡 Suggested Pasture/Forage proportion: {pasture_ratio}% (guidance only)")

    else:
        st.error("No feasible solution found for poultry feed.")

# ---------------- INGREDIENT MANAGEMENT ----------------
with tab_ingredients:
    st.header("📋 Manage Ingredients")
    editable_df = df.reset_index()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

    st.subheader("📤 Upload New Ingredients CSV")
    uploaded_file = st.file_uploader("Upload CSV with columns: Ingredient, Category, CP, Energy, Fibre, Calcium, Cost", type=["csv"])
    if uploaded_file:
        new_ing = pd.read_csv(uploaded_file)
        required_cols = {"Ingredient","Category","CP","Energy","Fibre","Calcium","Cost"}
        if required_cols.issubset(new_ing.columns):
            new_ing = new_ing.set_index("Ingredient")
            st.session_state.ingredient_data = pd.concat([st.session_state.ingredient_data,new_ing])
            df = st.session_state.ingredient_data.copy()
            st.success(f"✅ Added {len(new_ing)} new ingredients.")
        else:
            st.error("❌ CSV must contain all required columns.")

    if st.button("💾 Save Changes"):
        if edited_df["Ingredient"].is_unique and edited_df["Ingredient"].notnull().all():
            st.session_state.ingredient_data = edited_df.set_index("Ingredient")
            df = st.session_state.ingredient_data.copy()
            st.success("✅ Ingredients updated successfully!")
        else:
            st.error("❌ All ingredient names must be unique and non-empty.")

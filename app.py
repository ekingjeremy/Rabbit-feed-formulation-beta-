import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🐄🐰🐔 Nigerian Livestock Feed Optimizer", layout="wide")

# ---------------- INGREDIENT DATABASE ----------------
@st.cache_data
def load_ingredients():
    data = {
        "Ingredient": [
            "Maize (grain)","Soybean meal","Groundnut cake","Palm kernel cake","Wheat offal",
            "Rice bran","Brewers dried grains","Fish meal","Bone meal","Limestone",
            "Salt","Vitamin/Mineral premix","Methionine","Lysine",
            "Alfalfa hay","Elephant grass","Guinea grass","Cowpea haulms","Napier grass","Maize silage","Hay (grass/legume)"
        ],
        "Category": [
            "Concentrate","Concentrate","Concentrate","Concentrate","Concentrate",
            "Concentrate","Concentrate","Concentrate","Mineral","Mineral",
            "Mineral","Additive","Additive","Additive",
            "Fodder","Fodder","Fodder","Fodder","Fodder","Fodder","Fodder"
        ],
        "CP (%)":[9,42,45,18,17,12,18,60,0,0,0,0,0,0,18,8,10,20,12,9,8],
        "Energy (kcal/kg)":[3300,2700,2640,2175,1870,2860,1980,2800,0,0,0,0,0,0,2300,2200,2300,2100,2200,2000,1800],
        "Fibre (%)":[3,6.5,6,15,10,12,20,1,0,0,0,0,0,0,25,32,28,18,30,28,35],
        "Calcium (%)":[0.02,0.3,0.4,0.2,0.3,0.24,0.4,4,20,38,0,0,0,0,1.5,0.5,0.6,1.2,0.6,0.5,0.5],
        "Cost (₦/kg)":[550,760,900,135,150,230,350,5000,295,155,65,1500,3100,2400,180,120,120,200,150,140,170]
    }
    df = pd.DataFrame(data)
    df.set_index("Ingredient", inplace=True)
    return df

df = load_ingredients()

# ---------------- SPECIES DATABASES ----------------
rabbit_breeds = {
    "New Zealand White": {"adult_weight": 4.5, "growth_rate": 35, "cp_need": 16},
    "Californian": {"adult_weight": 4.0, "growth_rate": 32, "cp_need": 16},
    "Chinchilla": {"adult_weight": 3.5, "growth_rate": 28, "cp_need": 15},
    "Flemish Giant": {"adult_weight": 6.5, "growth_rate": 40, "cp_need": 17},
    "Dutch": {"adult_weight": 2.5, "growth_rate": 20, "cp_need": 15},
    "Local Nigerian": {"adult_weight": 2.8, "growth_rate": 18, "cp_need": 14}
}

poultry_types = {
    "Broiler Starter": {"CP": 22, "Energy": 2900},
    "Broiler Grower": {"CP": 19, "Energy": 3000},
    "Broiler Finisher": {"CP": 17, "Energy": 3000},
    "Layer Starter": {"CP": 20, "Energy": 2800},
    "Layer Grower": {"CP": 17, "Energy": 2800},
    "Layer Production": {"CP": 16, "Energy": 2600},
}

cattle_types = {
    "Dairy Cow": {"CP": 14, "Energy": 2500, "Fibre": 30, "adult_weight": 450, "growth_rate": 800},
    "Beef Cattle": {"CP": 13, "Energy": 2400, "Fibre": 32, "adult_weight": 400, "growth_rate": 700},
}

# ---------------- LANDING PAGE ----------------
st.title("🐄🐰🐔 Nigerian Livestock Feed Formulator")
st.markdown("""
This app allows you to **formulate feed for Rabbits, Poultry, and Cattle** using Nigerian ingredients.
You can select **Concentrate vs Fodder**, edit ingredient prices, and see **growth predictions**.
""")

animal_choice = st.selectbox("Select an animal to formulate feed for:", ["Rabbit", "Poultry", "Cattle"])

# ---------------- CREATE TABS ----------------
tab1, tab2, tab3 = st.tabs(["Feed Optimizer", "Manage Ingredients", "Growth Prediction"])

# ---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:
    st.header("Feed Settings")
    ration_type = st.selectbox("Ration Type", ["Mixed (Fodder+Concentrate)","Concentrate only","Fodder only"])
    
    if animal_choice=="Rabbit":
        breed = st.selectbox("Rabbit Breed", list(rabbit_breeds.keys()))
        age_weeks = st.slider("Age (weeks)", 4, 52, 12)
        binfo = rabbit_breeds[breed]
        cp_req = st.slider("Crude Protein (%)",10,50,binfo["cp_need"])
        energy_req = st.slider("Energy (kcal/kg)",1500,3500,2500)
        fibre_req = st.slider("Fibre (%)",5,40,12)
        calcium_req = st.slider("Calcium (%)",0.1,5.0,0.5)
    elif animal_choice=="Poultry":
        ptype = st.selectbox("Poultry Type", list(poultry_types.keys()))
        pinfo = poultry_types[ptype]
        cp_req = st.slider("Crude Protein (%)",15,30,pinfo["CP"])
        energy_req = st.slider("Energy (kcal/kg)",2500,3500,pinfo["Energy"])
        fibre_req = st.slider("Fibre (%)",0,10,5)
        calcium_req = st.slider("Calcium (%)",0.5,4.0,1.0)
        age_weeks = 12
    else:
        ctype = st.selectbox("Cattle Type", list(cattle_types.keys()))
        cinfo = cattle_types[ctype]
        cp_req = st.slider("Crude Protein (%)",10,20,cinfo["CP"])
        energy_req = st.slider("Energy (kcal/kg)",2000,3000,cinfo["Energy"])
        fibre_req = st.slider("Fibre (%)",25,40,cinfo["Fibre"])
        calcium_req = st.slider("Calcium (%)",0.1,4.0,1.0)
        age_weeks = st.slider("Age (weeks)",1,104,52)

# ---------------- FILTER INGREDIENTS ----------------
if ration_type=="Concentrate only":
    ingredients=df[df['Category'].isin(["Concentrate","Additive","Mineral"])]
elif ration_type=="Fodder only":
    ingredients=df[df['Category']=="Fodder"]
else:
    ingredients=df[df['Category'].isin(["Fodder","Concentrate","Additive","Mineral"])]

# ---------------- FEED OPTIMIZER ----------------
with tab1:
    st.header(f"🔬 {animal_choice} Feed Optimizer")
    model = LpProblem("FeedOptimization", LpMinimize)
    vars = {i: LpVariable(i,lowBound=0) for i in ingredients.index}

    # Objective: minimize cost
    model += lpSum(vars[i]*ingredients.loc[i,"Cost (₦/kg)"] for i in ingredients.index)
    
    # Total proportion = 1
    model += lpSum(vars[i] for i in ingredients.index)==1
    
    # Nutrient constraints
    model += lpSum(vars[i]*ingredients.loc[i,"CP (%)"] for i in ingredients.index)>=cp_req
    model += lpSum(vars[i]*ingredients.loc[i,"Energy (kcal/kg)"] for i in ingredients.index)>=energy_req
    model += lpSum(vars[i]*ingredients.loc[i,"Fibre (%)"] for i in ingredients.index)>=fibre_req
    model += lpSum(vars[i]*ingredients.loc[i,"Calcium (%)"] for i in ingredients.index)>=calcium_req

    # Limits for minerals and additives
    for i in ingredients.index:
        if ingredients.loc[i,"Category"]=="Mineral":
            model += vars[i]<=0.05
        elif ingredients.loc[i,"Category"]=="Additive":
            model += vars[i]<=0.02
        elif ingredients.loc[i,"Category"]=="Concentrate":
            model += vars[i]<=0.6

    # Solve
    model.solve()

    if LpStatus[model.status]=="Optimal":
        res={i: vars[i].varValue for i in ingredients.index if vars[i].varValue>0.0001}
        res_df=pd.DataFrame.from_dict(res,orient="index",columns=["Proportion (kg)"])
        res_df["Cost (₦)"]=res_df["Proportion (kg)"]*ingredients.loc[res_df.index,"Cost (₦/kg)"]
        st.dataframe(res_df)
        st.write(f"**💸 Cost/kg Feed: ₦{value(model.objective):.2f}**")
        st.plotly_chart(px.pie(res_df,values="Proportion (kg)",names=res_df.index))
        csv=res_df.to_csv().encode('utf-8')
        st.download_button("💾 Download CSV",data=csv,file_name=f"{animal_choice}_feed.csv",mime="text/csv")
    else:
        st.error("No feasible solution with current constraints.")

# ---------------- INGREDIENTS MANAGEMENT ----------------
with tab2:
    st.header("📋 Edit Ingredients")
    editable = ingredients.reset_index()
    edited = st.data_editor(editable,num_rows="dynamic",use_container_width=True)
    uploaded = st.file_uploader("Upload CSV to add ingredients", type=["csv"])
    if uploaded:
        new = pd.read_csv(uploaded).set_index("Ingredient")
        df = pd.concat([df,new])
        st.success("New ingredients added!")
    if st.button("💾 Save Edited"):
        df.update(edited.set_index("Ingredient"))
        st.success("Ingredients saved!")

# ---------------- GROWTH PREDICTION ----------------
with tab3:
    st.header("📈 Growth Prediction")
    if LpStatus[model.status]=="Optimal":
        props=np.array([vars[i].varValue for i in ingredients.index])
        cpvals=np.array([ingredients.loc[i,"CP (%)"] for i in ingredients.index])
        enervals=np.array([ingredients.loc[i,"Energy (kcal/kg)"] for i in ingredients.index])
        feed_cp=np.dot(props,cpvals)
        feed_energy=np.dot(props,enervals)

        if animal_choice=="Rabbit":
            base=binfo["growth_rate"]
            expected=binfo["adult_weight"]*(1-np.exp(-0.08*age_weeks))
        elif animal_choice=="Poultry":
            base=50  # simplified
            expected=2.5
        else:
            base=cinfo["growth_rate"]
            expected=cinfo["adult_weight"]*(1-np.exp(-0.005*age_weeks))

        wg=base*(0.5*(feed_cp/cp_req))*(0.3*(feed_energy/energy_req))
        st.metric("Daily gain",f"{wg:.1f} g/day")
        st.metric("Expected weight",f"{expected:.2f} kg")

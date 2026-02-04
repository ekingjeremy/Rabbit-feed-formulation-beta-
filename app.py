import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

st.set_page_config(
    page_title="🌾 Jeremiah's Nigerian Livestock Feed Formulator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLING --------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* {font-family: 'Inter', sans-serif;}
.main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1rem;}
.hero-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
padding: 2rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;}
.metric-card {background: white; padding: 1rem; border-radius: 12px; text-align: center;}
.metric-value {font-size: 2rem; font-weight: 800; color: #667eea;}
</style>""", unsafe_allow_html=True)

# -------------------- HELPER --------------------
def categorize_ingredient(name, cp, fiber, energy):
    name = name.lower()
    if cp > 30: return 'Protein Sources'
    if fiber > 20: return 'Fiber Sources'
    if energy > 2800 and cp < 15: return 'Energy Sources'
    return 'Other'

@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    for df in [rabbit, poultry, cattle]:
        df["Category"] = df.apply(lambda r: categorize_ingredient(r['Ingredient'], r['CP'], r['Fiber'], r['Energy']), axis=1)
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

@st.cache_resource
def train_model(data):
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal",
              "Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)
    return model

model = train_model(ml_df)

# -------------------- HERO --------------------
st.markdown("""<div class="hero-container">
<h1>🌾 Nigerian Livestock Feed Formulator</h1>
AI-Powered Precision Nutrition for Rabbits, Poultry & Cattle
</div>""", unsafe_allow_html=True)

# -------------------- ANIMAL SELECT --------------------
animal = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])

if animal == "Rabbit":
    df = rabbit_df.copy()
elif animal == "Poultry":
    df = poultry_df.copy()
else:
    df = cattle_df.copy()

# -------------------- BREED DATABASE --------------------
rabbit_breeds = ["New Zealand White","Californian","Chinchilla","Dutch","Flemish Giant","Angora","Rex","Lionhead"]
poultry_breeds = {
    "Broiler":["Ross 308","Cobb 500","Arbor Acres","Hubbard"],
    "Layer":["ISA Brown","Hy-Line","Lohmann Brown","Noiler"],
    "Indigenous":["Yoruba Ecotype","Fulani","Naked Neck"]
}
cattle_breeds = ["White Fulani","Sokoto Gudali","Red Bororo","Muturu","N'Dama","Friesian Cross"]

st.sidebar.header("🐾 Breed Selection")
if animal == "Rabbit":
    breed = st.sidebar.selectbox("Select Rabbit Breed", rabbit_breeds)
elif animal == "Poultry":
    bird_type = st.sidebar.selectbox("Production Type", list(poultry_breeds.keys()))
    breed = st.sidebar.selectbox("Select Breed", poultry_breeds[bird_type])
else:
    breed = st.sidebar.selectbox("Select Cattle Breed", cattle_breeds)

# -------------------- PARAMETERS --------------------
st.sidebar.header("🎯 Animal Parameters")
age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 8, 30, 18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 1800, 12000, 3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

# -------------------- NUTRIENT ENGINE --------------------
def get_nutrient_requirements(animal, age):
    if animal=="Rabbit":
        if age<=8: return "Weaner", {"CP":"18–20%","Energy":"2500–2700","Fiber":"12–14%"}
        elif age<=20: return "Grower", {"CP":"16–18%","Energy":"2400–2600","Fiber":"13–15%"}
        else: return "Adult", {"CP":"15–17%","Energy":"2300–2500","Fiber":"14–16%"}
    if animal=="Poultry":
        if age<=4: return "Starter", {"CP":"21–23%","Energy":"2900–3000","Calcium":"1%"}
        elif age<=8: return "Grower", {"CP":"18–20%","Energy":"2800–2900","Calcium":"0.9%"}
        else: return "Finisher/Layer", {"CP":"16–18%","Energy":"2700–2800","Calcium":"3.5–4%"}
    if animal=="Cattle":
        if age<=24: return "Calf/Grower", {"CP":"16–18%","Energy":"2600–2800","Fiber":"18%"}
        else: return "Adult/Fattening", {"CP":"12–14%","Energy":"2400–2600","Fiber":"20%"}

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Feed Optimizer","📈 AI Prediction","📋 Database","🧠 Nutrient Guide"])

# -------------------- TAB 1 OPTIMIZER --------------------
with tab1:
    if st.button("🚀 Optimize Feed Formula"):
        prob = LpProblem("FeedMix", LpMinimize)
        ingredients = df["Ingredient"].tolist()
        vars_dict = LpVariable.dicts("Ingr", ingredients, lowBound=0)

        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
        prob += lpSum(vars_dict[i] for i in ingredients) == 1
        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients) >= cp_req
        prob += lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients) >= energy_req

        prob.solve()

        if LpStatus[prob.status]=="Optimal":
            result = {i:vars_dict[i].value() for i in ingredients if vars_dict[i].value()>0.001}
            st.dataframe(pd.DataFrame(result.items(), columns=["Ingredient","Proportion"]))

# -------------------- TAB 2 AI --------------------
with tab2:
    avg_cp, avg_energy = df["CP"].mean(), df["Energy"].mean()
    X_input = np.array([[age,weight,cp_req,energy_req,feed_intake,avg_cp,avg_energy]])
    prediction = model.predict(X_input)[0]
    st.metric("Daily Gain (g)", f"{prediction:.1f}")

# -------------------- TAB 3 DATABASE --------------------
with tab3:
    st.dataframe(df)

# -------------------- TAB 4 NUTRIENT GUIDE --------------------
with tab4:
    stage, nutrients = get_nutrient_requirements(animal, age)
    st.markdown(f"### 🐾 {animal} | Breed: **{breed}** | Stage: **{stage}**")
    cols = st.columns(len(nutrients))
    for col,(nut,val) in zip(cols,nutrients.items()):
        col.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div>{nut}</div>", unsafe_allow_html=True)

    st.markdown("""
**Protein** → Growth & production  
**Energy** → Weight gain  
**Fiber** → Digestive health  
**Calcium** → Bones & eggshells  
""")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🌾 Powered by AI for Nigerian Farmers")

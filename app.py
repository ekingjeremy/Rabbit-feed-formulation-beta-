import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🌾 Necstech Livestock Feed Formulator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM GLOBAL STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

.main {
    background: linear-gradient(135deg, #eef2f3 0%, #d9e4f5 100%);
}

/* HERO */
.hero {
    background: linear-gradient(135deg,#2b5876,#4e4376);
    padding: 4rem 2rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {font-size:3rem;font-weight:800;}
.hero p {font-size:1.2rem;opacity:0.9;}

/* FEATURE CARDS */
.feature-card {
    background:white;
    padding:1.5rem;
    border-radius:16px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    text-align:center;
    transition:0.3s;
}
.feature-card:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 25px rgba(0,0,0,0.12);
}

/* SECTION TITLE */
.section-title {
    font-size:2rem;
    font-weight:800;
    margin:1rem 0;
    color:#2b5876;
}

/* METRIC CARD */
.metric-card {
    background:white;
    padding:1rem;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.08);
    text-align:center;
}
.metric-value {
    font-size:2rem;
    font-weight:800;
    color:#4e4376;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#2b5876,#4e4376);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LANDING PAGE
# =========================================================
st.markdown("""
<div class="hero">
<h1>🌾 Necstech Livestock Feed Formulator</h1>
<p>AI-Powered Precision Nutrition for Smarter, Profitable Farming in Nigeria</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Why Necstech Matters</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
features = [
    ("🥣 Least-Cost Formulation", "Balanced feed at minimum cost"),
    ("🧠 AI Growth Prediction", "Forecast animal performance"),
    ("📊 Data-Driven Nutrition", "Scientific feeding guidance"),
    ("🇳🇬 Built for Nigeria", "Local breeds & ingredients")
]
for col, (title, desc) in zip([c1, c2, c3, c4], features):
    col.markdown(f"<div class='feature-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# DATA + MODEL
# =========================================================
def categorize_ingredient(name, cp, fiber, energy):
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
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal","Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)
    return model

model = train_model(ml_df)

# =========================================================
# ANIMAL SELECTION
# =========================================================
st.markdown('<div class="section-title">Formulation System</div>', unsafe_allow_html=True)

animal = st.selectbox("🐾 Select Animal Type", ["Rabbit","Poultry","Cattle"])
df = rabbit_df.copy() if animal=="Rabbit" else poultry_df.copy() if animal=="Poultry" else cattle_df.copy()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("🐾 Breed Selection")

rabbit_breeds=["New Zealand White","Californian","Chinchilla","Dutch","Flemish Giant","Angora","Rex","Lionhead"]
poultry_breeds={"Broiler":["Ross 308","Cobb 500"],"Layer":["ISA Brown","Hy-Line"],"Indigenous":["Fulani","Naked Neck"]}
cattle_breeds=["White Fulani","Sokoto Gudali","Red Bororo","Muturu"]

if animal=="Rabbit":
    breed=st.sidebar.selectbox("Rabbit Breed",rabbit_breeds)
elif animal=="Poultry":
    bird_type=st.sidebar.selectbox("Production Type",list(poultry_breeds.keys()))
    breed=st.sidebar.selectbox("Breed",poultry_breeds[bird_type])
else:
    breed=st.sidebar.selectbox("Cattle Breed",cattle_breeds)

st.sidebar.header("🎯 Animal Parameters")
age=st.sidebar.slider("Age (weeks)",1,120,8)
weight=st.sidebar.slider("Body Weight (kg)",0.1,600.0,2.0)
cp_req=st.sidebar.slider("Crude Protein Requirement (%)",8,30,18)
energy_req=st.sidebar.slider("Energy Requirement",1800,12000,3000)
feed_intake=st.sidebar.slider("Feed Intake (kg/day)",0.05,30.0,0.5)

# =========================================================
# NUTRIENT ENGINE
# =========================================================
def get_nutrient_requirements(animal, age):
    if animal=="Rabbit":
        if age<=8: return "Weaner",{"CP":"18–20%","Energy":"2500–2700","Fiber":"12–14%"}
        elif age<=20: return "Grower",{"CP":"16–18%","Energy":"2400–2600","Fiber":"13–15%"}
        else: return "Adult",{"CP":"15–17%","Energy":"2300–2500","Fiber":"14–16%"}
    if animal=="Poultry":
        if age<=4: return "Starter",{"CP":"21–23%","Energy":"2900–3000","Calcium":"1%"}
        elif age<=8: return "Grower",{"CP":"18–20%","Energy":"2800–2900","Calcium":"0.9%"}
        else: return "Finisher/Layer",{"CP":"16–18%","Energy":"2700–2800","Calcium":"3.5–4%"}
    if animal=="Cattle":
        if age<=24: return "Calf/Grower",{"CP":"16–18%","Energy":"2600–2800","Fiber":"18%"}
        else: return "Adult/Fattening",{"CP":"12–14%","Energy":"2400–2600","Fiber":"20%"}

# =========================================================
# TABS
# =========================================================
tab1,tab2,tab3,tab4=st.tabs(["🔬 Feed Optimizer","📈 AI Prediction","📋 Database","🧠 Nutrient Guide"])

with tab1:
    st.subheader("Least-Cost Feed Optimization")
    if st.button("🚀 Optimize Feed Formula"):
        prob=LpProblem("FeedMix",LpMinimize)
        ingredients=df["Ingredient"].tolist()
        vars_dict=LpVariable.dicts("Ingr",ingredients,lowBound=0)
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
        prob+=lpSum(vars_dict[i] for i in ingredients)==1
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients)>=cp_req
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients)>=energy_req
        prob.solve()
        if LpStatus[prob.status]=="Optimal":
            result={i:vars_dict[i].value() for i in ingredients if vars_dict[i].value()>0.001}
            st.dataframe(pd.DataFrame(result.items(),columns=["Ingredient","Proportion"]))

with tab2:
    st.subheader("AI Growth Prediction")
    avg_cp,avg_energy=df["CP"].mean(),df["Energy"].mean()
    X_input=np.array([[age,weight,cp_req,energy_req,feed_intake,avg_cp,avg_energy]])
    prediction=model.predict(X_input)[0]
    st.metric("Daily Gain (g)",f"{prediction:.1f}")

with tab3:
    st.subheader("Ingredient Database")
    st.dataframe(df)

with tab4:
    stage,nutrients=get_nutrient_requirements(animal,age)
    st.markdown(f"### {animal} • {breed} • Stage: {stage}")
    cols=st.columns(len(nutrients))
    for col,(nut,val) in zip(cols,nutrients.items()):
        col.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div>{nut}</div>",unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center>🌾 Powered by Necstech AI for Nigerian Farmers</center>", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🌾 Necstech Livestock Feed Formulator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM GLOBAL STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

.main {
    background: linear-gradient(135deg, #eef2f3 0%, #d9e4f5 100%);
}

/* HERO */
.hero {
    background: linear-gradient(135deg,#2b5876,#4e4376);
    padding: 4rem 2rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {font-size:3rem;font-weight:800;}
.hero p {font-size:1.2rem;opacity:0.9;}

/* FEATURE CARDS */
.feature-card {
    background:white;
    padding:1.5rem;
    border-radius:16px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    text-align:center;
    transition:0.3s;
}
.feature-card:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 25px rgba(0,0,0,0.12);
}

/* SECTION TITLE */
.section-title {
    font-size:2rem;
    font-weight:800;
    margin:1rem 0;
    color:#2b5876;
}

/* METRIC CARD */
.metric-card {
    background:white;
    padding:1rem;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.08);
    text-align:center;
}
.metric-value {
    font-size:2rem;
    font-weight:800;
    color:#4e4376;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#2b5876,#4e4376);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LANDING PAGE
# =========================================================
st.markdown("""
<div class="hero">
<h1>🌾 Necstech Livestock Feed Formulator</h1>
<p>AI-Powered Precision Nutrition for Smarter, Profitable Farming in Nigeria</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Why Necstech Matters</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
features = [
    ("🥣 Least-Cost Formulation", "Balanced feed at minimum cost"),
    ("🧠 AI Growth Prediction", "Forecast animal performance"),
    ("📊 Data-Driven Nutrition", "Scientific feeding guidance"),
    ("🇳🇬 Built for Nigeria", "Local breeds & ingredients")
]
for col, (title, desc) in zip([c1, c2, c3, c4], features):
    col.markdown(f"<div class='feature-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# DATA + MODEL
# =========================================================
def categorize_ingredient(name, cp, fiber, energy):
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
    X = data[["Age_Weeks","Body_Weight_kg","CP_Requirement_%","Energy_Requirement_Kcal","Feed_Intake_kg","Ingredient_CP_%","Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)
    return model

model = train_model(ml_df)

# =========================================================
# ANIMAL SELECTION
# =========================================================
st.markdown('<div class="section-title">Formulation System</div>', unsafe_allow_html=True)

animal = st.selectbox("🐾 Select Animal Type", ["Rabbit","Poultry","Cattle"])
df = rabbit_df.copy() if animal=="Rabbit" else poultry_df.copy() if animal=="Poultry" else cattle_df.copy()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("🐾 Breed Selection")

rabbit_breeds=["New Zealand White","Californian","Chinchilla","Dutch","Flemish Giant","Angora","Rex","Lionhead"]
poultry_breeds={"Broiler":["Ross 308","Cobb 500"],"Layer":["ISA Brown","Hy-Line"],"Indigenous":["Fulani","Naked Neck"]}
cattle_breeds=["White Fulani","Sokoto Gudali","Red Bororo","Muturu"]

if animal=="Rabbit":
    breed=st.sidebar.selectbox("Rabbit Breed",rabbit_breeds)
elif animal=="Poultry":
    bird_type=st.sidebar.selectbox("Production Type",list(poultry_breeds.keys()))
    breed=st.sidebar.selectbox("Breed",poultry_breeds[bird_type])
else:
    breed=st.sidebar.selectbox("Cattle Breed",cattle_breeds)

st.sidebar.header("🎯 Animal Parameters")
age=st.sidebar.slider("Age (weeks)",1,120,8)
weight=st.sidebar.slider("Body Weight (kg)",0.1,600.0,2.0)
cp_req=st.sidebar.slider("Crude Protein Requirement (%)",8,30,18)
energy_req=st.sidebar.slider("Energy Requirement",1800,12000,3000)
feed_intake=st.sidebar.slider("Feed Intake (kg/day)",0.05,30.0,0.5)

# =========================================================
# NUTRIENT ENGINE
# =========================================================
def get_nutrient_requirements(animal, age):
    if animal=="Rabbit":
        if age<=8: return "Weaner",{"CP":"18–20%","Energy":"2500–2700","Fiber":"12–14%"}
        elif age<=20: return "Grower",{"CP":"16–18%","Energy":"2400–2600","Fiber":"13–15%"}
        else: return "Adult",{"CP":"15–17%","Energy":"2300–2500","Fiber":"14–16%"}
    if animal=="Poultry":
        if age<=4: return "Starter",{"CP":"21–23%","Energy":"2900–3000","Calcium":"1%"}
        elif age<=8: return "Grower",{"CP":"18–20%","Energy":"2800–2900","Calcium":"0.9%"}
        else: return "Finisher/Layer",{"CP":"16–18%","Energy":"2700–2800","Calcium":"3.5–4%"}
    if animal=="Cattle":
        if age<=24: return "Calf/Grower",{"CP":"16–18%","Energy":"2600–2800","Fiber":"18%"}
        else: return "Adult/Fattening",{"CP":"12–14%","Energy":"2400–2600","Fiber":"20%"}

# =========================================================
# TABS
# =========================================================
tab1,tab2,tab3,tab4=st.tabs(["🔬 Feed Optimizer","📈 AI Prediction","📋 Database","🧠 Nutrient Guide"])

with tab1:
    st.subheader("Least-Cost Feed Optimization")
    if st.button("🚀 Optimize Feed Formula"):
        prob=LpProblem("FeedMix",LpMinimize)
        ingredients=df["Ingredient"].tolist()
        vars_dict=LpVariable.dicts("Ingr",ingredients,lowBound=0)
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Cost"].values[0] for i in ingredients)
        prob+=lpSum(vars_dict[i] for i in ingredients)==1
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["CP"].values[0] for i in ingredients)>=cp_req
        prob+=lpSum(vars_dict[i]*df[df["Ingredient"]==i]["Energy"].values[0] for i in ingredients)>=energy_req
        prob.solve()
        if LpStatus[prob.status]=="Optimal":
            result={i:vars_dict[i].value() for i in ingredients if vars_dict[i].value()>0.001}
            st.dataframe(pd.DataFrame(result.items(),columns=["Ingredient","Proportion"]))

with tab2:
    st.subheader("AI Growth Prediction")
    avg_cp,avg_energy=df["CP"].mean(),df["Energy"].mean()
    X_input=np.array([[age,weight,cp_req,energy_req,feed_intake,avg_cp,avg_energy]])
    prediction=model.predict(X_input)[0]
    st.metric("Daily Gain (g)",f"{prediction:.1f}")

with tab3:
    st.subheader("Ingredient Database")
    st.dataframe(df)

with tab4:
    stage,nutrients=get_nutrient_requirements(animal,age)
    st.markdown(f"### {animal} • {breed} • Stage: {stage}")
    cols=st.columns(len(nutrients))
    for col,(nut,val) in zip(cols,nutrients.items()):
        col.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div>{nut}</div>",unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center>🌾 Powered by Necstech AI for Nigerian Farmers</center>", unsafe_allow_html=True)

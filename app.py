import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🌾 Jeremiah's Nigerian Livestock Feed Formulator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
    }
    h1 {
        font-size: clamp(1.5rem, 4vw, 3rem) !important;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: clamp(1.5rem, 4vw, 3rem);
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .feature-card {
        background: white;
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 800;
        color: #667eea;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: clamp(0.5rem, 2vw, 0.75rem) clamp(1.5rem, 4vw, 2.5rem);
        border: none;
        font-weight: 700;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Helper function: categorize ingredients
def categorize_ingredient(ingredient_name, cp, fiber, energy):
    ingredient_lower = ingredient_name.lower()
    if any(word in ingredient_lower for word in ['limestone', 'phosphate', 'salt', 'bone meal', 'mineral', 'premix']):
        return 'Minerals & Supplements'
    if any(word in ingredient_lower for word in ['lysine', 'methionine', 'threonine', 'urea']):
        return 'Supplements'
    if cp > 30:
        return 'Protein Sources'
    if fiber > 20:
        return 'Fiber Sources'
    if energy > 2800 and cp < 15:
        return 'Energy Sources'
    if any(word in ingredient_lower for word in ['grass', 'hay', 'stover', 'straw', 'silage', 'haulm', 'vine', 'leaves']):
        return 'Forages & Roughages'
    if 15 <= cp <= 30:
        return 'Protein Concentrates'
    return 'Energy Sources'

# Load data
@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    
    for df in [rabbit, poultry, cattle]:
        df['Category'] = df.apply(
            lambda row: categorize_ingredient(row['Ingredient'], row['CP'], row['Fiber'], row['Energy']), 
            axis=1
        )
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

# Train ML model
@st.cache_resource
def train_model(data):
    X = data[["Age_Weeks", "Body_Weight_kg", "CP_Requirement_%", "Energy_Requirement_Kcal",
              "Feed_Intake_kg", "Ingredient_CP_%", "Ingredient_Energy"]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)

# Hero section
st.markdown("""
<div class="hero-container">
    <div style="font-size: clamp(1.8rem, 5vw, 3.5rem); font-weight: 800;">🌾 Nigerian Livestock Feed Formulator</div>
    <div style="font-size: clamp(0.9rem, 2.5vw, 1.4rem); margin-top: 1rem;">AI-Powered Precision Nutrition for Rabbits, Poultry & Cattle</div>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)
features = [
    ("💰", "Least-Cost", "Optimize feed costs"),
    ("🤖", "AI Prediction", "ML growth forecasts"),
    ("🇳🇬", "Nigerian Data", "97 local ingredients"),
    ("🐰", "31 Breeds", "Complete database")
]

for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-weight: 700; margin: 0.5rem 0;">{title}</div>
            <div style="color: #666;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Animal selection
animal = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])

if animal == "Rabbit":
    df = rabbit_df.copy()
    st.info("🐰 **Rabbit Nutrition** - Formulating for herbivores with high fiber needs")
elif animal == "Poultry":
    df = poultry_df.copy()
    st.info("🐔 **Poultry Nutrition** - Optimizing for broilers and layers")
else:
    df = cattle_df.copy()
    st.info("🐄 **Cattle Nutrition** - Formulating for ruminants")

# Sidebar inputs
st.sidebar.header("🎯 Animal Parameters")
age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 8, 30, 18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 1800, 12000, 3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.metric("Animal", animal)
st.sidebar.metric("Ingredients Available", len(df))

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Feed Optimizer", "📈 AI Prediction", "📋 Database"])

# TAB 1: Feed Optimizer
with tab1:
    st.header("🔬 Least Cost Feed Formulation")
    
    if st.button("🚀 Optimize Feed Formula", type="primary"):
        with st.spinner("Calculating..."):
            try:
                prob = LpProblem("FeedMix", LpMinimize)
                ingredients = df["Ingredient"].tolist()
                vars_dict = LpVariable.dicts("Ingr", ingredients, lowBound=0)
                
                prob += lpSum(vars_dict[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
                prob += lpSum(vars_dict[i] for i in ingredients) == 1
                prob += lpSum(vars_dict[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req
                prob += lpSum(vars_dict[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req
                
                prob.solve()
                
                if LpStatus[prob.status] == "Optimal":
                    result = {i: vars_dict[i].value() for i in ingredients if vars_dict[i].value() > 0.001}
                    result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
                    result_df["Proportion (%)"] = (result_df["Proportion"] * 100).round(2)
                    result_df = result_df.merge(df[["Ingredient", "Cost", "CP", "Energy", "Category"]], on="Ingredient")
                    result_df["Cost Contribution (₦)"] = (result_df["Proportion"] * result_df["Cost"]).round(2)
                    result_df = result_df.sort_values("Proportion", ascending=False)
                    
                    total_cost = value(prob.objective)
                    total_cp = sum(result_df["Proportion"] * result_df["CP"])
                    total_energy = sum(result_df["Proportion"] * result_df["Energy"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("💰 Feed Cost/kg", f"₦{total_cost:.2f}")
                    with col2:
                        st.metric("📅 Daily Cost", f"₦{total_cost * feed_intake:.2f}")
                    with col3:
                        st.metric("📦 Ingredients", len(result))
                    
                    st.success(f"✅ Optimization Complete! Cost: ₦{total_cost:.2f}/kg")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(result_df, values="Proportion (%)", names="Ingredient", title="Feed Composition")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.bar(result_df, y="Ingredient", x="Cost Contribution (₦)", title="Cost Breakdown", orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button("📥 Download Formula", csv, f"{animal}_formula.csv", "text/csv")
                else:
                    st.error("❌ No solution found. Adjust requirements.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# TAB 2: AI Prediction
with tab2:
    st.header("📈 AI Growth Prediction")
    
    avg_cp = df["CP"].mean()
    avg_energy = df["Energy"].mean()
    X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
    prediction = model.predict(X_input)[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Daily Gain", f"{prediction:.1f} g")
    with col2:
        st.metric("Weekly Gain", f"{prediction * 7:.0f} g")
    with col3:
        st.metric("Monthly Gain", f"{prediction * 30 / 1000:.2f} kg")
    with col4:
        st.metric("90-Day Weight", f"{weight + (prediction * 90 / 1000):.1f} kg")
    
    days = np.arange(0, 91)
    projected_weights = weight + (prediction * days / 1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=projected_weights, mode='lines', name='Projected Weight',
                            line=dict(color='#667eea', width=3), fill='tozeroy'))
    fig.update_layout(xaxis_title="Days", yaxis_title="Weight (kg)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Database
with tab3:
    st.header("📋 Ingredient Database")
    
    search = st.text_input("🔍 Search ingredients")
    filtered_df = df[df["Ingredient"].str.contains(search, case=False)] if search else df
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ingredients", len(filtered_df))
    with col2:
        st.metric("Avg Cost", f"₦{filtered_df['Cost'].mean():.2f}")
    with col3:
        st.metric("Avg Protein", f"{filtered_df['CP'].mean():.1f}%")
    with col4:
        st.metric("Avg Energy", f"{filtered_df['Energy'].mean():.0f} kcal")
    
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download Database", csv, f"{animal}_ingredients.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <h3 style="color: #667eea;">🌾 Nigerian Livestock Feed Formulator</h3>
    <p>Powered by AI • Built for Nigerian Farmers</p>
    <p style="font-size: 0.85rem;">Data sources: NIAS, FAO, Nigerian markets (2026)</p>
</div>
""", unsafe_allow_html=True)

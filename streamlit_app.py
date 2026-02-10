import subprocess
import sys
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

st.set_page_config(page_title="🌍 Necstech Feed Optimizer", layout="wide")

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
    }
    
    .big-font {
        font-size: 3rem !important;
        font-weight: 700;
        color: #27ae60;
        text-align: center;
        margin-bottom: 0;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .nutrient-table {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stage-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .breed-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
    }
    
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    rabbit = pd.read_csv("rabbit_ingredients.csv")
    poultry = pd.read_csv("poultry_ingredients.csv")
    cattle = pd.read_csv("cattle_ingredients.csv")
    ml_data = pd.read_csv("livestock_feed_training_dataset.csv")
    return rabbit, poultry, cattle, ml_data

rabbit_df, poultry_df, cattle_df, ml_df = load_data()

# =====================================================
# TRAIN AI MODEL
# =====================================================
@st.cache_resource
def train_model(data):
    X = data[[
        "Age_Weeks", "Body_Weight_kg",
        "CP_Requirement_%", "Energy_Requirement_Kcal",
        "Feed_Intake_kg", "Ingredient_CP_%", "Ingredient_Energy"
    ]]
    y = data["Expected_Daily_Gain_g"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(ml_df)

# =====================================================
# BREED DATABASE
# =====================================================
def get_breed_database():
    """Returns comprehensive breed information"""
    
    rabbit_breeds = {
        "New Zealand White": {
            "Type": "Meat",
            "Mature Weight (kg)": "4.5-5.5",
            "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent",
            "Best For": "Commercial meat production",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "10-12"
        },
        "Californian": {
            "Type": "Meat",
            "Mature Weight (kg)": "4.0-5.0",
            "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent",
            "Best For": "Meat and show",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "10-12"
        },
        "Flemish Giant": {
            "Type": "Meat",
            "Mature Weight (kg)": "6.0-10.0",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Large-scale meat production",
            "Recommended CP (%)": "17-19",
            "Market Age (weeks)": "14-16"
        },
        "Dutch": {
            "Type": "Pet/Show",
            "Mature Weight (kg)": "2.0-2.5",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Pets and breeding",
            "Recommended CP (%)": "15-17",
            "Market Age (weeks)": "8-10"
        },
        "Rex": {
            "Type": "Meat/Fur",
            "Mature Weight (kg)": "3.5-4.5",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Fur and meat",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "10-12"
        }
    }
    
    poultry_breeds = {
        "Broiler (Cobb 500)": {
            "Type": "Meat",
            "Mature Weight (kg)": "2.5-3.0",
            "Growth Rate": "Very Fast",
            "Feed Efficiency": "Excellent (FCR 1.6-1.8)",
            "Best For": "Commercial meat production",
            "Recommended CP (%)": "20-22",
            "Market Age (weeks)": "5-6"
        },
        "Broiler (Ross 308)": {
            "Type": "Meat",
            "Mature Weight (kg)": "2.3-2.8",
            "Growth Rate": "Very Fast",
            "Feed Efficiency": "Excellent (FCR 1.65-1.85)",
            "Best For": "Commercial meat production",
            "Recommended CP (%)": "20-22",
            "Market Age (weeks)": "5-6"
        },
        "Layer (Isa Brown)": {
            "Type": "Eggs",
            "Mature Weight (kg)": "1.8-2.0",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "High egg production (300+ eggs/year)",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "18-20 (point of lay)"
        },
        "Layer (Lohmann Brown)": {
            "Type": "Eggs",
            "Mature Weight (kg)": "1.9-2.1",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Excellent",
            "Best For": "Egg production (320+ eggs/year)",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "18-20 (point of lay)"
        },
        "Noiler": {
            "Type": "Dual Purpose",
            "Mature Weight (kg)": "2.0-2.5",
            "Growth Rate": "Fast",
            "Feed Efficiency": "Good",
            "Best For": "Meat and eggs (Nigerian adapted)",
            "Recommended CP (%)": "18-20",
            "Market Age (weeks)": "12-16"
        },
        "Kuroiler": {
            "Type": "Dual Purpose",
            "Mature Weight (kg)": "2.5-3.5",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Free-range, dual purpose",
            "Recommended CP (%)": "16-18",
            "Market Age (weeks)": "14-18"
        },
        "Local Nigerian": {
            "Type": "Dual Purpose",
            "Mature Weight (kg)": "1.2-1.8",
            "Growth Rate": "Slow",
            "Feed Efficiency": "Moderate",
            "Best For": "Free-range, disease resistant",
            "Recommended CP (%)": "14-16",
            "Market Age (weeks)": "20-24"
        }
    }
    
    cattle_breeds = {
        "White Fulani": {
            "Type": "Beef/Dairy",
            "Mature Weight (kg)": "300-450",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Milk and beef (Nigerian indigenous)",
            "Recommended CP (%)": "14-16",
            "Market Age (months)": "24-30"
        },
        "Red Bororo": {
            "Type": "Beef",
            "Mature Weight (kg)": "250-350",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Beef production (heat tolerant)",
            "Recommended CP (%)": "13-15",
            "Market Age (months)": "24-28"
        },
        "Sokoto Gudali": {
            "Type": "Beef",
            "Mature Weight (kg)": "350-500",
            "Growth Rate": "Moderate-Fast",
            "Feed Efficiency": "Good",
            "Best For": "Beef (large frame)",
            "Recommended CP (%)": "14-16",
            "Market Age (months)": "24-30"
        },
        "N'Dama": {
            "Type": "Beef/Draft",
            "Mature Weight (kg)": "300-400",
            "Growth Rate": "Moderate",
            "Feed Efficiency": "Good",
            "Best For": "Trypanosomiasis resistant",
            "Recommended CP (%)": "12-14",
            "Market Age (months)": "30-36"
        },
        "Muturu": {
            "Type": "Beef/Draft",
            "Mature Weight (kg)": "200-300",
            "Growth Rate": "Slow",
            "Feed Efficiency": "Moderate",
            "Best For": "Small-holder, disease resistant",
            "Recommended CP (%)": "12-14",
            "Market Age (months)": "30-36"
        },
        "Holstein Friesian (Cross)": {
            "Type": "Dairy",
            "Mature Weight (kg)": "450-650",
            "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent",
            "Best For": "High milk production",
            "Recommended CP (%)": "16-18",
            "Market Age (months)": "24-28"
        },
        "Brahman Cross": {
            "Type": "Beef",
            "Mature Weight (kg)": "400-550",
            "Growth Rate": "Fast",
            "Feed Efficiency": "Excellent",
            "Best For": "Beef (heat adapted)",
            "Recommended CP (%)": "14-16",
            "Market Age (months)": "20-24"
        }
    }
    
    return {
        "Rabbit": rabbit_breeds,
        "Poultry": poultry_breeds,
        "Cattle": cattle_breeds
    }

# =====================================================
# NUTRIENT REQUIREMENTS DATA
# =====================================================
def get_nutrient_requirements():
    """Returns comprehensive nutrient requirements for all livestock types"""
    
    rabbit_nutrients = {
        "Grower (4-12 weeks)": {
            "Crude Protein (%)": "16-18",
            "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "12-16",
            "Calcium (%)": "0.4-0.8",
            "Phosphorus (%)": "0.3-0.5",
            "Lysine (%)": "0.65-0.75",
            "Feed Intake (g/day)": "80-120"
        },
        "Finisher (12-16 weeks)": {
            "Crude Protein (%)": "14-16",
            "Energy (kcal/kg)": "2400-2600",
            "Crude Fiber (%)": "14-18",
            "Calcium (%)": "0.4-0.7",
            "Phosphorus (%)": "0.3-0.5",
            "Lysine (%)": "0.55-0.65",
            "Feed Intake (g/day)": "120-180"
        },
        "Doe (Maintenance)": {
            "Crude Protein (%)": "15-16",
            "Energy (kcal/kg)": "2500-2600",
            "Crude Fiber (%)": "14-16",
            "Calcium (%)": "0.5-0.8",
            "Phosphorus (%)": "0.4-0.5",
            "Lysine (%)": "0.60-0.70",
            "Feed Intake (g/day)": "100-150"
        },
        "Doe (Pregnant)": {
            "Crude Protein (%)": "16-18",
            "Energy (kcal/kg)": "2600-2800",
            "Crude Fiber (%)": "12-15",
            "Calcium (%)": "0.8-1.2",
            "Phosphorus (%)": "0.5-0.7",
            "Lysine (%)": "0.70-0.80",
            "Feed Intake (g/day)": "150-200"
        },
        "Doe (Lactating)": {
            "Crude Protein (%)": "17-19",
            "Energy (kcal/kg)": "2700-3000",
            "Crude Fiber (%)": "12-14",
            "Calcium (%)": "1.0-1.5",
            "Phosphorus (%)": "0.6-0.8",
            "Lysine (%)": "0.75-0.90",
            "Feed Intake (g/day)": "200-400"
        },
        "Buck (Breeding)": {
            "Crude Protein (%)": "15-17",
            "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "14-16",
            "Calcium (%)": "0.5-0.8",
            "Phosphorus (%)": "0.4-0.6",
            "Lysine (%)": "0.65-0.75",
            "Feed Intake (g/day)": "120-170"
        }
    }
    
    poultry_nutrients = {
        "Broiler Starter (0-3 weeks)": {
            "Crude Protein (%)": "22-24",
            "Energy (kcal/kg)": "3000-3200",
            "Crude Fiber (%)": "3-4",
            "Calcium (%)": "0.9-1.0",
            "Phosphorus (%)": "0.45-0.50",
            "Lysine (%)": "1.20-1.35",
            "Methionine (%)": "0.50-0.55",
            "Feed Intake (g/day)": "25-35"
        },
        "Broiler Grower (3-6 weeks)": {
            "Crude Protein (%)": "20-22",
            "Energy (kcal/kg)": "3100-3300",
            "Crude Fiber (%)": "3-5",
            "Calcium (%)": "0.85-0.95",
            "Phosphorus (%)": "0.40-0.45",
            "Lysine (%)": "1.05-1.20",
            "Methionine (%)": "0.45-0.50",
            "Feed Intake (g/day)": "80-120"
        },
        "Broiler Finisher (6+ weeks)": {
            "Crude Protein (%)": "18-20",
            "Energy (kcal/kg)": "3200-3400",
            "Crude Fiber (%)": "3-5",
            "Calcium (%)": "0.80-0.90",
            "Phosphorus (%)": "0.35-0.40",
            "Lysine (%)": "0.95-1.10",
            "Methionine (%)": "0.40-0.45",
            "Feed Intake (g/day)": "140-180"
        },
        "Layer Starter (0-6 weeks)": {
            "Crude Protein (%)": "18-20",
            "Energy (kcal/kg)": "2800-3000",
            "Crude Fiber (%)": "3-5",
            "Calcium (%)": "0.9-1.0",
            "Phosphorus (%)": "0.45-0.50",
            "Lysine (%)": "0.95-1.05",
            "Methionine (%)": "0.40-0.45",
            "Feed Intake (g/day)": "20-40"
        },
        "Layer Grower (6-18 weeks)": {
            "Crude Protein (%)": "16-18",
            "Energy (kcal/kg)": "2700-2900",
            "Crude Fiber (%)": "4-6",
            "Calcium (%)": "0.8-0.9",
            "Phosphorus (%)": "0.40-0.45",
            "Lysine (%)": "0.75-0.85",
            "Methionine (%)": "0.35-0.40",
            "Feed Intake (g/day)": "60-90"
        },
        "Layer Production (18+ weeks)": {
            "Crude Protein (%)": "16-18",
            "Energy (kcal/kg)": "2750-2900",
            "Crude Fiber (%)": "4-6",
            "Calcium (%)": "3.5-4.0",
            "Phosphorus (%)": "0.35-0.40",
            "Lysine (%)": "0.75-0.85",
            "Methionine (%)": "0.38-0.42",
            "Feed Intake (g/day)": "110-130"
        }
    }
    
    cattle_nutrients = {
        "Calf Starter (0-3 months)": {
            "Crude Protein (%)": "18-20",
            "Energy (kcal/kg)": "3000-3200",
            "Crude Fiber (%)": "8-12",
            "Calcium (%)": "0.7-1.0",
            "Phosphorus (%)": "0.4-0.6",
            "TDN (%)": "72-78",
            "Feed Intake (kg/day)": "0.5-1.5"
        },
        "Calf Grower (3-6 months)": {
            "Crude Protein (%)": "16-18",
            "Energy (kcal/kg)": "2800-3000",
            "Crude Fiber (%)": "10-15",
            "Calcium (%)": "0.6-0.9",
            "Phosphorus (%)": "0.35-0.50",
            "TDN (%)": "68-74",
            "Feed Intake (kg/day)": "2-4"
        },
        "Heifer (6-12 months)": {
            "Crude Protein (%)": "14-16",
            "Energy (kcal/kg)": "2600-2800",
            "Crude Fiber (%)": "12-18",
            "Calcium (%)": "0.5-0.8",
            "Phosphorus (%)": "0.30-0.45",
            "TDN (%)": "65-70",
            "Feed Intake (kg/day)": "4-7"
        },
        "Bull (Breeding)": {
            "Crude Protein (%)": "12-14",
            "Energy (kcal/kg)": "2500-2700",
            "Crude Fiber (%)": "15-20",
            "Calcium (%)": "0.4-0.7",
            "Phosphorus (%)": "0.25-0.40",
            "TDN (%)": "62-68",
            "Feed Intake (kg/day)": "8-12"
        },
        "Cow (Dry)": {
            "Crude Protein (%)": "10-12",
            "Energy (kcal/kg)": "2400-2600",
            "Crude Fiber (%)": "18-25",
            "Calcium (%)": "0.4-0.6",
            "Phosphorus (%)": "0.25-0.35",
            "TDN (%)": "58-65",
            "Feed Intake (kg/day)": "10-15"
        },
        "Cow (Lactating)": {
            "Crude Protein (%)": "14-18",
            "Energy (kcal/kg)": "2700-3000",
            "Crude Fiber (%)": "15-22",
            "Calcium (%)": "0.6-0.9",
            "Phosphorus (%)": "0.35-0.50",
            "TDN (%)": "68-75",
            "Feed Intake (kg/day)": "12-20"
        },
        "Beef Finisher": {
            "Crude Protein (%)": "12-14",
            "Energy (kcal/kg)": "2800-3100",
            "Crude Fiber (%)": "8-15",
            "Calcium (%)": "0.5-0.7",
            "Phosphorus (%)": "0.30-0.45",
            "TDN (%)": "70-78",
            "Feed Intake (kg/day)": "8-14"
        }
    }
    
    return {
        "Rabbit": rabbit_nutrients,
        "Poultry": poultry_nutrients,
        "Cattle": cattle_nutrients
    }

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'formulation_history' not in st.session_state:
    st.session_state.formulation_history = []

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def generate_report(animal, age, weight, cp_req, energy_req, feed_intake, 
                   result_df=None, total_cost=None, prediction=None):
    """Generate a comprehensive PDF report"""
    
    report = f"""
    ═══════════════════════════════════════════════════════════════
                    NECSTECH FEED OPTIMIZER REPORT
                         Livestock Feed Formulation
    ═══════════════════════════════════════════════════════════════
    
    Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ANIMAL INFORMATION
    ─────────────────────────────────────────────────────────────
    Species: {animal}
    Age: {age} weeks
    Body Weight: {weight} kg
    Daily Feed Intake: {feed_intake} kg
    
    NUTRITIONAL REQUIREMENTS
    ─────────────────────────────────────────────────────────────
    Crude Protein: {cp_req}%
    Energy: {energy_req} kcal/kg
    
    """
    
    if result_df is not None and total_cost is not None:
        report += f"""
    OPTIMIZED FEED FORMULA
    ─────────────────────────────────────────────────────────────
    Total Cost per kg: ₦{total_cost:.2f}
    Daily Feed Cost: ₦{total_cost * feed_intake:.2f}
    Number of Ingredients: {len(result_df)}
    
    INGREDIENT COMPOSITION
    ─────────────────────────────────────────────────────────────
    """
        for _, row in result_df.iterrows():
            report += f"    {row['Ingredient']}: {row['Proportion (%)']:.2f}% (₦{row['Cost Contribution (₦)']:.2f})\n"
    
    if prediction is not None:
        weekly_gain = prediction * 7
        monthly_gain = prediction * 30
        projected_weight = weight + (monthly_gain * 3 / 1000)
        fcr = (feed_intake * 1000) / prediction if prediction > 0 else 0
        
        report += f"""
    
    GROWTH PREDICTIONS
    ─────────────────────────────────────────────────────────────
    Daily Weight Gain: {prediction:.1f} g/day
    Weekly Gain: {weekly_gain:.0f} g
    Monthly Gain: {monthly_gain/1000:.2f} kg
    90-Day Projected Weight: {projected_weight:.1f} kg
    Feed Conversion Ratio: {fcr:.2f}:1
    """
        
        if total_cost is not None:
            cost_per_kg_gain = (total_cost * feed_intake * 1000) / prediction
            report += f"    Cost per kg Gain: ₦{cost_per_kg_gain:.2f}\n"
    
    report += f"""
    
    ═══════════════════════════════════════════════════════════════
    Generated by Necstech Feed Optimizer
    Powered by Nigerian Agricultural Data
    ═══════════════════════════════════════════════════════════════
    """
    
    return report

# =====================================================
# NAVIGATION FUNCTIONS
# =====================================================
def show_home():
    st.markdown('<p class="big-font">🌍 Necstech Feed Optimizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered nutrition platform for Rabbits, Poultry, and Cattle</p>', unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #27ae60;">✔ Least-Cost</h3>
            <p>Optimize feed costs while meeting nutrition needs</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #3498db;">✔ AI Prediction</h3>
            <p>Machine learning growth forecasts</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #e74c3c;">✔ Nigerian Data</h3>
            <p>97 ingredients with local prices</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #f39c12;">✔ 31+ Breeds</h3>
            <p>Comprehensive breed database</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 Nutrient Guide", type="primary", use_container_width=True):
            st.session_state.page = 'nutrient_guide'
            st.rerun()
    
    with col2:
        if st.button("🐾 Breed Database", type="primary", use_container_width=True):
            st.session_state.page = 'breed_database'
            st.rerun()
    
    with col3:
        if st.button("🚀 Feed Formulator", type="primary", use_container_width=True):
            st.session_state.page = 'formulator'
            st.rerun()
    
    # Recent formulations
    if st.session_state.formulation_history:
        st.markdown("---")
        st.subheader("📊 Recent Formulations")
        
        for i, history in enumerate(st.session_state.formulation_history[-3:]):
            with st.expander(f"{history['animal']} - {history['timestamp']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", f"{history['age']} weeks")
                    st.metric("Weight", f"{history['weight']} kg")
                with col2:
                    st.metric("Protein Req", f"{history['cp_req']}%")
                    st.metric("Energy Req", f"{history['energy_req']} kcal")
                with col3:
                    if 'total_cost' in history:
                        st.metric("Cost/kg", f"₦{history['total_cost']:.2f}")
                    if 'prediction' in history:
                        st.metric("Daily Gain", f"{history['prediction']:.1f} g")

def show_breed_database():
    st.markdown('<p class="big-font">🐾 Breed Database</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive information on livestock breeds for Nigerian conditions</p>', unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    breed_data = get_breed_database()
    
    # Animal selector
    animal_type = st.selectbox("Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    
    breeds = breed_data[animal_type]
    
    st.markdown(f"## {animal_type} Breeds ({len(breeds)} breeds)")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search = st.text_input("🔍 Search breeds", placeholder="Type breed name...")
    
    with col2:
        if animal_type in ["Rabbit", "Cattle"]:
            type_filter = st.selectbox("Filter by Type", ["All"] + list(set([b["Type"] for b in breeds.values()])))
        else:
            type_filter = "All"
    
    # Display breeds
    for breed_name, breed_info in breeds.items():
        # Apply filters
        if search and search.lower() not in breed_name.lower():
            continue
        if type_filter != "All" and breed_info["Type"] != type_filter:
            continue
        
        with st.container():
            st.markdown(f'<div class="breed-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {breed_name}")
                st.markdown(f"**Type:** {breed_info['Type']}")
                st.markdown(f"**Best For:** {breed_info['Best For']}")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Mature Weight", breed_info["Mature Weight (kg)"])
                with col_b:
                    st.metric("Growth Rate", breed_info["Growth Rate"])
                with col_c:
                    st.metric("Feed Efficiency", breed_info["Feed Efficiency"])
            
            with col2:
                st.markdown("**Feeding Recommendation**")
                st.info(f"Protein: {breed_info['Recommended CP (%)']}")
                if animal_type == "Cattle":
                    st.info(f"Market Age: {breed_info['Market Age (months)']} months")
                else:
                    st.info(f"Market Age: {breed_info['Market Age (weeks)']} weeks")
                
                if st.button(f"Use {breed_name}", key=f"breed_{breed_name}"):
                    st.session_state.selected_breed = breed_name
                    st.session_state.page = 'formulator'
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    # Summary statistics
    st.markdown("### 📊 Breed Statistics")
    breed_df = pd.DataFrame(breeds).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Type distribution
        type_counts = breed_df['Type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Breed Distribution by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Growth rate distribution
        growth_counts = breed_df['Growth Rate'].value_counts()
        fig = px.bar(x=growth_counts.index, y=growth_counts.values,
                     title="Breeds by Growth Rate",
                     labels={'x': 'Growth Rate', 'y': 'Number of Breeds'})
        st.plotly_chart(fig, use_container_width=True)

def show_nutrient_guide():
    st.markdown('<p class="big-font">📖 Nutrient Requirements Guide</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive nutritional standards for all livestock types and development stages</p>', unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    nutrient_data = get_nutrient_requirements()
    
    # Animal selector
    animal_type = st.selectbox("🐾 Select Animal Type", ["Rabbit", "Poultry", "Cattle"])
    
    st.markdown(f"## {animal_type} Nutrient Requirements")
    
    requirements = nutrient_data[animal_type]
    
    # Display requirements for each stage
    for stage, nutrients in requirements.items():
        st.markdown(f'<div class="stage-header">🎯 {stage}</div>', unsafe_allow_html=True)
        
        df_nutrients = pd.DataFrame([nutrients]).T
        df_nutrients.columns = ['Requirement']
        df_nutrients.index.name = 'Nutrient Parameter'
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_nutrients, use_container_width=True)
        
        with col2:
            st.markdown("##### Key Nutrients")
            if "Crude Protein (%)" in nutrients:
                st.metric("Protein", nutrients["Crude Protein (%)"])
            if "Energy (kcal/kg)" in nutrients:
                st.metric("Energy", nutrients["Energy (kcal/kg)"])
            if "Crude Fiber (%)" in nutrients:
                st.metric("Fiber", nutrients["Crude Fiber (%)"])
        
        st.markdown("---")
    
    # Additional information
    st.markdown("### 📋 Important Notes")
    
    if animal_type == "Rabbit":
        st.info("""
        **Rabbit Feeding Guidelines:**
        - Provide fresh water at all times (rabbits drink 2-3x their feed weight)
        - Hay should make up 70-80% of adult rabbit diet
        - Introduce new feeds gradually over 7-10 days
        - Monitor body condition score regularly (ideal: ribs barely palpable)
        - Higher fiber content prevents digestive issues and hairballs
        - Avoid sudden diet changes which can cause enteritis
        """)
    elif animal_type == "Poultry":
        st.info("""
        **Poultry Feeding Guidelines:**
        - Layer birds require high calcium (3.5-4%) for strong eggshells
        - Grit (insoluble granite) aids digestion, especially for whole grains
        - Feed should be stored in cool, dry, rodent-proof conditions
        - Sudden feed changes can reduce performance by 10-20%
        - Water consumption is roughly 2x feed intake (more in hot weather)
        - Use feeders that minimize waste (adjust to bird back height)
        """)
    else:
        st.info("""
        **Cattle Feeding Guidelines:**
        - TDN = Total Digestible Nutrients (energy measure)
        - Ruminants require 15-20% fiber for proper rumen function
        - Transition periods are critical - allow 21 days minimum
        - Fresh, clean water must always be available (50-80L/day for lactating cows)
        - Monitor body condition score (BCS 1-9 scale, target: 5-6)
        - Mineral supplementation is essential (especially salt, calcium, phosphorus)
        - Avoid over-feeding grain (acidosis risk) - max 60% of diet
        """)
    
    # Download option
    st.markdown("### 📥 Download Guide")
    
    all_stages = []
    for stage, nutrients in requirements.items():
        row = {"Stage": stage}
        row.update(nutrients)
        all_stages.append(row)
    
    download_df = pd.DataFrame(all_stages)
    csv = download_df.to_csv(index=False)
    
    st.download_button(
        label=f"📥 Download {animal_type} Nutrient Guide (CSV)",
        data=csv,
        file_name=f"{animal_type.lower()}_nutrient_guide.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    if st.button("🚀 Proceed to Feed Formulation", type="primary", use_container_width=True):
        st.session_state.page = 'formulator'
        st.rerun()

def show_formulator():
    col_header1, col_header2 = st.columns([4, 1])
    
    with col_header1:
        st.markdown('<p class="big-font">🔬 Feed Formulation Center</p>', unsafe_allow_html=True)
    
    with col_header2:
        if st.button("← Home"):
            st.session_state.page = 'home'
            st.rerun()
    
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
    st.sidebar.markdown("---")
    
    # Show selected breed if available
    if 'selected_breed' in st.session_state:
        st.sidebar.success(f"✓ Breed: {st.session_state.selected_breed}")

    age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
    weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
    cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
    energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
    feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Selection")
    st.sidebar.metric("Animal", animal)
    st.sidebar.metric("Ingredients Available", len(df))
    
    # Quick links
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Links")
    if st.sidebar.button("📖 Nutrient Guide", use_container_width=True):
        st.session_state.page = 'nutrient_guide'
        st.rerun()
    if st.sidebar.button("🐾 Breed Database", use_container_width=True):
        st.session_state.page = 'breed_database'
        st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Feed Optimizer", 
        "📋 Ingredient Database", 
        "📈 Weight Gain Prediction",
        "📊 Cost Analysis Dashboard"
    ])

    # TAB 1: Feed Optimizer
    with tab1:
        st.header("🔬 Least Cost Feed Formulation")
        
        st.markdown("""
        This optimizer uses **linear programming** to find the cheapest combination of ingredients 
        that meets all your nutritional requirements.
        """)
        
        # Additional constraints
        with st.expander("⚙️ Advanced Constraints (Optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_fiber_constraint = st.checkbox("Add Fiber Constraint")
                if use_fiber_constraint:
                    min_fiber = st.slider("Minimum Fiber (%)", 0, 30, 12)
                    max_fiber = st.slider("Maximum Fiber (%)", 0, 40, 20)
            
            with col2:
                limit_ingredients = st.checkbox("Limit Number of Ingredients")
                if limit_ingredients:
                    max_ingredients = st.slider("Maximum Ingredients", 3, 15, 8)
        
        if st.button("🚀 Optimize Feed Formula", type="primary"):
            with st.spinner("Calculating optimal feed mix..."):
                try:
                    prob = LpProblem("FeedMix", LpMinimize)
                    ingredients = df["Ingredient"].tolist()
                    vars = LpVariable.dicts("Ingr", ingredients, lowBound=0, upBound=1)

                    # Objective: minimize cost
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
                    
                    # Constraints
                    prob += lpSum(vars[i] for i in ingredients) == 1  # Sum to 100%
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req
                    prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req

                    # Optional fiber constraints
                    if use_fiber_constraint and 'Fiber' in df.columns:
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) >= min_fiber
                        prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Fiber"].values[0] for i in ingredients) <= max_fiber

                    prob.solve()

                    if LpStatus[prob.status] == "Optimal":
                        result = {i: vars[i].value() for i in ingredients if vars[i].value() > 0.001}
                        
                        # Check ingredient limit
                        if limit_ingredients and len(result) > max_ingredients:
                            st.warning(f"⚠️ Solution uses {len(result)} ingredients (limit: {max_ingredients}). Adjust constraints or increase limit.")
                        
                        result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
                        result_df["Proportion (%)"] = (result_df["Proportion"] * 100).round(2)
                        result_df["Cost/kg (₦)"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["Cost"].values[0]
                        )
                        result_df["Cost Contribution (₦)"] = (result_df["Proportion"] * result_df["Cost/kg (₦)"]).round(2)
                        
                        # Calculate nutritional composition
                        result_df["CP Contribution"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["CP"].values[0]
                        ) * result_df["Proportion"]
                        
                        result_df["Energy Contribution"] = result_df["Ingredient"].apply(
                            lambda x: df[df["Ingredient"] == x]["Energy"].values[0]
                        ) * result_df["Proportion"]
                        
                        total_cp = result_df["CP Contribution"].sum()
                        total_energy = result_df["Energy Contribution"].sum()
                        
                        result_df = result_df.sort_values("Proportion", ascending=False)
                        
                        # Store results
                        total_cost = value(prob.objective)
                        st.session_state['optimization_result'] = result_df
                        st.session_state['total_cost'] = total_cost
                        st.session_state['total_cp'] = total_cp
                        st.session_state['total_energy'] = total_energy
                        
                        # Save to history
                        st.session_state.formulation_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'animal': animal,
                            'age': age,
                            'weight': weight,
                            'cp_req': cp_req,
                            'energy_req': energy_req,
                            'total_cost': total_cost
                        })
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("💰 Feed Cost/kg", f"₦{total_cost:.2f}")
                        with col2:
                            st.metric("📅 Daily Feed Cost", f"₦{total_cost * feed_intake:.2f}")
                        with col3:
                            st.metric("📦 Ingredients Used", len(result))
                        with col4:
                            monthly_cost = total_cost * feed_intake * 30
                            st.metric("📆 Monthly Cost", f"₦{monthly_cost:.2f}")
                        
                        # Nutritional achievement
                        st.markdown("---")
                        st.subheader("✅ Nutritional Achievement")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            cp_pct = (total_cp / cp_req) * 100 if cp_req > 0 else 0
                            st.metric("Crude Protein", f"{total_cp:.2f}%", 
                                     delta=f"{cp_pct:.1f}% of requirement")
                        with col2:
                            energy_pct = (total_energy / energy_req) * 100 if energy_req > 0 else 0
                            st.metric("Energy", f"{total_energy:.0f} kcal/kg",
                                     delta=f"{energy_pct:.1f}% of requirement")
                        
                        st.success(f"✅ Optimization Complete! Total Cost: ₦{total_cost:.2f}/kg")
                        
                        # Display table
                        st.dataframe(
                            result_df[["Ingredient", "Proportion (%)", "Cost/kg (₦)", "Cost Contribution (₦)"]],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                result_df,
                                values="Proportion (%)",
                                names="Ingredient",
                                title="Feed Composition",
                                color_discrete_sequence=px.colors.sequential.Greens
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                result_df,
                                x="Ingredient",
                                y="Cost Contribution (₦)",
                                title="Cost Breakdown by Ingredient",
                                color="Cost Contribution (₦)",
                                color_continuous_scale="Greens"
                            )
                            fig_bar.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Formula (CSV)",
                                data=csv,
                                file_name=f"{animal}_feed_formula_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            report = generate_report(animal, age, weight, cp_req, energy_req, 
                                                    feed_intake, result_df, total_cost)
                            st.download_button(
                                label="📄 Download Report (TXT)",
                                data=report,
                                file_name=f"{animal}_feed_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ No feasible solution found. Try adjusting your requirements or constraints.")
                
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")

    # TAB 2: Ingredient Database
    with tab2:
        st.header("📋 Ingredient Database Manager")
        
        st.markdown(f"""
        **{len(df)} ingredients** available for {animal} feed formulation. 
        You can view, edit, add, or remove ingredients below.
        """)
        
        # Search and filter
        col1, col2 = st.columns(2)
        
        with col1:
            search = st.text_input("🔍 Search ingredients", placeholder="Type to filter...")
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Ingredient", "CP", "Energy", "Cost"])
        
        if search:
            filtered_df = df[df["Ingredient"].str.contains(search, case=False, na=False)]
        else:
            filtered_df = df
        
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False if sort_by != "Ingredient" else True)
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ingredients", len(filtered_df))
        with col2:
            st.metric("Avg Cost/kg", f"₦{filtered_df['Cost'].mean():.2f}")
        with col3:
            st.metric("Avg Protein", f"{filtered_df['CP'].mean():.1f}%")
        with col4:
            st.metric("Avg Energy", f"{filtered_df['Energy'].mean():.0f} kcal")
        
        st.markdown("---")

        edited_df = st.data_editor(
            filtered_df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Ingredient": st.column_config.TextColumn("Ingredient", width="medium"),
                "CP": st.column_config.NumberColumn("Crude Protein (%)", format="%.1f"),
                "Energy": st.column_config.NumberColumn("Energy (kcal/kg)", format="%.0f"),
                "Fiber": st.column_config.NumberColumn("Crude Fiber (%)", format="%.1f"),
                "Cost": st.column_config.NumberColumn("Cost (₦/kg)", format="₦%.2f"),
            }
        )

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Changes to Database", use_container_width=True):
                edited_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
                st.success(f"✅ Ingredient database updated successfully!")
                st.cache_data.clear()
        
        with col2:
            csv = edited_df.to_csv(index=False)
            st.download_button(
                "📥 Download Database (CSV)",
                csv,
                f"{animal.lower()}_ingredients.csv",
                "text/csv",
                use_container_width=True
            )

    # TAB 3: Weight Gain Prediction
    with tab3:
        st.header("📈 AI Weight Gain Prediction")
        
        st.markdown("""
        Our **Random Forest machine learning model** predicts animal growth based on 
        110+ feeding trials from Nigerian farms.
        """)
        
        st.markdown("---")
        
        if st.button("🎯 Calculate Growth Prediction", type="primary"):
            with st.spinner("Calculating growth predictions..."):
                avg_cp = df["CP"].mean()
                avg_energy = df["Energy"].mean()

                X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
                prediction = model.predict(X_input)[0]
                
                st.session_state['prediction'] = prediction
                st.session_state['avg_cp'] = avg_cp
                st.session_state['avg_energy'] = avg_energy
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            
            weekly_gain = prediction * 7
            monthly_gain = prediction * 30
            projected_weight_90d = weight + (monthly_gain * 3 / 1000)

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Daily Weight Gain", f"{prediction:.1f} g/day")
            with col2:
                st.metric("Weekly Gain", f"{weekly_gain:.0f} g")
            with col3:
                st.metric("Monthly Gain", f"{monthly_gain / 1000:.2f} kg")
            with col4:
                st.metric("90-Day Weight", f"{projected_weight_90d:.1f} kg", 
                         delta=f"+{projected_weight_90d - weight:.1f} kg")
            
            st.subheader("📊 90-Day Weight Projection")
            
            days = np.arange(0, 91)
            projected_weights = weight + (prediction * days / 1000)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days,
                y=projected_weights,
                mode='lines',
                name='Projected Weight',
                line=dict(color='#27ae60', width=3),
                fill='tozeroy',
                fillcolor='rgba(39, 174, 96, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0],
                y=[weight],
                mode='markers',
                name='Current Weight',
                marker=dict(size=12, color='#e74c3c')
            ))
            
            fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Weight (kg)",
                hovermode='x unified',
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction > 0:
                    fcr = (feed_intake * 1000) / prediction
                else:
                    fcr = 0
                st.metric("Feed Conversion Ratio (FCR)", f"{fcr:.2f}:1")
                st.caption("Feed required to gain 1 kg of body weight")
            
            with col2:
                if 'total_cost' in st.session_state and prediction > 0:
                    total_cost = st.session_state['total_cost']
                    cost_per_kg_gain = (total_cost * feed_intake * 1000) / prediction
                    st.metric("Cost per kg Gain", f"₦{cost_per_kg_gain:.2f}")
                    st.caption("Feed cost to produce 1 kg of weight gain")
                else:
                    st.info("💡 Run the Feed Optimizer first to see cost metrics")
            
            st.markdown("---")
            st.subheader("🎯 Growth Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if animal == "Rabbit":
                    if prediction > 30:
                        performance = "🟢 Excellent"
                    elif prediction > 20:
                        performance = "🟡 Good"
                    else:
                        performance = "🔴 Below Average"
                elif animal == "Poultry":
                    if prediction > 50:
                        performance = "🟢 Excellent"
                    elif prediction > 35:
                        performance = "🟡 Good"
                    else:
                        performance = "🔴 Below Average"
                else:
                    if prediction > 800:
                        performance = "🟢 Excellent"
                    elif prediction > 500:
                        performance = "🟡 Good"
                    else:
                        performance = "🔴 Below Average"
                
                st.metric("Performance Rating", performance)
            
            with col2:
                days_to_target = 0
                if animal == "Rabbit":
                    target_weight = 2.5
                elif animal == "Poultry":
                    target_weight = 2.0
                else:
                    target_weight = 300
                
                if prediction > 0 and weight < target_weight:
                    days_to_target = int((target_weight - weight) * 1000 / prediction)
                    st.metric("Days to Market Weight", f"{days_to_target} days")
                    st.caption(f"Target: {target_weight} kg")
                else:
                    st.metric("Market Weight", "✅ Achieved")
        else:
            st.info("👆 Click the 'Calculate Growth Prediction' button above to see results")
    
    # TAB 4: Cost Analysis Dashboard
    with tab4:
        st.header("📊 Cost Analysis Dashboard")
        
        if 'optimization_result' not in st.session_state:
            st.warning("⚠️ Please run the Feed Optimizer first to see cost analysis")
        else:
            result_df = st.session_state['optimization_result']
            total_cost = st.session_state['total_cost']
            
            # Time-based cost projections
            st.subheader("💰 Cost Projections")
            
            col1, col2, col3, col4 = st.columns(4)
            
            daily_cost = total_cost * feed_intake
            weekly_cost = daily_cost * 7
            monthly_cost = daily_cost * 30
            yearly_cost = daily_cost * 365
            
            with col1:
                st.metric("Daily Cost", f"₦{daily_cost:.2f}")
            with col2:
                st.metric("Weekly Cost", f"₦{weekly_cost:.2f}")
            with col3:
                st.metric("Monthly Cost", f"₦{monthly_cost:.2f}")
            with col4:
                st.metric("Yearly Cost", f"₦{yearly_cost:,.2f}")
            
            # Herd/flock cost calculator
            st.markdown("---")
            st.subheader("🐾 Herd/Flock Cost Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_animals = st.number_input("Number of Animals", min_value=1, max_value=10000, value=100)
            
            with col2:
                duration_days = st.slider("Duration (days)", 1, 365, 90)
            
            total_herd_cost = daily_cost * num_animals * duration_days
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Feed Cost", f"₦{total_herd_cost:,.2f}")
            with col2:
                cost_per_animal = total_herd_cost / num_animals
                st.metric("Cost per Animal", f"₦{cost_per_animal:,.2f}")
            with col3:
                daily_herd_cost = daily_cost * num_animals
                st.metric("Daily Herd Cost", f"₦{daily_herd_cost:,.2f}")
            
            # Cost breakdown visualization
            st.markdown("---")
            st.subheader("📊 Cost Breakdown Analysis")
            
            fig = px.treemap(
                result_df,
                path=['Ingredient'],
                values='Cost Contribution (₦)',
                title='Cost Contribution by Ingredient',
                color='Cost Contribution (₦)',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Ingredient cost comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 5 most expensive ingredients by contribution
                top_5 = result_df.nlargest(5, 'Cost Contribution (₦)')
                fig = px.bar(
                    top_5,
                    x='Ingredient',
                    y='Cost Contribution (₦)',
                    title='Top 5 Cost Contributors',
                    color='Cost Contribution (₦)',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Proportion vs Cost scatter
                fig = px.scatter(
                    result_df,
                    x='Proportion (%)',
                    y='Cost/kg (₦)',
                    size='Cost Contribution (₦)',
                    hover_name='Ingredient',
                    title='Proportion vs Unit Cost',
                    color='Cost Contribution (₦)',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ROI Calculator (if prediction available)
            if 'prediction' in st.session_state:
                st.markdown("---")
                st.subheader("💵 Return on Investment Calculator")
                
                prediction = st.session_state['prediction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if animal == "Rabbit":
                        price_per_kg = st.number_input("Selling Price (₦/kg live weight)", 
                                                       min_value=500, max_value=5000, value=1500)
                    elif animal == "Poultry":
                        price_per_kg = st.number_input("Selling Price (₦/kg live weight)", 
                                                       min_value=500, max_value=3000, value=1200)
                    else:
                        price_per_kg = st.number_input("Selling Price (₦/kg live weight)", 
                                                       min_value=500, max_value=5000, value=2000)
                
                with col2:
                    production_days = st.number_input("Production Cycle (days)", 
                                                     min_value=30, max_value=365, value=90)
                
                # Calculate ROI
                total_feed_cost = daily_cost * production_days
                weight_gain_kg = (prediction * production_days) / 1000
                final_weight = weight + weight_gain_kg
                revenue = final_weight * price_per_kg
                profit = revenue - total_feed_cost
                roi_percent = (profit / total_feed_cost * 100) if total_feed_cost > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Feed Cost", f"₦{total_feed_cost:,.2f}")
                with col2:
                    st.metric("Final Weight", f"{final_weight:.2f} kg")
                with col3:
                    st.metric("Revenue", f"₦{revenue:,.2f}")
                with col4:
                    st.metric("Profit", f"₦{profit:,.2f}", delta=f"{roi_percent:.1f}% ROI")
                
                # ROI visualization
                roi_data = pd.DataFrame({
                    'Category': ['Feed Cost', 'Profit'],
                    'Amount': [total_feed_cost, profit if profit > 0 else 0]
                })
                
                fig = px.pie(
                    roi_data,
                    values='Amount',
                    names='Category',
                    title=f'Cost vs Profit (ROI: {roi_percent:.1f}%)',
                    color_discrete_sequence=['#e74c3c', '#27ae60']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if profit > 0:
                    st.success(f"✅ Profitable! Expected profit of ₦{profit:,.2f} per animal")
                else:
                    st.error(f"⚠️ Loss expected. Adjust feeding program or selling price.")

# =====================================================
# PAGE ROUTING
# =====================================================
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'nutrient_guide':
    show_nutrient_guide()
elif st.session_state.page == 'breed_database':
    show_breed_database()
elif st.session_state.page == 'formulator':
    show_formulator()

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p><strong>🌍 Necstech Feed Optimizer v2.0</strong></p>
    <p>Powered by Nigerian agricultural data | Built with Streamlit + ML</p>
    <p style='font-size: 0.85rem;'>Data sources: NIAS, FAO, Nigerian markets (2026)</p>
    <p style='font-size: 0.75rem; margin-top: 1rem;'>© 2026 Necstech - Optimizing African Agriculture</p>
</div>
""", unsafe_allow_html=True)


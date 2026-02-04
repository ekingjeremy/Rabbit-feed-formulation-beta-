import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🌍 AI Livestock Feed Formulator", layout="wide")

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
# LANDING PAGE
# =====================================================
st.markdown('<p class="big-font">🌍 Intelligent Livestock Feed Formulator</p>', unsafe_allow_html=True)
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
        <h3 style="color: #f39c12;">✔ 31 Breeds</h3>
        <p>Comprehensive breed database</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# ANIMAL SELECTION
# =====================================================
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

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🎯 Animal Parameters")
st.sidebar.markdown("---")

age = st.sidebar.slider("Age (weeks)", 1, 120, 8)
weight = st.sidebar.slider("Body Weight (kg)", 0.1, 600.0, 2.0)
cp_req = st.sidebar.slider("Crude Protein Requirement (%)", 10, 30, 18)
energy_req = st.sidebar.slider("Energy Requirement (Kcal/kg)", 2000, 12000, 3000)
feed_intake = st.sidebar.slider("Feed Intake (kg/day)", 0.05, 30.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Selection")
st.sidebar.metric("Animal", animal)
st.sidebar.metric("Ingredients Available", len(df))

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔬 Feed Optimizer", "📈 AI Growth Prediction", "📋 Ingredient Database"])

# =====================================================
# FEED OPTIMIZER
# =====================================================
with tab1:
    st.header("🔬 Least Cost Feed Formulation")
    
    st.markdown("""
    This optimizer uses **linear programming** to find the cheapest combination of ingredients 
    that meets all your nutritional requirements.
    """)
    
    if st.button("🚀 Optimize Feed Formula", type="primary"):
        with st.spinner("Calculating optimal feed mix..."):
            prob = LpProblem("FeedMix", LpMinimize)
            ingredients = df["Ingredient"].tolist()
            vars = LpVariable.dicts("Ingr", ingredients, lowBound=0)

            prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Cost"].values[0] for i in ingredients)
            prob += lpSum(vars[i] for i in ingredients) == 1
            prob += lpSum(vars[i] * df[df["Ingredient"] == i]["CP"].values[0] for i in ingredients) >= cp_req
            prob += lpSum(vars[i] * df[df["Ingredient"] == i]["Energy"].values[0] for i in ingredients) >= energy_req

            prob.solve()

            if LpStatus[prob.status] == "Optimal":
                result = {i: vars[i].value() for i in ingredients if vars[i].value() > 0.001}
                result_df = pd.DataFrame(result.items(), columns=["Ingredient", "Proportion"])
                result_df["Proportion (%)"] = (result_df["Proportion"] * 100).round(2)
                result_df["Cost/kg (₦)"] = result_df["Ingredient"].apply(
                    lambda x: df[df["Ingredient"] == x]["Cost"].values[0]
                )
                result_df["Cost Contribution (₦)"] = (result_df["Proportion"] * result_df["Cost/kg (₦)"]).round(2)
                
                # Sort by proportion
                result_df = result_df.sort_values("Proportion", ascending=False)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                total_cost = value(prob.objective)
                
                with col1:
                    st.metric("💰 Feed Cost per kg", f"₦{total_cost:.2f}")
                with col2:
                    st.metric("📅 Daily Feed Cost", f"₦{total_cost * feed_intake:.2f}")
                with col3:
                    st.metric("📦 Ingredients Used", len(result))
                
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
                
                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Formula as CSV",
                    data=csv,
                    file_name=f"{animal}_feed_formula.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ No feasible solution found. Try adjusting your requirements.")

# =====================================================
# AI GROWTH PREDICTION
# =====================================================
with tab2:
    st.header("📈 AI Growth Prediction")
    
    st.markdown("""
    Our **Random Forest machine learning model** predicts animal growth based on 
    110+ feeding trials from Nigerian farms.
    """)

    avg_cp = df["CP"].mean()
    avg_energy = df["Energy"].mean()

    X_input = np.array([[age, weight, cp_req, energy_req, feed_intake, avg_cp, avg_energy]])
    prediction = model.predict(X_input)[0]
    
    # Calculate projections
    weekly_gain = prediction * 7
    monthly_gain = prediction * 30
    projected_weight_90d = weight + (monthly_gain * 3 / 1000)

    # Display predictions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Weight Gain", f"{prediction:.1f} g/day")
    with col2:
        st.metric("Weekly Gain", f"{weekly_gain:.0f} g")
    with col3:
        st.metric("Monthly Gain", f"{monthly_gain / 1000:.2f} kg")
    with col4:
        st.metric("90-Day Weight", f"{projected_weight_90d:.1f} kg", delta=f"+{projected_weight_90d - weight:.1f} kg")
    
    # Growth projection chart
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
    
    # Performance metrics
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
        if 'total_cost' in locals() and prediction > 0:
            cost_per_kg_gain = (total_cost * feed_intake * 1000) / prediction
            st.metric("Cost per kg Gain", f"₦{cost_per_kg_gain:.2f}")
            st.caption("Feed cost to produce 1 kg of weight gain")

# =====================================================
# INGREDIENT MANAGER
# =====================================================
with tab3:
    st.header("📋 Ingredient Database Manager")
    
    st.markdown(f"""
    **{len(df)} ingredients** available for {animal} feed formulation. 
    You can edit, add, or remove ingredients below.
    """)
    
    # Search functionality
    search = st.text_input("🔍 Search ingredients", placeholder="Type to filter...")
    
    if search:
        filtered_df = df[df["Ingredient"].str.contains(search, case=False, na=False)]
    else:
        filtered_df = df
    
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
        if st.button("💾 Save Changes to Database"):
            edited_df.to_csv(f"{animal.lower()}_ingredients.csv", index=False)
            st.success(f"✅ Ingredient database updated successfully!")
    
    with col2:
        csv = edited_df.to_csv(index=False)
        st.download_button(
            "📥 Download Database as CSV",
            csv,
            f"{animal.lower()}_ingredients.csv",
            "text/csv",
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p><strong>🌍 AI Livestock Feed Formulator</strong></p>
    <p>Powered by Nigerian agricultural data | Built with Streamlit + ML</p>
    <p style='font-size: 0.85rem;'>Data sources: NIAS, FAO, Nigerian markets (2026)</p>
</div>
""", unsafe_allow_html=True)

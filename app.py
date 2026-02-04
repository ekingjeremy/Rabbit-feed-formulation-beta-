import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Necstech Feed Formulator", layout="wide")

st.title("🌾 Necstech Livestock Feed Formulator")
st.markdown("### AI-Powered Precision Nutrition for Smarter Farming in Nigeria")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Nutrient Guide",
    "🥣 Feed Optimizer",
    "🧮 Ingredient Optimizer",
    "📈 Weight Gain Prediction"
])

# =========================================================
# 📘 TAB 1 — Nutrient Guide
# =========================================================
with tab1:
    animal = st.selectbox("Select Animal", ["Rabbit", "Poultry", "Cattle"])
    stage = st.selectbox("Growth Stage", ["Starter", "Grower", "Finisher"])

    nutrient_data = {
        "Rabbit": {"Starter": [18, 2500, 14],
                   "Grower": [16, 2400, 13],
                   "Finisher": [15, 2300, 12]},
        "Poultry": {"Starter": [22, 3000, 5],
                    "Grower": [20, 2900, 5],
                    "Finisher": [18, 2800, 4]},
        "Cattle": {"Starter": [16, 2600, 18],
                   "Grower": [14, 2500, 20],
                   "Finisher": [12, 2400, 22]}
    }

    cp, energy, fibre = nutrient_data[animal][stage]

    df_export = pd.DataFrame({
        "Animal": [animal],
        "Stage": [stage],
        "Crude Protein %": [cp],
        "Energy kcal/kg": [energy],
        "Fibre %": [fibre]
    })

    st.dataframe(df_export)

    st.download_button("💾 Download Nutrient Guide",
                       df_export.to_csv(index=False),
                       "nutrient_guide.csv",
                       "text/csv")

# =========================================================
# 🥣 TAB 2 — Feed Optimizer
# =========================================================
with tab2:
    cp_req = st.slider("Crude Protein (%)", 10, 30, 18)
    energy_req = st.slider("Energy (kcal/kg)", 2000, 3500, 2800)
    fibre_req = st.slider("Fibre (%)", 3, 25, 10)

    if st.button("🔄 Recalculate Feed Formula"):
        ingredients = pd.DataFrame({
            "Ingredient": ["Maize", "Soybean Meal", "Wheat Bran"],
            "Protein (%)": [9, 44, 15],
            "Energy (kcal/kg)": [3300, 2500, 1800],
            "Fibre (%)": [2, 6, 12],
            "Cost (₦/kg)": [300, 500, 200]
        })

        st.dataframe(ingredients)

        fig = px.bar(ingredients, x="Ingredient", y="Cost (₦/kg)",
                     title="Ingredient Cost Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("💾 Download Feed Formula",
                           ingredients.to_csv(index=False),
                           "feed_formula.csv",
                           "text/csv")

# =========================================================
# 🧮 TAB 3 — Ingredient Optimizer
# =========================================================
with tab3:
    maize = st.slider("Maize (%)", 0, 100, 40)
    soy = st.slider("Soybean Meal (%)", 0, 100, 30)
    bran = st.slider("Wheat Bran (%)", 0, 100, 30)

    total = maize + soy + bran
    st.write(f"Total Mix: **{total}%**")

    if total == 100:
        if st.button("🔄 Update Nutrient Profile"):
            protein = maize*0.09 + soy*0.44 + bran*0.15
            energy = maize*3300 + soy*2500 + bran*1800
            fibre = maize*0.02 + soy*0.06 + bran*0.12

            df_mix = p_

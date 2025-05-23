import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, value
import plotly.express as px

st.set_page_config(page_title="🐰 Rabbit Feed Optimizer Nigeria", layout="wide")
st.title("🐰 Rabbit Feed Formulation Optimizer + Editor + Predictor")

# Initialize default Nigerian ingredient data with categories, nutrients, and costs
if "ingredient_data" not in st.session_state:
    st.session_state.ingredient_data = pd.DataFrame({
        "Category": [
            "Fodder", "Fodder", "Fodder", "Fodder", "Fodder",
            "Fodder", "Fodder", "Fodder", "Fodder",
            "Concentrate", "Concentrate", "Concentrate", "Concentrate", "Concentrate",
            "Concentrate", "Concentrate", "Concentrate"
        ],
        "CP": [
            18, 10, 15, 12, 13, 20, 11, 25, 22,
            9, 44, 45, 15, 40, 12, 18, 60
        ],
        "Energy": [
            2300, 2000, 2100, 1900, 1800, 2100, 1950, 2300, 2100,
            3400, 3200, 3000, 1800, 3100, 2700, 2600, 3200
        ],
        "Fibre": [
            25, 22, 18, 24, 20, 18, 23, 15, 19,
            2, 7, 6, 10, 8, 9, 15, 1
        ],
        "Calcium": [
            1.5, 1.3, 1.2, 1.0, 0.8, 1.3, 1.1, 1.8, 1.4,
            0.02, 0.3, 0.25, 0.1, 0.3, 0.12, 0.4, 4.5
        ],
        "Cost": [
            80, 75, 70, 60, 55, 65, 70, 90, 85,
            120, 150, 130, 90, 140, 100, 110, 200
        ]
    }, index=[
        "Alfalfa", "Elephant Grass", "Sweet Potato Vine", "Guinea Grass", "Banana Leaves",
        "Mulberry Leaves", "Napier Grass", "Seku (Leucaena)", "Cassava Leaves",
        "Maize", "Soybean Meal", "Groundnut Cake", "Wheat Bran", "Cottonseed Cake",
        "Rice Bran", "Palm Kernel Cake", "Fish Meal"
    ])

df = st.session_state.ingredient_data.copy()

# Tabs for functionality
tab1, tab2, tab3 = st.tabs(["🧪 Optimizer", "📝 Edit Ingredients", "📈 Performance Predictor"])

# -------------------- Optimizer Tab --------------------
with tab1:
    st.sidebar.header("Nutrient Requirements (per kg feed)")
    cp_req = st.sidebar.slider("Crude Protein (%)", 10, 25, 16)
    energy_req = st.sidebar.slider("Energy (Kcal/kg)", 1800, 3500, 2500)
    fibre_req = st.sidebar.slider("Fibre (%)", 5, 30, 10)
    calcium_req = st.sidebar.slider("Calcium (%)", 0.1, 1.5, 0.5)

    # Linear programming formulation
    model = LpProblem("Rabbit_Feed_Optimization", LpMinimize)
    vars = {ing: LpVariable(ing, lowBound=0) for ing in df.index}

    # Objective: Minimize cost
    model += lpSum([vars[i] * df.loc[i, 'Cost'] for i in df.index])

    # Nutrient constraints (weighted sum must meet or exceed requirements)
    model += lpSum([vars[i] * df.loc[i, 'CP'] for i in df.index]) >= cp_req
    model += lpSum([vars[i] * df.loc[i, 'Energy'] for i in df.index]) >= energy_req
    model += lpSum([vars[i] * df.loc[i, 'Fibre'] for i in df.index]) >= fibre_req
    model += lpSum([vars[i] * df.loc[i, 'Calcium'] for i in df.index]) >= calcium_req

    # Total mix sums to 1 kg
    model += lpSum([vars[i] for i in df.index]) == 1

    # Solve the LP problem
    model.solve()

    if LpStatus[model.status] == "Optimal":
        st.subheader("📊 Optimized Feed Mix")
        results = {i: vars[i].varValue for i in df.index if vars[i].varValue > 0.001}
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Proportion (kg)'])
        result_df["Category"] = df.loc[result_df.index, "Category"]
        result_df["Cost (₦)"] = result_df["Proportion (kg)"] * df.loc[result_df.index, 'Cost']

        st.dataframe(result_df.style.format({"Proportion (kg)": "{:.3f}", "Cost (₦)": "₦{:.2f}"}))
        st.write(f"**Total Cost per kg Feed: ₦{value(model.objective):.2f}**")

        # Pie chart with category labels
        fig = px.pie(result_df, values='Proportion (kg)', names=result_df.index,
                     title='Feed Ingredient Distribution',
                     color=result_df["Category"],
                     color_discrete_map={"Fodder": "green", "Concentrate": "orange"})
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("⚠️ No feasible solution found with current nutrient settings.")

# -------------------- Edit Ingredients Tab --------------------
with tab2:
    st.subheader("✍️ Edit or Add Ingredients")

    editable_df = df.reset_index().rename(columns={"index": "Ingredient"})

    # Show editable data editor with dynamic rows
    edited_df = st.data_editor(
        editable_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ingredient": st.column_config.TextColumn("Ingredient"),
            "Category": st.column_config.SelectboxColumn("Category", options=["Fodder", "Concentrate"]),
            "CP": st.column_config.NumberColumn("Crude Protein (%)", min_value=0.0, max_value=100.0, format="%.2f"),
            "Energy": st.column_config.NumberColumn("Energy (Kcal/kg)", min_value=0.0, format="%.1f"),
            "Fibre": st.column_config.NumberColumn("Fibre (%)", min_value=0.0, max_value=100.0, format="%.2f"),
            "Calcium": st.column_config.NumberColumn("Calcium (%)", min_value=0.0, max_value=100.0, format="%.3f"),
            "Cost": st.column_config.NumberColumn("Cost (₦/kg)", min_value=0.0, format="%.2f"),
        },
    )

    # Button to save edits
    if st.button("💾 Save Changes"):
        # Validate: All ingredient names present, unique, and no NaNs in category or nutrients
        if (
            "Ingredient" in edited_df.columns
            and edited_df["Ingredient"].notna().all()
            and edited_df["Ingredient"].is_unique
            and edited_df["Category"].notna().all()
            and (edited_df[["CP", "Energy", "Fibre", "Calcium", "Cost"]] >= 0).all().all()
        ):
            cleaned_df = edited_df.dropna(subset=["Ingredient", "Category"])
            cleaned_df = cleaned_df.set_index("Ingredient")
            st.session_state.ingredient_data = cleaned_df
            st.success("Ingredient list updated successfully!")
        else:
            st.error("❌ Please make sure all ingredients have unique names, valid categories, and valid numeric nutrient/cost values.")

    # Button to remove unused ingredients based on last optimization
    if "vars" in locals() and LpStatus[model.status] == "Optimal" and st.button("🧹 Remove unused ingredients"):
        used = [i for i in df.index if vars[i].varValue > 0.001]
        st.session_state.ingredient_data = df.loc[used]
        st.success("Unused ingredients removed.")

    # Upload new ingredients CSV
    st.markdown("---")
    st.subheader("⬆️ Upload New Ingredients CSV")
    st.markdown(
        """
        Upload a CSV file with columns: 
        `Ingredient`, `Category` (Fodder or Concentrate), `CP`, `Energy`, `Fibre`, `Calcium`, `Cost`.
        """
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            new_ingredients = pd.read_csv(uploaded_file)
            # Validate necessary columns
            required_cols = {"Ingredient", "Category", "CP", "Energy", "Fibre", "Calcium", "Cost"}
            if not required_cols.issubset(new_ingredients.columns):
                st.error(f"CSV missing required columns: {required_cols - set(new_ingredients.columns)}")
            else:
                # Clean and append new ingredients
                new_ingredients = new_ingredients.dropna(subset=["Ingredient", "Category"])
                new_ingredients = new_ingredients.set_index("Ingredient")

                # Validate Category values
                if not new_ingredients["Category"].isin(["Fodder", "Concentrate"]).all():
                    st.error("Category column must only contain 'Fodder' or 'Concentrate'.")
                else:
                    # Append avoiding duplicates (overwrite existing)
                    current_df = st.session_state.ingredient_data.copy()
                    current_df.update(new_ingredients)  # update existing with new
                    combined_df = pd.concat([current_df[~current_df.index.isin(new_ingredients.index)], new_ingredients])
                    st.session_state.ingredient_data = combined_df.sort_index()
                    st.success(f"Uploaded {len(new_ingredients)} ingredients successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# -------------------- Performance Predictor Tab --------------------
with tab3:
    st.subheader("🚀 Performance Predictor")

    if "vars" in locals() and LpStatus[model.status] == "Optimal":
        # Calculate weighted average nutrient values of optimized mix
        protein_val = sum(vars[i].varValue * df.loc[i, "CP"] for i in df.index)
        energy_val = sum(vars[i].varValue * df.loc[i, "Energy"] for i in df.index)
        fibre_val = sum(vars[i].varValue * df.loc[i, "Fibre"] for i in df.index)
        calcium_val = sum(vars[i].varValue * df.loc[i, "Calcium"] for i in df.index)

        # Simple mock-up prediction formula (replace with real model as needed)
        expected_gain = 10 + 0.02 * protein_val + 0.0015 * energy_val - 0.01 * fibre_val + 0.05 * calcium_val

        st.metric("📈 Expected Weight Gain (g/day)", f"{expected_gain:.2f}")
        st.markdown(
            """
            > This is a simulated estimate based on nutrient contents.
            > For a real AI prediction, integrate a trained ML model.
            """
        )

        # Show optimized feed summary again
        st.dataframe(pd.DataFrame({
            "Nutrient": ["Crude Protein (%)", "Energy (Kcal/kg)", "Fibre (%)", "Calcium (%)"],
            "Value": [protein_val, energy_val, fibre_val, calcium_val]
        }).style.format({"Value": "{:.2f}"}))
    else:
        st.info("Run the optimizer first to generate a feed mix.")

# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, and dataset
model = joblib.load('rf_crop_yield_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('processed_crop_yield.csv')

# Simplified descriptions for UI clarity
simple_descriptions = {
    'NDVI_Trend': "Vegetation growth rate",
    'Rain_Sum': "Total rainfall during season (mm)",
    'Temp_Avg': "Average temperature (°C)",
    'NDVI_Mean': "General plant health",
    'NDVI_PeakWeek': "Week of best NDVI (crop growth)"
}

# Get context-specific default values
def get_defaults(crop, region, soil):
    subset = df[(df['Crop'] == crop) & (df['Region'] == region) & (df['Soil'] == soil)]
    if subset.empty:
        subset = df
    return {
        'NDVI_Trend': subset['NDVI_Trend'].mean(),
        'Rain_Sum': subset['Rain_Sum'].mean(),
        'Temp_Avg': subset['Temp_Avg'].mean(),
        'NDVI_Mean': subset['NDVI_Mean'].mean(),
        'NDVI_PeakWeek': subset['NDVI_PeakWeek'].mean()
    }

# Make prediction and give advice
def generate_summary(user_data, defaults, prediction, crop, region, soil):
    st.subheader("📝 Field Input Summary")
    st.markdown(f"**Crop**: {crop}")
    st.markdown(f"**Region**: {region}")
    st.markdown(f"**Soil Type**: {soil}")

    for key in defaults:
        val = user_data[key]
        is_default = val == defaults[key]
        label = simple_descriptions[key]
        source_note = "Default" if is_default else "User Provided"
        st.text(f"{key} – {label} ({source_note}: {round(defaults[key], 2)}): {round(val, 2)}")

    st.subheader(f"🌾 Estimated Crop Yield: {prediction:.2f} tons/hectare")

    # Agronomic Suggestions
    st.subheader("📌 Agronomic Suggestions")
    suggestions = []

    if user_data['NDVI_Trend'] < 0.01:
        suggestions.append("🔎 Vegetation growth is slow — improve soil nutrition or irrigation.")
    if user_data['Rain_Sum'] < 400:
        suggestions.append("💧 Low rainfall — consider drought-tolerant crop varieties.")
    if user_data['Temp_Avg'] > 30:
        suggestions.append("🌡️ High temperatures — try early-season sowing next time.")
    if user_data['NDVI_PeakWeek'] > 20:
        suggestions.append("⏳ Late peak growth — check fertilizer and pest schedules.")

    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.success("✅ Excellent! All conditions are optimal for healthy yield.")

# Streamlit UI
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌱 Smart Crop Yield Prediction")
st.markdown("Provide your **field and environmental conditions** to predict expected yield and get personalized agronomic advice.")

# Input Section
with st.form("prediction_form"):
    crop = st.selectbox("🌾 Select Crop Type", sorted(df['Crop'].unique()))
    region = st.selectbox("📍 Select Region", sorted(df['Region'].unique()))
    soil = st.selectbox("🧱 Select Soil Type", sorted(df['Soil'].unique()))

    defaults = get_defaults(crop, region, soil)
    st.markdown("#### 🌦️ Optional Field Metrics (leave blank to use smart defaults)")

    user_data = {}
    for key, desc in simple_descriptions.items():
        default_val = round(defaults[key], 2)
        user_input = st.text_input(f"{desc} ({key})", value="", help=f"Leave empty to use default: {default_val}")
        if user_input.strip() == "":
            user_data[key] = default_val
        else:
            try:
                user_data[key] = float(user_input)
            except ValueError:
                st.warning(f"⚠️ Invalid value for {key}. Default ({default_val}) will be used.")
                user_data[key] = default_val

    submitted = st.form_submit_button("🔍 Predict Yield")

# Run prediction
if submitted:
    input_df = pd.DataFrame([user_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    generate_summary(user_data, defaults, prediction, crop, region, soil)

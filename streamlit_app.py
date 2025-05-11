import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
import base64

# Load model, scaler, and dataset
model = joblib.load('rf_crop_yield_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('processed_crop_yield.csv')

# Descriptions for UI and reporting
simple_descriptions = {
    'NDVI_Trend': "Vegetation growth rate",
    'Rain_Sum': "Total rainfall during season (mm)",
    'Temp_Avg': "Average temperature (°C)",
    'NDVI_Mean': "General plant health",
    'NDVI_PeakWeek': "Week of best NDVI (crop growth)"
}

# Get default values from filtered subset
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

# Create the prediction summary
def generate_summary(user_data, defaults, prediction, crop, region, soil):
    summary_html = f"<h4>📝 Field Input Summary</h4>"
    summary_html += f"<b>Crop:</b> {crop}<br><b>Region:</b> {region}<br><b>Soil Type:</b> {soil}<br><br>"

    summary_html += "<ul>"
    for key in defaults:
        val = user_data[key]['value']  # Actual value used (user or default)
        is_default = user_data[key]['is_default']
        source_note = "(Not Filled, So Default Values Considered)" if is_default else ""
        label = simple_descriptions[key]
        summary_html += f"<li><b>{key}</b> – {label} {source_note}: <b>{round(val, 2)}</b></li>"
    summary_html += "</ul>"

    summary_html += f"<h4>🌾 Estimated Crop Yield:</h4><p><b>{prediction:.2f} tons/hectare</b></p>"

    summary_html += "<h4>📌 Agronomic Suggestions</h4><ul>"
    suggestions = []
    val_dict = {k: user_data[k]['value'] for k in user_data}

    if val_dict['NDVI_Trend'] < 0.01:
        suggestions.append("🔎 Vegetation growth is slow — improve soil nutrition or irrigation.")
    if val_dict['Rain_Sum'] < 400:
        suggestions.append("💧 Low rainfall — consider drought-tolerant crop varieties.")
    if val_dict['Temp_Avg'] > 30:
        suggestions.append("🌡️ High temperatures — try early-season sowing next time.")
    if val_dict['NDVI_PeakWeek'] > 20:
        suggestions.append("⏳ Late peak growth — check fertilizer and pest schedules.")

    if suggestions:
        for s in suggestions:
            summary_html += f"<li>{s}</li>"
    else:
        summary_html += "<li>✅ All field conditions are optimal for strong crop performance.</li>"
    summary_html += "</ul>"

    st.markdown(summary_html, unsafe_allow_html=True)
    return summary_html

# Streamlit layout
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌱 Smart Crop Yield Prediction")
st.markdown("Enter your **field and environmental conditions** to predict yield and get expert advice.")

# Input form
with st.form("prediction_form"):
    crop = st.selectbox("🌾 Crop Type", sorted(df['Crop'].unique()))
    region = st.selectbox("📍 Region", sorted(df['Region'].unique()))
    soil = st.selectbox("🧱 Soil Type", sorted(df['Soil'].unique()))

    defaults = get_defaults(crop, region, soil)
    st.markdown("#### 🌦️ Optional Field Metrics (press Enter to use smart defaults)")

    user_data = {}
    for key, desc in simple_descriptions.items():
        default_val = round(defaults[key], 2)
        user_input = st.text_input(f"{desc} ({key})", value="", help=f"Leave blank to use default: {default_val}")
        if user_input.strip() == "":
            user_data[key] = {'value': default_val, 'is_default': True}
        else:
            try:
                user_data[key] = {'value': float(user_input), 'is_default': False}
            except ValueError:
                st.warning(f"⚠️ Invalid input for {key}. Using default {default_val}.")
                user_data[key] = {'value': default_val, 'is_default': True}

    submitted = st.form_submit_button("🔍 Predict Yield")

# Run prediction and show results
if submitted:
    values_only = [user_data[k]['value'] for k in user_data]
    input_scaled = scaler.transform([values_only])
    prediction = model.predict(input_scaled)[0]
    summary_html = generate_summary(user_data, defaults, prediction, crop, region, soil)

    # PDF-like export using HTML download
    b64 = base64.b64encode(summary_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="field_report.html">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

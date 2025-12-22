import os
import numpy as np
import streamlit as st
import joblib
from auth import require_login

require_login()

# --------------------------
# Paths to model files
# --------------------------
BASE_DIR = os.path.dirname(__file__)
st.title("🌾CropCast")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# --------------------------
# Load models, scaler, encoders
# --------------------------
@st.cache_resource
def load_models():
    missing_files = [p for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH] if not os.path.exists(p)]
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(os.path.basename(f) for f in missing_files)}")
        return None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, scaler, encoders
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

model, scaler, encoders = load_models()
if model is None or scaler is None or encoders is None:
    st.stop()

# ================================
# 🌾 Streamlit UI
# ================================
st.write("Predict estimated crop yield based on environmental and farming conditions.")

# -------------------------------
# 🧠 Input Features
# -------------------------------
region = st.selectbox("Region", encoders["Region"].classes_)
soil_type = st.selectbox("Soil Type", encoders["Soil_Type"].classes_)
crop = st.selectbox("Crop", encoders["Crop"].classes_)
weather = st.selectbox("Weather Condition", encoders["Weather_Condition"].classes_)

# ✅ Toggles for Boolean Inputs
fertilizer_used = st.toggle("Fertilizer Used?")
irrigation_used = st.toggle("Irrigation Used?")

fertilizer_value = "Yes" if fertilizer_used else "No"
irrigation_value = "Yes" if irrigation_used else "No"

# 🌦️ Numeric Inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
harvest = st.number_input("Days to harvest", min_value=0.0, step=0.1)

# -------------------------------
# 🧩 Encode + Scale Features
# -------------------------------
try:
    # Encode categorical features
    region_encoded = encoders["Region"].transform([region])[0]
    soil_encoded = encoders["Soil_Type"].transform([soil_type])[0]
    crop_encoded = encoders["Crop"].transform([crop])[0]
    weather_encoded = encoders["Weather_Condition"].transform([weather])[0]
    fertilizer_encoded = encoders["Fertilizer_Used"].transform([fertilizer_value])[0]
    irrigation_encoded = encoders["Irrigation_Used"].transform([irrigation_value])[0]

    # Scale numeric features
    numeric_features = np.array([[rainfall, temperature, harvest]])
    numeric_features_scaled = scaler.transform(numeric_features)

    # Combine categorical + scaled numeric features in training order
    features = np.array([[
        region_encoded,
        soil_encoded,
        crop_encoded,
        fertilizer_encoded,
        irrigation_encoded,
        weather_encoded,
        numeric_features_scaled[0][0],  # Rainfall
        numeric_features_scaled[0][1],  # Temperature
        numeric_features_scaled[0][2],  # Days_to_Harvest
    ]])

except Exception as e:
    st.error(f"Error processing features: {e}")
    st.stop()

# -------------------------------
# 🚀 Prediction
# -------------------------------
if st.button("Predict Yield"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"🌱 Estimated Crop Yield: **{prediction:.2f} tons/hectare**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

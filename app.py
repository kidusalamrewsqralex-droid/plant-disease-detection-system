import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib
from sklearn.preprocessing import StandardScaler
# -------------------------
# Tabs
# -------------------------
tab1, tab2,tab3 = st.tabs(["Green Thumb", "CYPM", "About"])

# -------------------------
# Tab 1: Plant Disease Detection
# -------------------------
with tab1:
    st.title("🌱 GREEN THUMB")
    st.write("Upload your leaf image. ")
    st.write("TIP:it is better for the background of the image to be 'BLACK'! ")

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "my_model.h5")


    @st.cache_resource
    def load_disease_model():
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}")
            return None
        try:
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_disease_model()

    # -------------------------
    # Class names
    # -------------------------
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
        'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
        'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy',
        'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]

    # -------------------------
    # Full disease responses
    # -------------------------
    disease_responses = {
        "Apple___Apple_scab": """
**Diagnosis:** Apple Scab is a fungal disease caused by *Venturia inaequalis*.

**Cause:** High humidity and wet conditions promote fungal spore growth.

**Treatment:** Apply fungicides such as captan or mancozeb during early stages. Remove fallen leaves and infected fruit.

**Prevention:** Ensure proper pruning for airflow, avoid overhead watering, and clear debris around the tree base.
""",
        "Apple___Black_rot": """
**Diagnosis:** Black Rot is caused by the fungus *Botryosphaeria obtusa*.

**Cause:** Wet, warm weather and infected pruning wounds.

**Treatment:** Prune out cankers and apply fungicide during growing season.

**Prevention:** Sanitize tools, remove mummified fruits, and improve tree spacing.
""",
        "Apple___Cedar_apple_rust": """
**Diagnosis:** Cedar Apple Rust is a fungal disease linked to both apple and cedar trees.

**Cause:** Caused by *Gymnosporangium juniperi-virginianae*, spreads between cedar and apple trees.

**Treatment:** Apply fungicides during early growth stages. Remove nearby cedar trees if possible.

**Prevention:** Use rust-resistant varieties and avoid planting near cedars.
""",
        "Apple___healthy": "✅ The apple plant is healthy. Continue proper watering, pruning, and disease monitoring.",
        "Blueberry___healthy": "✅ The blueberry plant is healthy. Maintain well-drained, acidic soil and avoid waterlogging.",
        "Cherry___healthy": "✅ The cherry plant is healthy. Monitor for signs of mildew or rot during humid seasons.",
        "Cherry___Powdery_mildew": """
**Diagnosis:** Powdery mildew is a fungal infection that forms a white powder on leaves.

**Cause:** High humidity, poor air circulation.

**Treatment:** Apply sulfur-based or neem oil sprays.

**Prevention:** Prune regularly and avoid watering late in the day.
""",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": """
**Diagnosis:** Gray Leaf Spot is caused by *Cercospora zeae-maydis*.

**Cause:** Warm, humid environments with high leaf moisture.

**Treatment:** Use fungicides like strobilurins or triazoles.

**Prevention:** Rotate crops and select resistant hybrids.
""",
        "Corn___Common_rust": """
**Diagnosis:** Common Rust is caused by *Puccinia sorghi*.

**Cause:** Spread by wind-borne spores under moist conditions.

**Treatment:** Use fungicides if infection is severe.

**Prevention:** Plant resistant corn varieties.
""",
        "Corn___Northern_Leaf_Blight": """
**Diagnosis:** Northern Leaf Blight is caused by *Exserohilum turcicum*, leads to cigar-shaped lesions.

**Cause:** Prolonged wetness and mild temperatures.

**Treatment:** Apply fungicides early in disease cycle.

**Prevention:** Rotate crops, use disease-resistant hybrids.
""",
        "Corn___healthy": "✅ The corn plant is healthy. Monitor for discoloration and maintain fertilizer schedule.",
        "Grape___Black_rot": """
**Diagnosis:** Black rot is a common fungal disease in grapes caused by *Guignardia bidwellii*.

**Cause:** Wet weather and poor airflow.

**Treatment:** Apply fungicides and remove infected berries and leaves.

**Prevention:** Train vines properly and prune regularly.
""",
        "Grape___Esca_(Black_Measles)": """
**Diagnosis:** Esca (Black Measles) is a trunk disease caused by multiple fungi.

**Cause:** Enters through pruning wounds, worsened by drought stress.

**Treatment:** No cure, remove infected vines.

**Prevention:** Prune carefully and avoid stress to vines.
""",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": """
**Diagnosis:** Leaf Blight causes dark, angular spots on grape leaves.

**Cause:** Caused by *Isariopsis clavispora*, thrives in wet environments.

**Treatment:** Use copper fungicides and remove infected leaves.

**Prevention:** Improve air circulation, reduce overhead irrigation.
""",
        "Grape___healthy": "✅ The grape plant is healthy. Continue proper training and disease monitoring.",
        "Orange___Haunglongbing_(Citrus_greening)": """
**Diagnosis:** Huanglongbing (HLB), or citrus greening, is caused by a bacterium spread by psyllids.

**Cause:** Insect vector *Diaphorina citri* transmits the bacteria.

**Treatment:** No cure. Remove infected trees.

**Prevention:** Control psyllids and plant resistant rootstocks.
""",
        "Peach___Bacterial_spot": """
**Diagnosis:** Bacterial spot causes lesions on leaves and fruit.

**Cause:** Caused by *Xanthomonas campestris*, thrives in rainy weather.

**Treatment:** Copper sprays and resistant cultivars.

**Prevention:** Avoid overhead irrigation and prune for airflow.
""",
        "Peach___healthy": "✅ The peach tree is healthy. Monitor during wet seasons for leaf spots or fruit pitting.",
        "Pepper,_bell___Bacterial_spot": """
**Diagnosis:** Bacterial spot affects leaves and fruit of bell peppers.

**Cause:** Spread by contaminated tools and wet conditions.

**Treatment:** Use copper-based sprays.

**Prevention:** Avoid working with wet plants, sanitize tools.
""",
        "Pepper,_bell___healthy": "✅ The pepper plant is healthy. Maintain warm, dry soil and avoid splash-back using mulch.",
        "Potato___Early_blight": """
**Diagnosis:** Early blight is caused by *Alternaria solani*.

**Cause:** Warm temperatures and humidity.

**Treatment:** Use chlorothalonil or mancozeb sprays.

**Prevention:** Rotate crops, avoid overhead watering.
""",
        "Potato___Late_blight": """
**Diagnosis:** Late blight is caused by *Phytophthora infestans*.

**Cause:** Cool, wet conditions.

**Treatment:** Apply fungicides such as cymoxanil.

**Prevention:** Remove infected plants immediately and rotate crops.
""",
        "Potato___healthy": "✅ The potato plant is healthy. Hill soil around stems and avoid waterlogging.",
        "Raspberry___healthy": "✅ The raspberry plant is healthy. Mulch properly and prune regularly.",
        "Soybean___healthy": "✅ The soybean plant is healthy. Check for aphids and fungal symptoms during humid weather.",
        "Squash___Powdery_mildew": """
**Diagnosis:** Powdery mildew is caused by *Podosphaera xanthii*.

**Cause:** Dry days followed by humid nights.

**Treatment:** Apply sulfur or neem oil-based sprays.

**Prevention:** Plant in sunny areas and space properly.
""",
        "Strawberry___Leaf_scorch": """
**Diagnosis:** Leaf scorch is caused by fungal pathogens.

**Cause:** High humidity and poor air movement.

**Treatment:** Use fungicides and remove infected leaves.

**Prevention:** Avoid overcrowding and improve drainage.
""",
        "Strawberry___healthy": "✅ The strawberry plant is healthy. Maintain spacing and moist (not wet) soil.",
        "Tomato___Bacterial_spot": """
**Diagnosis:** Bacterial spot causes black lesions on leaves and fruit.

**Cause:** Wet, warm conditions.

**Treatment:** Copper-based fungicides.

**Prevention:** Use clean seeds and avoid overhead watering.
""",
        "Tomato___Early_blight": """
**Diagnosis:** Early blight is caused by *Alternaria solani*.

**Cause:** Poor air circulation and leaf wetness.

**Treatment:** Use mancozeb or chlorothalonil sprays.

**Prevention:** Rotate crops and remove infected debris.
""",
        "Tomato___Late_blight": """
**Diagnosis:** Late blight is caused by *Phytophthora infestans*.

**Cause:** Cool, moist conditions.

**Treatment:** Apply fungicides quickly and remove affected plants.

**Prevention:** Avoid overhead watering and use resistant varieties.
""",
        "Tomato___Leaf_Mold": """
**Diagnosis:** Leaf mold is caused by *Fulvia fulva*.

**Cause:** High humidity in greenhouses or shaded areas.

**Treatment:** Use fungicides and increase ventilation.

**Prevention:** Prune regularly and avoid dense foliage.
""",
        "Tomato___Septoria_leaf_spot": """
**Diagnosis:** Caused by *Septoria lycopersici*, shows small spots on leaves.

**Cause:** High humidity, wet foliage.

**Treatment:** Use chlorothalonil-based fungicides.

**Prevention:** Space plants well and avoid overhead watering.
""",
        "Tomato___Spider_mites Two-spotted_spider_mite": """
**Diagnosis:** Two-spotted spider mite infestation causes stippling and webbing on leaves.

**Cause:** Dry conditions, lack of predators.

**Treatment:** Use miticides or neem oil.

**Prevention:** Maintain moderate humidity, encourage natural predators, rotate crops.
""",
        "Tomato___Target_Spot": """
**Diagnosis:** Target spot is caused by *Corynespora cassiicola*.

**Cause:** Warm, moist conditions and poor airflow.

**Treatment:** Apply appropriate fungicides like chlorothalonil early in infection.

**Prevention:** Increase plant spacing, ensure good drainage, and remove affected leaves.
""",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
**Diagnosis:** Tomato Yellow Leaf Curl Virus (TYLCV) causes leaf curling, yellowing, and stunted growth.

**Cause:** Spread by whiteflies, especially in hot and dry climates.

**Treatment:** No cure — infected plants should be removed immediately.

**Prevention:** Use whitefly-resistant tomato varieties, apply insecticidal soap, and use physical barriers like nets.
""",
        "Tomato___Tomato_mosaic_virus": """
**Diagnosis:** Tomato Mosaic Virus leads to mottled or curled leaves and deformed fruits.

**Cause:** Spread by contaminated tools, hands, or infected seeds.

**Treatment:** No chemical cure — remove infected plants and disinfect tools.

**Prevention:** Wash hands before handling, sterilize equipment, and avoid smoking near plants.
""",
        "Tomato___healthy": "✅ The tomato plant is healthy. Keep monitoring for early signs of pests or disease."
    }

    # -------------------------
    # File upload & prediction
    # -------------------------
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file and model:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img = img.resize((128,128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)/255.0

            with st.spinner("Analyzing image..."):
                preds = model.predict(x)
                pred_class = class_names[np.argmax(preds)]
                confidence = np.max(preds)

            st.success(f"Prediction: {pred_class}")
            st.write(f"Confidence: {confidence:.2f}")

            # Show disease info automatically
            with st.expander("💬 Disease Info"):
                response = disease_responses.get(pred_class, "No additional info available.")
                st.markdown(response)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------
# Tab 2: Crop Yield Prediction
# -------------------------
with tab2:
    import os
    import streamlit as st
    import joblib
    import numpy as np

    # --------------------------
    # Paths to model files
    # --------------------------
    BASE_DIR = os.path.dirname(__file__)
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
    st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌱", layout="centered")
    st.title("🌾 Crop Yield Prediction App")
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

with tab3:
    about_text = """
    ## About This App

    Welcome to the **Crop Yield Prediction & Plant Disease Detection System** — a smart, AI-powered platform designed to help farmers, agronomists, and agriculture enthusiasts make data-driven decisions for healthier crops and better yields.

    ### Crop Yield Prediction
    Using advanced **machine learning models**, this feature predicts the expected yield of your crops based on inputs like soil parameters, weather conditions, and crop type. It helps farmers:
    - Plan better for harvests
    - Optimize resource usage (fertilizers, water, labor)
    - Make informed decisions for sustainable farming

    ### Plant Disease Detection
    This feature leverages **computer vision and deep learning** to identify common diseases in crops from images of leaves. Simply upload a photo of your plant, and the system will:
    - Detect potential diseases accurately
    - Suggest preventive measures and best practices
    - Reduce crop loss by enabling timely intervention

    ### Why Use This Platform
    - **AI-Powered:** Built with modern machine learning and computer vision techniques.
    - **User-Friendly:** No technical expertise required — just enter your data or upload a leaf image.
    - **Reliable Insights:** Provides actionable information to improve crop health and yield.
    - **Sustainable Agriculture:** Supports data-driven, eco-friendly farming practices.

    **Empowering farmers with technology, one crop at a time.**
    """

    st.markdown(about_text)
    st.write("Developer: Kidus Alamrew")



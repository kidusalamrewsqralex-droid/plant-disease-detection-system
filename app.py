import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib
import urllib.parse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="GREEN HAND", page_icon="🌱", layout="centered")
# -------------------------
# Tabs
# -------------------------
tab1, tab2,tab3,tab4 = st.tabs(["Green Thumb","PSDM", "CYPM", "About"])

# -------------------------
# Tab 1: Plant Disease Detection
# -------------------------
with tab1:

    # ===============================
    # Plant Disease Detection App
    # ===============================

    # ------------------------------
    # Imports
    # ------------------------------
    # -----------------------------
    # Imports
    # -----------------------------
    import streamlit as st
    import os
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model

    # -----------------------------
    # Model file path
    # -----------------------------
    # Load the model
    # -----------------------------
    from tensorflow.keras.models import load_model

    model = load_model("models/model.keras", compile=False)

    # Class names
    # -----------------------------
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

    # -----------------------------
    # Disease responses (short example)
    # -----------------------------
    disease_responses = {"Apple___Apple_scab": """
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
                         "Tomato___healthy": "✅ The tomato plant is healthy. Keep monitoring for early signs of pests or disease."}


    # Helper function: preprocess
    # -----------------------------
    def preprocess_image(img):
        img = img.convert("RGB")
        img = img.resize((128, 128))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        return x


    # -----------------------------
    # Streamlit App
    # -----------------------------
    st.title("🌱 GREEN THUMB")
    st.write("Upload a leaf image and detect the plant disease.")
    st.write("TIP:it is better for the background of the image to be 'BLACK'! ")

    uploaded_file = st.file_uploader("Upload an image of your plant:", type=["jpg", "jpeg", "png"])

    from PIL import Image
    import numpy as np
    import streamlit as st

    if uploaded_file is not None and model is not None:
        try:
            # Open the image
            img = Image.open(uploaded_file).convert("RGB")  # ensure 3 channels

            # Display image safely
            st.image(img, caption="Uploaded Image", use_column_width=True)  # alternative to use_container_width

            # Preprocess
            x = preprocess_image(img)  # make sure this returns a shape (1, H, W, 3)

            with st.spinner("Analyzing image..."):
                preds = model.predict(x)
                pred_idx = np.argmax(preds)
                pred_class = class_names[pred_idx]
                confidence = np.max(preds)

            st.success(f"Prediction: {pred_class}")
            st.write(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

            # Show disease info
    with st.expander("💬 Disease Info"):
        response = disease_responses.get(pred_class, "No additional info available.")
        st.markdown(response)

# -------------------------
    # Streamlit UI
    # -------------------------


'''with tab2:
    import os
    import sys
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import streamlit as st
    import pandas as pd
    import altair as alt

    # ---------------------------
    # CONFIG
    # ---------------------------
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pest_identifier_model.h5")  # path to your trained model
    IMAGE_SIZE = (224, 224)
    TOP_K = 5

    # ---------------------------
    # CLASS NAMES (IP102)
    # ---------------------------
    CLASS_NAMES = {
        0: "rice leaf roller", 1: "rice leaf caterpillar", 2: "paddy stem maggot",
        3: "asiatic rice borer", 4: "yellow rice borer", 5: "rice gall midge",
        6: "rice stemfly", 7: "brown plant hopper", 8: "white backed plant hopper",
        9: "small brown plant hopper", 10: "rice water weevil", 11: "rice leafhopper",
        12: "grain spreader thrips", 13: "rice shell pest", 14: "grub",
        15: "mole cricket", 16: "wireworm", 17: "white margined moth",
        18: "black cutworm", 19: "large cutworm", 20: "yellow cutworm",
        21: "red spider", 22: "corn borer", 23: "army worm", 24: "aphids",
        25: "Potosiabre vitarsis", 26: "peach borer", 27: "english grain aphid",
        28: "green bug", 29: "bird cherry-oat aphid", 30: "wheat blossom midge",
        31: "penthaleus major", 32: "longlegged spider mite", 33: "wheat phloeothrips",
        34: "wheat sawfly", 35: "cerodonta denticornis", 36: "beet fly",
        37: "flea beetle", 38: "cabbage army worm", 39: "beet army worm",
        40: "beet spot flies", 41: "meadow moth", 42: "beet weevil",
        43: "serica orientalis motschulsky", 44: "alfalfa weevil", 45: "flax budworm",
        46: "alfalfa plant bug", 47: "tarnished plant bug", 48: "Locustoidea",
        49: "lytta polita", 50: "legume blister beetle", 51: "blister beetle",
        52: "therioaphis maculata Buckton", 53: "odontothrips loti", 54: "Thrips",
        55: "alfalfa seed chalcid", 56: "Pieris canidia", 57: "Apolygus lucorum",
        58: "Limacodidae", 59: "Viteus vitifoliae", 60: "Colomerus vitis",
        61: "Brevipoalpus lewisi McGregor", 62: "oides decempunctata",
        63: "Polyphagotarsonemus latus", 64: "Pseudococcus comstocki Kuwana",
        65: "parathrene regalis", 66: "Ampelophaga", 67: "Lycorma delicatula",
        68: "Xylotrechus", 69: "Cicadella viridis", 70: "Miridae",
        71: "Trialeurodes vaporariorum", 72: "Erythroneura apicalis", 73: "Papilio xuthus",
        74: "Panonchus citri McGregor", 75: "Phyllocoptes oleiverus ashmead",
        76: "Icerya purchasi Maskell", 77: "Unaspis yanonensis", 78: "Ceroplastes rubens",
        79: "Chrysomphalus aonidum", 80: "Parlatoria zizyphus Lucus",
        81: "Nipaecoccus vastalor", 82: "Aleurocanthus spiniferus",
        83: "Tetradacus c Bactrocera minax", 84: "Dacus dorsalis(Hendel)",
        85: "Bactrocera tsuneonis", 86: "Prodenia litura", 87: "Adristyrannus",
        88: "Phyllocnistis citrella Stainton", 89: "Toxoptera citricidus",
        90: "Toxoptera aurantii", 91: "Aphis citricola Vander Goot",
        92: "Scirtothrips dorsalis Hood", 93: "Dasineura sp",
        94: "Lawana imitata Melichar", 95: "Salurnis marginella Guerr",
        96: "Deporaus marginatus Pascoe", 97: "Chlumetia transversa",
        98: "Mango flat beak leafhopper", 99: "Rhytidodera bowrinii white",
        100: "Sternochetus frigidus", 101: "Cicadellidae"
    }
    CLASS_LIST = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]


    # ---------------------------
    # HELPER FUNCTIONS
    # ---------------------------
    @st.cache_resource
    def load_model():
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.stop()
        model = tf.keras.models.load_model(MODEL_PATH)
        return model


    def preprocess_image(image):
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


    def predict(model, img_array):
        preds = model.predict(img_array)
        preds = tf.nn.softmax(preds).numpy().flatten()
        return preds


    def top_k_predictions(preds, k=TOP_K):
        idxs = np.argsort(preds)[::-1][:k]
        return [(CLASS_LIST[i], preds[i]) for i in idxs]


    # ---------------------------
    # STREAMLIT APP
    # ---------------------------
    st.set_page_config(page_title="Pest Identification", layout="wide")
    st.title("🪲 AI Pest Species Identification (IP102)")
    st.write("Upload an image of a pest to identify its species using a trained CNN model.")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload pest image", type=["jpg", "jpeg", "png"])
        st.info("The model predicts one of 102 pest species from the IP102 dataset.")

    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            model = load_model()
            img_array = preprocess_image(image)
            preds = predict(model, img_array)
            topk = top_k_predictions(preds)

            top_pred, top_prob = topk[0]
            st.subheader(f"Prediction: **{top_pred}** ({top_prob * 100:.2f}%)")

            df = pd.DataFrame(topk, columns=["Pest", "Probability"])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Pest", sort="-y"),
                y="Probability",
                color="Pest"
            )
            st.altair_chart(chart, use_container_width=True)

            st.write("**Top Predictions**")
            df["Probability (%)"] = (df["Probability"] * 100).round(2)
            st.dataframe(df[["Pest", "Probability (%)"]])
        else:
            st.warning("Please upload an image to start.")

    st.markdown("---")
    st.caption("This demo uses a CNN trained on the IP102 pest dataset. Always verify results with expert agronomists.")
'''
with tab3:
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





with tab4:
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
    
    import streamlit as st
    import urllib.parse

    your_email = "kidusalamrewsqralex@gmail.com"
    subject = urllib.parse.quote("Green hand - Support Request")
    body = urllib.parse.quote("Hi, what do you want to know about my app, mate?")

    gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={your_email}&su={subject}&body={body}"

    # Use markdown link for reliable behavior
    st.markdown(f"[📧 Contact Developer]({gmail_link})", unsafe_allow_html=True)

    




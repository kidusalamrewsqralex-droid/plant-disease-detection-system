# pages/Plant_Disease_Detection.py
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from auth import require_login
require_login()

# -----------------------------
# Load Model (cached for performance)
# -----------------------------
@st.cache_resource
def load_my_model():
    model = load_model("models/model.keras", compile=False)
    return model

model = load_my_model()


# -----------------------------
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
# Disease responses (example)
# -----------------------------
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
                         "Tomato___healthy": "✅ The tomato plant is healthy. Keep monitoring for early signs of pests or disease."}



# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # shape: (1, H, W, 3)
    return x

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌱 GREEN THUMB")
st.write("Upload a leaf image and detect the plant disease.")
st.write("TIP: For best results, use a clean background (black or white).")

uploaded_file = st.file_uploader("Upload an image of your plant:", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        # Load image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        x = preprocess_image(img)

        # Predict
        with st.spinner("Analyzing image..."):
            preds = model.predict(x)
            pred_idx = np.argmax(preds)
            pred_class = class_names[pred_idx]
            confidence = np.max(preds)

        # Display results
        st.success(f"Prediction: {pred_class}")
        st.write(f"Confidence: {confidence:.2f}")

        # Show disease info
        with st.expander("💬 Disease Info"):
            response = disease_responses.get(pred_class, "No additional info available.")
            st.markdown(response)

    except Exception as e:
        st.error(f"Prediction error: {e}")
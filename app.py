
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Suppress TensorFlow warnings
tab1, tab2,tab3= st.tabs(["PDDM", "CYPM", "About"])
with tab1:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load model
    model = load_model("models/my_model.h5")

    # Define class names (update these for your dataset)
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
        'Cherry___healthy', 'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn___Common_rust',
        'Corn___healthy', 'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

    st.title("🌱 Green Thumb")
    st.write("Built by Kidus Alamrew")

    st.write(
        "Upload a plant leaf image to detect disease (NOTICE:it is better for the background of the image to be 'black'!)")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img = img.resize((128, 128))  # Match your training size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # Predict
        preds = model.predict(x)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds)

        st.write(f"### Prediction: **{pred_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")
        #########################################

        # 🧠 Manual disease response database
        import streamlit as st
        from PIL import Image
        import os

        # ---- Disease response dictionary ----
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

            "Cherry___Powdery_mildew": """
            **Diagnosis:** Powdery mildew is a fungal infection that forms a white powder on leaves.

            **Cause:** High humidity, poor air circulation.

            **Treatment:** Apply sulfur-based or neem oil sprays.

            **Prevention:** Prune regularly and avoid watering late in the day.
            """,

            "Cherry___healthy": "✅ The cherry plant is healthy. Monitor for signs of mildew or rot during humid seasons.",

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

            **Prevention:** Wash hands before handling, sterilize equipment, and avoid smoking near plants (tobacco can carry the virus).
            """,

            "Tomato___healthy": "✅ The tomato plant is healthy. Keep monitoring for early signs of pests or disease. Ensure good air circulation and balanced fertilization."

        }

        with st.expander("💬 Want info about the disease?"):
            st.info("Copy and paste the disease name below (e.g., `Tomato___Late_blight`):")
            user_input = st.text_input("🔍 Disease name")

            if user_input:
                response = disease_responses.get(
                    user_input,
                    "⚠️ Disease not found. Please check the spelling or format.")
                st.markdown("### 🧠 Response:")
                st.markdown(response)
    st.write("Support and reach for us either for help or appreciation!")
with tab2:
    import streamlit as st
    import numpy as np
    import joblib
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression  # or the model class you used
    # Load your trained model

    # get the path relative to app.py
    model_path = os.path.join(os.path.dirname(__file__), "model2.pkl")
    model = joblib.load(model_path)

    st.title("🌾 Crop Yield Prediction App")

    st.markdown("Enter the field details below to predict expected crop yield (tons/hectare).")

    # --- Categorical Inputs ---
    region = st.selectbox("Region", ["West", "South", "North", "East"])
    soil = st.selectbox("Soil Type", ["Sandy", "Clay", "Loam", "Silt", "Peaty", "Chalky"])
    crop = st.selectbox("Crop", ["Cotton", "Rice", "Barley", "Soybean", "Wheat", "Maize"])
    weather = st.selectbox("Weather Condition", ["Cloudy", "Rainy", "Sunny"])

    # --- Numeric Inputs ---
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)
    Days_to_Harvest=st.number_input("Days that were required to harvest", min_value=-10.0, max_value=60.0, step=0.1)
    # --- Boolean Inputs ---
    fertilizer = st.toggle("Fertilizer Used")
    irrigation = st.toggle("Irrigation Used")

    # --- Encoding maps ---
    region_map = {"West": 3, "South": 2, "North": 1, "East": 0}
    soil_map = {"Sandy": 4, "Clay": 1, "Loam": 2, "Silt": 5, "Peaty": 3, "Chalky": 0}
    crop_map = {"Cotton": 1, "Rice": 3, "Barley": 0, "Soybean": 4, "Wheat": 5, "Maize": 2}
    weather_map = {"Cloudy": 0, "Rainy": 1, "Sunny": 2}

    # --- Prepare input array ---
    input_data = np.array([[
        region_map[region],
        soil_map[soil],
        crop_map[crop],
        rainfall,
        temperature,
        int(fertilizer),
        int(irrigation),
        weather_map[weather],
        Days_to_Harvest,  
    ]])

    # --- Prediction ---
    if st.button("Predict Crop Yield"):
        prediction = model.predict(input_data)
        st.success(f"🌱 Estimated Crop Yield: **{prediction[0]:.2f} tons/hectare**")






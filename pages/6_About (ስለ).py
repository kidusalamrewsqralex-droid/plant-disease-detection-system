import streamlit as st
from auth import require_login

require_login()
about_text = """
    ## About This App

    Welcome to the **Crop Yield Prediction & Plant Disease Detection System** — a smart, AI-powered platform designed to help farmers, agronomists, and agriculture enthusiasts make data-driven decisions for healthier crops and better yields.

    ### Green Thumb (Plant Disease Detection Model)
    This feature leverages **computer vision and deep learning** to identify common diseases in crops from images of leaves. Simply upload a photo of your plant, and the system will:
    - Detect potential diseases accurately
    - Suggest preventive measures and best practices
    - Reduce crop loss by enabling timely intervention

    ### Crop Cast (Crop Yield Prediction Model)
    Using advanced **machine learning models**, this feature predicts the expected yield of your crops based on inputs like soil parameters, weather conditions, and crop type. It helps farmers:
    - Plan better for harvests
    - Optimize resource usage (fertilizers, water, labor)
    - Make informed decisions for sustainable farming

    ### Crop Recommender
    The **Crop Recommender** is a dedicated module that suggests the **best crop to grow** based on your local soil and environmental conditions. By entering data such as:
    - Nitrogen, Phosphorus, and Potassium levels
    - Temperature and Humidity
    - Soil pH and Rainfall

    The system predicts the **most suitable crop** using trained machine learning models like **Random Forest** and **XGBoost**, helping farmers:
    - Maximize yield potential
    - Reduce crop failure
    - Make smart, data-driven planting decisions

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

st.markdown(f"[📧 Contact Developer]({gmail_link})", unsafe_allow_html=True)

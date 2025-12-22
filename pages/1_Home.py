import streamlit as st
from auth import require_login

require_login()
st.set_page_config(
    page_title="Green Hand",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar branding
st.sidebar.title("🌱 Green Hand")
st.sidebar.caption("AI for Smart Agriculture")

# Main welcome
st.title("Welcome to Green Hand 🌿")
st.markdown("""
Use the sidebar to navigate between:
- 🌿 Green Thumb (Plant Disease Detection Model)
- 🌾 Crop Cast (Crop Yield Prediction Model)
- ℹ️ About Models
""")

st.info("Empowering farmers with AI-driven insights 🌍")

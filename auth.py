import streamlit as st

def require_login():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Please log in to access this page.")
        st.switch_page("app.py")
        st.stop()

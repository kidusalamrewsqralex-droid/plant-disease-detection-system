import streamlit as st

def require_login():
    if not st.session_state.get("logged_in", False):
        st.query_params["redirected"] = "true"
        st.switch_page("app.py")
        st.stop()

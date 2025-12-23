import streamlit as st
import hashlib
import json
import os
import pandas as pd
USERS_FILE = "users.json"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

import streamlit as st

st.set_page_config(
    page_title="Login/Sign-up",
    page_icon="🌱",
    layout="centered"
)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""


def login(username, password):
    users = load_users()
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

        st.header("🛠 Admin Dashboard")
        st.write(f"Logged in as Admin: {st.session_state.username}")

            # Load users
        users = load_users()  # This is your function that reads users.json

        if users:
            st.subheader("Registered Users")
                # Convert to DataFrame for nice display
            df_users = pd.DataFrame([
                    {"Username": u, "Role": info["role"]} for u, info in users.items()
                ])
            st.dataframe(df_users)
        else:
            st.write("No users registered yet.")

        st.session_state.logged_in = True
        st.session_state.username = ADMIN_USERNAME
        st.session_state.role = "admin"
        return True
    elif username in users and hash_password(password) == users[username]["password"]:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = "user"
        st.success(f"Logged in as {username} ✅")
        return True
    else:
        st.error("❌ Invalid username or password")
        return False


def signup(username, password):
    if username == ADMIN_USERNAME:
        st.error("❌ This username is reserved for admin")
        return
    users = load_users()
    if username in users:
        st.error("❌ Username already exists")
        return
    users[username] = {"password": hash_password(password), "role": "user"}
    save_users(users)
    st.success("Signup successful! You can now log in.")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.success("Logged out successfully ✅")


st.title("🌱 GREEN HAND LOGIN SYSTEM")

# Display login/signup or home UI depending on session state
if st.session_state.logged_in:
    st.write(f"Welcome, **{st.session_state.username}**! Role: {st.session_state.role}")
    if st.button("Logout"):
        logout()
else:
    choice = st.radio("Login / Signup", ["Login", "Signup"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(username, password)  # Session state changes immediately take effect

    elif choice == "Signup":
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        if st.button("Sign Up"):
            if password != confirm:
                st.error("❌ Passwords do not match")
            else:
                signup(username, password)

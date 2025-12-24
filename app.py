import streamlit as st
import hashlib
import json
import os
import pandas as pd

# ---------------- CONFIG ----------------
USERS_FILE = "users.json"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

st.set_page_config(
    page_title="Login / Signup",
    page_icon="🌱",
    layout="centered"
)

# ---------------- HELPERS ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # File exists but is empty or invalid → start fresh
            return {}
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ---------------- AUTH LOGIC (NO UI HERE) ----------------
def login(username, password):
    users = load_users()

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.username = ADMIN_USERNAME
        st.session_state.role = "admin"
        return True

    if username in users and hash_password(password) == users[username]["password"]:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = "user"
        return True

    st.error("❌ Invalid username or password")
    return False

def signup(username, password):
    if username == ADMIN_USERNAME:
        st.error("❌ This username is reserved")
        return

    users = load_users()
    if username in users:
        st.error("❌ Username already exists")
        return

    users[username] = {
        "password": hash_password(password),
        "role": "user"
    }
    save_users(users)
    st.success("✅ Signup successful! You can now log in.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# ---------------- UI ----------------
st.title("🌱 GREEN HAND LOGIN SYSTEM")
if "redirected" in st.query_params:
    st.warning("🔒 Please log in to access that page.")
if st.session_state.logged_in:

    st.success(f"Welcome, {st.session_state.username} 👋")

    # ---------- ADMIN DASHBOARD ----------
    if st.session_state.role == "admin":
        st.header("🛠 Admin Dashboard")

        users = load_users()
        if users:
            df = pd.DataFrame([
                {"Username": u, "Role": info["role"]}
                for u, info in users.items()
            ])
            st.dataframe(df)
        else:
            st.write("No users registered yet.")

    # ---------- USER DASHBOARD ----------
    else:
        st.header("👤 User Dashboard")
        st.write("You are logged in as a normal user.")

    if st.button("Logout"):
        logout()

# ---------------- LOGIN / SIGNUP ----------------
else:
    choice = st.radio("Login / Signup", ["Login", "Signup"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            login(username, password)

    else:
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        confirm = st.text_input("Confirm password", type="password")

        if st.button("Sign Up"):
            if password != confirm:
                st.error("❌ Passwords do not match")
            else:
                signup(username, password)

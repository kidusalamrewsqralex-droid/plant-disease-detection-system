import streamlit as st
import pandas as pd
from auth import require_login
require_login()

st.title("📊 Model Comparison Dashboard")

# 1️⃣ Load your precomputed comparison table
results_df = pd.read_csv("data/crop cast model_comparison.csv")  # Make sure path is correct

# 2️⃣ Display full table
st.subheader("Crop_Cast Models")
st.dataframe(results_df)

st.write("R2-score graph")
st.bar_chart(results_df.set_index("Model")["R2"])
# 3️⃣ Determine the best model (lowest RMSE)
best_model = results_df.sort_values("RMSE").iloc[0]

# 4️⃣ Show the winning model
st.subheader("🏆 Winning Model")
st.success(f"**{best_model['Model']}** is the best model! \n"
           f"RMSE: {best_model['RMSE']:.2f}, R²: {best_model['R2']:.2f}")


st.subheader("Green_Thumb Models")


@st.cache_data
def load_results2():
    return pd.read_csv("data/green thumb model_comparison.csv")

df = load_results2()

st.dataframe(df)

st.write("Val_accuracy graph")
st.bar_chart(df.set_index("Model")["Val_accuracy"])

# Sort and pick the first row (best model)
best_model2 = df.sort_values("Val_accuracy", ascending=False).iloc[0]

# Access scalar values
val_acc = float(best_model2['Val_accuracy'])
val_loss = float(best_model2['val_loss'])

# Now format safely
st.subheader("🏆 Winning Model")
st.success(
    f"**{best_model2['Model']}** is the best model! \n"
    f"Val Accuracy: {val_acc:.2f}, Val Loss: {val_loss:.2f}"
)

import streamlit as st
import pandas as pd
import joblib
import os
from geopy.distance import geodesic
import hashlib
import base64

# ---------------------- Background Image Setup ----------------------
def get_base64_of_bin_file(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def set_local_bg(image_path):
    encoded = get_base64_of_bin_file(image_path)
    if encoded:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
            }}
            .money-legit {{
                font-size: 40px;
                color: green;
                animation: moneyFlow 2s infinite;
            }}
            .money-fraud {{
                font-size: 40px;
                color: red;
                animation: redAlert 2s infinite;
            }}
            @keyframes moneyFlow {{
                0% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-10px); }}
                100% {{ transform: translateY(0); }}
            }}
            @keyframes redAlert {{
                0%, 100% {{ color: red; }}
                50% {{ color: darkred; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Background image not found. Using default white background.")

# Set background image
set_local_bg("background.jpg")

# ---------------------- Load Model and Encoders ----------------------
try:
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")
except FileNotFoundError:
    st.error("❌ Model or encoder file not found.")
    st.stop()

# ---------------------- Utility ----------------------
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# ---------------------- App UI ----------------------
st.title("💳 Fraud Detection System")
st.write("### Enter the transaction details below:")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude", format="%.6f")
long = st.number_input("Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)
gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number", type="password")

# Calculate distance
distance = haversine(lat, long, merch_lat, merch_long)

# ---------------------- Prediction ----------------------
if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categoricals
        for col in ['merchant', 'category', 'gender']:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1  # Unknown category

        # Hash CC number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16) % (10 ** 8))

        # Prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.markdown("<div class='money-fraud'>🚨 Fraudulent Transaction 🚨</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='money-legit'>💵 Legitimate Transaction 💵</div>", unsafe_allow_html=True)

        st.subheader(f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
    else:
        st.error("❗ Please fill all required fields.")

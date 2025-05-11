import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic
from streamlit_lottie import st_lottie
import requests

# Load model and encoders
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

# Load money animation from LottieFiles
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

money_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5ngs2ksb.json")

# Calculate distance between user and merchant
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Title and animation
st.title("💰 Fraud Detection System")
st_lottie(money_lottie, height=200, key="intro_animation")
st.write("Enter the transaction details below:")

# Input fields
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

# Compute distance
distance = haversine(lat, long, merch_lat, merch_long)

# Prediction logic
if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categorical columns
        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1  # Assign -1 for unknown categories

        # Hash credit card number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Cast to correct types
        input_data[['amt', 'distance', 'hour', 'day', 'month']] = input_data[['amt', 'distance', 'hour', 'day', 'month']].astype(float)

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.subheader(f"Prediction: {result}")

        # Show animation again after result
        st_lottie(money_lottie, height=200, key="result_animation")
    else:
        st.error("❗ Please fill in all required fields.")

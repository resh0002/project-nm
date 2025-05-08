import streamlit as st
import pandas as pd
import joblib
import hashlib
from geopy.distance import geodesic

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")  # should be a dict: {col: encoder}
    return model, encoder

model, encoder = load_model_and_encoders()

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def consistent_hash(x):
    return int(hashlib.sha256(x.encode()).hexdigest(), 16) % (10 ** 2)

# Streamlit UI
st.title("🛡️ Fraud Detection System")
st.write("Enter the transaction details below to check if it's fraudulent.")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("User Latitude", format="%.6f")
long = st.number_input("User Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)
gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

distance = haversine(lat, long, merch_lat, merch_long)

if st.button("Check for Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([{
            'merchant': merchant,
            'category': category,
            'amt': amt,
            'distance': distance,
            'hour': hour,
            'day': day,
            'month': month,
            'gender': gender,
            'cc_num': cc_num
        }])

        # Encode categorical features
        categorical_cols = ['merchant', 'category', 'gender']
        for col in categorical_cols:
            try:
                input_data[col] = input_data[col].apply(lambda x: encoder[col].transform([x])[0])
            except Exception:
                input_data[col] = -1  # Unknown category

        # Hash the credit card number consistently
        input_data['cc_num'] = input_data['cc_num'].apply(consistent_hash)

        # Predict
        prediction = model.predict(input_data)[0]
        result = "🛑 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please fill in all required fields.")

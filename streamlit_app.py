import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# --- Load model and encoders ---
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

# --- Set full-page background image with working CSS ---
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://rulesware.com/wp-content/uploads/2021/09/fraud.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Customize the layout to show content properly over the background */
        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 10px 20px rgba(0,0,0,0.2);
        }

        /* Add specific styling to improve readability */
        .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stButton {
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 10px;
            border-radius: 8px;
        }

        /* Heading and prediction results */
        .stTitle, .stSubheader, .stError {
            color: #333;
        }

        /* Money Animation for legitimate transaction */
        .money-legit {{
            font-size: 40px;
            color: green;
            animation: moneyFlow 2s infinite;
        }}

        /* Money Animation for fraudulent transaction */
        .money-fraud {{
            font-size: 40px;
            color: red;
            animation: redAlert 2s infinite;
        }}

        /* Keyframes for money flowing */
        @keyframes moneyFlow {{
            0% {{
                transform: translateY(0);
            }}
            50% {{
                transform: translateY(-20px);
            }}
            100% {{
                transform: translateY(0);
            }}
        }}

        /* Keyframes for red alert */
        @keyframes redAlert {{
            0% {{
                color: red;
            }}
            50% {{
                color: darkred;
            }}
            100% {{
                color: red;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

# --- Haversine distance function ---
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# --- App Title ---
st.title("💳 Fraud Detection System")
st.write("### Enter the transaction details below:")

# --- Inputs ---
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.

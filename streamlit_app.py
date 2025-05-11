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

# --- Calculate distance ---
distance = haversine(lat, long, merch_lat, merch_long)

# --- Predict button ---
if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categoricals
        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        # Hash credit card number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Predict
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        
        # Display the result with animation
        if prediction == 1:
            st.markdown(
                f"<div class='money-fraud'>🚨 Fraudulent Transaction 🚨</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='money-legit'>💵 Legitimate Transaction 💵</div>",
                unsafe_allow_html=True
            )
        st.subheader(f"Prediction: {result}")
    else:
        st.error("❗ Please fill all required fields.")

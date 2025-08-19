import requests

MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
MODEL_PATH = "expresso_churn_model.pkl"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")


import streamlit as st
import pandas as pd
import joblib
import os

# Load model
if not os.path.exists("expresso_churn_model.pkl"):
    st.error("Model file not found. Please make sure expresso_churn_model.pkl is in the same folder as this app.py")
else:
    model = joblib.load("expresso_churn_model.pkl")

    st.title("Expresso Churn Prediction")

    st.write("Enter the client's details to predict churn:")

    # Input fields (excluding REGION, because it was dropped during training)
    TENURE = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    MONTANT = st.number_input("Montant", min_value=0.0, value=0.0)
    FREQUENCE_RECH = st.number_input("Frequence Rech", min_value=0.0, value=0.0)
    REVENUE = st.number_input("Revenue", min_value=0.0, value=0.0)
    ARPU_SEGMENT = st.number_input("ARPU Segment", min_value=0.0, value=0.0)
    FREQUENCE = st.number_input("Frequence", min_value=0.0, value=0.0)
    DATA_VOLUME = st.number_input("Data Volume", min_value=0.0, value=0.0)
    ON_NET = st.number_input("On Net", min_value=0.0, value=0.0)
    ORANGE = st.number_input("Orange", min_value=0.0, value=0.0)
    TIGO = st.number_input("Tigo", min_value=0.0, value=0.0)
    MRG = st.number_input("MRG (encoded as number)", min_value=0, value=0)
    REGULARITY = st.number_input("Regularity", min_value=0, value=0)
    TOP_PACK = st.number_input("Top Pack (encoded as number)", min_value=0, value=0)
    FREQ_TOP_PACK = st.number_input("Frequency Top Pack", min_value=0.0, value=0.0)

    # Predict button
    if st.button("Predict Churn"):
        input_data = pd.DataFrame([[
            TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
            FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
            MRG, REGULARITY, TOP_PACK, FREQ_TOP_PACK
        ]], columns=[
            "TENURE", "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT",
            "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
            "MRG", "REGULARITY", "TOP_PACK", "FREQ_TOP_PACK"
        ])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"The client is likely to churn (Probability: {proba:.2f})")
        else:
            st.success(f"The client is likely to stay (Probability: {proba:.2f})")


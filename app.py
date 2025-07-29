# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("corrosion_model.pkl")
le_inhibitor = joblib.load("le_inhibitor.pkl")
le_source = joblib.load("le_source.pkl")

# UI
st.title(" Corrosion Inhibitor Selector")
st.write("Predict the inhibition efficiency based on lab conditions.")

inhibitor = st.selectbox("Inhibitor", le_inhibitor.classes_)
source = st.selectbox("Source", le_source.classes_)
concentration = st.number_input("Concentration (ppm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
time = st.number_input("Exposure Time (h)", min_value=0.0)

if st.button("Predict Efficiency"):
    inhibitor_code = le_inhibitor.transform([inhibitor])[0]
    source_code = le_source.transform([source])[0]

    X_input = pd.DataFrame([[
        inhibitor_code, source_code, concentration, temperature, time
    ]], columns=["Inhibitor_Code", "Source_Code", "Concentration (ppm)",
                 "Temperature (°C)", "Exposure Time (h)"])

    prediction = model.predict(X_input)[0]
    st.success(f"📈 Predicted Inhibition Efficiency: {prediction:.2f}%")

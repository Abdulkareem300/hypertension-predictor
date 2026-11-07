import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('hypertension_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. Design the Streamlit UI
st.title("🩺 Hypertension Stage Predictor")
st.write("Input the demographic, lifestyle, and clinical parameters below to predict SBP, DBP, and classification level simultaneously.")

# Create a structured layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("Age Category", [0, 1, 2], help="Choose your age category index")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    BMI = st.slider("BMI Category", 0, 3, 1)

with col2:
    resi = st.selectbox("Residence", [0, 1], format_func=lambda x: "Urban" if x == 1 else "Rural")
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    odisease = st.slider("Other Diseases Index", 0, 3, 0)

with col3:
    creantine = st.selectbox("Creatinine level high?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bun = st.selectbox("BUN level high?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    noofmed = st.selectbox("Number of Meds", [0, 1])

# 3. Model Prediction logic
if st.button("Predict Hypertension Stages", type="primary"):
    # Assemble inputs in correct feature sequence
    input_data = pd.DataFrame([[
        age, sex, BMI, resi, smoking, odisease, creantine, bun, noofmed
    ]], columns=['age', 'sex', 'BMI', 'Resi', 'Smoking', 'odisease', 'creantine', 'BUN', 'Noofmed'])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Render results nicely
    st.success("### Prediction Results:")
    st.metric(label="Predicted Systolic Blood Pressure Category (SBP)", value=prediction[0])
    st.metric(label="Predicted Diastolic Blood Pressure Category (DBP)", value=prediction[1])
    st.metric(label="Hypertension Class Level", value=prediction[2])

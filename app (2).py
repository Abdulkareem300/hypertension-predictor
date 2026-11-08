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

# 2. Design the Streamlit UI with medical explanations
st.set_page_config(page_title="Hypertension Stage Predictor", layout="wide")

st.title("🩺 Hypertension Stage & Blood Pressure Predictor")

# Prominent Disclaimer at the top
st.warning("⚠️ **DISCLAIMER**: This application is a machine learning prediction tool designed for educational purposes only. It is **NOT** a diagnostic tool and does not replace professional medical advice, diagnosis, or treatment.")

st.write("""
This tool estimates **Systolic Blood Pressure (SBP)**, **Diastolic Blood Pressure (DBP)**, and **Hypertension Stage**
simultaneously based on patient demographics, lifestyle indicators, and kidney function markers.
""")

# Adding a collapsible section with medical definitions
with st.expander("📖 Detailed Medical Terms, Abbreviations & Categorizations"):
    st.markdown("""
    * **Systolic Blood Pressure (SBP)**: The peak pressure exerted against arterial walls during heart contraction (top value).
    * **Diastolic Blood Pressure (DBP)**: The residual pressure in arteries during heart relaxation between beats (bottom value).
    * **BMI (Body Mass Index)**: Metric associating height & weight to estimate body fat categories.
    * **BUN (Blood Urea Nitrogen)**: Waste product indicating renal health (Elevated levels suggest reduced kidney filtration rate).
    * **Creatinine (creantine)**: Metabolic waste cleared by kidneys. High serum values suggest compromised kidney performance.
    * **Other Comorbidities (odisease)**: Score indicating presence of other active chronic medical conditions.
    * **Noofmed (Number of Medications)**: Indication of whether the patient is already on active antihypertensive or cardiovascular drug therapy.
    """)

st.divider()

# Translation dictionaries for user-friendly inputs
age_options = {
    "Young Adult (Under 35 years of age)": 0,
    "Middle-aged Adult (35 to 59 years of age)": 1,
    "Senior / Elderly (60+ years of age)": 2
}

bmi_options = {
    "Underweight or Normal weight (BMI < 25.0)": 0,
    "Overweight (BMI 25.0 to 29.9)": 1,
    "Obese Class I or II (BMI 30.0 to 39.9)": 2,
    "Severely/Morbidly Obese (BMI 40.0 or higher)": 3
}

sex_options = {
    "Female": 0,
    "Male": 1
}

residence_options = {
    "Urban Area (City / High Density)": 1,
    "Rural Area (Country / Low Density)": 0
}

smoking_options = {
    "No / Never Smoked": 0,
    "Yes / Active or Former Smoker": 1
}

comorbidity_options = {
    "0 (No other chronic diseases)": 0,
    "1 (Mild - One other chronic disease)": 1,
    "2 (Moderate - Two other chronic diseases)": 2,
    "3 (Severe - Three or more other chronic conditions)": 3
}

creatinine_options = {
    "Normal Serum Creatinine levels": 0,
    "Abnormal / Elevated Serum Creatinine levels": 1
}

bun_options = {
    "Normal Blood Urea Nitrogen levels": 0,
    "Abnormal / Elevated Blood Urea Nitrogen levels": 1
}

medication_options = {
    "None (Not currently taking blood pressure medications)": 0,
    "On Treatment (Currently taking 1 or more blood pressure medications)": 1
}

# Target categories decoding
sbp_meanings = {
    "0": "Normal / Prehypertension (Under 130 mmHg)",
    "1": "Stage 1 Hypertension (130 - 139 mmHg)",
    "2": "Stage 2 Hypertension / Crisis (140+ mmHg)"
}

dbp_meanings = {
    "0": "Normal / Prehypertension (Under 80 mmHg)",
    "1": "Stage 1 Hypertension (80 - 89 mmHg)",
    "2": "Stage 2 Hypertension / Crisis (90+ mmHg)"
}

# Create a structured layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    selected_age = st.selectbox("Age Range Classification", list(age_options.keys()))
    age = age_options[selected_age]

    selected_sex = st.selectbox("Biological Sex", list(sex_options.keys()))
    sex = sex_options[selected_sex]

    selected_bmi = st.selectbox("Weight Profile (BMI Category)", list(bmi_options.keys()))
    BMI = bmi_options[selected_bmi]

with col2:
    st.subheader("🏠 Lifestyle & Environment")
    selected_resi = st.selectbox("Residential Environment", list(residence_options.keys()))
    resi = residence_options[selected_resi]

    selected_smoking = st.selectbox("Tobacco Exposure / Smoking Status", list(smoking_options.keys()))
    smoking = smoking_options[selected_smoking]

    selected_odisease = st.selectbox("Co-existing Chronic Illnesses (Comorbidities)", list(comorbidity_options.keys()))
    odisease = comorbidity_options[selected_odisease]

with col3:
    st.subheader("🧪 Laboratory & Clinical Markers")
    selected_creantine = st.selectbox("Serum Creatinine Status", list(creatinine_options.keys()))
    creantine = creatinine_options[selected_creantine]

    selected_bun = st.selectbox("BUN (Blood Urea Nitrogen) Status", list(bun_options.keys()))
    bun = bun_options[selected_bun]

    selected_noofmed = st.selectbox("Current Cardiovascular Therapy", list(medication_options.keys()))
    noofmed = medication_options[selected_noofmed]

st.divider()

# 3. Model Prediction logic
if st.button("🔮 Predict Cardiovascular & Hypertension Profile", type="primary", use_container_width=True):
    # Assemble inputs in correct feature sequence expected by model
    input_data = pd.DataFrame([[
        age, sex, BMI, resi, smoking, odisease, creantine, bun, noofmed
    ]], columns=['age', 'sex', 'BMI', 'Resi', 'Smoking', 'odisease', 'creantine', 'BUN', 'Noofmed'])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Decode predictions securely
    predicted_sbp_code = str(prediction[0])
    predicted_dbp_code = str(prediction[1])
    predicted_level = str(prediction[2]).strip()

    sbp_text = sbp_meanings.get(predicted_sbp_code, f"Category {predicted_sbp_code}")
    dbp_text = dbp_meanings.get(predicted_dbp_code, f"Category {predicted_dbp_code}")

    # Display Output Analysis
    st.markdown("### 📋 Simulated Clinical Profile Summary")
    
    # Create an impactful card container with status alerts based on severity
    stage_upper = predicted_level.upper()
    
    if "CRISIS" in stage_upper:
        st.error(f"### 🚨 CRITICAL PROFILE DETECTED: {stage_upper}")
    elif "STAGE 2" in stage_upper:
        st.warning(f"### ⚠️ HIGH RISK PROFILE DETECTED: {stage_upper}")
    else:
        st.success(f"###  STABLE PROFILE DETECTED: {stage_upper}")
        
    # Display metrics side-by-side inside this status frame
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.info(f"**Systolic Blood Pressure (SBP)**\n### {sbp_text}")
    with col_res2:
        st.info(f"**Diastolic Blood Pressure (DBP)**\n### {dbp_text}")

    # Footer disclaimer
    st.info("🚨 **Notice**: This system uses statistical probability from historical datasets to make predictions. Accuracy is dependent on model limits. Always obtain medical diagnostic advice from a registered professional doctor using calibrated hardware.")

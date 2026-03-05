# -*- coding: utf-8 -*-
"""
Health Risk Prediction Web App (Streamlit)
- Multi-factor risk assessment
- Clinical history integration
- Advanced visualizations
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# =============== App Constants ===============
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "trained_model.sav")
DEFAULT_SCALER_PATH = os.path.join(APP_DIR, "scaler.sav")
DEFAULT_GENDER_ENCODER_PATH = os.path.join(APP_DIR, "gender_encoder.sav")
DEFAULT_DATA_PATH = os.path.join(APP_DIR, "diabetes.csv")

FEATURE_NAMES = [
    "Age", "Gender", "Pulse Rate", "Systolic BP", "Diastolic BP",
    "Glucose", "Height", "Weight", "BMI",
    "Family Diabetes", "Hypertensive", "Family Hypertension",
    "Cardiovascular Disease", "Stroke"
]

# =============== Utils ===============
@st.cache_resource
def load_model_and_encoders():
    """Load model, scaler, and gender encoder"""
    model = pickle.load(open(DEFAULT_MODEL_PATH, "rb"))
    scaler = pickle.load(open(DEFAULT_SCALER_PATH, "rb"))
    gender_encoder = pickle.load(open(DEFAULT_GENDER_ENCODER_PATH, "rb"))
    return model, scaler, gender_encoder

def predict_risk(model, scaler, gender_encoder, inputs_dict):
    """Make risk prediction"""
    
    # Convert gender to numeric
    gender_encoded = 1 if inputs_dict['gender'] == 'Male' else 0
    
    # Create feature array
    features = [
        inputs_dict['age'],
        gender_encoded,
        inputs_dict['pulse_rate'],
        inputs_dict['systolic_bp'],
        inputs_dict['diastolic_bp'],
        inputs_dict['glucose'],
        inputs_dict['height'],
        inputs_dict['weight'],
        inputs_dict['bmi'],
        inputs_dict['family_diabetes'],
        inputs_dict['hypertensive'],
        inputs_dict['family_hypertension'],
        inputs_dict['cardiovascular'],
        inputs_dict['stroke']
    ]
    
    # Convert to array and scale
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    
    # Predict
    prediction = model.predict(x_scaled)[0]
    probability = model.predict_proba(x_scaled)[0]
    
    # Get risk percentage
    risk_percentage = probability[1] * 100  # Probability of diabetic
    
    # Determine risk level
    if risk_percentage < 30:
        risk_level = "LOW"
        risk_color = "green"
    elif risk_percentage < 60:
        risk_level = "MODERATE"
        risk_color = "orange"
    else:
        risk_level = "HIGH"
        risk_color = "red"
    
    # Confidence (how sure the model is)
    confidence = max(probability) * 100
    
    # Validation (if confidence is high enough)
    validation = "Passed" if confidence > 70 else "Review Needed"
    
    return {
        'prediction': prediction,
        'risk_percentage': risk_percentage,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'confidence': confidence,
        'validation': validation,
        'probability': probability
    }

# =============== PWA Support ===============
def add_pwa_support():
    """Add PWA support for mobile installation"""
    pwa_html = """
    <link rel="manifest" href="manifest.json">
    <link rel="icon" href="icon-192.png">
    <link rel="apple-touch-icon" href="icon-192.png">
    <meta name="theme-color" content="#667eea">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="HealthRisk">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """
    components.html(pwa_html, height=0)

# =============== UI ===============
st.set_page_config(
    page_title="DiaPredict",
    page_icon="🩺",
    layout="centered"
)

add_pwa_support()

st.title("🩺 DiaPredict")

# Load model
try:
    model, scaler, gender_encoder = load_model_and_encoders()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =============== MAIN INTERFACE ===============

st.markdown("### 👤 Physical Statistics")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Biological Gender", ["Male", "Female"])
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

with col2:
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=175, step=1)
    
    # Auto-calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    st.metric("Body Mass Index", f"{bmi:.2f}")

st.markdown("### 💓 Vital Indicators")

col3, col4 = st.columns(2)

with col3:
    pulse_rate = st.number_input("Resting HR (BPM)", min_value=40, max_value=150, value=72, step=1)
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120, step=1)

with col4:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=90.0, step=0.1)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=80, step=1)

# Age
age = st.number_input("Age (years)", min_value=18, max_value=100, value=35, step=1)

st.markdown("### 📋 Clinical History")

col5, col6 = st.columns(2)

with col5:
    hypertension = st.checkbox("Hypertension")
    cardiovascular = st.checkbox("Cardiovascular Disease")

with col6:
    stroke = st.checkbox("Stroke History")
    family_diabetes = st.checkbox("Lineal Diabetes")

family_hypertension = st.checkbox("Lineal Hypertension")

# =============== PREDICTION ===============

if st.button("🔍 Analyze Risk Factor", use_container_width=True):
    
    # Prepare inputs
    inputs = {
        'age': age,
        'gender': gender,
        'pulse_rate': pulse_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose': glucose,
        'height': height_m,
        'weight': weight_kg,
        'bmi': bmi,
        'family_diabetes': 1 if family_diabetes else 0,
        'hypertensive': 1 if hypertension else 0,
        'family_hypertension': 1 if family_hypertension else 0,
        'cardiovascular': 1 if cardiovascular else 0,
        'stroke': 1 if stroke else 0
    }
    
    # Make prediction
    result = predict_risk(model, scaler, gender_encoder, inputs)
    
    # =============== DISPLAY RESULTS (Like your image) ===============
    
    st.markdown("---")
    st.markdown("### 📊 Risk Assessment Results")
    
    # Main risk display (circular gauge style)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Risk percentage display
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;'>
            <h4 style='margin: 0; opacity: 0.9;'>PROBABILITY</h4>
            <h1 style='margin: 20px 0; font-size: 4em;'>{result['risk_percentage']:.0f}%</h1>
            <div style='display: inline-block; background: rgba(255,255,255,0.2); padding: 10px 30px; border-radius: 25px; border: 2px solid white;'>
                <strong>{result['risk_level']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Confidence and Validation
    col_conf, col_val = st.columns(2)
    
    with col_conf:
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 10px;'>
            <p style='margin: 0; color: #666; font-size: 0.9em;'>CONFIDENCE</p>
            <h2 style='margin: 10px 0; color: #667eea;'>{result['confidence']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_val:
        val_color = "#28a745" if result['validation'] == "Passed" else "#ffc107"
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 10px;'>
            <p style='margin: 0; color: #666; font-size: 0.9em;'>VALIDATION</p>
            <h2 style='margin: 10px 0; color: {val_color};'>{result['validation']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 📈 Risk Factors Analysis")
    
    # Show which factors contribute most
    risk_factors = []
    
    if glucose > 100:
        risk_factors.append("⚠️ Elevated glucose level")
    if systolic_bp > 140 or diastolic_bp > 90:
        risk_factors.append("⚠️ High blood pressure")
    if bmi > 25:
        risk_factors.append("⚠️ BMI above normal range")
    if hypertension:
        risk_factors.append("⚠️ Hypertension history")
    if cardiovascular:
        risk_factors.append("⚠️ Cardiovascular disease history")
    if family_diabetes:
        risk_factors.append("⚠️ Family history of diabetes")
    
    if risk_factors:
        st.warning("**Contributing Risk Factors:**")
        for factor in risk_factors:
            st.write(factor)
    else:
        st.success("✅ No major risk factors detected")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if result['risk_level'] == "HIGH":
        st.error("""
        **High Risk Detected**
        - Consult a healthcare professional immediately
        - Regular monitoring of glucose and blood pressure
        - Lifestyle modifications recommended
        - Consider preventive medication if advised
        """)
    elif result['risk_level'] == "MODERATE":
        st.warning("""
        **Moderate Risk**
        - Schedule a check-up with your doctor
        - Monitor vitals regularly
        - Adopt healthy diet and exercise routine
        - Re-assess in 3-6 months
        """)
    else:
        st.success("""
        **Low Risk**
        - Maintain current healthy lifestyle
        - Annual health check-ups recommended
        - Stay active and eat balanced diet
        - Monitor any changes in health status
        """)
    
    st.info("**Disclaimer:** This is a screening tool for educational purposes only. Always consult healthcare professionals for medical advice.")

# =============== ABOUT ===============
with st.expander("ℹ️ About This Tool"):
    st.write("""
    **DiaPredict Tool**
    
    This application uses machine learning to assess health risks based on:
    - Physical statistics and vital signs
    - Clinical and family history
    - Advanced predictive analytics
    
    **Model Information:**
    - Algorithm: Support Vector Machine (SVM)
    - Training Data: 5,289 patient records
    - Features: 14 health indicators
    
    **Creator:** Intan Abdali && Shahadat Hossain Shahed
    
    *For educational and screening purposes only. Not a substitute for professional medical diagnosis.*
    """)


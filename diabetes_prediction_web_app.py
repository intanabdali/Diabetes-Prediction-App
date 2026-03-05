# -*- coding: utf-8 -*-
"""
DiaPredict: Advanced Diabetes Risk Assessment
Designed for: Intan Abdali & Shahadat Hossain Shahed
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# =============== Page Configuration ===============
st.set_page_config(
    page_title="DiaPredict | Precision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============== Professional Theme CSS ===============
def local_css():
    st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Sidebar Branding */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
        }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {
            color: #f1f5f9 !important;
        }

        /* Card-like containers */
        div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
            background-color: white;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }

        /* Metric Styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #2563eb;
            font-weight: 700;
        }

        /* Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            height: 3.5em;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        }

        /* Result Box Gradient */
        .result-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 40px;
            border-radius: 24px;
            color: white;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Custom Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-weight: 600;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =============== Utils & Prediction Logic ===============
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "trained_model.sav")
DEFAULT_SCALER_PATH = os.path.join(APP_DIR, "scaler.sav")
DEFAULT_GENDER_ENCODER_PATH = os.path.join(APP_DIR, "gender_encoder.sav")

@st.cache_resource
def load_model_and_encoders():
    try:
        model = pickle.load(open(DEFAULT_MODEL_PATH, "rb"))
        scaler = pickle.load(open(DEFAULT_SCALER_PATH, "rb"))
        gender_encoder = pickle.load(open(DEFAULT_GENDER_ENCODER_PATH, "rb"))
        return model, scaler, gender_encoder
    except Exception as e:
        return None, None, None

def predict_risk(model, scaler, gender_encoder, inputs_dict):
    gender_encoded = 1 if inputs_dict['gender'] == 'Male' else 0
    features = [
        inputs_dict['age'], gender_encoded, inputs_dict['pulse_rate'],
        inputs_dict['systolic_bp'], inputs_dict['diastolic_bp'],
        inputs_dict['glucose'], inputs_dict['height'],
        inputs_dict['weight'], inputs_dict['bmi'],
        inputs_dict['family_diabetes'], inputs_dict['hypertensive'],
        inputs_dict['family_hypertension'], inputs_dict['cardiovascular'],
        inputs_dict['stroke']
    ]
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    
    prediction = model.predict(x_scaled)[0]
    probability = model.predict_proba(x_scaled)[0]
    risk_percentage = probability[1] * 100
    
    if risk_percentage < 30:
        level, color = "LOW", "green"
    elif risk_percentage < 60:
        level, color = "MODERATE", "orange"
    else:
        level, color = "HIGH", "red"
        
    confidence = max(probability) * 100
    return {
        'risk_percentage': risk_percentage,
        'risk_level': level,
        'risk_color': color,
        'confidence': confidence,
        'validation': "Passed" if confidence > 70 else "Review Needed"
    }

# =============== Sidebar Dashboard ===============
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("DiaPredict AI")
    st.markdown("---")
    st.markdown("### 🏥 System Status")
    st.success("✨ Engine: SVM-RBF Active")
    st.info("📊 Database: 5.2k Records")
    st.markdown("---")
    st.caption("v2.4.0-Stable | © 2024")

# =============== Main UI ===============
model, scaler, gender_encoder = load_model_and_encoders()

st.title("🩺 DiaPredict: Precision Risk Analytics")
st.markdown("Utilize our clinical-grade AI to forecast metabolic health risks based on multi-factor patient data.")

if model is None:
    st.error("🚨 Critical Error: Model files (trained_model.sav, scaler.sav, gender_encoder.sav) not found in directory.")
    st.stop()

tab1, tab2 = st.tabs(["📝 Diagnostic Input", "ℹ️ About Engine"])

with tab1:
    st.markdown("### 👤 Patient Demographics & Body Composition")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", 18, 100, 35)
        gender = st.selectbox("Biological Gender", ["Male", "Female"])
    with col2:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 72.0)
        height_cm = st.number_input("Height (cm)", 100, 250, 170)
    with col3:
        h_m = height_cm / 100
        bmi = weight / (h_m ** 2)
        st.metric("Calculated BMI", f"{bmi:.1f}", delta="Normal" if 18.5<=bmi<=25 else "Attention")

    st.markdown("### 💓 Clinical Vital Indicators")
    col4, col5, col6 = st.columns(3)
    with col4:
        glucose = st.slider("Fasting Glucose (mg/dL)", 50, 350, 100)
    with col5:
        pulse = st.number_input("Pulse Rate (BPM)", 40, 160, 72)
    with col6:
        sbp = st.number_input("Systolic BP", 80, 200, 120)
        dbp = st.number_input("Diastolic BP", 40, 130, 80)

    st.markdown("### 📋 Medical & Family History")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        hist_hyper = st.toggle("Patient Hypertension")
        hist_cardio = st.toggle("CVD / Heart Disease")
    with h_col2:
        hist_stroke = st.toggle("Stroke History")
        fam_dia = st.toggle("Lineal Diabetes History")
    with h_col3:
        fam_hyper = st.toggle("Family Hypertension")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate Risk Assessment Report"):
        inputs = {
            'age': age, 'gender': gender, 'pulse_rate': pulse, 'systolic_bp': sbp,
            'diastolic_bp': dbp, 'glucose': glucose, 'height': h_m,
            'weight': weight, 'bmi': bmi, 'family_diabetes': 1 if fam_dia else 0,
            'hypertensive': 1 if hist_hyper else 0, 'family_hypertension': 1 if fam_hyper else 0,
            'cardiovascular': 1 if hist_cardio else 0, 'stroke': 1 if hist_stroke else 0
        }
        
        result = predict_risk(model, scaler, gender_encoder, inputs)

        st.markdown("---")
        res_left, res_right = st.columns([1, 1.2])

        with res_left:
            st.markdown(f"""
            <div class="result-card">
                <p style='text-transform: uppercase; letter-spacing: 2px; font-size: 0.9rem; opacity: 0.7;'>Probability of Diabetes</p>
                <h1 style='font-size: 5.5rem; margin: 15px 0;'>{result['risk_percentage']:.0f}%</h1>
                <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 12px; border: 1px solid white;'>
                    <span style='font-weight: bold;'>STATUS: {result['risk_level']} RISK</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.subheader("📊 Engine Insights")
            st.write(f"**Confidence Score:** `{result['confidence']:.1f}%`")
            st.write(f"**Validation Check:** `{result['validation']}`")
            
            if result['risk_level'] == "HIGH":
                st.error("⚠️ **Clinical Warning:** High probability detected. Immediate consultation with a healthcare professional and fasting A1c blood test is advised.")
            elif result['risk_level'] == "MODERATE":
                st.warning("⚖️ **Moderate Risk:** Pre-diabetic markers present. Recommend lifestyle intervention, increased physical activity, and carbohydrate management.")
            else:
                st.success("✅ **Stable Status:** Risk factors within safe range. Maintain current healthy lifestyle and conduct annual check-ups.")

with tab2:
    st.markdown("""
    ### ⚙️ Predictive Engine Methodology
    **DiaPredict** utilizes a sophisticated Machine Learning pipeline to evaluate metabolic risk.
    
    * **Algorithm:** Support Vector Machine (SVM) with RBF Kernel.
    * **Dataset:** Trained on 5,200+ clinical records.
    * **Feature Weights:** Fasting Glucose and BMI carry the highest statistical weight in our current model version.
    
    **Developed by:** Intan Abdali & Shahadat Hossain Shahed
    """)

st.caption("Disclaimer: This tool is for screening purposes and does not constitute a formal medical diagnosis.")

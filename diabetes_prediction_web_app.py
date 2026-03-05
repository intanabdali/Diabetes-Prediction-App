# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import streamlit as st

# ================= PATHS =================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "trained_model.sav")
SCALER_PATH = os.path.join(APP_DIR, "scaler.sav")
ENCODER_PATH = os.path.join(APP_DIR, "gender_encoder.sav")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DiaPredict Premium",
    page_icon="🩺",
    layout="wide"
)

# ================= UPDATED DARK-THEME UI STYLE =================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* MATCHING ATTACHED IMAGE BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #1c3b44 0%, #2c5364 50%, #203a43 100%);
        color: white;
    }

    /* Glassmorphism Card Styling */
    .premium-card {
        background: rgba(255, 255, 255, 0.07);
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        margin-bottom: 25px;
    }

    /* Subheader Styling */
    .section-header {
        color: #00d2ff;
        font-weight: 800;
        font-size: 1.3rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Input Fields Customization for Dark Theme */
    .stNumberInput div div input, .stSelectbox div div div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    label p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
    }

    /* Hero Prediction Button */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 20px !important;
        font-size: 20px !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 210, 255, 0.5);
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
    }

    /* Result Dashboard */
    .result-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        border: 2px solid #00d2ff;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
    }

    .risk-score {
        font-size: 85px;
        font-weight: 800;
        color: #00d2ff;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
        margin: 0;
    }
    
    /* Metrics for Dark Theme */
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
    }

    /* Sidebar Fix */
    section[data-testid="stSidebar"] {
        background-color: #162a33 !important;
    }

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    m = pickle.load(open(MODEL_PATH, "rb"))
    s = pickle.load(open(SCALER_PATH, "rb"))
    e = pickle.load(open(ENCODER_PATH, "rb"))
    return m, s, e

try:
    model, scaler, encoder = load_model()
except:
    st.error("⚠️ System Offline: Model synchronization failed. Check your .sav files.")
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown(f"""
        <div style='text-align:center; padding-bottom:20px;'>
            <h1 style='color:#00d2ff; font-size: 32px; margin-bottom:0;'>🩺 DiaPredict</h1>
            <p style='color:rgba(255,255,255,0.6);'>Clinical Risk AI v3.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Research Leads")
    st.success("**Intan Abdali**")
    st.success("**S.H. Shahed**")
    
    st.info("💡 **Clinical Note:** Fasting glucose data collected within the last 24 hours provides the most accurate risk forecasting.")

# ================= MAIN CONTENT =================
st.markdown("<h1 style='text-align: center; color: white; margin-bottom:0;'>Diabetes Risk Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 1.1rem; margin-bottom:40px;'>AI-Powered Metabolic Screening Interface</p>", unsafe_allow_html=True)

# --- PHYSICAL STATS ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">👤 Biometric Statistics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Biological Gender", ["Male", "Female"])
    age = st.number_input("Patient Age", 18, 100, 35)

with col2:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Height (cm)", 100, 250, 175)

with col3:
    height_m = height/100
    bmi = weight/(height_m**2)
    st.metric("Body Mass Index (BMI)", f"{bmi:.1f}")
    if 18.5 <= bmi <= 25:
        st.caption("✅ Optimal weight range detected.")
    else:
        st.caption("⚠️ Review weight management markers.")

st.markdown("</div>", unsafe_allow_html=True)

# --- VITAL SIGNS ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">💓 Vital Indicator Dashboard</div>', unsafe_allow_html=True)

vcol1, vcol2, vcol3 = st.columns(3)

with vcol1:
    glucose = st.number_input("Fasting Glucose (mg/dL) 🩸", 50.0, 350.0, 95.0)

with vcol2:
    sys_bp = st.number_input("Systolic BP (mmHg) 🩺", 80, 220, 120)
    dia_bp = st.number_input("Diastolic BP (mmHg) 🩺", 40, 130, 80)

with vcol3:
    pulse = st.number_input("Resting Pulse (BPM) 💓", 40, 160, 72)

st.markdown("</div>", unsafe_allow_html=True)

# --- MEDICAL HISTORY ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Clinical History Review</div>', unsafe_allow_html=True)

hcol1, hcol2, hcol3 = st.columns(3)
with hcol1:
    hypertension = st.toggle("History of Hypertension")
    cardiovascular = st.toggle("CVD History")
with hcol2:
    stroke = st.toggle("History of Stroke")
    family_diabetes = st.toggle("Genetic Diabetes History")
with hcol3:
    family_hypertension = st.toggle("Genetic Hypertension")

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION ENGINE =================
if st.button("🔍 RUN ANALYTICAL SCREENING"):
    with st.spinner("Processing Multi-Factor Data..."):
        # Encoding and Logic
        gender_encoded = 1 if gender=="Male" else 0
        features = [age, gender_encoded, pulse, sys_bp, dia_bp, glucose, height_m, weight, bmi, 
                    1 if family_diabetes else 0, 1 if hypertension else 0, 
                    1 if family_hypertension else 0, 1 if cardiovascular else 0, 1 if stroke else 0]
        
        x = np.array(features).reshape(1,-1)
        x_scaled = scaler.transform(x)
        prob = model.predict_proba(x_scaled)[0]
        risk = prob[1]*100

        # Logic for Results
        if risk < 25:
            level, color, icon = "LOW", "#00ffcc", "✅"
        elif risk < 60:
            level, color, icon = "MODERATE", "#ffcc00", "⚠️"
        else:
            level, color, icon = "CRITICAL", "#ff4b4b", "🚨"

        # --- PREMIUM RESULT DISPLAY ---
        st.markdown(f"""
            <div class="result-container">
                <p style="color: rgba(255,255,255,0.7); font-weight: 600; text-transform: uppercase; letter-spacing: 2px;">Predicted Risk Probability</p>
                <h1 class="risk-score">{risk:.1f}%</h1>
                <h2 style="color: {color}; font-weight: 800; margin-top: -10px;">{icon} {level} RISK DETECTED</h2>
                <p style="color: rgba(255,255,255,0.8); max-width: 600px; margin: 0 auto; line-height: 1.6;">
                    The AI engine has analyzed your biometric profile against clinical datasets. 
                    A score of {risk:.1f}% indicates a {level.lower()} probability of diabetic markers based on current vital inputs.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # --- DYNAMIC ADVISORY ---
        st.markdown("<br>", unsafe_allow_html=True)
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown(f"### 🎯 Key Observations")
            if glucose > 125: st.error("Fasting Glucose: Critical Elevation")
            if bmi > 30: st.warning("Metabolic: Obesity Marker Present")
            if family_diabetes: st.info("Genetic: High Predisposition")
            if not any([glucose > 125, bmi > 30, family_diabetes]): st.success("Safe Zone: Primary biomarkers are within range.")

        with col_adv2:
            st.markdown("### 👨‍⚕️ Suggested Protocol")
            if level == "CRITICAL":
                st.markdown("1. Immediate Physician Consultation\n2. Lab-Grade HbA1c Blood Panel\n3. Strict Carbohydrate Restriction")
            elif level == "MODERATE":
                st.markdown("1. Increased Physical Activity (150m+/wk)\n2. Quarterly Glucose Self-Monitoring\n3. Dietary Sugar Reduction")
            else:
                st.markdown("1. Maintain Current Wellness Routine\n2. Stay Hydrated & Physically Active\n3. Regular Annual Health Screening")

# ================= FOOTER =================
st.markdown("""
    <div style="margin-top: 80px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center; color: rgba(255,255,255,0.4); font-size: 0.9rem;">
        DiaPredict Pro • AI-Powered Health Screening • HIPAA Compliant Layout<br>
        Developed by Intan Abdali & Shahadat Hossain Shahed
    </div>
""", unsafe_allow_html=True)

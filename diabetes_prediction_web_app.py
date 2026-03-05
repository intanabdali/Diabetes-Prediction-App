# -*- coding: utf-8 -*-
"""
Professional Diabetes Risk Prediction App
Improved UI Version
"""

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
    page_title="DiaPredict",
    page_icon="🩺",
    layout="centered"
)

# ================= UI STYLE =================

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

h1{
text-align:center;
color:white;
font-weight:700;
}

.card{
background: rgba(255,255,255,0.08);
padding:25px;
border-radius:15px;
backdrop-filter: blur(10px);
box-shadow:0 8px 20px rgba(0,0,0,0.3);
margin-bottom:20px;
color:white;
}

.stButton>button{
background: linear-gradient(135deg,#667eea,#764ba2);
color:white;
border:none;
padding:12px;
font-size:18px;
border-radius:10px;
font-weight:600;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================

@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    encoder = pickle.load(open(ENCODER_PATH, "rb"))
    return model, scaler, encoder

try:
    model, scaler, encoder = load_model()
except:
    st.error("Model files not found")
    st.stop()

# ================= SIDEBAR =================

with st.sidebar:

    st.title("🩺 DiaPredict")

    st.markdown("""
AI Powered Diabetes Risk Screening

Developer  
**Intan Abdali**  
**Shahadat Hossain Shahed**

Machine learning based prediction tool.
""")

    st.info("Educational health screening tool")

# ================= TITLE =================

st.title("🩺 Diabetes Risk Prediction")

# ================= PHYSICAL STATS =================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Physical Statistics")

col1,col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    weight = st.number_input(
        "Weight (kg)",
        30.0,200.0,70.0
    )

with col2:

    height = st.number_input(
        "Height (cm)",
        100,250,175
    )

height_m = height/100
bmi = weight/(height_m**2)

st.metric("Body Mass Index", f"{bmi:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ================= VITAL SIGNS =================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💓 Vital Indicators")

col3,col4 = st.columns(2)

with col3:

    pulse = st.number_input(
        "Pulse Rate (BPM)",
        40,150,72
    )

    sys_bp = st.number_input(
        "Systolic BP",
        80,200,120
    )

with col4:

    glucose = st.number_input(
        "Glucose Level",
        50.0,300.0,90.0
    )

    dia_bp = st.number_input(
        "Diastolic BP",
        50,130,80
    )

age = st.number_input(
    "Age",
    18,100,35
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= MEDICAL HISTORY =================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Medical History")

col5,col6 = st.columns(2)

with col5:

    hypertension = st.checkbox("Hypertension")

    cardiovascular = st.checkbox("Cardiovascular Disease")

with col6:

    stroke = st.checkbox("Stroke")

    family_diabetes = st.checkbox("Family Diabetes")

family_hypertension = st.checkbox("Family Hypertension")

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================

if st.button("🔍 Analyze Risk Factor", use_container_width=True):

    with st.spinner("Analyzing health data..."):

        gender_encoded = 1 if gender=="Male" else 0

        features = [
            age,
            gender_encoded,
            pulse,
            sys_bp,
            dia_bp,
            glucose,
            height_m,
            weight,
            bmi,
            1 if family_diabetes else 0,
            1 if hypertension else 0,
            1 if family_hypertension else 0,
            1 if cardiovascular else 0,
            1 if stroke else 0
        ]

        x = np.array(features).reshape(1,-1)

        x_scaled = scaler.transform(x)

        pred = model.predict(x_scaled)[0]

        prob = model.predict_proba(x_scaled)[0]

        risk = prob[1]*100

# ================= RESULT =================

    st.markdown("---")
    st.subheader("📊 Risk Assessment")

    if risk < 30:
        level="LOW"
        color="green"
    elif risk <60:
        level="MODERATE"
        color="orange"
    else:
        level="HIGH"
        color="red"

    st.markdown(f"""
<div style="text-align:center;
padding:40px;
border-radius:20px;
background:linear-gradient(135deg,#667eea,#764ba2);
color:white;
box-shadow:0 10px 30px rgba(0,0,0,0.3);">

<h3>Prediction Result</h3>

<h1 style="font-size:60px">{risk:.0f}%</h1>

<h2>{level} RISK</h2>

</div>
""", unsafe_allow_html=True)

    st.progress(int(risk))

# ================= RISK FACTORS =================

    st.subheader("⚠️ Risk Factors")

    factors=[]

    if glucose>100:
        factors.append("High glucose")

    if sys_bp>140 or dia_bp>90:
        factors.append("High blood pressure")

    if bmi>25:
        factors.append("High BMI")

    if family_diabetes:
        factors.append("Family diabetes history")

    if factors:

        for f in factors:
            st.warning(f)

    else:

        st.success("No major risk factors detected")

# ================= RECOMMENDATION =================

    st.subheader("💡 Recommendation")

    if level=="HIGH":

        st.error("""
Consult doctor immediately

Monitor glucose regularly

Adopt strict healthy lifestyle
""")

    elif level=="MODERATE":

        st.warning("""
Schedule medical checkup

Improve diet and exercise
""")

    else:

        st.success("""
Maintain healthy lifestyle
""")

# ================= FOOTER =================

st.markdown("""
<hr>
<center style="color:white">
DiaPredict • AI Health Screening Tool
<br>
Developed by Intan Abdali
</center>
""", unsafe_allow_html=True)

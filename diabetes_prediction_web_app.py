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

# ================= PREMIUM UI STYLE =================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #F0F4F8;
    }

    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #E6F0F3 100%);
    }

    /* Professional Card Styling */
    .premium-card {
        background: white;
        padding: 30px;
        border-radius: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 25px rgba(0, 180, 216, 0.05);
        margin-bottom: 25px;
    }

    /* Subheader Styling */
    .section-header {
        color: #1A365D;
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ================================== */
    /* TOGGLE BUTTON VISIBILITY           */
    /* ================================== */
    
    .stCheckbox {
        background: #F8FAFC;
        padding: 12px 16px;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        border-color: #00B4D8;
        background: #F0F9FF;
    }
    
    .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        font-size: 15px !important;
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"] {
        width: 50px !important;
        height: 28px !important;
        border-radius: 14px !important;
        background-color: #CBD5E1 !important;
        border: 2px solid #94A3B8 !important;
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"][aria-checked="true"] {
        background-color: #00B4D8 !important;
        border-color: #0077B6 !important;
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"] > div {
        width: 22px !important;
        height: 22px !important;
        background-color: white !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }

    /* Input Fields */
    .stNumberInput div div input, .stSelectbox div div div {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #1E293B !important;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #0077B6 0%, #00B4D8 100%);
        color: white;
        border: none;
        padding: 20px !important;
        font-size: 20px !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(0, 180, 216, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(0, 180, 216, 0.4);
        background: linear-gradient(90deg, #00B4D8 0%, #0077B6 100%);
    }

    /* Result Dashboard */
    .result-container {
        background: #FFFFFF;
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        border: 2px solid #00B4D8;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    }

    .risk-score {
        font-size: 80px;
        font-weight: 800;
        background: -webkit-linear-gradient(#023E8A, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    label p {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* ===================================== */
    /* FIX: CLINICAL INSIGHTS - DARK TEXT!   */
    /* ===================================== */
    
    /* Override ALL Streamlit alert text colors to be DARK */
    
    /* Success alerts - DARK GREEN TEXT */
    .stSuccess {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%) !important;
        border-left: 4px solid #10B981 !important;
        padding: 16px 20px !important;
        border-radius: 12px !important;
    }
    
    .stSuccess > div {
        color: #064E3B !important;  /* Very dark green */
    }
    
    .stSuccess [data-testid="stMarkdownContainer"] {
        color: #064E3B !important;
    }
    
    .stSuccess [data-testid="stMarkdownContainer"] p,
    .stSuccess [data-testid="stMarkdownContainer"] strong,
    .stSuccess [data-testid="stMarkdownContainer"] b {
        color: #064E3B !important;  /* Force dark green */
        font-weight: 600 !important;
    }
    
    /* Error alerts - DARK RED TEXT */
    .stAlert[kind="error"],
    div[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%) !important;
        border-left: 4px solid #DC2626 !important;
    }
    
    .stAlert[kind="error"] > div,
    .stAlert[kind="error"] [data-testid="stMarkdownContainer"],
    .stAlert[kind="error"] [data-testid="stMarkdownContainer"] p,
    .stAlert[kind="error"] [data-testid="stMarkdownContainer"] strong,
    .stAlert[kind="error"] [data-testid="stMarkdownContainer"] b {
        color: #7F1D1D !important;  /* Very dark red */
        font-weight: 600 !important;
    }
    
    /* Warning alerts - DARK ORANGE TEXT */
    .stAlert[kind="warning"],
    div[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%) !important;
        border-left: 4px solid #F59E0B !important;
    }
    
    .stAlert[kind="warning"] > div,
    .stAlert[kind="warning"] [data-testid="stMarkdownContainer"],
    .stAlert[kind="warning"] [data-testid="stMarkdownContainer"] p,
    .stAlert[kind="warning"] [data-testid="stMarkdownContainer"] strong,
    .stAlert[kind="warning"] [data-testid="stMarkdownContainer"] b {
        color: #78350F !important;  /* Very dark brown */
        font-weight: 600 !important;
    }
    
    /* Info alerts - DARK BLUE TEXT */
    .stAlert[kind="info"],
    div[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%) !important;
        border-left: 4px solid #3B82F6 !important;
    }
    
    .stAlert[kind="info"] > div,
    .stAlert[kind="info"] [data-testid="stMarkdownContainer"],
    .stAlert[kind="info"] [data-testid="stMarkdownContainer"] p,
    .stAlert[kind="info"] [data-testid="stMarkdownContainer"] strong,
    .stAlert[kind="info"] [data-testid="stMarkdownContainer"] b {
        color: #1E3A8A !important;  /* Very dark blue */
        font-weight: 600 !important;
    }
    
    /* Caption text - INHERIT dark color from parent */
    .stCaptionContainer,
    div[data-testid="stCaptionContainer"],
    .caption,
    [class*="caption"] {
        color: inherit !important;
        opacity: 1 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
        margin-top: 8px !important;
    }
    
    /* Force all text in alerts to be visible */
    .stSuccess p,
    .stAlert p,
    div[data-baseweb="notification"] p {
        color: inherit !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* ========================================= */
    /* PROTOCOL BOXES                            */
    /* ========================================= */
    
    .protocol-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        font-size: 15px;
        line-height: 1.7;
    }
    
    .protocol-critical {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 5px solid #DC2626;
        color: #7F1D1D;
    }
    
    .protocol-critical b {
        color: #991B1B !important;
        font-weight: 700;
    }
    
    .protocol-moderate {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 5px solid #F59E0B;
        color: #78350F;
    }
    
    .protocol-moderate b {
        color: #92400E !important;
        font-weight: 700;
    }
    
    .protocol-low {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 5px solid #10B981;
        color: #064E3B;
    }
    
    .protocol-low b {
        color: #065F46 !important;
        font-weight: 700;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .protocol-box {
            padding: 18px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .risk-score {
            font-size: 60px;
        }
        
        .section-header {
            font-size: 1.1rem;
        }
        
        .stCheckbox {
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        
        .stCheckbox > label {
            font-size: 16px !important;
        }
        
        .stCheckbox > label > div[data-baseweb="checkbox"] {
            width: 55px !important;
            height: 32px !important;
        }
        
        .stCheckbox > label > div[data-baseweb="checkbox"] > div {
            width: 26px !important;
            height: 26px !important;
        }
        
        .stAlert,
        .stSuccess {
            font-size: 14px !important;
            padding: 14px 16px !important;
        }
        
        .stCaptionContainer,
        div[data-testid="stCaptionContainer"] {
            font-size: 13px !important;
        }
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
    st.error("⚠️ System Offline: Model synchronization failed.")
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding-bottom:20px;'>
            <h1 style='color:#00B4D8; font-size: 32px;'>🩺 DiaPredict</h1>
            <p style='color:#64748B;'>Clinical Risk Intelligence v3.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Research Leads")
    st.success("**Intan Abdali**")
    st.success("**S.H. Shahed**")
    
    st.info("💡 **Clinical Note:** Predictive models are most accurate when using fasting glucose data collected within the last 24 hours.")

# ================= MAIN CONTENT =================
st.markdown("<h1 style='text-align: center; color: #1A365D;'>Diabetes Risk Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.1rem;'>Input patient biometrics to generate a high-precision probability report.</p>", unsafe_allow_html=True)

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
    bmi_color = "normal" if 18.5 <= bmi <= 25 else "off"
    st.metric("Body Mass Index (BMI)", f"{bmi:.1f}", delta="Optimal Range" if bmi_color=="normal" else "Review Needed")

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
    hypertension = st.checkbox("History of Hypertension")
    cardiovascular = st.checkbox("CVD History")
with hcol2:
    stroke = st.checkbox("History of Stroke")
    family_diabetes = st.checkbox("Genetic Diabetes History")
with hcol3:
    family_hypertension = st.checkbox("Genetic Hypertension")

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION ENGINE =================
if st.button("🚀 GENERATE PREDICTIVE REPORT"):
    with st.spinner("Processing Clinical Data..."):
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
            level, color, icon = "LOW", "#06D6A0", "✅"
        elif risk < 60:
            level, color, icon = "MODERATE", "#FFD166", "⚠️"
        else:
            level, color, icon = "CRITICAL", "#EF476F", "🚨"

        # --- RESULT DISPLAY ---
        st.markdown(f"""
            <div class="result-container">
                <p style="color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 2px;">Probability Assessment</p>
                <h1 class="risk-score">{risk:.1f}%</h1>
                <h2 style="color: {color}; font-weight: 800; margin-top: -10px;">{icon} {level} RISK LEVEL</h2>
                <p style="color: #475569; max-width: 500px; margin: 0 auto;">Based on the current SVM algorithm parameters, the patient shows a {risk:.1f}% statistical likelihood of diabetic markers.</p>
            </div>
        """, unsafe_allow_html=True)

        # ================= ADVISORY - WITH DARK TEXT! =================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 0.5px solid #CBD5E1; margin: 30px 0;'>", unsafe_allow_html=True)
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown("### 🎯 Clinical Insights")
            
            # --- Glucose Observation ---
            if glucose > 125:
                st.error(f"**Critical Glucose:** {glucose} mg/dL")
                st.caption("🚨 **Hyperglycemia detected.** Levels above 125 mg/dL in a fasting state are primary indicators for Type 2 Diabetes.")
            elif glucose > 100:
                st.warning(f"**Pre-diabetic Range:** {glucose} mg/dL")
                st.caption("⚠️ Your glucose is in the 'Impaired Fasting Glucose' range. **Early intervention can reverse this trend.**")
            else:
                st.success("**Healthy Glucose Metabolism**")
                st.caption("✅ **Your fasting blood sugar is within the optimal clinical range (70-100 mg/dL).**")

            # --- BMI Observation ---
            if bmi > 30:
                st.warning(f"**Weight Factor: Obesity (BMI {bmi:.1f})**")
                st.caption("⚖️ **High BMI increases insulin resistance,** making it harder for your body to regulate blood sugar effectively.")
            elif bmi > 25:
                st.info(f"**Weight Factor: Overweight (BMI {bmi:.1f})**")
                st.caption("🏃 Slight elevation in BMI detected. **Modest weight loss (5-7%) can significantly reduce diabetes risk.**")

            # --- Genetic Observation ---
            if family_diabetes:
                st.info("**Genetic Predisposition: Present**")
                st.caption("🧬 **A lineal history of diabetes increases your baseline risk.** Environmental factors (diet/exercise) are now your primary defense.")

        with col_adv2:
            st.markdown("### 👨‍⚕️ Clinical Protocol")
            
            if level == "CRITICAL":
                st.markdown("""
                <div class="protocol-box protocol-critical">
                    <b>⚠️ IMMEDIATE ACTIONS:</b><br><br>
                    <b>1. Physician Consultation:</b> Present this report to a GP or Endocrinologist within 48 hours.<br><br>
                    <b>2. Diagnostic Testing:</b> Request a Laboratory HbA1c test and an Oral Glucose Tolerance Test (OGTT).<br><br>
                    <b>3. Nutritional Crisis Management:</b> Adopt a 'Low Glycemic Index' diet immediately; eliminate all refined sugars.
                </div>
                """, unsafe_allow_html=True)
                
            elif level == "MODERATE":
                st.markdown("""
                <div class="protocol-box protocol-moderate">
                    <b>📋 PREVENTATIVE PROTOCOL:</b><br><br>
                    <b>1. Metabolic Activation:</b> Target 150 minutes of moderate aerobic activity (brisk walking) per week.<br><br>
                    <b>2. Glucose Tracking:</b> Begin a 'Sugar Journal' to identify which specific meals cause energy crashes or spikes.<br><br>
                    <b>3. Fiber Integration:</b> Increase daily fiber to 25g+ to slow down glucose absorption in the bloodstream.
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="protocol-box protocol-low">
                    <b>✅ MAINTENANCE STRATEGY:</b><br><br>
                    <b>1. Annual Bio-Screening:</b> Continue yearly fasting glucose checks to maintain this healthy baseline.<br><br>
                    <b>2. Hydration Focus:</b> Maintain optimal kidney function by consuming 2-3 liters of water daily.<br><br>
                    <b>3. Stress Management:</b> High cortisol can spike glucose; maintain consistent sleep cycles (7-8 hours).
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
    <div style="margin-top: 100px; padding: 20px; border-top: 1px solid #E2E8F0; text-align: center; color: #94A3B8;">
        DiaPredict Pro • Secure HIPAA-Aligned Interface<br>
        Developed with ❤️ by the Clinical AI Team
    </div>
""", unsafe_allow_html=True)

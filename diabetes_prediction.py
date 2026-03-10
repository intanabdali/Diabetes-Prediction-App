import streamlit as st
import numpy as np
import pickle
import requests

# 1. LOAD CREDENTIALS FROM SECRETS
API_KEY = st.secrets["FIREBASE_API_KEY"]

# --- FIREBASE HELPER FUNCTIONS ---
def login_with_email_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

def signup_with_email_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

# --- LOGIN UI ---
if 'user' not in st.session_state:
    st.title("🔐 Secure Diabetes Predictor")
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            res = login_with_email_password(email, password)
            if "idToken" in res:
                st.session_state.user = res['email']
                st.rerun()
            else:
                st.error("Invalid Login. Please check your email/password.")
    
    with tab2:
        new_email = st.text_input("New Email")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            res = signup_with_email_password(new_email, new_pass)
            if "idToken" in res:
                st.success("Account created! You can now log in.")
            else:
                st.error(f"Error: {res.get('error', {}).get('message')}")
    st.stop() # Stops the rest of the code until logged in

# --- LOGGED IN CONTENT ---
st.sidebar.write(f"Logged in as: {st.session_state.user}")
if st.sidebar.button("Logout"):
    del st.session_state.user
    st.rerun()

# 2. LOAD YOUR SAVED MODEL AND SCALER
# Note: You need to save your 'scaler' object using pickle too, just like the model!
try:
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    # Make sure to upload 'scaler.sav' to your GitHub repo as well
    scaler = pickle.load(open('scaler.sav', 'rb')) 
except Exception as e:
    st.error("Model files not found. Please upload 'trained_model.sav' and 'scaler.sav' to GitHub.")
    st.stop()

st.title('Diabetes Prediction App')

# 3. GET INPUT FROM USER VIA UI
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0)

with col2:
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI value', format="%.1f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', format="%.3f")
    Age = st.number_input('Age of the Person', min_value=0, step=1)

# Code for Prediction
diagnosis = ''

if st.button('Test Result'):
    # Prepare input data
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
    
    # Standardize the input
    std_data = scaler.transform(input_data_as_numpy_array)
    
    # Predict
    prediction = loaded_model.predict(std_data)
    
    if (prediction[0] == 1):
        diagnosis = 'The person is diabetic'
        st.error(diagnosis)
    else:
        diagnosis = 'The person is not diabetic'
        st.success(diagnosis)

# auth_ui.py
# Authentication UI Components

import streamlit as st
from firebase_auth import FirebaseAuth
from datetime import datetime

def show_login_page(firebase_auth):
    """Display login/signup page"""
    
    # Header
    st.markdown("""
        <div style='text-align:center; padding:40px 0 30px 0;'>
            <h1 style='color:#00B4D8; font-size:3rem; margin-bottom:10px;'>🩺 DiaPredict</h1>
            <p style='color:#64748B; font-size:1.2rem;'>Clinical Risk Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Welcome Back")
            email = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔒 Password", type="password", placeholder="••••••••")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Login", use_container_width=True)
            with col2:
                forgot = st.form_submit_button("Forgot Password?", use_container_width=True)
            
            if submit:
                if email and password:
                    st.error("❌ Invalid email or password")
                else:
                    st.warning("⚠️ Please enter both email and password")
            
            if forgot:
                pass  # Static - no action
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("signup_form", clear_on_submit=False):
            st.subheader("Create Account")
            name = st.text_input("👤 Full Name", placeholder="John Doe")
            email = st.text_input("📧 Email", placeholder="your@email.com", key="signup_email")
            password = st.text_input("🔒 Password", type="password", placeholder="••••••••", 
                                    help="Minimum 6 characters", key="signup_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", 
                                            placeholder="••••••••", key="confirm_password")
            
            agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                pass  # Static - no action

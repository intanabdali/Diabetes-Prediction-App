# auth_ui.py
# Authentication UI Components
# Copy this entire file into GitHub

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
                    with st.spinner("Logging in..."):
                        result = firebase_auth.sign_in(email, password)
                        
                        if result['success']:
                            st.session_state.user = {
                                'email': result['email'],
                                'user_id': result['user_id'],
                                'display_name': result.get('display_name', '')
                            }
                            st.session_state.token = result['token']
                            st.session_state.refresh_token = result['refresh_token']
                            st.session_state.login_time = datetime.now()
                            
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")
                else:
                    st.warning("⚠️ Please enter both email and password")
            
            if forgot:
                if email:
                    result = firebase_auth.send_password_reset_email(email)
                    if result['success']:
                        st.success("✅ Password reset email sent! Check your inbox.")
                    else:
                        st.error(f"❌ {result['error']}")
                else:
                    st.warning("⚠️ Please enter your email address")
    
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
                if not agree:
                    st.warning("⚠️ Please agree to the Terms of Service")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif email and password and name:
                    with st.spinner("Creating account..."):
                        result = firebase_auth.sign_up(email, password, name)
                        
                        if result['success']:
                            st.success("✅ Account created successfully! Please login.")
                            st.balloons()
                        else:
                            st.error(f"❌ {result['error']}")
                else:
                    st.warning("⚠️ Please fill in all fields")

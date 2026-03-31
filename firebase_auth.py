# firebase_auth.py
# Firebase Authentication Module for DiaPredict

import streamlit as st
import requests
from datetime import datetime, timedelta

class FirebaseAuth:
    def __init__(self, api_key, auth_domain, project_id):
        self.api_key = api_key
        self.auth_domain = auth_domain
        self.project_id = project_id
        self.rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts"
        
    def sign_up(self, email, password, display_name=None):
        """Create new user account"""
        try:
            url = f"{self.rest_api_url}:signUp?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                if display_name:
                    self.update_profile(data['idToken'], display_name)
                
                return {
                    'success': True,
                    'user_id': data['localId'],
                    'email': data['email'],
                    'token': data['idToken'],
                    'refresh_token': data['refreshToken']
                }
            else:
                error_message = data.get('error', {}).get('message', 'Unknown error')
                return {'success': False, 'error': self._format_error(error_message)}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_in(self, email, password):
        """Sign in existing user"""
        try:
            url = f"{self.rest_api_url}:signInWithPassword?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'user_id': data['localId'],
                    'email': data['email'],
                    'token': data['idToken'],
                    'refresh_token': data['refreshToken'],
                    'display_name': data.get('displayName', '')
                }
            else:
                error_message = data.get('error', {}).get('message', 'Unknown error')
                return {'success': False, 'error': self._format_error(error_message)}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_profile(self, id_token, display_name=None):
        """Update user profile"""
        try:
            url = f"{self.rest_api_url}:update?key={self.api_key}"
            payload = {"idToken": id_token}
            
            if display_name:
                payload["displayName"] = display_name
            
            response = requests.post(url, json=payload)
            return response.status_code == 200
            
        except:
            return False
    
    def send_password_reset_email(self, email):
        """Send password reset email"""
        try:
            url = f"{self.rest_api_url}:sendOobCode?key={self.api_key}"
            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return {'success': True, 'message': 'Password reset email sent'}
            else:
                data = response.json()
                error_message = data.get('error', {}).get('message', 'Unknown error')
                return {'success': False, 'error': self._format_error(error_message)}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def refresh_token(self, refresh_token):
        """Refresh authentication token"""
        try:
            url = f"https://securetoken.googleapis.com/v1/token?key={self.api_key}"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'token': data['id_token'],
                    'refresh_token': data['refresh_token']
                }
            else:
                return {'success': False, 'error': 'Token refresh failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _format_error(self, error_message):
        """Format Firebase error messages"""
        error_map = {
            'EMAIL_EXISTS': 'This email is already registered',
            'OPERATION_NOT_ALLOWED': 'Password sign-in is disabled',
            'TOO_MANY_ATTEMPTS_TRY_LATER': 'Too many attempts. Try again later',
            'EMAIL_NOT_FOUND': 'Email not found. Please sign up first',
            'INVALID_PASSWORD': 'Incorrect password',
            'USER_DISABLED': 'This account has been disabled',
            'INVALID_EMAIL': 'Invalid email address',
            'WEAK_PASSWORD': 'Password must be at least 6 characters',
            'INVALID_LOGIN_CREDENTIALS': 'Invalid email or password'
        }
        
        return error_map.get(error_message, error_message)


def init_session_state():
    """Initialize session state"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None


def is_logged_in():
    """Always returns False - login disabled"""
    return False


def logout():
    """Log out user"""
    st.session_state.user = None
    st.session_state.token = None
    st.session_state.refresh_token = None
    st.session_state.login_time = None


def get_current_user():
    """Get current user info"""
    return st.session_state.user

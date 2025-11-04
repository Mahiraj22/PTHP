"""
Authentication Module for Sentiment Analysis Pipeline
Premium Modern UI Design - Fixed Layout Version
"""

import streamlit as st
import sqlite3
import hashlib
import time
import base64
import binascii
import hmac
import os
import re

# ==================== CONFIGURATION ====================

SECRET_KEY = "mahiyaraj@04"

# ==================== PASSWORD HASHING ====================

def hash_password(password):
    """Hash password using SHA256 with salt"""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    """Verify a stored password against provided password"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password

# ==================== TOKEN MANAGEMENT ====================

def create_token(username):
    """Create a simple JWT-like token"""
    timestamp = str(time.time())
    payload = f"{username}:{timestamp}"
    signature = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = base64.b64encode(f"{payload}:{signature}".encode()).decode()
    return token

def verify_token(token):
    """Verify JWT-like token"""
    try:
        decoded = base64.b64decode(token.encode()).decode()
        payload, signature = decoded.rsplit(':', 1)
        username, timestamp = payload.split(':', 1)
        expected_signature = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        
        if signature == expected_signature:
            if time.time() - float(timestamp) < 86400:
                return username
    except:
        pass
    return None

# ==================== DATABASE OPERATIONS ====================

def init_auth_db():
    """Initialize authentication database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, email, password):
    """Register a new user"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        hashed_pw = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_pw)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Login user and return token"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and verify_password(result[0], password):
        token = create_token(username)
        return True, token, "Welcome back!"
    return False, None, "Invalid credentials. Please try again."

# ==================== SESSION MANAGEMENT ====================

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'login_error' not in st.session_state:
        st.session_state.login_error = None
    if 'register_error' not in st.session_state:
        st.session_state.register_error = None
    if 'register_success' not in st.session_state:
        st.session_state.register_success = None

def logout():
    """Logout user and clear session"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.token = None

def check_authentication():
    """Check if user is authenticated with valid token"""
    if st.session_state.authenticated and st.session_state.token:
        username = verify_token(st.session_state.token)
        if username:
            return True
        else:
            logout()
            return False
    return False

# ==================== PREMIUM CSS STYLING ====================

def load_auth_css():
    """Load premium modern CSS for authentication UI"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    
    /* Container setup */
    .block-container {
        padding: 4rem 2rem !important;
        max-width: 600px !important;
        margin: 0 auto !important;
        width: 100% !important;
    }
    
    /* Card */
    .auth-card {
        background: transparent;
        border-radius: 0;
        padding: 0;
        border: none;
        box-shadow: none;
        overflow: visible;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Title */
    .form-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        word-wrap: break-word;
    }
    
    .form-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 2.5rem;
        text-align: center;
        line-height: 1.6;
    }
    
    /* Input fields - FIXED FOR CONSISTENT SIZING */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input[type="text"],
    .stTextInput > div > div > input[type="password"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 14px !important;
        height: 58px !important;
        min-height: 58px !important;
        max-height: 58px !important;
        padding: 0 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-sizing: border-box !important;
        width: 100% !important;
        text-align: center !important;
        display: block !important;
        line-height: 58px !important;
        margin: 0 !important;
        vertical-align: middle !important;
    }
    
    .stTextInput > div > div {
        width: 100% !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        overflow: hidden !important;
        min-height: 58px !important;
        max-height: 58px !important;
        height: 58px !important;
        position: relative !important;
    }
    
    .stTextInput > div {
        width: 100% !important;
        border: none !important;
        background: transparent !important;
        overflow: hidden !important;
        padding: 0 !important;
        min-height: 58px !important;
        max-height: 58px !important;
        height: 58px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextInput > div > div > input[type="password"]:focus {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1.5px solid rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
        outline: none !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextInput > div > div > input[type="password"]::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
        font-weight: 400 !important;
        text-align: center !important;
    }
    
    /* Hide the "Press Enter to submit" text */
    .stTextInput > div > div > div,
    .stTextInput div[data-testid="InputInstructions"],
    .stTextInput [data-testid="InputInstructions"],
    .stTextInput span[data-testid="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* Additional safeguard for any text elements within input containers */
    .stTextInput > div > div > span,
    .stTextInput > div > div > div > span {
        display: none !important;
        visibility: hidden !important;
    }
    
    .stTextInput > label {
        display: none !important;
    }
    
    .stTextInput {
        margin-bottom: 1.25rem !important;
        width: 100% !important;
        overflow: hidden !important;
        padding: 0 !important;
        height: 58px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        height: 58px !important;
        width: 100% !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        font-family: 'Inter', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        margin-top: 0.5rem !important;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary button */
    button[key="toggle_to_register"],
    button[key="toggle_to_login"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        box-shadow: none !important;
        margin-top: 1rem !important;
        text-transform: none !important;
    }
    
    button[key="toggle_to_register"]:hover,
    button[key="toggle_to_login"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.25) !important;
        color: white !important;
    }
    
    /* Form container */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
    
    /* Divider */
    .auth-divider {
        display: flex;
        align-items: center;
        margin: 2rem 0 1.5rem 0;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
    }
    
    .auth-divider::before,
    .auth-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255, 255, 255, 0.15);
    }
    
    .auth-divider::before {
        margin-right: 1rem;
    }
    
    .auth-divider::after {
        margin-left: 1rem;
    }
    
    /* Social buttons */
    .social-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.875rem;
        margin-bottom: 1.5rem;
        width: 100%;
    }
    
    .social-btn {
        background: rgba(255, 255, 255, 0.08);
        border: 1.5px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .social-btn:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }
    
    .social-btn i {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Messages */
    .success-message {
        background: rgba(74, 222, 128, 0.15);
        border: 1.5px solid rgba(74, 222, 128, 0.4);
        color: #4ade80;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 0.925rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        box-sizing: border-box;
    }
    
    .error-message {
        background: rgba(248, 113, 113, 0.15);
        border: 1.5px solid rgba(248, 113, 113, 0.4);
        color: #f87171;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 0.925rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Toggle text */
    .toggle-text {
        text-align: center;
        margin-top: 1.5rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.925rem;
    }
    
    .toggle-link {
        color: #818cf8;
        font-weight: 600;
        cursor: pointer;
        transition: color 0.2s ease;
    }
    
    .toggle-link:hover {
        color: #a5b4fc;
        text-decoration: underline;
    }
    
    /* Feature list */
    .feature-list {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        width: 100%;
        box-sizing: border-box;
    }
    
    .feature-item {
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.75rem;
        padding-left: 1.75rem;
        position: relative;
        font-size: 0.925rem;
        line-height: 1.6;
    }
    
    .feature-item:last-child {
        margin-bottom: 0;
    }
    
    .feature-item::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #4ade80;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        
        .auth-card {
            padding: 2rem 1.5rem;
        }
        
        .form-title {
            font-size: 2rem;
        }
        
        .social-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

# ==================== UI COMPONENTS ====================

def show_features():
    """Display feature list"""
    st.markdown("""
    <div class="feature-list">
        <div class="feature-item">Advanced AI-powered sentiment analysis</div>
        <div class="feature-item">Real-time data processing</div>
        <div class="feature-item">Comprehensive comparison reports</div>
        <div class="feature-item">Secure and encrypted data storage</div>
        <div class="feature-item">Export and email functionality</div>
    </div>
    """, unsafe_allow_html=True)

def show_social_buttons():
    """Display social login buttons"""
    st.markdown("""
    <div class="auth-divider">Continue with social</div>
    <div class="social-grid">
        <div class="social-btn" title="Continue with Google">
            <i class="fab fa-google"></i>
        </div>
        <div class="social-btn" title="Continue with Facebook">
            <i class="fab fa-facebook-f"></i>
        </div>
        <div class="social-btn" title="Continue with Twitter">
            <i class="fab fa-twitter"></i>
        </div>
        <div class="social-btn" title="Continue with LinkedIn">
            <i class="fab fa-linkedin-in"></i>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_login_form():
    """Display modern login form"""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="form-title">Welcome Back</h2>', unsafe_allow_html=True)
    st.markdown('<p class="form-subtitle">Sign in to access your sentiment analysis dashboard and continue your work</p>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.login_error:
        st.markdown(f'<div class="error-message"><i class="fas fa-exclamation-circle"></i> {st.session_state.login_error}</div>', unsafe_allow_html=True)
        st.session_state.login_error = None
    
    if st.session_state.register_success:
        st.markdown(f'<div class="success-message"><i class="fas fa-check-circle"></i> {st.session_state.register_success}</div>', unsafe_allow_html=True)
        st.session_state.register_success = None
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username", key="login_username", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password", label_visibility="collapsed")
        
        submit = st.form_submit_button("Sign In", use_container_width=True)
        
        if submit:
            if username and password:
                success, token, message = login_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.token = token
                    st.rerun()
                else:
                    st.session_state.login_error = message
                    st.rerun()
            else:
                st.session_state.login_error = "Please fill in all fields"
                st.rerun()
    
    show_social_buttons()
    
    st.markdown("""
    <div class="toggle-text">
        Don't have an account? <span class="toggle-link">Create one now</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Create New Account", key="toggle_to_register", use_container_width=True):
        st.session_state.show_register = True
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_register_form():
    """Display modern registration form"""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="form-title">Create Account</h2>', unsafe_allow_html=True)
    st.markdown('<p class="form-subtitle">Join us and start analyzing sentiments with powerful AI tools</p>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.register_error:
        st.markdown(f'<div class="error-message"><i class="fas fa-exclamation-circle"></i> {st.session_state.register_error}</div>', unsafe_allow_html=True)
        st.session_state.register_error = None
    
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Choose a username", key="register_username", label_visibility="collapsed")
        email = st.text_input("Email", placeholder="Enter your email address", key="register_email", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Create a strong password", key="register_password", label_visibility="collapsed")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="register_confirm", label_visibility="collapsed")
        
        submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            if username and email and password and confirm_password:
                if password != confirm_password:
                    st.session_state.register_error = "Passwords don't match"
                    st.rerun()
                elif len(password) < 6:
                    st.session_state.register_error = "Password must be at least 6 characters"
                    st.rerun()
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.session_state.register_error = "Invalid email format"
                    st.rerun()
                else:
                    success, message = register_user(username, email, password)
                    if success:
                        st.session_state.register_success = message
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.session_state.register_error = message
                        st.rerun()
            else:
                st.session_state.register_error = "Please fill in all fields"
                st.rerun()
    
    show_social_buttons()
    
    st.markdown("""
    <div class="toggle-text">
        Already have an account? <span class="toggle-link">Sign in instead</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Back to Sign In", key="toggle_to_login", use_container_width=True):
        st.session_state.show_register = False
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_auth_page():
    """Display the authentication page"""
    load_auth_css()
    
    if st.session_state.show_register:
        show_register_form()
    else:
        show_login_form()

# ==================== INITIALIZE ====================

init_auth_db()
init_session_state()
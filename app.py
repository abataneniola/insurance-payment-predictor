import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------
# Load model, scaler, and encoders
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(BASE_DIR, "label_encoder_gender.pkl"))
le_diabetic = joblib.load(os.path.join(BASE_DIR, "label_encoder_diabetic.pkl"))
le_smoker = joblib.load(os.path.join(BASE_DIR, "label_encoder_smoker.pkl"))

# ------------------------
# Helper function to safely encode any input
# ------------------------
def safe_encode(encoder, input_value, field_name="value"):
    """
    Safely encode a value by trying multiple formats.
    Returns encoded value if successful, raises ValueError if all attempts fail.
    """
    # Store original input for debugging
    original_input = input_value
    
    # Try the exact input first
    try:
        return encoder.transform([input_value])[0]
    except:
        pass
    
    # Try lowercase
    try:
        return encoder.transform([input_value.lower()])[0]
    except:
        pass
    
    # Try uppercase
    try:
        return encoder.transform([input_value.upper()])[0]
    except:
        pass
    
    # Try capitalized (first letter uppercase)
    try:
        return encoder.transform([input_value.capitalize()])[0]
    except:
        pass
    
    # Try title case
    try:
        return encoder.transform([input_value.title()])[0]
    except:
        pass
    
    # If encoder expects numbers (0/1), map common text to numbers
    if set(encoder.classes_) == {0, 1} or set(encoder.classes_) == {0.0, 1.0}:
        # Create mapping for common text values
        text_to_num = {
            'male': 1, 'female': 0,
            'm': 1, 'f': 0,
            'yes': 1, 'no': 0,
            'y': 1, 'n': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0
        }
        
        # Try to map the input
        input_lower = input_value.lower().strip()
        if input_lower in text_to_num:
            try:
                return encoder.transform([text_to_num[input_lower]])[0]
            except:
                pass
    
    # If we get here, none of the attempts worked
    available = list(encoder.classes_)
    raise ValueError(
        f"Cannot encode '{original_input}' for {field_name}. "
        f"Encoder expects one of: {available}"
    )

# ------------------------
# Streamlit app config
# ------------------------
st.set_page_config(
    page_title="Insurance Claim Predictor", 
    layout="centered",
    page_icon="🏥",
    initial_sidebar_state="collapsed"
)

# ------------------------
# Custom CSS for beautiful styling
# ------------------------
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Container styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        border: 2px solid white;
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: #34495e;
        font-size: 1.1rem;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #3498db, #2980b9);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Form container */
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button[data-baseweb="button"]:first-child {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    
    .stButton > button[data-baseweb="button"]:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.4);
    }
    
    .stButton > button[data-baseweb="button"]:last-child {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
    }
    
    .stButton > button[data-baseweb="button"]:last-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4);
    }
    
    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, #f1c40f, #f39c12);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(243, 156, 18, 0.3);
        border: 2px solid white;
    }
    
    .result-box h2 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .result-box p {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #34495e;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Beautiful Header
# ------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 Health Insurance Payment Predictor</h1>
    <p>Predict your insurance costs with AI-powered accuracy</p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# Info Box
# ------------------------
st.markdown("""
<div class="info-box">
    <h3>✨ Welcome!</h3>
    <p>This model predicts insurance cost based on your health and demographic data.</p>
    <p style='font-size: 0.9rem; opacity: 0.9;'>Enter your details below for an instant estimate</p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# Optional debug section (hidden by default)
# ------------------------
#with st.expander("🔧 Debug Info (click to expand)"):
    #st.write("**Encoder Classes:**")
    #st.write(f"Gender encoder expects: {list(le_gender.classes_)}")
    #st.write(f"Diabetic encoder expects: {list(le_diabetic.classes_)}")
    #st.write(f"Smoker encoder expects: {list(le_smoker.classes_)}")

# ------------------------
# Default values
# ------------------------
DEFAULTS = {
    "age": 18,
    "bmi": 25.0,
    "children": 0,
    "bloodpressure": 120,
    "gender": "Male",
    "diabetic": "No",
    "smoker": "No"
}

# Initialize session state
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------
# Reset function
# ------------------------
def reset_inputs():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# ------------------------
# User input form
# ------------------------
st.markdown('<div class="form-container">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">📋 Enter Your Information</h3>', unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 👤 Personal Information")
        st.number_input("Age", min_value=0, max_value=120, key="age", help="Your age in years")
        st.number_input("BMI", min_value=10.0, max_value=70.0, key="bmi", help="Body Mass Index")
        st.number_input("Number of Children", min_value=0, max_value=10, key="children", help="Number of dependents")
        
    with col2:
        st.markdown("##### ❤️ Health Status")
        st.number_input("Blood Pressure", min_value=60, max_value=200, key="bloodpressure", help="Systolic blood pressure")
        st.selectbox("Gender", options=["Male", "Female"], key="gender")
        st.selectbox("Diabetic", options=["No", "Yes"], key="diabetic", help="Do you have diabetes?")
        st.selectbox("Smoker", options=["No", "Yes"], key="smoker", help="Do you smoke?")
    
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.form_submit_button("💰 Calculate Payment")
    with col2:
        reset = st.form_submit_button("🔄 Reset Form", on_click=reset_inputs)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# Prediction logic
# ------------------------
if submitted:
    try:
        # Use safe encoding that tries multiple formats
        with st.spinner("🤖 AI is calculating your estimate..."):
            gender_encoded = safe_encode(le_gender, st.session_state.gender, "gender")
            diabetic_encoded = safe_encode(le_diabetic, st.session_state.diabetic, "diabetic")
            smoker_encoded = safe_encode(le_smoker, st.session_state.smoker, "smoker")
        
        input_data = pd.DataFrame({
            "age": [st.session_state.age],
            "gender": [gender_encoded],
            "bmi": [st.session_state.bmi],
            "bloodpressure": [st.session_state.bloodpressure],
            "diabetic": [diabetic_encoded],
            "children": [st.session_state.children],
            "smoker": [smoker_encoded]
        })
        
        # Scale numeric columns
        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Show success with beautiful formatting
        st.balloons()
        
        # Determine color based on prediction amount
        if prediction < 5000:
            emoji = "🟢"
            bg_color = "linear-gradient(135deg, #27ae60, #2ecc71)"
        elif prediction < 15000:
            emoji = "🟡"
            bg_color = "linear-gradient(135deg, #f1c40f, #f39c12)"
        else:
            emoji = "🔴"
            bg_color = "linear-gradient(135deg, #e74c3c, #c0392b)"
        
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 2rem; border-radius: 20px; text-align: center; margin-top: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: 2px solid white;">
            <h2 style="color: white; font-size: 2rem; margin-bottom: 1rem;">{emoji} Your Estimated Cost</h2>
            <p style="color: white; font-size: 3.5rem; font-weight: bold; margin: 0;">${prediction:,.2f}</p>
            <p style="color: white; margin-top: 1rem; opacity: 0.9;">Based on your personal health information</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show what was encoded (optional, remove if not wanted)
        #with st.expander("📊 View encoded values"):
            #st.write("**Values sent to model:**")
            #st.write(f"- Gender encoded as: {gender_encoded}")
            #st.write(f"- Diabetic encoded as: {diabetic_encoded}")
            #st.write(f"- Smoker encoded as: {smoker_encoded}")
        
    except Exception as e:
        st.error("⚠️ There was an error with the encoding. Please check the debug info below.")
        st.exception(e)
        
        # Show helpful error message
        st.info("""
        **Troubleshooting tips:**
        1. Check the Debug Info section above to see what values your encoder expects
        2. Make sure you're selecting from the dropdown options
        3. If the problem persists, your encoder might have been trained on different values
        """)

# ------------------------
# Footer
# ------------------------
st.markdown("""
<div class="footer">
    <p>Made with ❤️ using Streamlit | Data-driven predictions for better health planning</p>
    <p style="font-size: 0.8rem;">© 2026 Insurance Payment Predictor</p>
</div>
""", unsafe_allow_html=True)
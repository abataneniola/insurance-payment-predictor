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
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("💡 Health Insurance Payment Prediction App")
st.info("This model predicts insurance cost based on your health and demographic data.")

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
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Age", min_value=0, max_value=120, key="age")
        st.number_input("BMI", min_value=10.0, max_value=70.0, key="bmi")
        st.number_input("Number of Children", min_value=0, max_value=10, key="children")
        
    with col2:
        st.number_input("Blood Pressure", min_value=60, max_value=200, key="bloodpressure")
        st.selectbox("Gender", options=["Male", "Female"], key="gender")
        st.selectbox("Diabetic", options=["No", "Yes"], key="diabetic")
        st.selectbox("Smoker", options=["No", "Yes"], key="smoker")
    
    submitted = st.form_submit_button("Predict Payment")
    reset = st.form_submit_button("Reset Inputs", on_click=reset_inputs)

# ------------------------
# Prediction logic
# ------------------------
if submitted:
    try:
        # Use safe encoding that tries multiple formats
        with st.spinner("Calculating prediction..."):
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
        
        # Show success with nice formatting
        st.balloons()
        st.success(f"💰 **Estimated Insurance Cost:** ${prediction:,.2f}")
        
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
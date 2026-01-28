import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
import re  # Added for Regex Extraction
from models import Autoencoder, MultiTaskNet
from data_processor import ClinicalPreprocessor

# --- 1. Setup ---
st.set_page_config(page_title="Deep Learning Clinical Risk Engine", layout="wide", page_icon="🩺")

@st.cache_resource
def load_system():
    # Load Preprocessor
    pre = ClinicalPreprocessor.load('preprocessor.joblib')
    # Load Models (weights_only=False required for full class loading)
    ae = torch.load('autoencoder.pth', map_location='cpu', weights_only=False)
    mt = torch.load('multitask_model.pth', map_location='cpu', weights_only=False)
    kmeans = joblib.load('kmeans.joblib')
    return pre, ae, mt, kmeans

try:
    pre, ae, mt, kmeans = load_system()
    st.sidebar.success("✅ Neural Systems Online")
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- 2. The "Smart Regex Brain" (No OpenAI Needed) ---
def extract_clinical_data_local(text):
    """
    V2: Handles typos, qualitative terms ('high'), and natural phrasing.
    """
    text = text.lower()
    extracted = {}
    
    # --- A. Smart Number Extraction ---
    
    # 1. AGE: Catch "Patient is 72", "72 years old", or just "72" at start
    # Looks for 2 digits followed by 'years' OR preceded by 'is'/'patient'
    age_match = re.search(r'(?:is|age|old|patient)\s*(\d{2})', text)
    if age_match:
        extracted['age'] = int(age_match.group(1))

    # 2. GLUCOSE: Catch numbers OR "High Glucose"
    # First, look for specific number
    gluc_match = re.search(r'(?:glucose|sugar)[\w\s:]*(\d{2,3})', text)
    if gluc_match:
        extracted['avg_glucose_level'] = int(gluc_match.group(1))
    elif 'high glucose' in text or 'high sugar' in text:
        extracted['avg_glucose_level'] = 200 # Default 'High' value

    # 3. BMI: Look for explicit BMI
    bmi_match = re.search(r'bmi[\s:]*(\d{2}(?:\.\d)?)', text)
    if bmi_match:
        extracted['bmi'] = float(bmi_match.group(1))

    # --- B. Symptom & Condition Mapping ---
    symptom_map = {
        # Habits (Added specific checks for typos could go here, but regex is better)
        'smok': 'smoking',  # Catches "smokes", "smoking", "smoker"
        'cigar': 'smoking',
        'drink': 'alcohol_use', 'alcohol': 'alcohol_use', 'liquor': 'alcohol_use',
        'fat': 'obesity', 'obese': 'obesity', 'overweight': 'obesity',
        
        # Vitals/Conditions
        'bp': 'hypertension', 'pressure': 'hypertension', 'hypertension': 'hypertension',
        
        # Symptoms
        'cough': 'dry_cough', 
        'snore': 'snoring',
        'pain': 'chest_pain', 'chest': 'chest_pain',
        'breath': 'shortness_of_breath', 'short': 'shortness_of_breath',
        'swallow': 'swallowing_difficulty',
        'tired': 'fatigue',
        'blood': 'coughing_of_blood'
    }
    
    found_concepts = []
    
    # Check for keywords
    for word, feature in symptom_map.items():
        if word in text:
            # For hypertension, we set the internal flag 
            if feature == 'hypertension':
                extracted['hypertension'] = 1
            # For smoking, we set the slider value (1-8 scale logic)
            elif feature == 'smoking':
                extracted['smoking'] = 7
            # For others, set high severity
            else:
                extracted[feature] = 7
                
            if feature not in found_concepts:
                found_concepts.append(feature)

    # --- C. Typo Handling (Manual) ---
    # Specifically catching your typo "amoke"
    if 'amoke' in text or 'moke' in text:
        extracted['smoking'] = 7
        if 'smoking' not in found_concepts: found_concepts.append('smoking')

    return extracted, found_concepts

# --- 3. UI Layout ---
col_chat, col_results = st.columns([1, 1])

with col_chat:
    st.title("🩺 AI Health Assistant")
    st.caption("Natural Language Clinical Extraction Engine")
    
    # Input Area
    user_input = st.text_area("Patient Notes / Transcription:", height=100,
                             placeholder="Ex: Patient is 67 years old. He smokes regularly, drinks alcohol, and complains of snoring. Glucose is 155.")
    
    # Session State
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    # Analysis Button
    if st.button("📝 Extract Clinical Data"):
        if user_input:
            with st.spinner("Processing text..."):
                # Call Local Engine
                data, keywords = extract_clinical_data_local(user_input)
                
                # Update State
                st.session_state.form_data.update(data)
                
                # Feedback
                if keywords or data:
                    st.success(f"Extracted: {len(data)} data points")
                    st.json(data)
                else:
                    st.warning("No specific symptoms detected. Try being more specific.")
        else:
            st.warning("Please enter notes first.")

    st.markdown("---")
    st.subheader("Verify & Edit Vitals")
    
    # Form with Defaults from Extraction
    with st.form("vitals"):
        def get_val(key, default=1):
            return st.session_state.form_data.get(key, default)
        
        # Dynamic Defaults for Numbers
        default_age = st.session_state.form_data.get('age', 45)
        default_bmi = st.session_state.form_data.get('bmi', 25.0)
        default_gluc = st.session_state.form_data.get('avg_glucose_level', 100)

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 90, default_age)
            bmi = st.number_input("BMI", 15.0, 60.0, float(default_bmi))
            glucose = st.number_input("Avg Glucose", 50, 400, default_gluc)
            pollution = st.slider("Air Pollution Exposure", 1, 8, 3)
            
        with c2:
            st.markdown("**Symptoms (1-8)**")
            # If extracted, default to High (7), else Low (1)
            cough = st.slider("Dry Cough", 1, 8, get_val('dry_cough'))
            snoring = st.slider("Snoring", 1, 8, get_val('snoring'))
            chest_pain = st.slider("Chest Pain", 1, 8, get_val('chest_pain'))
            swallowing = st.slider("Swallowing Difficulty", 1, 8, get_val('swallowing_difficulty'))
            
            # Hidden fields that might be set by text
            obesity_val = get_val('obesity', 2)
            alcohol_val = get_val('alcohol_use', 2)
            smoking_val = get_val('smoking', 1) # 1=Low, 7=High
            
        analyze = st.form_submit_button("Run Deep Learning Diagnosis")

with col_results:
    if analyze:
        # Build Input Vector
        # We map the text-extracted 'smoking' (1 or 7) to the model's expected format
        smoking_status = 'smokes' if smoking_val > 4 else 'never smoked'
        
        input_dict = {
            'age': age, 'bmi': bmi, 'avg_glucose_level': glucose,
            'dry_cough': cough, 'snoring': snoring, 
            'swallowing_difficulty': swallowing, 'chest_pain': chest_pain,
            'air_pollution': pollution,
            'obesity': obesity_val, 'alcohol_use': alcohol_val,
            'smoking': smoking_val,
            # Defaults
            'gender': 'Male', 'smoking_status': smoking_status,
            'genetic_risk': 3, 'passive_smoker': 2, 'coughing_of_blood': 1,
            'fatigue': 1, 'weight_loss': 1, 'shortness_of_breath': 1,
            'wheezing': 1, 'clubbing_of_finger_nails': 1, 'frequent_cold': 1,
            'hypertension': 0, 'heart_disease': 0,
            'work_type': 'Private', 'residence_type': 'Urban'
        }
        
        # Inference
        X_proc = pre.transform(pd.DataFrame([input_dict])).values
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        
        ae.eval()
        mt.eval()
        with torch.no_grad():
            s_logit, l_logit, _ = mt(X_tensor)
            prob_stroke = torch.sigmoid(s_logit).item()
            prob_lung = torch.sigmoid(l_logit).item()
            _, latent = ae(X_tensor)
            cluster = kmeans.predict(latent.numpy())[0]

        # Dashboard
        st.markdown(f"## 🏥 Patient Phenotype: Type {cluster}")
        
        # Stroke Meter
        s_color = "red" if prob_stroke > 0.5 else "green"
        st.markdown(f"### 🧠 Stroke Risk: :{s_color}[{prob_stroke:.1%}]")
        st.progress(prob_stroke)
        
        # Lung Meter
        l_color = "red" if prob_lung > 0.5 else "green"
        st.markdown(f"### 🫁 Lung Cancer Risk: :{l_color}[{prob_lung:.1%}]")
        st.progress(prob_lung)
        
        if prob_lung > 0.65 or prob_stroke > 0.65:
             st.error("⚠️ HIGH RISK DETECTED: Recommend Clinical Referral")
        elif prob_lung > 0.35 or prob_stroke > 0.35 and not (prob_lung > 0.65 or prob_stroke > 0.65):
             st.warning("⚠️ MODERATE RISK: Suggest Further Testing")
        else:
             st.success("✅ LOW RISK: Routine checkup recommended.")
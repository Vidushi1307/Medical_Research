import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set Page Config
st.set_page_config(page_title="Surgical Risk Predictor", layout="wide")

# 1. LOAD THE BRAIN
@st.cache_resource
def load_assets():
    models = joblib.load('surgery_models.pkl')
    le_dict = joblib.load('encoders.pkl')
    ui_data = joblib.load('ui_data.pkl')
    return models, le_dict, ui_data

models, le_dict, ui_data = load_assets()

# Header
st.title("🏥 Surgical Complication Risk Predictor")
st.markdown("---")

# 2. SIDEBAR - PATIENT INPUTS
st.sidebar.header("Patient Clinical Profile")

with st.sidebar:
    age = st.number_input("Age", 1, 100, 55)
    sex = st.selectbox("Sex", ["MALE", "FEMALE"])
    height = st.number_input("Height (cm)", 100, 250, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    
    # Auto-calculate BMI
    bmi = weight / ((height/100)**2)
    st.info(f"Calculated BMI: {bmi:.1f}")
    
    diabetes = st.selectbox("Diabetes", ["NO", "ORAL", "INSULIN"])
    htn = st.selectbox("Hypertension (Medication)", ["NO", "YES"])
    chf = st.selectbox("Congestive Heart Failure", ["NO", "YES"])
    copd = st.selectbox("History of Severe COPD", ["NO", "YES"])
    smoker = st.selectbox("Current Smoker (within 1 yr)", ["NO", "YES"])
    dyspnea = st.selectbox("Dyspnea Level", ["NO", "On Exertion", "At Rest"])
    
    st.sidebar.header("Surgical Details")
    diagnosis = st.selectbox("Diagnosis", ui_data['diagnoses'])
    surgery = st.selectbox("Planned Surgery", ui_data['surgeries'])
    asa = st.select_slider("ASA Class", options=[1, 2, 3, 4, 5], value=2)
    los = st.slider("Expected Length of Stay (Days)", 1, 30, 5)

# 3. PREDICTION LOGIC
if st.button("Generate Risk Assessment"):
    # Prepare input data dictionary
    raw_input = {
        'AGE_clean': age,
        'SEX': sex,
        'DIAGNOSIS': diagnosis,
        'SURGERY': surgery,
        'LOS_clean': los,
        'Diabetes': diabetes,
        'Hypertension requiring Medication': htn,
        'Congestive heart failure in 30 days prior to surgery': chf,
        'Dyspnea': dyspnea,
        'Current Smoker within 1 year': smoker,
        'History of severe COPD': copd,
        'BMI': bmi,
        'ASA Class': asa
    }

    # Encode inputs
    encoded_input = pd.DataFrame([raw_input])
    for col, le in le_dict.items():
        try:
            encoded_input[col] = le.transform(encoded_input[col].astype(str))
        except:
            encoded_input[col] = 0 # Default if label unknown

    # Calculate percentages for all 13 complications
    results = {}
    for comp_name, model in models.items():
        probs = model.predict_proba(encoded_input)[0]
        classes = model.classes_
        p_dict = dict(zip(classes, probs))
        
        # Weighted Index Logic: P(Avg)*50 + P(Above)*100
        score = p_dict.get(1, 0) * 50 + p_dict.get(2, 0) * 100
        results[comp_name] = score

    # 4. DISPLAY RESULTS
    st.subheader("Critical Risk Summary")
    c1, c2, c3 = st.columns(3)
    
    # Top Metrics
    c1.metric("Serious Complication", f"{results['Serious complication']:.1f}%")
    c2.metric("Cardiac Risk", f"{results['Cardiac complication']:.1f}%")
    c3.metric("Mortality Risk", f"{results['Death']:.1f}%")

    st.markdown("---")
    st.subheader("Detailed Complication Breakdown")
    
    # Grid of other complications
    cols = st.columns(3)
    other_comps = [c for c in results.keys() if c not in ['Serious complication', 'Cardiac complication', 'Death']]
    
    for i, comp in enumerate(other_comps):
        val = results[comp]
        color = "red" if val > 70 else "orange" if val > 30 else "green"
        cols[i % 3].markdown(f"**{comp}**")
        cols[i % 3].progress(val/100)
        cols[i % 3].write(f":{color}[{val:.1f}% Risk]")

else:
    st.info("Fill in the patient details in the sidebar and click the button to see the risk profile.")
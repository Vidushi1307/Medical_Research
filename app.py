import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set Page Config
st.set_page_config(page_title="Surgical Risk Predictor", layout="wide")

# 1. LOAD THE BRAIN
@st.cache_resource
def load_assets():
    # Note: Ensure these files exist in your directory
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
    # Existing Demographic Inputs
    age = st.number_input("Age", 1, 100, 55)
    sex = st.selectbox("Sex", ["MALE", "FEMALE"])
    height = st.number_input("Height (cm)", 100, 250, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    
    bmi = weight / ((height/100)**2)
    st.info(f"Calculated BMI: {bmi:.1f}")
    
    st.markdown("### Comorbidities")
    diabetes = st.selectbox("Diabetes", ["NO", "ORAL", "INSULIN"])
    htn = st.selectbox("Hypertension (Medication)", ["NO", "YES"])
    chf = st.selectbox("Congestive Heart Failure", ["NO", "YES"])
    copd = st.selectbox("History of Severe COPD", ["NO", "YES"])
    smoker = st.selectbox("Current Smoker (within 1 yr)", ["NO", "YES"])
    dyspnea = st.selectbox("Dyspnea Level", ["NO", "On Exertion", "At Rest"])

    # --- NEW LAB INPUTS (Visual Only) ---
    st.markdown("### Laboratory Values")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        hb = st.number_input("Hb (g/dL)", value=12.0)
        tlc = st.number_input("TLC (cells/mm³)", value=7000)
        plt = st.number_input("Plt (10³/µL)", value=250)
        na = st.number_input("Na+ (mEq/L)", value=140)
        k = st.number_input("K+ (mEq/L)", value=4.0)
        urea = st.number_input("Urea (mg/dL)", value=30)
        ph = st.number_input("Blood pH", min_value=6.50, max_value=8.00, value=7.40, step=0.01, format="%.2f")
    with col_l2:
        creat = st.number_input("Creat (mg/dL)", value=1.0)
        tbili = st.number_input("T. Billirubin (mg/dL)", value=0.8)
        alp = st.number_input("ALP (U/L)", value=100)
        sgot = st.number_input("SGOT (U/L)", value=25)
        sgpt = st.number_input("SGPT (U/L)", value=25)
        lactate = st.number_input("Lactate (mmol/L)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, format="%.1f")
        tprot = st.number_input("T. Proteins (g/dL)", value=7.0)
        alb = st.number_input("Albumin (g/dL)", value=4.0)

    st.sidebar.header("Surgical Details")
    diagnosis = st.selectbox("Diagnosis", ui_data['diagnoses'])
    surgery = st.selectbox("Planned Surgery", ui_data['surgeries'])
    asa = st.select_slider("ASA Class", options=[1, 2, 3, 4, 5], value=2)
    los = st.slider("Expected Length of Stay (Days)", 1, 30, 5)

# 3. PREDICTION LOGIC
if st.button("Generate Risk Assessment"):
    # Current features only for the model
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
            encoded_input[col] = 0 

    # Calculate percentages
    results = {}
    for comp_name, model in models.items():
        # RENAME LOGIC
        display_name = "Complication Rate" if comp_name == "Any complication" else comp_name
        
        # REMOVE Serious Complication
        if comp_name == "Serious complication":
            continue
            
        probs = model.predict_proba(encoded_input)[0]
        classes = model.classes_
        p_dict = dict(zip(classes, probs))
        
        score = p_dict.get(1, 0) * 50 + p_dict.get(2, 0) * 100
        results[display_name] = score

    # 4. DISPLAY RESULTS
    st.subheader("Critical Risk Summary")
    c1, c2, c3 = st.columns(3)
    
    # Updated logic for top metrics
    c1.metric("Complication Rate", f"{results.get('Complication Rate', 0):.1f}%")
    c2.metric("Cardiac Risk", f"{results.get('Cardiac complication', 0):.1f}%")
    c3.metric("Mortality Risk", f"{results.get('Death', 0):.1f}%")

    st.markdown("---")
    st.subheader("Detailed Complication Breakdown")
    
    cols = st.columns(3)
    # Filter out the top metrics for the grid
    other_comps = [c for c in results.keys() if c not in ['Complication Rate', 'Cardiac complication', 'Death']]
    
    for i, comp in enumerate(other_comps):
        val = results[comp]
        color = "red" if val > 70 else "orange" if val > 30 else "green"
        cols[i % 3].markdown(f"**{comp}**")
        cols[i % 3].progress(val/100)
        cols[i % 3].write(f":{color}[{val:.1f}% Risk]")

else:
    st.info("Fill in the patient details in the sidebar and click the button to see the risk profile.")
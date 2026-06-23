import pickle
import streamlit as st
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt
from backend.train_kidney import MODEL_PATH, SCALER_PATH
from frontend.utils import api  # Force reload
from frontend.components import charts


# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# --- 2. Load the Scaler ---
@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

def render_kidney_page():
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h2 style="margin:0; font-size: 1.75rem;">🦠 Kidney Function Screening</h2>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Screen for renal health indicators using clinical test values.
    </p>
</div>
""", unsafe_allow_html=True)

    model = load_model()
    scaler = load_scaler()
    
    with st.form("kidney_form"):
        profile = api.fetch_profile() or {}
        # 1. Age Calculation
        default_age = 50
        if profile.get('dob'):
            try:
                from datetime import datetime
                birth_date = datetime.strptime(str(profile['dob']).split()[0], "%Y-%m-%d")
                today = datetime.today()
                default_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except:
                pass

        # Section 1: Demographics & Vitals
        st.subheader("Patient Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, default_age)
        with c2:
            bp = st.number_input("Blood Pressure (mm/Hg)", 50.0, 200.0, 80.0)
        with c3:
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])

        # Section 2: Laboratory Results
        st.subheader("Lab Reports")
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            al = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
            su = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
            bgr = st.number_input("Blood Glucose Random", 50.0, 500.0, 120.0)
        with l2:
            bu = st.number_input("Blood Urea", 10.0, 300.0, 36.0)
            sc = st.number_input("Serum Creatinine", 0.0, 50.0, 1.2)
            sod = st.number_input("Sodium", 100.0, 200.0, 135.0)
        with l3:
            pot = st.number_input("Potassium", 1.0, 10.0, 4.0)
            hemo = st.number_input("Hemoglobin", 3.0, 20.0, 15.0)
            pcv = st.number_input("Packed Cell Volume", 10.0, 60.0, 44.0)
        with l4:
            wc = st.number_input("White Blood Cell Count", 1000.0, 30000.0, 7800.0)
            rc = st.number_input("Red Blood Cell Count", 1.0, 10.0, 5.2)

        # Section 3: Medical History & Symptoms
        st.subheader("History & Status")
        m1, m2, m3 = st.columns(3)
        with m1:
            rbc = st.selectbox("Red Blood Cells (Urine)", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cells", ["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
            ba = st.selectbox("Bacteria", ["notpresent", "present"])
        with m2:
            htn = st.selectbox("Hypertension", ["no", "yes"])
            dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
            cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        with m3:
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["no", "yes"])
            ane = st.selectbox("Anemia", ["no", "yes"])

        if st.form_submit_button("Predict Kidney Health"):
            # Map inputs to Schema
            data = {
                "age": age, 
                "bp": bp, 
                "sg": sg, 
                "al": al, 
                "su": su,
                "rbc": 1 if rbc == "abnormal" else 0,
                "pc": 1 if pc == "abnormal" else 0,
                "pcc": 1 if pcc == "present" else 0,
                "ba": 1 if ba == "present" else 0,
                "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot, 
                "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
                "htn": 1 if htn == "yes" else 0,
                "dm": 1 if dm == "yes" else 0,
                "cad": 1 if cad == "yes" else 0,
                "appet": 1 if appet == "poor" else 0, 
                "pe": 1 if pe == "yes" else 0,
                "ane": 1 if ane == "yes" else 0
            }
            
            with st.spinner("Analyzing..."):
                result = api.get_prediction("kidney", data)
                
            if "error" in result:
                st.error(result['error'])
            else:
                st.subheader("Prediction Result")
                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0.0)

                if confidence >= 50:
                    st.error(f"Result: **{prediction}** (Probability: {confidence}%)")
                else:
                    st.success(f"Result: **{prediction}** (Probability: {confidence}%)")
                    
                api.save_record("Kidney", data, prediction)
                
                c1, c2 = st.columns(2)
                with c1: 
                    st.subheader("Risk Profile")
                    charts.render_radar_chart(data)
                with c2: 
                    st.subheader("Explanation (SHAP)")

                    # --- SHAP ADDITION START ---
                    # 1. Map to a clean DataFrame
                    input_df = pd.DataFrame([data])

                    # 2. Enforce the canonical order of KIDNEY_FEATURES
                    kidney_features_order = [
                        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 
                        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
                    ]
                    input_df = input_df[kidney_features_order]

                    # Preserve unscaled values for rendering real inputs on chart lines
                    display_df = input_df.copy()

                    # 3. Transform using the loaded StandardScaler
                    scaled_array = scaler.transform(input_df)
                    input_df_final = pd.DataFrame(scaled_array, columns=input_df.columns)

                    # 4. Generate SHAP values safely on the structured array
                    explainer = shap.Explainer(model)
                    shap_values = explainer(input_df_final)

                    # 5. Overwrite display attributes with readable numbers
                    shap_values.data = display_df.values

                    # 6. Render the layout container object
                    fig, ax = plt.subplots(figsize=(9, 10))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                    
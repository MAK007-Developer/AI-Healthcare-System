import os
import pickle
import streamlit as st
from backend.train_diabetes import MODEL_PATH
from frontend.utils import api
from frontend.components import charts
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt


# --- 1. Load the Model ---
# We use @st.cache_resource so the model loads only once when the app starts, making it fast.
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def render_diabetes_page():
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h2 style="margin:0; font-size: 1.75rem;">🩸 Diabetes Health Screening</h2>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Use your latest lab report to screen for potential diabetes indicators.
    </p>
</div>
""", unsafe_allow_html=True)

    model = load_model()
    # --- Autofill Logic ---
    profile = api.fetch_profile() or {}
    
    # 1. Age Calculation
    default_age = 30
    if profile.get('dob'):
        try:
            from datetime import datetime
            birth_date = datetime.strptime(str(profile['dob']).split()[0], "%Y-%m-%d")
            today = datetime.today()
            default_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except:
            pass
            
    # 2. Gender
    p_gender = str(profile.get('gender', 'Male')).strip()
    gender_idx = 0 if p_gender.lower() == "female" else 1

    # 3. Smoking history
    smoking_idx = 0
    if str(profile.get('smoking_history', '')).lower() in ["1", "yes", "true", "current", "former", "ever"]:
        smoking_idx = 1

    # 4. BMI Calculation
    default_bmi = 25.0
    if profile.get('height') and profile.get('weight'):
        try:
            h_m = float(profile['height']) / 100
            w_kg = float(profile['weight'])
            default_bmi = round(w_kg / (h_m ** 2), 1)
        except:
            pass

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Details")
        gender = st.selectbox("Gender", ["Female", "Male"], index=gender_idx)
        age = st.number_input("Age", 1, 120, default_age)
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, default_bmi)
        hba1c = st.number_input("HbA1c Level (From Lab Report)", 0.0, 15.0, 5.5, help="Hemoglobin A1c is your average blood sugar levels over the past 3 months.")
        glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 300, 100)
    
    with col2:
        st.subheader("Medical History")
        hypertension = st.selectbox("Hypertension (High BP)", ["No", "Yes"])
        heart_disease = st.selectbox("History of Heart Disease", ["No", "Yes"])
        smoking = st.selectbox("Smoking History", ["No", "Yes"], index=smoking_idx)
        # Advanced Inputs (Optional)
        with st.expander("Additional Health Factors", expanded=False):
            high_chol = st.selectbox("High Cholesterol", ["No", "Yes"])
            activity = st.selectbox("Physically Active (Past 30d)", ["No", "Yes"])
            gen_health = st.slider("General Health Rating", 1, 5, 3, help="1=Excellent, 5=Poor")

    if st.button("Run Screening Analysis", type="primary", use_container_width=True):
        # Map Inputs
        inputs = {
            "gender": 1 if gender == "Male" else 0,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "smoking_history": 1 if smoking == "Yes" else 0,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "glucose": glucose,
            "high_chol": 1 if high_chol == "Yes" else 0,
            "physical_activity": 1 if activity == "Yes" else 0,
            "general_health": gen_health,
        }
        inputsSHAP = {
            "hypertension": 1 if hypertension == "Yes" else 0,
            "high_chol": 1 if high_chol == "Yes" else 0,
            "bmi": bmi,
            "smoking_history": 1 if smoking == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "physical_activity": 1 if activity == "Yes" else 0,
            "general_health": gen_health,
            "gender": 1 if gender == "Male" else 0,
            "age_bucket": age,
        } 
        # Override smoking mapping if the user chose specific strings? 
        # Schema documentation was "0: No, 1: Yes". 
        # Ideally, we should trust the schema.
        
        with st.spinner("Analyzing..."):
            result = api.get_prediction("diabetes", inputs)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("Prediction Result")
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            if confidence >= 50:
                st.error(f"Result: **{prediction}** (Probability: {confidence}%)")
            else:
                st.success(f"Result: **{prediction}** (Probability: {confidence}%)")

            # Save Record
            api.save_record("Diabetes", inputs, prediction)
            
            # Show Charts
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Risk Profile")
                charts.render_radar_chart(inputs)
            with c2:
                st.subheader("Explanation (SHAP)")
                # Convert to DataFrame
                # 1. Wrap the inputs dict in a list so Pandas reads it as a single row
                input_df = pd.DataFrame([inputsSHAP])

                # 2. Filter out extra lab values (like HbA1c and glucose) if your trained model 
                # only expects the 9 features defined in DIABETES_FEATURES.
                # from backend.features import DIABETES_DATASET_MAP
                # input_df = input_df[DIABETES_DATASET_MAP.values()]

                # 3. Initialize the SHAP Explainer using your loaded model
                explainer = shap.Explainer(model)

                # 4. Compute the SHAP values safely using the cleaned DataFrame
                shap_values = explainer(input_df)

                # 5. Create the Matplotlib figure container (w , h) width and height in inches
                fig, ax = plt.subplots(figsize=(9, 10))

                # 6. Generate the Waterfall plot for our patient (row index 0)
                shap.plots.waterfall(shap_values[0], show=False)

                # 7. Safely push the visual into the Streamlit UI layout
                st.pyplot(fig)
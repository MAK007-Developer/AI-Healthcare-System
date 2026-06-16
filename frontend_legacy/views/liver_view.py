import streamlit as st
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np # Added for log transformations
import shap
from backend.train_liver import MODEL_PATH, SCALER_PATH
from frontend_legacy.utils import api
from frontend_legacy.components import charts

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# Added a cache resource to load our RobustScaler
@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

def render_liver_page():
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h2 style="margin:0; font-size: 1.75rem;">🥃 Liver Health Screening</h2>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Analyze liver function markers from your latest blood panel.
    </p>
</div>
""", unsafe_allow_html=True)
    
    profile = api.fetch_profile() or {}
    model = load_model()
    scaler = load_scaler() # Load the scaler here
    
    # 1. Age Calculation
    default_age = 45
    if profile.get('dob'):
        try:
            from datetime import datetime
            birth_date = datetime.strptime(str(profile['dob']).split()[0], "%Y-%m-%d")
            today = datetime.today()
            default_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except:
            pass

    # 2. Gender
    p_gender = profile.get('gender', 'Male')
    gender_idx = 0 if p_gender == "Female" else 1

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, default_age)
        gender = st.selectbox("Gender", ["Female", "Male"], index=gender_idx)
        tot_bili = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
        alk_phos = st.number_input("Alkaline Phosphotase", 0.0, 2000.0, 100.0)
        alamine = st.number_input("Alamine Aminotransferase", 0.0, 2000.0, 30.0)
    
    with col2:
        albumin = st.number_input("Albumin", 0.0, 10.0, 3.0)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 10.0, 1.0)
        # Extras (Hidden defaults or simple inputs)
        direct_bili = 0.5
        aspartate = 30.0
        total_prot = 6.0

    if st.button("Predict Liver Risk", type="primary"):
        inputs = {
            "age": float(age),
            "gender": 1 if gender == "Male" else 0,
            "total_bilirubin": tot_bili,
            "direct_bilirubin": direct_bili,
            "alkaline_phosphotase": alk_phos,
            "alamine_aminotransferase": alamine,
            "aspartate_aminotransferase": aspartate,
            "total_proteins": total_prot,
            "albumin": albumin,
            "albumin_and_globulin_ratio": ag_ratio
        }
        
        with st.spinner("Analyzing Liver..."):
            result = api.get_prediction("liver", inputs)
            
        if "error" in result:
            st.error(result['error'])
        else:
            st.subheader("Prediction Result")
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            st.success(f"Result: **{prediction}** (Probability: {confidence}%)")
            api.save_record("Liver", inputs, prediction)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Risk Profile") 
                charts.render_radar_chart(inputs)
            with c2:
                st.subheader("Explanation (SHAP)") 

                # 1. Convert to DataFrame
                input_df = pd.DataFrame([inputs])

                # 2. Rename Columns to match what the model was trained on
                column_mapping = {
                    'age': 'Age', 
                    'gender': 'Gender', 
                    'total_bilirubin': 'Total_Bilirubin',
                    'direct_bilirubin': 'Direct_Bilirubin', 
                    'alkaline_phosphotase': 'Alkaline_Phosphotase',
                    'alamine_aminotransferase': 'Alamine_Aminotransferase',
                    'aspartate_aminotransferase': 'Aspartate_Aminotransferase',
                    'total_proteins': 'Total_Proteins', 
                    'albumin': 'Albumin',
                    'albumin_and_globulin_ratio': 'Albumin_and_Globulin_Ratio'
                }
                input_df_renamed = input_df.rename(columns=column_mapping)
                
                # Keep a copy of the raw renamed data so the waterfall chart shows real values
                display_df = input_df_renamed.copy()

                # 3. Apply Log Transformation to skewed features
                skewed = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Albumin_and_Globulin_Ratio']
                input_df_renamed[skewed] = np.log1p(input_df_renamed[skewed])

                # 4. Scale the features using our loaded RobustScaler
                scaled_array = scaler.transform(input_df_renamed)
                input_df_final = pd.DataFrame(scaled_array, columns=input_df_renamed.columns)

                # 5. Initialize SHAP Explainer and calculate values using the PROCESSED data
                explainer = shap.Explainer(model)
                shap_values = explainer(input_df_final)

                # 6. Swap out the scaled data in the SHAP object for the RAW data for user-readability
                shap_values.data = display_df.values

                # 7. Create the Matplotlib figure and render the waterfall plot
                fig, ax = plt.subplots(figsize=(9, 10))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
import streamlit as st
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
import shap
from backend.train_lungs import MODEL_PATH, SCALER_PATH
from frontend_legacy.utils import api
from frontend_legacy.components import charts

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

def render_lungs_page():
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h2 style="margin:0; font-size: 1.75rem;">🫁 Respiratory Health Screening</h2>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        A self-assessment tool for lung cancer risk factors and symptoms.
    </p>
</div>
""", unsafe_allow_html=True)

    with st.form("lungs_form"):
        profile = api.fetch_profile() or {}
        model = load_model()
        scaler = load_scaler()
        
        # 1. Age & Gender
        default_age = 60
        gender_idx = 0 # Default Male [Male, Female]
        
        if profile.get('dob'):
            try:
                from datetime import datetime
                birth_date = datetime.strptime(str(profile['dob']).split()[0], "%Y-%m-%d")
                today = datetime.today()
                default_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except:
                pass
        
        if profile.get('gender') == 'Female':
            gender_idx = 1
            
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, default_age)
            gender = st.selectbox("Gender", ["Male", "Female"], index=gender_idx)
        
        st.subheader("Symptoms & Habits")
        
        # Grid layout for many checkboxes
        c1, c2, c3 = st.columns(3)
        with c1:
            smoking = st.checkbox("Smoking History")
            yellow_fingers = st.checkbox("Yellow Fingers")
            anxiety = st.checkbox("Anxiety")
            peer_pressure = st.checkbox("Peer Pressure")
            chronic_disease = st.checkbox("Chronic Disease")
        with c2:
            fatigue = st.checkbox("Fatigue / Tiredness")
            allergy = st.checkbox("Allergies")
            wheezing = st.checkbox("Wheezing")
            alcohol = st.checkbox("Alcohol Consumption")
            coughing = st.checkbox("Persistent Coughing")
        with c3:
            shortness_of_breath = st.checkbox("Shortness of Breath")
            swallowing_difficulty = st.checkbox("Swallowing Difficulty")
            chest_pain = st.checkbox("Chest Pain")

        if st.form_submit_button("Assess Risk"):
            
            data = {
                "gender": 1 if gender == "Male" else 0,
                "age": age,
                "smoking": int(smoking),
                "yellow_fingers": int(yellow_fingers),
                "anxiety": int(anxiety),
                "peer_pressure": int(peer_pressure),
                "chronic_disease": int(chronic_disease),
                "fatigue": int(fatigue),
                "allergy": int(allergy),
                "wheezing": int(wheezing),
                "alcohol": int(alcohol),
                "coughing": int(coughing),
                "shortness_of_breath": int(shortness_of_breath),
                "swallowing_difficulty": int(swallowing_difficulty),
                "chest_pain": int(chest_pain)
            }
            
            with st.spinner("Analyzing..."):
                result = api.get_prediction("lungs", data)
                
            if "error" in result:
                st.error(result['error'])
            else:
                st.subheader("Prediction Result")
                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0.0)
                st.success(f"Result: **{prediction}** (Probability: {confidence}%)")
                api.save_record("Lungs", data, prediction)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Risk Profile")
                    charts.render_radar_chart(data)
                with c2:
                    st.subheader("Explanation (SHAP)")

                    
                    # 1. Convert dictionary to DataFrame
                    input_df = pd.DataFrame([data])

                    # 2. Map Streamlit input keys to the exact LUNG_FEATURES names
                    column_mapping = {
                        "gender": "GENDER",
                        "age": "AGE",
                        "smoking": "SMOKING",
                        "yellow_fingers": "YELLOW_FINGERS",
                        "anxiety": "ANXIETY",
                        "peer_pressure": "PEER_PRESSURE",
                        "chronic_disease": "CHRONIC_DISEASE",
                        "fatigue": "FATIGUE",
                        "allergy": "ALLERGY",
                        "wheezing": "WHEEZING",
                        "alcohol": "ALCOHOL_CONSUMING", # Special mapping!
                        "coughing": "COUGHING",
                        "shortness_of_breath": "SHORTNESS_OF_BREATH",
                        "swallowing_difficulty": "SWALLOWING_DIFFICULTY",
                        "chest_pain": "CHEST_PAIN"
                    }
                    input_df_renamed = input_df.rename(columns=column_mapping)
                    
                    # 3. Force the exact column order used during training
                    lung_features_order = [
                        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                        'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 
                        'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
                    ]
                    input_df_renamed = input_df_renamed[lung_features_order]

                    # Keep a raw copy for the waterfall chart display
                    display_df = input_df_renamed.copy()

                    # 4. Scale the features using our loaded StandardScaler
                    # (Removed the log transformation since it wasn't used in train_lungs.py)
                    scaled_array = scaler.transform(input_df_renamed)
                    input_df_final = pd.DataFrame(scaled_array, columns=input_df_renamed.columns)

                    # 5. Initialize SHAP Explainer and calculate values on the scaled data
                    explainer = shap.Explainer(model)
                    shap_values = explainer(input_df_final)

                    # 6. Swap out the scaled data in the SHAP object for the RAW data
                    shap_values.data = display_df.values

                    # 7. Create the Matplotlib figure and render the waterfall plot
                    fig, ax = plt.subplots(figsize=(9, 10))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                    
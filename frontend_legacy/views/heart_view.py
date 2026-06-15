import pickle
from matplotlib import pyplot as plt
import pandas as pd
import shap
import streamlit as st
from datetime import datetime
from backend.train_heart import MODEL_PATH
from frontend_legacy.utils import api
from frontend_legacy.components import charts


# ---------------------------------------------------------------------------
# Field reference (UCI Cleveland Heart Disease dataset)
# ---------------------------------------------------------------------------
# age       – years
# sex       – 1 = Male, 0 = Female
# cp        – chest pain type: 0 = typical angina, 1 = atypical angina,
#              2 = non-anginal pain, 3 = asymptomatic
# trestbps  – resting blood pressure (mm Hg on admission)
# chol      – serum cholesterol (mg/dl)
# fbs       – fasting blood sugar > 120 mg/dl: 1 = True, 0 = False
# restecg   – resting ECG: 0 = normal, 1 = ST-T abnormality,
#              2 = left ventricular hypertrophy (Estes' criteria)
# thalach   – maximum heart rate achieved (bpm)
# exang     – exercise-induced angina: 1 = Yes, 0 = No
# oldpeak   – ST depression induced by exercise relative to rest
# slope     – slope of peak exercise ST segment:
#              0 = upsloping, 1 = flat, 2 = downsloping
# ca        – number of major vessels coloured by fluoroscopy (0–3)
# thal      – thalassemia: 1 = normal, 2 = fixed defect, 3 = reversible defect
# ---------------------------------------------------------------------------

# --- 1. Load the Model ---
# We use @st.cache_resource so the model loads only once when the app starts, making it fast.
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def render_heart_page():
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h2 style="margin:0; font-size: 1.75rem;">❤️ Heart Health Screening</h2>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Assess cardiovascular risk using clinical metrics from your check-up.
        Values are sourced from the UCI Cleveland Heart Disease dataset.
    </p>
</div>
""", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Autofill from profile
    # ------------------------------------------------------------------
    profile = api.fetch_profile() or {}
    model = load_model()    

    # Age
    default_age = 45
    if profile.get("dob"):
        try:
            birth_date = datetime.strptime(
                str(profile["dob"]).split()[0], "%Y-%m-%d"
            )
            today = datetime.today()
            default_age = (
                today.year
                - birth_date.year
                - ((today.month, today.day) < (birth_date.month, birth_date.day))
            )
        except Exception:
            pass

    # Sex (profile stores "Male" / "Female")
    p_gender = profile.get("gender", "Male")
    gender_idx = 0 if p_gender == "Female" else 1  # selectbox: [Female, Male]

    # Resting BP – use profile systolic if available
    default_bp = 120
    if profile.get("systolic_bp"):
        try:
            default_bp = int(profile["systolic_bp"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Form inputs – two columns
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=default_age)

        gender = st.selectbox("Sex", ["Female", "Male"], index=gender_idx)

        cp = st.selectbox(
            "Chest pain type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Typical angina",
                1: "1 – Atypical angina",
                2: "2 – Non-anginal pain",
                3: "3 – Asymptomatic",
            }[x],
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=default_bp
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=200
        )

        fbs_raw = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"])

    with col2:
        restecg = st.selectbox(
            "Resting ECG result",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – ST-T wave abnormality",
                2: "2 – Left ventricular hypertrophy",
            }[x],
        )

        thalach = st.number_input(
            "Maximum heart rate achieved (bpm)", min_value=60, max_value=220, value=150
        )

        exang_raw = st.selectbox("Exercise-induced angina?", ["No", "Yes"])

        oldpeak = st.number_input(
            "ST depression (exercise vs rest)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1, format="%.1f"
        )

        slope = st.selectbox(
            "Slope of peak exercise ST segment",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Upsloping",
                1: "1 – Flat",
                2: "2 – Downsloping",
            }[x],
        )

        ca = st.selectbox(
            "Major vessels coloured by fluoroscopy",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}",
        )

        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 – Normal",
                2: "2 – Fixed defect",
                3: "3 – Reversible defect",
            }[x],
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    if st.button("Predict Heart Risk", type="primary"):
        inputs = {
            "age":      age,
            "sex":      1 if gender == "Male" else 0,
            "cp":       cp,
            "trestbps": trestbps,
            "chol":     chol,
            "fbs":      1 if fbs_raw == "Yes" else 0,
            "restecg":  restecg,
            "thalach":  thalach,
            "exang":    1 if exang_raw == "Yes" else 0,
            "oldpeak":  oldpeak,
            "slope":    slope,
            "ca":       ca,
            "thal":     thal,
        }

        with st.spinner("Analyzing heart health..."):
            result = api.get_prediction("heart", inputs)

        if "error" in result:
            st.error(result["error"])
        else:
            prediction = result.get("prediction", "Unknown")
            disclaimer = result.get("disclaimer", "")

            st.success(f"Result: **{prediction}**")
            if disclaimer:
                st.caption(disclaimer)

            api.save_record("Heart", inputs, prediction)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Risk Profile")
                charts.render_radar_chart(inputs)
            with c2:
                st.subheader("Explanation (SHAP)")
                # Convert to DataFrame
                # 1. Wrap the inputs dict in a list so Pandas reads it as a single row
                input_df = pd.DataFrame([inputs])

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
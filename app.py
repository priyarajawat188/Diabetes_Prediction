import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ---------- Header ----------
st.title("🩺 Diabetes Prediction App")
st.write("Apni health details enter karein aur model batayega ki diabetes hone ki sambhavna kitni hai.")
st.divider()

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=40, step=1)
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=140, step=1)

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    heart_choice = st.selectbox("Heart Disease", options=["No", "Yes"])
    heart_disease = 1 if heart_choice == "Yes" else 0

# ---------- Predict ----------
if st.button("Predict", type="primary", use_container_width=True):
    patient = pd.DataFrame({
        "age": [age],
        "blood_glucose_level": [glucose],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
    })

    # Model ke training order mein columns arrange karo
    patient = patient[list(model.feature_names_in_)]

    predicted_class = model.predict(patient)[0]
    probs = model.predict_proba(patient)[0]

    st.divider()
    st.subheader("Result")

    if predicted_class == 1:
        st.error("Predicted Result: Diabetes")
    else:
        st.success("Predicted Result: Not Diabetes")

    m1, m2 = st.columns(2)
    m1.metric("Not Diabetes Probability", f"{probs[0] * 100:.1f}%")
    m2.metric("Diabetes Probability", f"{probs[1] * 100:.1f}%")

    st.progress(float(probs[1]), text="Diabetes risk")

    with st.expander("Input summary"):
        st.dataframe(patient, use_container_width=True)

st.divider()
st.caption("Ye sirf educational ML project hai, medical diagnosis nahi. Sahi jaanch ke liye doctor se milein.")

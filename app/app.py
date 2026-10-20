"""Step 5: Streamlit web app.   Run:  streamlit run app/app.py"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction")
st.caption("Educational ML project - not a medical diagnosis.")


@st.cache_resource
def load():
    return (joblib.load(MODELS / "best_model.joblib"),
            joblib.load(MODELS / "imputer.joblib"),
            joblib.load(MODELS / "scaler.joblib"),
            joblib.load(MODELS / "feature_names.joblib"))


try:
    model, imputer, scaler, features = load()
except FileNotFoundError:
    st.error("Model not found. Run `python main.py` first.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
    bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
    skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
with c2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    row = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                       columns=features)
    # Same cleaning as training: 0 in these columns means "unknown"
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        row[col] = row[col].replace(0, np.nan)
    X = scaler.transform(imputer.transform(row))
    prob = model.predict_proba(X)[0, 1]
    if prob >= 0.5:
        st.error(f"Likely DIABETIC (probability {prob:.0%})")
    else:
        st.success(f"Likely NOT diabetic (probability {1 - prob:.0%})")
    st.progress(float(prob))

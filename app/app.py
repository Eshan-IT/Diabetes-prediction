"""Step 5: Streamlit web app.   Run:  streamlit run app/app.py"""
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

_cfg = Path.cwd() / ".streamlit" / "config.toml"
if not _cfg.exists():  # takes effect the next time the app starts
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.write_text('[theme]\nbase = "light"\nprimaryColor = "#2563EB"\nbackgroundColor = "#F3F8FF"\n'
                    'secondaryBackgroundColor = "#E3ECFD"\ntextColor = "#0F2A55"\n')

st.set_page_config(page_title="Diabetes Risk Check", page_icon="🩺", layout="wide")

TEAL, CORAL, INK, MUTED = "#2563EB", "#D9534F", "#0F2A55", "#5B6F94"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Figtree:wght@400;500;600&display=swap');
:root {{ color-scheme: light; }}
html, body, [class*="css"], .stApp {{ font-family: 'Figtree', sans-serif; color: {INK}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: linear-gradient(180deg, #DCEAFF 0%, #F3F8FF 38%, #FFFFFF 100%) !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1100px; }}
.stApp p, .stApp label, .stApp span, .stApp li {{ color: {INK}; }}
.hero {{ background: linear-gradient(120deg, #1E40AF 0%, #2563EB 55%, #60A5FA 100%); border-radius: 26px; padding: 2.4rem 2.4rem 2rem;
        margin-bottom: 1.8rem; box-shadow: 0 16px 36px rgba(37,99,235,.28); }}
.hero h1 {{ font-family: 'Fraunces', serif; font-size: 2.8rem; line-height: 1.1; margin: 0 0 .6rem; color: #FFFFFF !important; }}
.hero p {{ font-size: 1.1rem; color: #E6F0FF !important; max-width: 36rem; margin: 0 0 1.3rem; }}
.pill {{ display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); color: #FFFFFF !important;
        border-radius: 999px; padding: .35rem .95rem; margin: 0 .5rem .4rem 0; font-size: .9rem; font-weight: 500; }}
.group-title {{ font-family: 'Fraunces', serif; font-size: 1.35rem; margin: 1.1rem 0 .15rem; color: {INK}; }}
.group-sub {{ color: {MUTED} !important; font-size: .93rem; margin-bottom: .6rem; }}
[data-testid="stSliderThumbValue"] {{ color: {TEAL} !important; font-weight: 600; }}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {MUTED} !important; }}
[data-testid="stExpander"] {{ background: #FFFFFF; border: 1px solid #D6E2F8; border-radius: 16px; }}
[data-baseweb="select"] > div {{ background: #FFFFFF !important; border-color: #C9D9F5 !important; }}
.result {{ background: #FFFFFF; border: 1px solid #D6E2F8; border-top: 6px solid var(--accent, {TEAL}); border-radius: 22px;
          padding: 1.6rem 1.5rem 1.4rem; text-align: center; box-shadow: 0 14px 34px rgba(37,99,235,.14); }}
.verdict {{ font-family: 'Fraunces', serif; font-size: 1.55rem; margin: .2rem 0 .3rem; }}
.chance {{ color: {MUTED} !important; font-size: 1rem; margin-bottom: 1rem; }}
.notes {{ text-align: left; margin-top: 1rem; border-top: 1px solid #E3EBFA; padding-top: .9rem; font-size: .93rem; }}
.notes div {{ padding: .25rem 0; color: {INK}; }}
.dot {{ display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; margin-right: .55rem; }}
.disclaimer {{ color: {MUTED} !important; font-size: .82rem; margin-top: 1rem; text-align: left; }}
.placeholder {{ background: #FFFFFF; border: 1.5px dashed #9DBBF7; border-radius: 22px; padding: 3rem 1.5rem; text-align: center; color: {MUTED} !important; }}
div.stButton > button {{ background: linear-gradient(120deg, #1D4ED8, #3B82F6); color: #FFFFFF !important; border: 0; border-radius: 14px;
        padding: .75rem 2rem; font-weight: 600; font-size: 1.05rem; width: 100%; box-shadow: 0 8px 18px rgba(37,99,235,.30); }}
div.stButton > button:hover {{ background: linear-gradient(120deg, #1E40AF, #2563EB); color: #FFFFFF !important; }}
div.stButton > button:focus-visible {{ outline: 3px solid #9DBBF7; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return (joblib.load(MODELS / "best_model.joblib"),
            joblib.load(MODELS / "imputer.joblib"),
            joblib.load(MODELS / "scaler.joblib"),
            joblib.load(MODELS / "feature_names.joblib"))


def gauge(prob: float, threshold: float) -> str:
    """Semicircle meter: filled arc = chance of diabetes, tick = decision threshold."""
    r, cx, cy = 90, 120, 120
    arc_len = math.pi * r
    color = CORAL if prob >= threshold else TEAL
    ang = math.pi - math.pi * threshold
    x1, y1 = cx + (r - 14) * math.cos(ang), cy - (r - 14) * math.sin(ang)
    x2, y2 = cx + (r + 14) * math.cos(ang), cy - (r + 14) * math.sin(ang)
    return (
        f'<svg viewBox="0 0 240 150" width="100%" style="max-width:320px" role="img" '
        f'aria-label="Estimated chance of diabetes {prob:.0%}">'
        f'<path d="M30 120 A90 90 0 0 1 210 120" fill="none" stroke="#E3ECFD" stroke-width="20" stroke-linecap="round"/>'
        f'<path d="M30 120 A90 90 0 0 1 210 120" fill="none" stroke="{color}" stroke-width="20" stroke-linecap="round" '
        f'stroke-dasharray="{arc_len * prob:.1f} {arc_len:.1f}"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="2.5"/>'
        f'<text x="120" y="112" text-anchor="middle" font-family="Fraunces, serif" font-size="40" font-weight="700" fill="{INK}">{prob:.0%}</text>'
        f'<text x="120" y="136" text-anchor="middle" font-family="Figtree, sans-serif" font-size="11" fill="{MUTED}">chance of diabetes</text>'
        f'</svg>')


try:
    model, imputer, scaler, features = load()
except FileNotFoundError:
    st.error("Model files not found. Run `python main.py` first to train and save the model.")
    st.stop()

st.markdown("""
<div class="hero">
<h1>Check your diabetes risk</h1>
<p>Enter a few health measurements and a machine learning model will estimate how likely diabetes is.</p>
<span class="pill">Trained on 768 patient records</span><span class="pill">5 models compared</span><span class="pill">Random Forest, ROC-AUC 0.82</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="group-title">Blood and body measurements</div>'
                '<div class="group-sub">Use your latest lab or clinic values.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    glucose = a.slider("Glucose (mg/dL)", 40, 250, 120)
    bmi = b.slider("BMI", 12.0, 60.0, 25.0, step=0.1)
    bp = a.slider("Blood pressure, lower number (mm Hg)", 30, 130, 70)
    insulin = b.slider("Insulin (mu U/ml)", 10, 850, 80)
    skin = a.slider("Skin thickness (mm)", 5, 100, 20)

    st.markdown('<div class="group-title">About you</div>'
                '<div class="group-sub">Background factors that change risk.</div>', unsafe_allow_html=True)
    c, d = st.columns(2)
    age = c.slider("Age (years)", 18, 90, 30)
    preg = d.slider("Number of pregnancies", 0, 17, 0)
    dpf = c.slider("Family history score", 0.05, 2.5, 0.50, step=0.01,
                   help="Diabetes Pedigree Function: higher means more relatives with diabetes.")

    with st.expander("Options"):
        skipped = st.multiselect(
            "Measurements I don't have",
            ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
            help="These are treated as missing and filled with the typical value from the training data.")
        threshold = st.slider("Flag as higher likelihood at or above", 0.20, 0.80, 0.50, step=0.05,
                              help="Lower values catch more diabetic cases but raise false alarms.")
    go = st.button("Estimate my risk")

with right:
    if go:
        row = pd.DataFrame([{"Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
                             "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
                             "DiabetesPedigreeFunction": dpf, "Age": age}])[features]
        for col in skipped:
            row[col] = np.nan
        prob = float(model.predict_proba(scaler.transform(imputer.transform(row)))[0, 1])
        high = prob >= threshold
        color = CORAL if high else TEAL
        verdict = "Higher likelihood of diabetes" if high else "Lower likelihood of diabetes"

        notes = []
        if "Glucose" not in skipped and glucose >= 140:
            notes.append((CORAL, f"Glucose of {glucose} mg/dL is above the usual 140 mg/dL cut-off"))
        if "BMI" not in skipped and bmi >= 30:
            notes.append((CORAL, f"BMI of {bmi:.1f} is in the obesity range (30 or more)"))
        if "BloodPressure" not in skipped and bp >= 90:
            notes.append((CORAL, f"Lower blood pressure of {bp} mm Hg is high (90 or more)"))
        if age >= 45:
            notes.append(("#E0A030", "Risk rises with age, especially after 45"))
        if not notes:
            notes.append((TEAL, "None of your main readings are in a high range"))
        notes_html = "".join(f'<div><span class="dot" style="background:{c}"></span>{t}</div>' for c, t in notes)

        st.markdown(f"""
<div class="result" style="--accent:{color}">
{gauge(prob, threshold)}
<div class="verdict" style="color:{color}">{verdict}</div>
<div class="chance">The tick on the meter marks your {threshold:.0%} cut-off.</div>
<div class="notes">{notes_html}</div>
<div class="disclaimer">Educational project, not a medical diagnosis. Only a blood test reviewed by a doctor can confirm diabetes.</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder">Your result will appear here.<br>Set your measurements, then choose <b>Estimate my risk</b>.</div>',
                    unsafe_allow_html=True)

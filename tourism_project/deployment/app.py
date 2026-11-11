"""
Wellness Tourism Package – Purchase Predictor (Streamlit app)
Loads the model that the GitHub Actions pipeline committed to this folder,
collects the customer's details into a dataframe and shows the purchase prediction.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
META_PATH = APP_DIR / "model_meta.json"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="wide")


@st.cache_resource
def load_model():
    """Load the trained pipeline (preprocessing + XGBoost) and its metadata."""
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return model, meta


model, meta = load_model()
threshold = float(meta.get("decision_threshold", 0.5))

st.title("🧘 Wellness Tourism Package – Purchase Predictor")
st.write(
    "**Visit with Us** · Enter a customer's profile and sales-interaction details to "
    "estimate how likely they are to buy the Wellness Tourism Package **before** "
    "the sales team contacts them."
)

# ---------------- Inputs ----------------
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
with c2:
    income = st.number_input("Monthly income", 1000, 100000, 23000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.radio("Valid passport?", ["Yes", "No"], horizontal=True)
    own_car = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)
    trips = st.number_input("Trips per year (avg.)", 1, 25, 3)
with c3:
    persons = st.number_input("People visiting", 1, 6, 3)
    children = st.number_input("Children (< 5 yrs) visiting", 0, 5, 1)
    star = st.selectbox("Preferred property star", [3, 4, 5])

st.subheader("Sales interaction")
d1, d2, d3 = st.columns(3)
with d1:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
with d2:
    duration = st.number_input("Pitch duration (min)", 1, 130, 15)
    followups = st.number_input("Number of follow-ups", 0, 10, 4)
with d3:
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect the inputs into a single-row dataframe with the training column names
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df, hide_index=True)

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_df)[0, 1])
    if prob >= threshold:
        st.success(f"✅ **Likely to purchase** – probability {prob:.1%}. Prioritise this customer.")
    else:
        st.warning(f"❌ **Unlikely to purchase** – probability {prob:.1%}. Lower contact priority.")
    st.progress(min(prob, 1.0))
    st.caption(f"Decision threshold: {threshold:.2f} (tuned for F1 during training)")

if meta.get("test_metrics"):
    with st.sidebar:
        st.header("Model card")
        st.write("XGBoost pipeline trained by GitHub Actions")
        m = meta["test_metrics"]
        st.metric("Test F1", f"{m.get('test_f1', 0):.3f}")
        st.metric("Test recall", f"{m.get('test_recall', 0):.3f}")
        st.metric("Test ROC-AUC", f"{m.get('test_roc_auc', 0):.3f}")

"""
app.py
Lightweight Streamlit app for single-customer churn prediction.

RUN THIS (from project root, .venv activated, after train_model.py has run):
    streamlit run app.py
"""

from pathlib import Path
import sys
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import add_features, TARGET

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction")
st.caption("Enter customer details to estimate churn risk.")

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure), step=1.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    customer_features = add_features(customer)
    prob = model.predict_proba(customer_features)[0, 1]
    pred = "Yes" if prob >= 0.5 else "No"

    if prob >= 0.60:
        risk, color = "HIGH", "red"
    elif prob >= 0.30:
        risk, color = "MEDIUM", "orange"
    else:
        risk, color = "LOW", "green"

    st.subheader("Prediction Result")
    st.metric("Churn Prediction", pred)
    st.metric("Churn Probability", f"{prob * 100:.1f}%")
    st.markdown(f"**Risk Level:** :{color}[{risk}]")

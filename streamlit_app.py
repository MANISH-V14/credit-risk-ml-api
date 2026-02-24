import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Credit Risk AI System", layout="wide")

st.title("💳 AI-Powered Credit Risk Scoring")
st.markdown("This system predicts the probability of default using a machine learning model trained on synthetic financial data.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Financial Details")
    annual_income = st.slider("Annual Income ($)", 20000, 200000, 60000)
    loan_amount = st.slider("Loan Amount ($)", 1000, 100000, 20000)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with col2:
    st.subheader("Employment & Debt Info")
    years_employed = st.slider("Years Employed", 0, 30, 5)
    existing_debt = st.slider("Existing Debt ($)", 0, 100000, 10000)
    num_credit_cards = st.slider("Number of Credit Cards", 0, 10, 3)

st.markdown("---")

if st.button("🔍 Predict Credit Risk"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "years_employed": years_employed,
            "existing_debt": existing_debt,
            "num_credit_cards": num_credit_cards
        }
    )

    if response.status_code == 200:
        result = response.json()
        probability = result["risk_probability"]

        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error(f"⚠️ High Credit Risk")
        else:
            st.success(f"✅ Low Credit Risk")

        st.metric(label="Risk Probability", value=f"{probability:.2%}")

    else:
        st.error("API connection failed.")

st.markdown("---")
st.subheader("Model Performance")

col3, col4 = st.columns(2)

with col3:
    try:
        cm_img = Image.open("confusion_matrix.png")
        st.image(cm_img, caption="Confusion Matrix")
    except:
        st.write("Run training script to generate confusion matrix.")

with col4:
    try:
        fi_img = Image.open("feature_importance.png")
        st.image(fi_img, caption="Feature Importance")
    except:
        st.write("Run training script to generate feature importance.")
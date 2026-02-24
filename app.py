from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("credit_model.pkl")

class Features(BaseModel):
    annual_income: float
    loan_amount: float
    credit_score: float
    years_employed: float
    existing_debt: float
    num_credit_cards: float

@app.get("/")
def read_root():
    return {"message": "Credit Risk API is running"}

@app.post("/predict")
def predict(data: Features):
    input_data = np.array([
        data.annual_income,
        data.loan_amount,
        data.credit_score,
        data.years_employed,
        data.existing_debt,
        data.num_credit_cards
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction[0]),
        "risk_probability": float(probability)
    }
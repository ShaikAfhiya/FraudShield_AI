
# ==========================================================
# FRAUDSHIELD AI - LIGHTWEIGHT FASTAPI
# Random Forest Fraud Prediction
# ==========================================================

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# 1. CREATE API

app = FastAPI(
    title="FraudShieldAI",
    description="Financial Fraud Detection API",
    version="1.0"
)


# 2. LOAD MODEL

rf_model = joblib.load("models/random_forest.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


# 3. INPUT FORMAT

class TransactionInput(BaseModel):
    features: list[float]


# 4. HOME ROUTE

@app.get("/")
def home():

    return {
        "message": "Welcome to FraudShieldAI",
        "status": "API is running"
    }


# 5. PREDICTION ROUTE

@app.post("/predict")
def predict_transaction(data: TransactionInput):

    if len(data.features) != len(feature_columns):

        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_columns)} features"
        )

    input_data = pd.DataFrame(
        [data.features],
        columns=feature_columns
    )

    prediction = int(
        rf_model.predict(input_data)[0]
    )

    probability = float(
        rf_model.predict_proba(input_data)[0][1]
    )

    result = (
        "Illicit Transaction"
        if prediction == 1
        else "Licit Transaction"
    )

    return {
        "prediction": result,
        "fraud_probability": round(
            probability * 100, 2
        )
    }

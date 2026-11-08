
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="FraudShieldAI",
    description="Financial Fraud Detection using Machine Learning",
    version="1.0"
)

# Load trained model
model = joblib.load("models/random_forest.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Dataset path
DATA_PATH = "data/elliptic/raw/elliptic_txs_features.csv"


class TransactionInput(BaseModel):
    transaction_id: int


@app.get("/")
def home():
    return {
        "message": "Welcome to FraudShieldAI",
        "status": "API is running"
    }


@app.post("/predict")
def predict(data: TransactionInput):

    transaction_found = False

    # Read dataset in smaller parts
    for chunk in pd.read_csv(DATA_PATH, header=None, chunksize=10000):

        # Find transaction using ID
        matching_row = chunk[chunk.iloc[:, 0] == data.transaction_id]

        if not matching_row.empty:

            transaction_found = True

            # Extract 165 features
            features = matching_row.iloc[0, 2:].tolist()

            # Prepare input for model
            input_data = pd.DataFrame(
                [features],
                columns=feature_columns
            )

            # Predict transaction
            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][
                list(model.classes_).index(1)
            ]

            result = "Illicit Transaction" if prediction == 1 else "Licit Transaction"

            return {
                "transaction_id": data.transaction_id,
                "prediction": result,
                "fraud_probability": round(float(probability) * 100, 2)
            }

    if not transaction_found:
        raise HTTPException(
            status_code=404,
            detail="Transaction ID not found in dataset"
        )

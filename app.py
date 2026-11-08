
# ==========================================================
# FRAUDSHIELD AI - FASTAPI
# Financial Fraud Detection using Random Forest + GraphSAGE
# ==========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch_geometric.nn import SAGEConv


# ==========================================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="FraudShieldAI",
    description="Financial Transaction Fraud Detection API",
    version="1.0"
)


# ==========================================================
# 2. INPUT DATA
# ==========================================================

class TransactionInput(BaseModel):
    transaction_id: int


# ==========================================================
# 3. LOAD SAVED MODELS
# ==========================================================

rf_model = joblib.load("models/random_forest.pkl")

scaler = joblib.load("models/feature_scaler.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

features = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_features.csv",
    header=None
)

features.columns = (
    ["txId", "time_step"] +
    [f"feature_{i}" for i in range(1, 166)]
)

features["txId"] = pd.to_numeric(features["txId"])

edges = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_edgelist.csv"
)

transaction_mapping = {
    tx_id: index
    for index, tx_id in enumerate(features["txId"])
}


# ==========================================================
# 5. PREPARE GRAPH
# ==========================================================

edges["source"] = edges["txId1"].map(transaction_mapping)
edges["target"] = edges["txId2"].map(transaction_mapping)

edges = edges.dropna(
    subset=["source", "target"]
)

edges["source"] = edges["source"].astype(int)
edges["target"] = edges["target"].astype(int)

time_steps = features["time_step"].values

source_time = time_steps[edges["source"].values]
target_time = time_steps[edges["target"].values]

causal_edges = edges[
    source_time <= target_time
]

edge_index = torch.tensor(
    causal_edges[["source", "target"]].values.T,
    dtype=torch.long
)


# ==========================================================
# 6. PREPARE FEATURES
# ==========================================================

X = features[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

X_graph = scaler.transform(X)

X_graph = np.nan_to_num(
    X_graph,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

x = torch.tensor(
    X_graph,
    dtype=torch.float
)


# ==========================================================
# 7. GRAPHSAGE MODEL
# ==========================================================

class GraphSAGE(torch.nn.Module):

    def __init__(self, input_features, hidden_features, output_classes):

        super().__init__()

        self.conv1 = SAGEConv(
            input_features,
            hidden_features
        )

        self.conv2 = SAGEConv(
            hidden_features,
            output_classes
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


# ==========================================================
# 8. LOAD GRAPHSAGE MODEL
# ==========================================================

graph_model = GraphSAGE(
    input_features=x.shape[1],
    hidden_features=32,
    output_classes=2
)

graph_model.load_state_dict(
    torch.load(
        "models/graphsage_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

graph_model.eval()


# ==========================================================
# 9. HOME ROUTE
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to FraudShieldAI",
        "status": "API is running"
    }


# ==========================================================
# 10. FRAUD PREDICTION ROUTE
# ==========================================================

@app.post("/predict")
def predict_transaction(data: TransactionInput):

    transaction_rows = features[
        features["txId"] == data.transaction_id
    ]

    if transaction_rows.empty:

        raise HTTPException(
            status_code=404,
            detail="Transaction ID not found"
        )

    transaction_index = transaction_rows.index[0]

    # Random Forest Prediction

    transaction_features = X.loc[
        [transaction_index]
    ]

    rf_prediction = int(
        rf_model.predict(transaction_features)[0]
    )

    rf_probability = rf_model.predict_proba(
        transaction_features
    )[0][1]

    # GraphSAGE Prediction

    with torch.no_grad():

        graph_output = graph_model(
            x,
            edge_index
        )

        graph_probabilities = F.softmax(
            graph_output[transaction_index],
            dim=0
        )

        graph_prediction = int(
            graph_probabilities.argmax().item()
        )

        graph_probability = float(
            graph_probabilities[1].item()
        )

    # Convert prediction into readable labels

    rf_result = (
        "Illicit" if rf_prediction == 1 else "Licit"
    )

    graph_result = (
        "Illicit" if graph_prediction == 1 else "Licit"
    )

    return {
        "transaction_id": data.transaction_id,

        "random_forest": {
            "prediction": rf_result,
            "illicit_probability": round(
                float(rf_probability) * 100, 2
            )
        },

        "graphsage": {
            "prediction": graph_result,
            "illicit_probability": round(
                graph_probability * 100, 2
            )
        }
    }

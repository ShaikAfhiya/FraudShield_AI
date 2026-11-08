
# ==========================================================
# FRAUDSHIELD AI - TRANSACTION PREDICTION
# Random Forest + GraphSAGE
# No custom functions
# ==========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib

from torch_geometric.nn import SAGEConv


# ==========================================================
# 1. ENTER TRANSACTION ID
# ==========================================================

transaction_id = 232438397  # Replace with an actual txId from your dataset


# ==========================================================
# 2. LOAD SAVED MODELS
# ==========================================================

rf_model = joblib.load("models/random_forest.pkl")

scaler = joblib.load("models/feature_scaler.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

features = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_features.csv",
    header=None
)

features.columns = (
    ["txId", "time_step"] +
    [f"feature_{i}" for i in range(1, 166)]
)

edges = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_edgelist.csv"
)

# Find the requested transaction

transaction_rows = features[
    features["txId"] == transaction_id
]

if transaction_rows.empty:

    print("Transaction ID not found in dataset.")

else:

    # ======================================================
    # 4. PREPARE TRANSACTION FEATURES
    # ======================================================

    X = features[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    transaction_index = transaction_rows.index[0]

    transaction_features = X.loc[[transaction_index]]

    transaction_features.columns = (
        transaction_features.columns.astype(str)
    )

    # ======================================================
    # 5. RANDOM FOREST PREDICTION
    # ======================================================

    rf_prediction = rf_model.predict(
        transaction_features
    )[0]

    rf_probability = rf_model.predict_proba(
        transaction_features
    )[0]

    print("\n========== RANDOM FOREST PREDICTION ==========")

    if rf_prediction == 1:
        print("Prediction: ILLICIT TRANSACTION")
    else:
        print("Prediction: LICIT TRANSACTION")

    print(
        "Illicit probability:",
        round(rf_probability[1] * 100, 2),
        "%"
    )

    # ======================================================
    # 6. PREPARE GRAPH DATA
    # ======================================================

    transaction_mapping = {
        tx_id: index
        for index, tx_id in enumerate(features["txId"])
    }

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

    # ======================================================
    # 7. NORMALIZE GRAPH FEATURES
    # ======================================================

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

    # ======================================================
    # 8. DEFINE GRAPHSAGE MODEL
    # ======================================================

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

    # ======================================================
    # 9. LOAD GRAPHSAGE MODEL
    # ======================================================

    model = GraphSAGE(
        input_features=x.shape[1],
        hidden_features=32,
        output_classes=2
    )

    model.load_state_dict(
        torch.load(
            "models/graphsage_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # ======================================================
    # 10. GRAPHSAGE PREDICTION
    # ======================================================

    with torch.no_grad():

        output = model(x, edge_index)

        probabilities = F.softmax(
            output[transaction_index],
            dim=0
        )

        graph_prediction = probabilities.argmax().item()

    print("\n========== GRAPHSAGE PREDICTION ==========")

    if graph_prediction == 1:
        print("Prediction: ILLICIT TRANSACTION")
    else:
        print("Prediction: LICIT TRANSACTION")

    print(
        "Illicit probability:",
        round(probabilities[1].item() * 100, 2),
        "%"
    )

    print("\nPrediction completed!")

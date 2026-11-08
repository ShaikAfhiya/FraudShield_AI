
# ==========================================================
# FRAUDSHIELD AI - FINANCIAL FRAUD DETECTION
# Random Forest + GraphSAGE
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    f1_score
)

import copy


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("\nLoading Elliptic Dataset...")

features = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_features.csv",
    header=None
)

edges = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_edgelist.csv"
)

classes = pd.read_csv(
    "data/elliptic/raw/elliptic_txs_classes.csv"
)

# Rename feature columns
features.columns = (
    ["txId", "time_step"] +
    [f"feature_{i}" for i in range(1, 166)]
)

# Convert transaction IDs and time steps
features["txId"] = pd.to_numeric(features["txId"])
features["time_step"] = pd.to_numeric(features["time_step"])

edges["txId1"] = pd.to_numeric(edges["txId1"])
edges["txId2"] = pd.to_numeric(edges["txId2"])

classes["txId"] = pd.to_numeric(classes["txId"])
classes["class"] = classes["class"].astype(str)

print("Features shape:", features.shape)
print("Edges shape:", edges.shape)
print("Classes shape:", classes.shape)


# ==========================================================
# 2. PREPARE LABELS
# ==========================================================

# 1 = Illicit transaction
# 2 = Licit transaction
# unknown = Unlabelled transaction

label_mapping = {
    "1": 1,
    "2": 0
}

label_dictionary = dict(
    zip(classes["txId"], classes["class"])
)

features["label"] = (
    features["txId"]
    .map(label_dictionary)
    .map(label_mapping)
    .fillna(-1)
    .astype(int)
)

print("\nLabel Distribution:")
print(features["label"].value_counts())


# ==========================================================
# 3. PREPARE FEATURES
# ==========================================================

feature_columns = [
    column for column in features.columns
    if column.startswith("feature_")
]

X = features[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

y = features["label"].values
time_steps = features["time_step"].values


# ==========================================================
# 4. TRAIN, VALIDATION AND TEST SPLIT
# ==========================================================

# Use earlier time steps for training.
# Use later time steps for validation and testing.

train_mask = (time_steps <= 29) & (y != -1)

validation_mask = (
    (time_steps >= 30) &
    (time_steps <= 34) &
    (y != -1)
)

test_mask = (time_steps > 34) & (y != -1)

train_indices = np.where(train_mask)[0]
validation_indices = np.where(validation_mask)[0]
test_indices = np.where(test_mask)[0]

print("\nDataset Split:")
print("Training samples:", len(train_indices))
print("Validation samples:", len(validation_indices))
print("Testing samples:", len(test_indices))


# ==========================================================
# 5. RANDOM FOREST BASELINE MODEL
# ==========================================================

print("\n========== RANDOM FOREST MODEL ==========")

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

# Keep feature names consistent
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_predictions))

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    target_names=["Licit", "Illicit"],
    zero_division=0
))

if len(np.unique(y_test)) == 2:
    print(
        "Random Forest ROC-AUC:",
        roc_auc_score(y_test, rf_probabilities)
    )

    print(
        "Random Forest PR-AUC:",
        average_precision_score(y_test, rf_probabilities)
    )


# ==========================================================
# 6. PREPARE GRAPH DATA FOR GRAPHSAGE
# ==========================================================

print("\n========== PREPARING GRAPH DATA ==========")

# Map transaction IDs to zero-based node indexes

transaction_mapping = {
    transaction_id: index
    for index, transaction_id in enumerate(features["txId"])
}

edges["source"] = edges["txId1"].map(transaction_mapping)
edges["target"] = edges["txId2"].map(transaction_mapping)

# Remove edges with missing transaction IDs

edges = edges.dropna(subset=["source", "target"])

edges["source"] = edges["source"].astype(int)
edges["target"] = edges["target"].astype(int)

# Keep edges that flow from an earlier or same-time
# transaction to a later or same-time transaction.

source_time = time_steps[edges["source"].values]
target_time = time_steps[edges["target"].values]

causal_edges = edges[
    source_time <= target_time
]

edge_index = torch.tensor(
    causal_edges[["source", "target"]].values.T,
    dtype=torch.long
)

print("Number of graph nodes:", len(features))
print("Number of graph edges:", edge_index.shape[1])


# ==========================================================
# 7. NORMALIZE FEATURES FOR GRAPHSAGE
# ==========================================================

scaler = StandardScaler()

# Fit scaler only on training nodes

X_train_graph = scaler.fit_transform(
    X.iloc[train_indices]
)

# Transform all nodes using the training scaler

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

y_tensor = torch.tensor(
    np.where(y == -1, 0, y),
    dtype=torch.long
)

train_idx = torch.tensor(
    train_indices,
    dtype=torch.long
)

validation_idx = torch.tensor(
    validation_indices,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_indices,
    dtype=torch.long
)

print("Node feature shape:", x.shape)
print("Edge index shape:", edge_index.shape)


# ==========================================================
# 8. CREATE GRAPHSAGE MODEL
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

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.conv2(x, edge_index)

        return x


# ==========================================================
# 9. INITIALIZE MODEL
# ==========================================================

print("\n========== GRAPHSAGE MODEL ==========")

model = GraphSAGE(
    input_features=x.shape[1],
    hidden_features=32,
    output_classes=2
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005
)

# Handle class imbalance with class weights

class_counts = np.bincount(
    y[train_indices],
    minlength=2
)

class_weights = len(train_indices) / (
    2 * np.maximum(class_counts, 1)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("Class weights:", class_weights)


# ==========================================================
# 10. TRAIN GRAPHSAGE MODEL
# ==========================================================

print("\nTraining GraphSAGE...")

best_validation_f1 = -1
best_model_weights = None

epochs = 15

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    # Forward propagation

    output = model(x, edge_index)

    # Calculate training loss

    loss = F.cross_entropy(
        output[train_idx],
        y_tensor[train_idx],
        weight=class_weights
    )

    # Backpropagation

    loss.backward()

    optimizer.step()

    # Validation

    model.eval()

    with torch.no_grad():

        validation_output = model(x, edge_index)

        validation_predictions = (
            validation_output[validation_idx]
            .argmax(dim=1)
        )

        validation_f1 = f1_score(
            y_tensor[validation_idx].numpy(),
            validation_predictions.numpy(),
            zero_division=0
        )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss.item():.4f} "
        f"Validation F1: {validation_f1:.4f}"
    )

    # Save the best model

    if validation_f1 > best_validation_f1:

        best_validation_f1 = validation_f1

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


# ==========================================================
# 11. TEST GRAPHSAGE MODEL
# ==========================================================

print("\n========== GRAPHSAGE EVALUATION ==========")

if best_model_weights is not None:

    model.load_state_dict(best_model_weights)

model.eval()

with torch.no_grad():

    final_output = model(x, edge_index)

    graph_probabilities = F.softmax(
        final_output[test_idx],
        dim=1
    )[:, 1]

    graph_predictions = (
        final_output[test_idx]
        .argmax(dim=1)
    )

y_test_graph = y_tensor[test_idx]

print("\nGraphSAGE Accuracy:")

print(
    accuracy_score(
        y_test_graph.numpy(),
        graph_predictions.numpy()
    )
)

print("\nGraphSAGE Classification Report:")

print(
    classification_report(
        y_test_graph.numpy(),
        graph_predictions.numpy(),
        target_names=["Licit", "Illicit"],
        zero_division=0
    )
)

if len(torch.unique(y_test_graph)) == 2:

    print(
        "GraphSAGE ROC-AUC:",
        roc_auc_score(
            y_test_graph.numpy(),
            graph_probabilities.numpy()
        )
    )

    print(
        "GraphSAGE PR-AUC:",
        average_precision_score(
            y_test_graph.numpy(),
            graph_probabilities.numpy()
        )
    )


# ==========================================================
# 12. COMPARE BOTH MODELS
# ==========================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

graph_accuracy = accuracy_score(
    y_test_graph.numpy(),
    graph_predictions.numpy()
)

print("\n========== MODEL COMPARISON ==========")

print("Random Forest Accuracy:", rf_accuracy)

print("GraphSAGE Accuracy:", graph_accuracy)


# ==========================================================
# 13. GRAPHSAGE CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test_graph.numpy(),
    graph_predictions.numpy()
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Licit", "Illicit"]
)

display.plot()

plt.title("GraphSAGE Confusion Matrix")

plt.show()


# ==========================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_columns
)

importance = importance.sort_values(
    ascending=False
).head(15)

importance.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top 15 Random Forest Features")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

print("\nFraudShieldAI Model Training Completed!")


# ==========================================================
# 15. SAVE TRAINED MODELS
# ==========================================================

import os
import joblib

# Create models folder

os.makedirs("models", exist_ok=True)

# Save Random Forest model

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

# Save GraphSAGE model

torch.save(
    model.state_dict(),
    "models/graphsage_model.pth"
)

# Save feature scaler

joblib.dump(
    scaler,
    "models/feature_scaler.pkl"
)

# Save feature column names

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("\nModels saved successfully!")

print("Random Forest: models/random_forest.pkl")
print("GraphSAGE: models/graphsage_model.pth")
print("Scaler: models/feature_scaler.pkl")

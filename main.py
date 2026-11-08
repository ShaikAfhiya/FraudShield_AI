
# FRAUDSHIELD GRAPH ML
# EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load datasets

DATA_PATH = "data/elliptic/raw"

features = pd.read_csv(
    DATA_PATH + "/elliptic_txs_features.csv",
    header=None
)

edges = pd.read_csv(
    DATA_PATH + "/elliptic_txs_edgelist.csv"
)

classes = pd.read_csv(
    DATA_PATH + "/elliptic_txs_classes.csv"
)

print("All datasets loaded successfully!")

# 2. Dataset information and shapes

print("\nFEATURES SHAPE:", features.shape)
print("\nEDGES SHAPE:", edges.shape)
print("\nCLASSES SHAPE:", classes.shape)

print("\nFeatures:")
print(features.head())

print("\nEdges:")
print(edges.head())

print("\nClasses:")
print(classes.head())

print("\nFeature Information:")
features.info()

# 3. Missing values

print("\nMissing Values:")
print("Features:", features.isnull().sum().sum())
print("Edges:", edges.isnull().sum().sum())
print("Classes:", classes.isnull().sum().sum())

# 4. Duplicate analysis

print("\nDuplicate Rows:")
print("Features:", features.duplicated().sum())
print("Edges:", edges.duplicated().sum())
print("Classes:", classes.duplicated().sum())

print("\nDuplicate Transaction IDs:")
print(features.iloc[:, 0].duplicated().sum())

# 5. Class distribution

print("\nClass Distribution:")
print(classes["class"].value_counts())

print("\nClass Percentages:")
print(classes["class"].value_counts(normalize=True) * 100)

plt.figure(figsize=(7, 5))
sns.countplot(data=classes, x="class")
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 6. Time-step analysis

print("\nTime Step Distribution:")
print(features.iloc[:, 1].value_counts().sort_index())

plt.figure(figsize=(10, 5))
sns.countplot(x=features.iloc[:, 1])
plt.title("Transactions Per Time Step")
plt.xlabel("Time Step")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=90)
plt.show()

# 7. Feature statistics

feature_data = features.iloc[:, 2:].apply(
    pd.to_numeric, errors="coerce"
)

print("\nFeature Statistics:")
print(feature_data.describe())

# 8. Feature distributions

feature_data.iloc[:, :3].hist(figsize=(10, 6), bins=30)
plt.suptitle("First Three Feature Distributions")
plt.tight_layout()
plt.show()

# 9. Outlier analysis

Q1 = feature_data.quantile(0.25)
Q3 = feature_data.quantile(0.75)

IQR = Q3 - Q1

outliers = (
    (feature_data < Q1 - 1.5 * IQR) |
    (feature_data > Q3 + 1.5 * IQR)
)

print("\nOutlier Count in First 5 Features:")
print(outliers.sum().head())

plt.figure(figsize=(10, 5))
sns.boxplot(data=feature_data.iloc[:, :5])
plt.title("Outlier Analysis - First 5 Features")
plt.show()

# 10. Correlation analysis

correlation = feature_data.iloc[:, :10].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 11. Graph connection analysis

source = edges.iloc[:, 0]
target = edges.iloc[:, 1]

print("\nTotal Connections:", len(edges))

print("Unique Source Transactions:", source.nunique())
print("Unique Target Transactions:", target.nunique())

degree = pd.concat([source, target]).value_counts()

print("\nTransaction Degree:")
print(degree.describe())

degree.head(20).plot(kind="bar", figsize=(10, 5))
plt.title("Top 20 Connected Transactions")
plt.xlabel("Transaction ID")
plt.ylabel("Number of Connections")
plt.show()

# 12. Check edge transaction IDs

feature_ids = set(features.iloc[:, 0].astype(str))

source_ids = set(source.astype(str))
target_ids = set(target.astype(str))

print("\nEdge Validation:")
print("Source IDs missing from features:", len(source_ids - feature_ids))
print("Target IDs missing from features:", len(target_ids - feature_ids))

# 13. Class distribution across time steps

feature_info = pd.DataFrame({
    "txId": features.iloc[:, 0].astype(str),
    "time_step": features.iloc[:, 1]
})

class_info = classes.copy()
class_info["txId"] = class_info["txId"].astype(str)

time_data = pd.merge(
    feature_info,
    class_info,
    on="txId"
)

time_class = pd.crosstab(
    time_data["time_step"],
    time_data["class"]
)

print("\nClass Distribution Across Time:")
print(time_class)

time_class.plot(figsize=(10, 5))
plt.title("Transaction Classes Across Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Number of Transactions")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")

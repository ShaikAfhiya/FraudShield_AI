# FraudShieldAI – Financial Fraud Detection Using Machine Learning

FraudShieldAI is a machine learning-based financial fraud detection system that identifies potentially illicit Bitcoin transactions using the Elliptic Bitcoin Transaction Dataset.

The project uses transaction features and graph-based relationships to analyze financial transactions and classify them as **licit or illicit**.

## Project Overview

Financial fraud detection is an important challenge in digital payment systems and cryptocurrency networks. FraudShieldAI uses machine learning and graph learning techniques to identify suspicious transaction patterns and help detect potentially fraudulent activities.

The project includes data analysis, graph construction, machine learning model training, transaction prediction, and REST API deployment using FastAPI.

## Key Features

- Exploratory Data Analysis (EDA)
- Bitcoin transaction graph analysis
- Transaction feature preprocessing
- Random Forest classification
- GraphSAGE graph neural network
- Model evaluation and performance analysis
- Transaction ID-based fraud prediction
- FastAPI REST API
- Swagger API documentation

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Graph Machine Learning:** PyTorch, PyTorch Geometric
- **API Development:** FastAPI, Uvicorn
- **Model Persistence:** Joblib
- **Deployment:** Render
- **Version Control:** Git and GitHub

## Dataset

This project uses the Elliptic Bitcoin Transaction Dataset.

The dataset contains three files:

| File                        | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `elliptic_txs_features.csv` | Transaction IDs, time steps, and 165 transaction features |
| `elliptic_txs_edgelist.csv` | Relationships between transactions                        |
| `elliptic_txs_classes.csv`  | Transaction class labels                                  |

### Dataset Labels

- `1` – Illicit transaction
- `2` – Licit transaction
- `unknown` – Unclassified transaction

Unknown transactions are excluded from supervised model training and evaluation.

**Dataset source:** Elliptic Bitcoin Transaction Dataset, available through Kaggle and the associated research publication.

The dataset is not included in this GitHub repository because of its large file size.

### Dataset Setup

Download the Elliptic dataset and place the three CSV files in:

```text
data/
└── elliptic/
    └── raw/
        ├── elliptic_txs_features.csv
        ├── elliptic_txs_edgelist.csv
        └── elliptic_txs_classes.csv
```

## Project Structure

```text
FraudShieldAI/
│
├── data/
│   └── elliptic/
│       └── raw/
│
├── models/
│   ├── random_forest.pkl
│   ├── graphsage_model.pth
│   ├── feature_scaler.pkl
│   └── feature_columns.pkl
│
├── main.py
├── graph_basics.py
├── ml_model.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShaikAfhiya/FraudShield_AI
cd FraudShieldAI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and Configure the Dataset

Download the Elliptic dataset and place the CSV files in the directory described above.

### 5. Run Exploratory Data Analysis

```bash
python main.py
```

### 6. Perform Graph Analysis

```bash
python graph_basics.py
```

### 7. Train Machine Learning Models

```bash
python ml_model.py
```

The trained models and preprocessing files are saved in the `models/` directory.

### 8. Run Transaction Prediction

```bash
python predict.py
```

### 9. Start FastAPI

```bash
uvicorn app:app --reload
```

## API Documentation

FraudShieldAI provides a REST API using FastAPI.

**Local API URL:**

```text
http://127.0.0.1:8000
```

**Swagger Documentation:**

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint   | Description                |
| ------ | ---------- | -------------------------- |
| GET    | `/`        | Check API status           |
| POST   | `/predict` | Predict transaction status |

### Prediction Request

The prediction endpoint accepts a transaction ID.

```json
{
  "transaction_id": 232438397
}
```

The API searches for the transaction in the dataset, extracts its 165 features, and passes them to the trained Random Forest model.

## Machine Learning Approach

### 1. Data Preprocessing

- Load transaction features and labels.
- Handle licit, illicit, and unknown labels.
- Separate transaction IDs and time steps from model features.
- Prepare the 165 transaction features.
- Scale features using StandardScaler.

### 2. Graph Construction

Transaction relationships are represented as a graph:

- Nodes represent Bitcoin transactions.
- Edges represent relationships between transactions.
- Transaction IDs are mapped to graph node indices.

### 3. Model Training

Two approaches are explored:

**Random Forest Classifier**

A traditional machine learning model used as a baseline for classifying licit and illicit transactions.

**GraphSAGE**

A graph neural network that learns transaction representations by aggregating information from neighboring nodes.

### 4. Model Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix


## Future Improvements

- Improve fraud detection performance using advanced graph neural networks.
- Implement real-time transaction monitoring.
- Add transaction risk scoring.
- Explore explainable AI techniques.
- Optimize API response time.
- Integrate additional graph-based fraud detection techniques.

## Project Objective

The objective of FraudShieldAI is to combine traditional machine learning and graph-based learning techniques to identify suspicious Bitcoin transactions and demonstrate an end-to-end financial fraud detection pipeline.

---

**Developed as an ML project using Python, Scikit-learn, PyTorch Geometric, and FastAPI.**

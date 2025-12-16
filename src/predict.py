import mlflow
import pandas as pd
from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0"
)

# -------------------------
# Load model from MLflow
# -------------------------
MODEL_NAME = "Credit_Risk_Model"
MODEL_STAGE = "Production"  # or "None" if not staged yet

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

FEATURE_COLUMNS = [
    "Total_Transaction_Amount",
    "Average_Transaction_Amount",
    "Transaction_Count",
    "Std_Transaction_Amount",
    "Transaction_Recency",
    "Avg_Amount_By_Category",
    "Count_By_FraudResult",
    "Night_Transactions",
    "Amount_CV",
    "Dormant_Flag",
    "Night_Txn_Ratio",
    "Log_Total_Amount",
]

# -------------------------
# Health check
# -------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):

    df = pd.DataFrame([data.dict()])[FEATURE_COLUMNS]

    # Probability of high-risk (class 1)
    prob = model.predict(df)[0]

    return PredictionResponse(
        risk_probability=float(prob),
        risk_label=int(prob >= 0.5)
    )

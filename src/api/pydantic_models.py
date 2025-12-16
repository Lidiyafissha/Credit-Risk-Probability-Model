from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: float
    Std_Transaction_Amount: float
    Transaction_Recency: float
    Avg_Amount_By_Category: float
    Count_By_FraudResult: float
    Night_Transactions: float
    Amount_CV: float
    Dormant_Flag: int
    Night_Txn_Ratio: float
    Log_Total_Amount: float


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int

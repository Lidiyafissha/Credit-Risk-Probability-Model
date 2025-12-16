# Credit-Risk-Probability-Model

## Credit Scoring Business Understannding

### 1. Basel II's Emphasis on Risk Measurement and Model Requirements

#### 1.1. The Link Between Basel II and Model Interpretability/Documentation

The Basel II Accord (specifically under the **Internal Ratings-Based (IRB) approach**) requires banks to use their own models to estimate key risk parameters like Probability of Default (**PD**), Loss Given Default (**LGD**), and Exposure at Default (**EAD**). These models directly determine the bank's minimum regulatory capital requirement.

This direct link to capital dictates two strict requirements:

* **Requirement for Interpretability:** Regulators (**Pillar 2: Supervisory Review**) must validate and approve the bank's internal capital assessment process. A model whose logic cannot be easily explained (a "black box") is almost impossible to approve. The bank's management also needs to understand the risk drivers to make informed lending and provisioning decisions. The coefficients and variable effects must have clear economic reasoning.
* **Requirement for Well-Documentation:** Extensive documentation is mandatory (**Pillar 2 and Pillar 3: Market Discipline**). This covers the model development process, data sources, performance, ongoing monitoring, and governance. This documentation ensures transparency for supervisors and demonstrates that the bank is following sound risk management practices.

---

### 2. The Necessity and Risks of Using a Proxy Variable for "Default"

#### 2.1. Why a Proxy Variable is Necessary

In credit risk modeling, the true event of regulatory default (e.g., legal charge-off or bankruptcy) may be too sparse, take too long to materialize, or be inconsistently defined across different portfolios.

A **proxy variable**—or a **Performance/Default Event Definition**—is created to define the target variable (e.g., $Y=1$ for a "bad" account) based on an earlier, more common, and observable signal of financial distress within a specific window. A common definition might be: "An account is defined as 'bad' if it has been **90 days Past Due (DPD)** or more at any point in the next 12 months."

This proxy is essential because it allows for:

* **Sufficient Sample Size:** Generating enough 'bad' observations to train a statistically robust model.
* **Timely Prediction:** Providing an earlier warning signal for risk management interventions and required provisioning *before* the costly true default occurs.

#### 2.2. Potential Business Risks of Predictions Based on a Proxy

The main risk stems from the potential mismatch between the predicted **proxy event** and the actual **true regulatory default**.

1.  **Proxy Mismatch (Over-Provisioning):** The model accurately predicts the proxy event (e.g., 90 DPD), but the borrower subsequently cures and **never** reaches true regulatory default. This leads to classifying accounts as risky that ultimately perform, resulting in unnecessarily high loan loss provisions and capital charges, which harms profitability.
2.  **Model Calibration Risk (Regulatory Non-Compliance):** The statistical model is trained to predict $\text{PD}_{\text{proxy}}$. Basel II, however, requires the final PD to be calibrated to the long-run average of the **true regulatory default rate ($\text{PD}_{\text{Basel}}$)**. If the bank fails to properly adjust or scale the model's output to the true definition, the Risk-Weighted Assets (RWA) calculation will be incorrect, risking regulatory sanctions.

---

### 3. Trade-offs: Simple vs. Complex Models in a Regulated Context

In a Basel II environment, the choice between a simple, interpretable model (like Logistic Regression with Weight of Evidence, or WoE) and a complex, high-performance model (like Gradient Boosting) is a crucial trade-off between **Regulatory Compliance** and **Predictive Accuracy**.

#### Simple/Interpretable Models (Logistic Regression/WoE)

**Trade-Off Advantages:**

* **High Interpretability:** The linear relationship and monotonic effects from WoE transformation make the model easy to validate, audit, and explain to regulators and internal stakeholders. This directly addresses the core **Pillar 2** requirements.
* **Stability & Auditability:** The model is stable, and changes in input variables result in predictable changes in output. This makes model monitoring straightforward and reduces the risk of regulatory rejection.
* **Ease of Deployment:** Widely understood and easier to implement in traditional banking IT systems.

**Trade-Off Disadvantages:**

* **Lower Predictive Performance:** These models are less capable of capturing complex non-linear relationships and interactions, potentially leading to less accurate PD estimates and, consequently, inefficient capital allocation (higher RWA than necessary).
* **Requires Extensive Feature Engineering:** Requires manual binning and WoE calculation, which consumes time and may lead to information loss.

#### Complex/High-Performance Models (Gradient Boosting)

**Trade-Off Advantages:**

* **Superior Predictive Performance:** Captures complex data structures, leading to more precise PD estimation. Higher accuracy can translate into more efficient capital use, potentially leading to a lower (but more accurate) RWA.
* **Less Feature Engineering:** Can automatically discover and exploit high-order feature interactions.

**Trade-Off Disadvantages:**

* **"Black Box" Nature (High Regulatory Risk):** It is extremely difficult to explain *why* a specific complex prediction was made. This opacity is a significant barrier to **Pillar 2** approval, as regulators may reject the model due to a lack of causal clarity.
* **Difficult to Audit and Monitor:** Complex models are more prone to overfitting and can be unstable. Monitoring for parameter drift and ensuring the economic logic remains sound is much harder, increasing operational and model risk.

**The Contextual Decision:**

In practice, simple, interpretable models are often mandated for **Regulatory Capital (Pillar 1)** because they minimize the risk of regulatory rejection. Complex models are increasingly used for **internal "Economic Capital" models** or for **operational decisions** (like optimizing collections strategy), where high accuracy is valued over regulatory-grade transparency. Using a complex model for regulatory capital requires exhaustive, sophisticated, and often *post-hoc* explainability techniques (like SHAP or LIME) to satisfy the high burden of proof for interpretability.

---

## Task 3 — Feature Engineering (Customer-Level Features)

The feature engineering process transforms raw transaction-level data into **customer-level features** suitable for predictive modeling. Key steps include:

1. **Customer Aggregation**:

   * Summarizes transactions at the customer level.
   * Example features:

     * `Total_Transaction_Amount`
     * `Average_Transaction_Amount`
     * `Transaction_Count`
     * `Transaction_Recency`
     * `Night_Transactions`
     * `Dormant_Flag`

2. **Feature Engineering**:

   * Compute derived features such as:

     * `Avg_Amount_By_Category`
     * `Count_By_FraudResult`
     * `Night_Txn_Ratio`
     * `Amount_CV` (coefficient of variation)
     * `Log_Total_Amount`

3. **Missing Value Handling**:

   * Numeric features: imputed using median.
   * Categorical features: imputed using most frequent value.

4. **Binning & Encoding**:

   * Continuous features are discretized using quantile binning.
   * Categorical features are transformed using Weight-of-Evidence (WoE) encoding for interpretability.

5. **Modular Pipeline Implementation**:

   * All transformations are implemented in `src/data_processing.py` and wrapped in a scikit-learn `Pipeline`.
   * This ensures reproducibility, easy experimentation, and proper integration with downstream model training.

**Example Usage**:

```python
from src.data_processing import create_feature_pipeline

# Load raw data
df = pd.read_csv("data/raw/data.csv")

# Build feature pipeline
feature_pipeline = create_feature_pipeline(df)
X = feature_pipeline.fit_transform(df)
```

---

## Task 4 — Proxy Target Variable Engineering

Regulatory default events are rare or delayed, requiring a **proxy target** for supervised learning.

1. **RFM Clustering**:

   * Customers are clustered based on **Recency, Frequency, and Monetary (RFM)** behavior.
   * K-Means clustering identifies high-risk segments.

2. **Proxy Target Definition**:

   * `is_high_risk` is set to 1 for customers in clusters representing high default likelihood.
   * This target is merged with the customer-level features for model training.

**Example Usage**:

```python
from src.data_processing import create_proxy_target

proxy_target, cluster_summary = create_proxy_target(df)
y = proxy_target["is_high_risk"]
```

---

## Task 5 — Model Training and Experiment Tracking

1. **Train/Test Split**:

   * Split customer-level features (`X`) and target (`y`) into training and testing sets using stratification to preserve class distribution.

```python
from src.train import split_data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
```

2. **Model Selection**:

   * Logistic Regression (interpretable, regulatory-friendly).
   * Decision Tree (handles non-linearities).

3. **Training & MLflow Logging**:

   * Each model is trained and evaluated with metrics: `accuracy`, `precision`, `recall`, `f1`.
   * MLflow automatically logs parameters, metrics, and model artifacts for experiment tracking.

```python
from src.train import train_logistic_regression, train_decision_tree

lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
```

4. **Best Model Selection**:

   * The model with the highest F1 score (or another chosen metric) is selected.
   * Registered in the MLflow Model Registry for deployment.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
best_run = max(runs, key=lambda r: r.data.metrics["f1"])
mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/model", name="Credit_Risk_Model")
```

---

## Task 6 — Model Deployment and Continuous Integration

The best model is exposed via a **REST API** and containerized for production deployment.

### 1. API Development (FastAPI)

* **Location**: `src/api/main.py`
* **Endpoints**:

  * `/predict`: Accepts a POST request with customer-level features and returns predicted risk probability.
* **Input/Output Validation**: Implemented using Pydantic models (`src/api/pydantic_models.py`).

**Example Input**:

```json
{
  "Total_Transaction_Amount": 15000,
  "Average_Transaction_Amount": 2000,
  "Transaction_Count": 7,
  ...
}
```

**Example Response**:

```json
{
  "risk_probability": 0.87
}
```

**Start API locally**:

```bash
uvicorn src.api.main:app --reload
```

### 2. Containerization

* **Dockerfile** sets up the Python environment and installs dependencies.
* **docker-compose.yml** orchestrates container build and execution.

**Build and run**:

```bash
docker-compose build
docker-compose up
```

* Service is available at `http://localhost:8000/predict`.

### 3. Continuous Integration (GitHub Actions)

* Workflow located at `.github/workflows/ci.yml`.
* Executes on every push to `main`.
* Steps include:

  * **Linting**: Using `flake8` to enforce code quality.
  * **Unit Testing**: Using `pytest` to validate feature engineering and helper functions.
* Builds fail if linter or tests fail.

---

## Unit Testing

* Tests are located in `test/test_data_processing.py`.
* Covers:

  * Correct output of feature engineering functions.
  * Proxy target variable creation.
* Run tests:

```bash
pytest -v
```

---

## Deployment Steps for New Users

1. **Clone Repository**:

```bash
git clone https://github.com/yourusername/Credit-Risk-Probability-Model.git
cd Credit-Risk-Probability-Model
```

2. **Set Up Environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. **Prepare Data**:

* Place raw transaction data in `data/raw/data.csv`.
* Ensure column names match the pipeline expectations.

4. **Feature Engineering & Target Creation**:

```python
from src.data_processing import create_feature_pipeline, create_proxy_target

df = pd.read_csv("data/raw/data.csv")
proxy_target, cluster_summary = create_proxy_target(df)
y = proxy_target["is_high_risk"]
feature_pipeline = create_feature_pipeline(df)
X = feature_pipeline.fit_transform(df, y=y)
```

5. **Train and Register Models**:

```python
from src.train import split_data, train_logistic_regression, train_decision_tree
X_train, X_test, y_train, y_test = split_data(X, y)
lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
```

6. **Select and Register Best Model**:

* Choose model based on F1 score.
* Register with MLflow:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
# Select best run ID here and register
mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name="Credit_Risk_Model")
```

7. **Deploy API**:

* Start FastAPI service locally:

```bash
uvicorn src.api.main:app --reload
```

* Or run via Docker:

```bash
docker-compose up
```

* Test `/predict` endpoint with customer features.



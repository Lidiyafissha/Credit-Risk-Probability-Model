import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_feature_pipeline(df):
    # Aggregate + engineer
    pipeline = Pipeline([
        ('aggregate', CustomerAggregator()),
        ('engineer', FeatureEngineer())
    ])

    X = pipeline.fit_transform(df)

    # ColumnTransformer: select only existing columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(
    exclude=['int64', 'float64']).columns.tolist()
    categorical_cols = [
    c for c in categorical_cols
    if c != 'CustomerId']  # Exclude ID column

    print("numeric_cols:", numeric_cols)
    print("categorical_cols:", categorical_cols)


    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    transformers = [
        ('num', SimpleImputer(strategy='median'), numeric_cols),
    ]
    if categorical_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    full_pipeline = Pipeline([
        ('agg_eng', pipeline),
        ('preprocess', preprocessor)
    ])

    return full_pipeline


# ==========================
# 1. Aggregation Transformer
# ==========================
class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates raw transaction data to customer-level features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        snapshot_date = X['TransactionStartTime'].max()

        agg = X.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Value', 'sum'),
            Average_Transaction_Amount=('Value', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Value', 'std'),
            Transaction_Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            Avg_Amount_By_Category=('Value', 'mean'),  # placeholder; could be extended
            Count_By_FraudResult=('FraudResult', 'sum'),
            Night_Transactions=('TransactionStartTime', lambda x: ((x.dt.hour < 6) | (x.dt.hour >= 22)).sum())
        ).reset_index()

        # Fill NaNs in std
        agg['Std_Transaction_Amount'] = agg['Std_Transaction_Amount'].fillna(0)
        return agg


# ==========================
# 2. Feature Engineering Transformer
# ==========================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature transformations: volatility, recency flags, log transform.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Volatility
        X['Amount_CV'] = X['Std_Transaction_Amount'] / (X['Average_Transaction_Amount'] + 1)
        # Recency flags
        X['Dormant_Flag'] = (X['Transaction_Recency'] > 90).astype(int)
        # Night transaction ratio
        X['Night_Txn_Ratio'] = X['Night_Transactions'] / X['Transaction_Count']
        # Log transform skewed variables
        X['Log_Total_Amount'] = np.log1p(X['Total_Transaction_Amount'])
        return X


# ==========================
# 3. Imputer Transformer
# ==========================
class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        return X_imp


# ==========================
# 4. Quantile Binner
# ==========================
class QuantileBinner(BaseEstimator, TransformerMixin):
    """
    Bins numeric features into quantiles for WoE/IV calculation.
    """
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.bins = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                self.bins[col] = pd.qcut(X[col], self.n_bins, duplicates='drop')
        return self

    def transform(self, X):
        X_binned = X.copy()
        for col in self.bins.keys():
            X_binned[col] = pd.qcut(X[col], self.n_bins, labels=False, duplicates='drop')
        return X_binned


# ==========================
# 5. WoE Encoder
# ==========================
class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight-of-Evidence encoding for categorical features.
    """
    def __init__(self):
        self.woe_dict = {}

    def fit(self, X, y):
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].nunique() < 20:
                grouped = pd.DataFrame({'y': y, 'feature': X[col]}).groupby('feature')['y']
                good = grouped.sum()
                bad = grouped.count() - good
                woe = np.log((good + 0.5) / (bad + 0.5))
                self.woe_dict[col] = woe.to_dict()
        return self

    def transform(self, X):
        X_enc = X.copy()
        for col, mapping in self.woe_dict.items():
            X_enc[col] = X_enc[col].map(mapping).fillna(0)
        return X_enc

# ==========================
# 6. RFM Calculation
# ==========================
def calculate_rfm(df):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby('CustomerId')
        .agg(
            Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum')
        )
        .reset_index()
    )
    return rfm

    # ==========================
    # 7. PROXY Target Creation
    # ==========================

def create_proxy_target(df, n_clusters=3):
    rfm = calculate_rfm(df)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_profile = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    high_risk_cluster = (
        cluster_profile
        .sort_values(by=['Recency', 'Frequency', 'Monetary'],
                     ascending=[False, True, True])
        .index[0]
    )

    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    return rfm[['CustomerId', 'is_high_risk']], cluster_profile

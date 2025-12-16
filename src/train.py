import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


# Use the 20-80 rule for splitting data
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

# Logistic Regression with Random Search and MLFlow
def train_logistic_regression(X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name="Logistic_Regression"):

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return model, metrics

# Decision Tree (with Random Search + MLflow)
def train_decision_tree(X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name="Decision_Tree"):

        model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return model, metrics

# mlflow.register_model(
#     model_uri="runs:/<RUN_ID>/model",
#     name="Credit_Risk_Model"
# )



"""
Feature Engineering Review
Baseline vs. log-transformed campaign feature.

Run this script BEFORE recording once to verify the environment and
again DURING the recording from VS Code.

Input:
    data/bank.csv

Outputs:
    results/model_comparison.csv
    results/feature_diagnostics.csv
    results/feature_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path("data/bank.csv")
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

TARGET = "y"
NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

# Keep the analysis focused on information available in the source data.
X = df[NUMERIC + CATEGORICAL].copy()
y = (df[TARGET] == "yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Diagnostic output for the recording.
skew = X_train[NUMERIC].skew().sort_values(ascending=False)
diagnostics = pd.DataFrame({
    "feature": skew.index,
    "skewness": skew.values,
    "absolute_skewness": skew.abs().values
}).sort_values("absolute_skewness", ascending=False)

diagnostics.to_csv(RESULTS / "feature_diagnostics.csv", index=False)

def make_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

# -----------------------
# Baseline
# -----------------------
baseline = make_pipeline()
baseline.fit(X_train, y_train)

baseline_prob = baseline.predict_proba(X_test)[:, 1]
baseline_pred = (baseline_prob >= 0.5).astype(int)

baseline_auc = roc_auc_score(y_test, baseline_prob)
baseline_acc = accuracy_score(y_test, baseline_pred)

# -----------------------
# Feature engineering
# -----------------------
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# campaign is non-negative and typically right-skewed.
X_train_fe["campaign"] = np.log1p(X_train_fe["campaign"])
X_test_fe["campaign"] = np.log1p(X_test_fe["campaign"])

updated = make_pipeline()
updated.fit(X_train_fe, y_train)

updated_prob = updated.predict_proba(X_test_fe)[:, 1]
updated_pred = (updated_prob >= 0.5).astype(int)

updated_auc = roc_auc_score(y_test, updated_prob)
updated_acc = accuracy_score(y_test, updated_pred)

comparison = pd.DataFrame([
    {
        "model": "Baseline",
        "feature_treatment": "Raw campaign",
        "roc_auc": baseline_auc,
        "accuracy": baseline_acc
    },
    {
        "model": "Updated",
        "feature_treatment": "log1p(campaign)",
        "roc_auc": updated_auc,
        "accuracy": updated_acc
    }
])

comparison["roc_auc_change"] = comparison["roc_auc"] - baseline_auc
comparison["accuracy_change"] = comparison["accuracy"] - baseline_acc
comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

summary = pd.DataFrame([
    ["Dataset rows", len(df)],
    ["Positive class rate", y.mean()],
    ["Most skewed numeric feature", diagnostics.iloc[0]["feature"]],
    ["Most skewed feature skewness", diagnostics.iloc[0]["skewness"]],
    ["Baseline ROC-AUC", baseline_auc],
    ["Updated ROC-AUC", updated_auc],
    ["ROC-AUC change", updated_auc - baseline_auc],
    ["Baseline Accuracy", baseline_acc],
    ["Updated Accuracy", updated_acc],
    ["Accuracy change", updated_acc - baseline_acc],
], columns=["metric", "value"])

summary.to_csv(RESULTS / "feature_summary.csv", index=False)

print("\n=== Feature Engineering Review ===")
print(f"Rows: {len(df):,}")
print("\nTop skewed features:")
print(diagnostics.head(5).to_string(index=False))
print("\nModel comparison:")
print(comparison[["model", "feature_treatment", "roc_auc", "accuracy"]].to_string(index=False))
print("\nRecommended review:")
if updated_auc > baseline_auc:
    print("The transformed feature improved ROC-AUC on this holdout split.")
else:
    print("The transformed feature did not improve ROC-AUC on this holdout split.")
print("Use cross-validation before treating the change as a final modeling decision.")

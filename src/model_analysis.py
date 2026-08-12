from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "bank.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

print("=" * 70)
print("FEATURE ENGINEERING REVIEW")
print("=" * 70)

print()
print(f"Loading dataset:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print()
print(f"Dataset shape: {df.shape}")


# -------------------------------------------------------------------
# Validate target
# -------------------------------------------------------------------

if "y" not in df.columns:
    raise ValueError(
        "Target column 'y' was not found in data/bank.csv."
    )


# The preparation script converts:
#
# yes -> 1
# no  -> 0
#
# Therefore we explicitly treat y as numeric here.

df["y"] = pd.to_numeric(
    df["y"],
    errors="coerce"
)

if df["y"].isna().any():
    raise ValueError(
        "Target column contains missing or non-numeric values."
    )

df["y"] = df["y"].astype(int)


# -------------------------------------------------------------------
# Target validation
# -------------------------------------------------------------------

print()
print("Target distribution:")
print(df["y"].value_counts().sort_index())

print()
print("Target proportions:")
print(
    df["y"]
    .value_counts(normalize=True)
    .sort_index()
)

if df["y"].nunique() < 2:
    raise ValueError(
        "The target contains only one class. "
        "Run download_and_prepare.py again and verify data/bank.csv."
    )


# -------------------------------------------------------------------
# Define features
# -------------------------------------------------------------------

TARGET = "y"

numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


# Make sure all expected columns exist.

expected_features = (
    numeric_features +
    categorical_features
)

missing_features = [
    column
    for column in expected_features
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        "The following expected features are missing: "
        + ", ".join(missing_features)
    )


X = df[expected_features].copy()
y = df[TARGET].copy()


# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("Train/test split:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

print()
print("Training target distribution:")
print(y_train.value_counts().sort_index())

print()
print("Testing target distribution:")
print(y_test.value_counts().sort_index())


# -------------------------------------------------------------------
# Feature diagnostics
# -------------------------------------------------------------------

print()
print("=" * 70)
print("FEATURE SKEWNESS")
print("=" * 70)

skewness = (
    X_train[numeric_features]
    .skew()
    .sort_values(ascending=False)
)

print()
print(skewness)


# Save feature diagnostics.

feature_diagnostics = pd.DataFrame({
    "feature": skewness.index,
    "skewness": skewness.values,
})

feature_diagnostics.to_csv(
    RESULTS_DIR / "feature_diagnostics.csv",
    index=False
)


# -------------------------------------------------------------------
# Identify the feature we want to investigate
# -------------------------------------------------------------------
#
# campaign represents the number of contacts performed during the
# current campaign.
#
# It is deliberately selected for the feature-engineering experiment
# because it is typically right-skewed.
# -------------------------------------------------------------------

TRANSFORM_FEATURE = "campaign"

print()
print("=" * 70)
print("FEATURE ENGINEERING CANDIDATE")
print("=" * 70)

print()
print(
    f"Selected feature: {TRANSFORM_FEATURE}"
)

print(
    f"Original skewness: "
    f"{X_train[TRANSFORM_FEATURE].skew():.3f}"
)


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# -------------------------------------------------------------------
# Baseline model
# -------------------------------------------------------------------

baseline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),
    ]
)


print()
print("=" * 70)
print("BASELINE MODEL")
print("=" * 70)

baseline.fit(
    X_train,
    y_train
)

baseline_probability = baseline.predict_proba(
    X_test
)[:, 1]

baseline_prediction = (
    baseline_probability >= 0.5
).astype(int)

baseline_auc = roc_auc_score(
    y_test,
    baseline_probability
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

print()
print(
    f"Baseline ROC-AUC: "
    f"{baseline_auc:.4f}"
)

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy:.4f}"
)


# -------------------------------------------------------------------
# Feature-engineered dataset
# -------------------------------------------------------------------

X_train_fe = X_train.copy()
X_test_fe = X_test.copy()


# Apply log1p transformation.
#
# log1p(x) = log(1 + x)
#
# This is safe for zero values and compresses large values.

X_train_fe[TRANSFORM_FEATURE] = np.log1p(
    X_train_fe[TRANSFORM_FEATURE]
)

X_test_fe[TRANSFORM_FEATURE] = np.log1p(
    X_test_fe[TRANSFORM_FEATURE]
)


# -------------------------------------------------------------------
# Check transformed skewness
# -------------------------------------------------------------------

transformed_skewness = (
    X_train_fe[TRANSFORM_FEATURE]
    .skew()
)

print()
print("=" * 70)
print("TRANSFORMED FEATURE")
print("=" * 70)

print()
print(
    f"Feature: {TRANSFORM_FEATURE}"
)

print(
    f"Original skewness: "
    f"{X_train[TRANSFORM_FEATURE].skew():.4f}"
)

print(
    f"Transformed skewness: "
    f"{transformed_skewness:.4f}"
)


# -------------------------------------------------------------------
# Feature-engineered model
# -------------------------------------------------------------------

feature_engineered = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),
    ]
)


print()
print("=" * 70)
print("FEATURE-ENGINEERED MODEL")
print("=" * 70)

feature_engineered.fit(
    X_train_fe,
    y_train
)

fe_probability = feature_engineered.predict_proba(
    X_test_fe
)[:, 1]

fe_prediction = (
    fe_probability >= 0.5
).astype(int)


fe_auc = roc_auc_score(
    y_test,
    fe_probability
)

fe_accuracy = accuracy_score(
    y_test,
    fe_prediction
)


print()
print(
    f"Updated ROC-AUC: "
    f"{fe_auc:.4f}"
)

print(
    f"Updated Accuracy: "
    f"{fe_accuracy:.4f}"
)


# -------------------------------------------------------------------
# Compare models
# -------------------------------------------------------------------

auc_change = fe_auc - baseline_auc
accuracy_change = fe_accuracy - baseline_accuracy


comparison = pd.DataFrame(
    [
        {
            "model": "Baseline",
            "feature_treatment": "Raw campaign",
            "roc_auc": baseline_auc,
            "accuracy": baseline_accuracy,
        },
        {
            "model": "Updated",
            "feature_treatment": "log1p(campaign)",
            "roc_auc": fe_auc,
            "accuracy": fe_accuracy,
        },
    ]
)


comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)


# -------------------------------------------------------------------
# Feature summary
# -------------------------------------------------------------------

feature_summary = pd.DataFrame(
    [
        {
            "feature": TRANSFORM_FEATURE,
            "original_skewness": X_train[
                TRANSFORM_FEATURE
            ].skew(),
            "transformed_skewness": transformed_skewness,
            "transformation": "log1p",
            "roc_auc_change": auc_change,
            "accuracy_change": accuracy_change,
        }
    ]
)


feature_summary.to_csv(
    RESULTS_DIR / "feature_summary.csv",
    index=False
)


# -------------------------------------------------------------------
# Final interpretation
# -------------------------------------------------------------------

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()

print(
    f"ROC-AUC change: "
    f"{auc_change:+.4f}"
)

print(
    f"Accuracy change: "
    f"{accuracy_change:+.4f}"
)


if auc_change > 0:

    print()
    print(
        "The log1p transformation improved ROC-AUC "
        "on the holdout set."
    )

    print(
        "Recommendation: retain the transformed feature "
        "for subsequent cross-validation."
    )

elif auc_change < 0:

    print()
    print(
        "The log1p transformation reduced ROC-AUC "
        "on the holdout set."
    )

    print(
        "Recommendation: do not adopt the transformation "
        "without additional validation."
    )

else:

    print()
    print(
        "The log1p transformation produced no change "
        "in ROC-AUC on the holdout set."
    )

    print(
        "Recommendation: retain the simpler representation "
        "unless cross-validation provides additional evidence."
    )


# -------------------------------------------------------------------
# Output files
# -------------------------------------------------------------------

print()
print("=" * 70)
print("RESULT FILES")
print("=" * 70)

print()

print(
    RESULTS_DIR / "feature_diagnostics.csv"
)

print(
    RESULTS_DIR / "feature_summary.csv"
)

print(
    RESULTS_DIR / "model_comparison.csv"
)

print()
print("Analysis complete.")
"""
preprocessing.py
Feature engineering + preprocessing pipeline builder for the Telco churn
dataset. Imported by train_model.py, evaluate_model.py, predict.py — not run
standalone.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServices"]

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two engineered features. Both are justified and leakage-free
    (computed only from information known before/at the time of the
    snapshot, not derived from the target):

    - TenureGroup: bins raw tenure (months) into bands. Churn risk is known
      to behave non-linearly with tenure (new customers churn far more than
      long-tenured ones); this makes that relationship directly available
      to linear models like Logistic Regression, not just tree models.

    - TotalServices: count of optional add-on services a customer has
      subscribed to (0-6). More add-ons generally means a more "sticky"
      customer relationship.
    """
    df = df.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    ).astype(str)

    df["TotalServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    return df


def get_feature_lists(df: pd.DataFrame):
    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_features = [
        c for c in df.columns
        if c not in numeric_features and c != TARGET
    ]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    ColumnTransformer fitted ONLY inside a Pipeline on training data
    (scikit-learn's Pipeline handles this automatically: .fit() on training
    data, .transform() on test/new data, so the scaler/encoder never sees
    test data statistics -> no leakage).

    Numeric -> median imputation (robust to outliers) + StandardScaler.
        Scaling matters for Logistic Regression because its coefficients
        are sensitive to feature magnitude (MonthlyCharges ~20-120 would
        otherwise dominate over a 0/1 flag like SeniorCitizen).
    Categorical -> most-frequent imputation + OneHotEncoder(handle_unknown="ignore").
        Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
        do not require scaling since they split on raw thresholds — sharing
        one pipeline here doesn't hurt them, it's just unnecessary for them.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor

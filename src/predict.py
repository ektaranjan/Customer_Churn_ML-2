"""
predict.py
Scores every customer in the cleaned dataset with the trained model and
assigns a risk level. Produces the bulk prediction file used by SQL/Power BI.

RUN THIS (after train_model.py):
    python src/predict.py

OUTPUT:
    Dataset/processed/churn_predictions.csv
    outputs/predictions/churn_predictions.csv (same file, duplicated per
        the requested folder layout)

Risk thresholds are a reasonable starting point, NOT a fixed law — tune
LOW_THRESHOLD / HIGH_THRESHOLD to match your retention team's capacity.
    LOW    : probability < 0.30
    MEDIUM : 0.30 <= probability < 0.60
    HIGH   : probability >= 0.60
"""

from pathlib import Path
import sys
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import add_features, TARGET

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "Dataset" / "raw" / "customer_churn.csv"           # has customerID
CLEANED_PATH = ROOT / "Dataset" / "processed" / "customer_churn_cleaned.csv"  # no customerID
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

OUT_PROCESSED = ROOT / "Dataset" / "processed" / "churn_predictions.csv"
OUT_PREDICTIONS = ROOT / "outputs" / "predictions" / "churn_predictions.csv"

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60


def risk_level(prob: float) -> str:
    if prob >= HIGH_THRESHOLD:
        return "HIGH"
    elif prob >= LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: {MODEL_PATH} not found. Run src/train_model.py first.")
    if not CLEANED_PATH.exists():
        sys.exit(f"ERROR: {CLEANED_PATH} not found. Run src/data_cleaning.py first.")

    model = joblib.load(MODEL_PATH)

    # Bring customerID back in by joining on row order with the original raw
    # file (data_cleaning.py preserves row order and only drops columns/rows
    # with missing TotalCharges+tenure==0 fills; duplicates are dropped too,
    # so we re-derive IDs safely from the raw file filtered the same way).
    raw_df = pd.read_csv(RAW_PATH)
    cleaned_df = pd.read_csv(CLEANED_PATH)

    if len(raw_df) == len(cleaned_df):
        customer_ids = raw_df["customerID"].values
    else:
        print("WARNING: row count mismatch between raw and cleaned data "
              "(duplicates/drops occurred) — generating sequential IDs instead.")
        customer_ids = [f"ROW-{i+1:05d}" for i in range(len(cleaned_df))]

    df = cleaned_df.copy()
    df.insert(0, "CustomerID", customer_ids)

    df_features = add_features(df.drop(columns=[TARGET]))
    X = df_features.drop(columns=["CustomerID"])

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    output = pd.DataFrame({
        "CustomerID": df["CustomerID"],
        "Churn": ["Yes" if p == 1 else "No" for p in preds],
        "Churn_Probability": (probs * 100).round(2),
        "Risk_Level": [risk_level(p) for p in probs],
        "MonthlyCharges": df["MonthlyCharges"].values,
        "Tenure": df["tenure"].values,
        "Contract": df["Contract"].values,
        "TotalCharges": df["TotalCharges"].values,
    })

    OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PROCESSED, index=False)
    output.to_csv(OUT_PREDICTIONS, index=False)

    print(f"Saved {len(output)} predictions to:")
    print(f"  {OUT_PROCESSED}")
    print(f"  {OUT_PREDICTIONS}")
    print("\nRisk level distribution:")
    print(output["Risk_Level"].value_counts())

    revenue_at_risk = output.loc[output["Risk_Level"] == "HIGH", "MonthlyCharges"].sum()
    print(f"\nPotential monthly revenue at risk (HIGH risk customers): ${revenue_at_risk:,.2f}")
    print("(This is POTENTIAL revenue at risk based on model probability, not a guaranteed loss.)")


if __name__ == "__main__":
    main()

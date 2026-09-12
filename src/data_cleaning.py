"""
data_cleaning.py
Loads Dataset/raw/customer_churn.csv, validates it, cleans it, and saves
Dataset/processed/customer_churn_cleaned.csv.

RUN THIS (from project root, with .venv activated):
    python src/data_cleaning.py
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "Dataset" / "raw" / "customer_churn.csv"
PROCESSED_PATH = ROOT / "Dataset" / "processed" / "customer_churn_cleaned.csv"

EXPECTED_TARGET = "Churn"
EXPECTED_ID_COL = "customerID"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"ERROR: Could not find dataset at {path}\n"
            f"Make sure customer_churn.csv is placed at Dataset/raw/customer_churn.csv"
        )
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns found: {list(df.columns)}")

    if EXPECTED_TARGET not in df.columns:
        sys.exit(f"ERROR: Expected target column '{EXPECTED_TARGET}' not found in dataset.")

    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- TotalCharges: stored as text with a few blank strings ----
    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
        before_type = df["TotalCharges"].dtype
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_missing = df["TotalCharges"].isna().sum()
        print(f"TotalCharges converted from {before_type} to numeric. "
              f"Blank/invalid values found: {n_missing}")

        # These are customers with tenure == 0 (brand new, not billed yet).
        # Filling with 0 reflects the real business meaning, not a guess.
        if "tenure" in df.columns:
            mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
            print(f"  -> {mask.sum()} of these have tenure == 0; filling TotalCharges with 0")
            df.loc[mask, "TotalCharges"] = 0

        remaining = df["TotalCharges"].isna().sum()
        if remaining > 0:
            print(f"  -> {remaining} rows still missing TotalCharges with tenure > 0; dropping them")
            df = df.dropna(subset=["TotalCharges"])

    # ---- Duplicates ----
    dup_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"  -> Dropped {dup_count} duplicate rows")

    # ---- Standardize categorical text ----
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # ---- Target encoding ----
    unique_target_vals = set(df[EXPECTED_TARGET].unique())
    if unique_target_vals <= {"Yes", "No"}:
        df[EXPECTED_TARGET] = df[EXPECTED_TARGET].map({"Yes": 1, "No": 0})
        print("Target 'Churn' encoded: Yes -> 1, No -> 0")
    else:
        sys.exit(f"ERROR: Unexpected values in target column: {unique_target_vals}")

    # ---- Drop ID column (no predictive value, prevents accidental leakage) ----
    if EXPECTED_ID_COL in df.columns:
        df = df.drop(columns=[EXPECTED_ID_COL])
        print(f"Dropped identifier column: {EXPECTED_ID_COL}")

    # ---- Outlier note (NOT auto-removed) ----
    # MonthlyCharges and tenure both have wide, legitimate ranges in this
    # dataset (e.g. tenure from 0 to 72 months, MonthlyCharges ~$18-$120).
    # These reflect real differences in customer plans/loyalty, not data
    # entry errors, so we do not strip them out here. Extreme values are
    # instead reviewed visually in EDA (boxplots) before any decision.

    print(f"\nFinal cleaned shape: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset -> {path}")


if __name__ == "__main__":
    raw_df = load_data(RAW_PATH)
    cleaned_df = validate_and_clean(raw_df)
    save_data(cleaned_df, PROCESSED_PATH)

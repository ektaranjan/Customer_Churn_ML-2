"""
train_model.py
Trains and compares Logistic Regression, Decision Tree, Random Forest, and
Gradient Boosting on the cleaned Telco churn dataset using stratified 5-fold
CV, tunes the best model with GridSearchCV, and saves the final pipeline.

RUN THIS (from project root, .venv activated):
    python src/train_model.py

INPUT:  Dataset/processed/customer_churn_cleaned.csv
OUTPUT: models/churn_model.joblib
        outputs/metrics/cross_validation_results.csv
        outputs/metrics/best_parameters.txt
        Dataset/processed/train_set.csv, test_set.csv (so evaluate_model.py
        and predict.py use the exact same split)
"""

from pathlib import Path
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import add_features, get_feature_lists, build_preprocessor, TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "processed" / "customer_churn_cleaned.csv"

# NOTE on XGBoost: not used here. XGBoost's precompiled wheels for very new
# Python releases (3.14) and Apple Silicon can lag or need a from-source
# build, which is unreliable for a portfolio project. GradientBoostingClassifier
# (built into scikit-learn) gives the same boosting-family approach with zero
# extra install risk. If `pip install xgboost` works fine in your environment,
# you can drop it into the MODELS dict below with the same interface.
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: {DATA_PATH} not found. Run src/data_cleaning.py first.")

    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features, categorical_features = get_feature_lists(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Stratified 80/20 split preserves the churn ratio in both sets — this
    # matters because churn is imbalanced (~26.5% positive class here); a
    # plain random split could over/under-represent churners in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, model in MODELS.items():
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        cv_scores = cross_validate(pipe, X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = cv_scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = cv_scores[f"test_{metric}"].std()
        results.append(row)
        print(f"{name}: " + ", ".join(f"{m}={row[m]:.3f}" for m in SCORING))

    cv_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    out_metrics_dir = ROOT / "outputs" / "metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(out_metrics_dir / "cross_validation_results.csv", index=False)
    print(f"\nSaved cross-validation results -> {out_metrics_dir / 'cross_validation_results.csv'}")

    best_model_name = cv_df.iloc[0]["model"]
    print(f"\nTop model by mean CV ROC-AUC: {best_model_name}")
    print("(Final model choice also considers evaluate_model.py results, not just this number.)")

    # Deliberately small, fast-to-run hyperparameter grids.
    if best_model_name == "RandomForest":
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
        }
        base_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    elif best_model_name == "GradientBoosting":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
        base_model = GradientBoostingClassifier(random_state=42)
    elif best_model_name == "LogisticRegression":
        param_grid = {"model__C": [0.01, 0.1, 1, 10]}
        base_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        param_grid = {"model__max_depth": [3, 5, 8, None]}
        base_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipe = Pipeline([("preprocess", preprocessor), ("model", base_model)])

    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.3f}")

    with open(out_metrics_dir / "best_parameters.txt", "w") as f:
        f.write(f"Best model type: {best_model_name}\n")
        f.write(f"Best params: {grid.best_params_}\n")
        f.write(f"Best CV ROC-AUC: {grid.best_score_:.4f}\n")

    final_pipeline = grid.best_estimator_
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "churn_model.joblib")
    print(f"\nSaved final tuned pipeline -> {models_dir / 'churn_model.joblib'}")

    # Persist the exact split so evaluate_model.py / predict.py match this run
    processed_dir = ROOT / "Dataset" / "processed"
    X_train.assign(**{TARGET: y_train}).to_csv(processed_dir / "train_set.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(processed_dir / "test_set.csv", index=False)
    print(f"Saved train/test splits -> {processed_dir}")


if __name__ == "__main__":
    main()

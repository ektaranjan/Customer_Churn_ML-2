"""
evaluate_model.py
Loads the saved model and held-out test set, computes final metrics, and
saves confusion matrix / ROC / precision-recall / feature importance charts.

RUN THIS (after train_model.py):
    python src/evaluate_model.py

OUTPUT:
    outputs/metrics/model_comparison.csv
    outputs/figures/confusion_matrix.png
    outputs/figures/roc_curve.png
    outputs/figures/precision_recall_curve.png
    outputs/figures/feature_importance.png
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import add_features, TARGET

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "Dataset" / "processed" / "test_set.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: {MODEL_PATH} not found. Run src/train_model.py first.")
    if not TEST_PATH.exists():
        sys.exit(f"ERROR: {TEST_PATH} not found. Run src/train_model.py first.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_test = add_features(df_test)

    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Final test-set metrics (best tuned model):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Churn Prediction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    print(f"""
    TN={tn} (correctly predicted stay)   FP={fp} (false alarm)
    FN={fn} (MISSED churner)             TP={tp} (correctly caught churner)

    Business note: a missed churner (FN) usually costs more than a false
    alarm (FP) — a false alarm just means an unneeded retention offer, a
    missed churner is lost revenue. This is why recall is often weighted
    heavily for churn use cases, but it's a business trade-off, not a fixed
    rule: a company with very expensive retention offers might instead
    prioritize precision.
    """)

    # ---- ROC Curve ----
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve - Churn Prediction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # ---- Precision-Recall Curve ----
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve - Churn Prediction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    # ---- Feature importance ----
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        final_estimator = model.named_steps["model"]

        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            label = "Feature Importance (impurity-based)"
        elif hasattr(final_estimator, "coef_"):
            importances = np.abs(final_estimator.coef_[0])
            label = "Feature Importance (|coefficient|)"
        else:
            importances = None

        if importances is not None:
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2E86AB")
            ax.set_xlabel(label)
            ax.set_title("Top 15 Features Associated with Churn Prediction")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
            plt.close()
            imp_df.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
            print(f"Saved feature importance -> {FIG_DIR / 'feature_importance.png'}")
            print("\nNOTE: This shows what the model relied on to make predictions,")
            print("NOT proof of what causes churn. Association, not causation.")
    except Exception as e:
        print(f"Feature importance could not be computed: {e}")

    print(f"\nSaved: {METRICS_DIR / 'model_comparison.csv'}")
    print(f"Saved: {FIG_DIR / 'confusion_matrix.png'}")
    print(f"Saved: {FIG_DIR / 'roc_curve.png'}")
    print(f"Saved: {FIG_DIR / 'precision_recall_curve.png'}")


if __name__ == "__main__":
    main()

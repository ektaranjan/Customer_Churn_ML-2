# Customer Churn Prediction & Retention Intelligence using Machine Learning

## Business Problem
Subscription businesses lose recurring revenue every time a customer churns,
and acquiring a new customer costs significantly more than retaining an
existing one. This project identifies which customers are likely to churn
**before** they leave, so a retention team can act.

## Project Objective
Build an end-to-end ML pipeline that predicts churn probability per
customer, explains what's associated with that risk, and turns the output
into a risk-scored, actionable customer list.

## Dataset
- **Name:** Telco Customer Churn (IBM sample dataset)
- **Source:** Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  (mirrored copy used here: IBM's own public GitHub repository,
  `IBM/telco-customer-churn-on-icp4d`)
- **Rows:** 7,043 customers | **Columns:** 21
- **Target:** `Churn` (Yes/No) — 5,174 No / 1,869 Yes (26.5% churn rate)
- 11 rows had a blank `TotalCharges` (all customers with `tenure = 0`,
  i.e. brand-new/not yet billed); filled with 0. 0 duplicate rows found.

## Technologies
Python (pandas, numpy, scikit-learn, matplotlib, seaborn), Jupyter, MySQL,
Power BI, Streamlit, Git/GitHub.

## Architecture
```
Raw CSV → Validation → Cleaning → EDA → Feature Engineering →
Preprocessing (Pipeline/ColumnTransformer) → Train/Test Split (stratified) →
Cross-Validation (4 models) → Hyperparameter Tuning (GridSearchCV) →
Final Model → Evaluation → Feature Importance → Bulk Prediction →
SQL Analytics + Power BI Dashboard + Streamlit App
```

## Data Preprocessing
- `TotalCharges` converted from text to numeric; blanks tied to `tenure=0`
  filled with 0 (real business meaning, not a guess).
- `customerID` dropped before modeling (no predictive value, avoids
  accidental leakage).
- `Churn` encoded Yes→1, No→0.
- Numeric features: median-imputed + `StandardScaler`.
- Categorical features: most-frequent-imputed + `OneHotEncoder(handle_unknown="ignore")`.
- All preprocessing lives inside a scikit-learn `Pipeline`/`ColumnTransformer`
  fitted only on the training fold — no leakage into test data.

## EDA
12 charts generated in `outputs/figures/` (churn distribution, churn by
contract/tenure/monthly charges/payment method/internet service/tech
support/online security, correlation heatmap, boxplots). Confirmed pattern:
**month-to-month contracts churn far more** than one/two-year contracts
(see `03_churn_by_contract.png`).

## Feature Engineering
- `TenureGroup`: bins tenure into 0-12/13-24/25-48/49-72 months — makes the
  known non-linear tenure/churn relationship directly available to linear
  models.
- `TotalServices`: count of optional add-ons subscribed (0-6) — proxy for
  customer engagement/stickiness.
- No leakage-risk features (e.g. no hand-built "risk score" duplicating
  Contract/PaymentMethod) were added.

## ML Models Compared (Stratified 5-Fold Cross-Validation, on training data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **GradientBoosting** | 0.803 | 0.662 | 0.530 | 0.588 | **0.848** |
| LogisticRegression | 0.749 | 0.518 | 0.796 | 0.627 | 0.846 |
| RandomForest | 0.776 | 0.569 | 0.633 | 0.600 | 0.824 |
| DecisionTree | 0.741 | 0.511 | 0.497 | 0.504 | 0.663 |

Full run saved in `outputs/metrics/cross_validation_results.csv`.

**Note on model choice:** GradientBoosting had the best ROC-AUC, but Logistic
Regression had far higher recall (0.796 vs 0.530). For a business that
values *catching every possible churner* over precision, Logistic Regression
could be the better real-world pick despite the lower ROC-AUC — this is a
business trade-off decision, not purely a metrics decision.

## Hyperparameter Tuning
`GridSearchCV` (5-fold, scoring=`roc_auc`) on GradientBoosting:
**Best params:** `{'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 200}`
**Best CV ROC-AUC:** 0.850 (full detail: `outputs/metrics/best_parameters.txt`)

## Final Model — Test Set Results (held-out 20%, 1,409 customers)
| Metric | Value |
|---|---|
| Accuracy | 0.798 |
| Precision | 0.655 |
| Recall | 0.503 |
| F1 Score | 0.569 |
| ROC-AUC | 0.846 |

Confusion matrix: TN=936, FP=99, FN=186, TP=188
(`outputs/figures/confusion_matrix.png`, `roc_curve.png`,
`precision_recall_curve.png`)

**Why recall matters here:** a missed churner (FN) usually costs more than a
false alarm (FP) — an unneeded retention offer is cheap, a lost customer is
lost recurring revenue. This model currently misses about half of actual
churners (186 of 374 in the test set), which is a real limitation worth
stating plainly, not hiding.

## Explainability
Feature importance (`outputs/figures/feature_importance.png`,
`outputs/metrics/feature_importance.csv`) shows what the model relied on to
make predictions — **this is association, not causation.** We do not claim
"Contract causes churn," only that contract type is strongly associated with
the model's prediction.

## Business Insights (from actual model output)
- 960 of 7,043 customers (13.6%) are flagged **HIGH risk** (probability ≥60%)
- **Potential monthly revenue at risk** from HIGH-risk customers: **$76,843.85**
  (this is potential, model-estimated risk, not a guaranteed future loss)
- Full breakdown: `outputs/predictions/churn_predictions.csv`

## Retention Strategy (risk-tier based)
- **HIGH risk:** priority retention outreach (call/offer) — small group,
  worth the highest-touch intervention
- **MEDIUM risk:** targeted automated engagement (email offers, service
  check-ins)
- **LOW risk:** standard loyalty/upsell touchpoints, no urgent action

## Power BI Dashboard
Data source: `dashboard/powerbi_data/churn_dashboard_data.csv` (merged
cleaned data + predictions, 7,043 rows). 3-page dashboard — see
`PROJECT_GUIDE.md` for exact build steps and DAX measures.

## Project Structure
```
Customer_Churn_ML/
├── Dataset/
│   ├── raw/customer_churn.csv
│   └── processed/  (cleaned data, train/test split, predictions)
├── notebooks/       (7 executed, error-free notebooks)
├── src/              (data_cleaning, preprocessing, eda, train_model,
│                       evaluate_model, predict)
├── models/churn_model.joblib
├── outputs/
│   ├── figures/     (16 charts)
│   ├── metrics/     (CV results, final metrics, feature importance)
│   └── predictions/churn_predictions.csv
├── sql/create_database.sql, churn_analysis.sql
├── dashboard/powerbi_data/churn_dashboard_data.csv
├── app.py            (Streamlit single-customer predictor)
├── requirements.txt, README.md, .gitignore, PROJECT_GUIDE.md
```

## Installation & How to Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python src/data_cleaning.py
python src/eda.py
python src/train_model.py
python src/evaluate_model.py
python src/predict.py
streamlit run app.py
```
See `PROJECT_GUIDE.md` for the full step-by-step (including SQL and Power BI).

## Limitations
- Static historical snapshot, not real-time customer behavior
- Feature importance shows association, not causation
- ~half of actual churners are currently missed (recall = 0.503) — a
  business could choose Logistic Regression instead to trade precision for
  higher recall, depending on retention-offer cost
- Dataset reflects one (fictional) telecom company's customer base; may not
  generalize directly to other industries

## Future Improvements
- Real-time scoring API (FastAPI)
- Model monitoring for data/concept drift
- Incorporate support-ticket text via NLP
- A/B test retention offers and measure actual ROI against model predictions



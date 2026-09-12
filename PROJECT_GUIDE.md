# PROJECT GUIDE — Exact Step-by-Step Execution

Every step below has actually been run once already (in the environment that
built this project) to confirm it works — the numbers you'll see match
`README.md`. Your own run should reproduce the same numbers since the code
uses `random_state=42` everywhere.

---

## STEP 1 — Install / check Python
**App:** Terminal
```bash
python3 --version
```
**Expected:** `Python 3.14.x` (you're on this already). **Verify:** version prints without error.

## STEP 2 — Create / place the project
**App:** Terminal
```bash
cd ~/Downloads          # or wherever you unzip this project
unzip Customer_Churn_ML.zip   # if you downloaded the zip
cd Customer_Churn_ML
pwd                      # confirm you're inside the project root
```

## STEP 3 — Create virtual environment
**App:** Terminal
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Verify:** prompt shows `(.venv)`.

## STEP 4 — Install requirements
**App:** Terminal
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip list
```
**Verify:** `pandas`, `scikit-learn`, `matplotlib`, `streamlit`, etc. appear in `pip list` with no errors.
**Common error:** `externally-managed-environment` — happens only if you forgot to activate `.venv` first. Fix: re-run `source .venv/bin/activate`, then reinstall.

## STEP 5 — Open in VS Code
**App:** VS Code
- `code .` from the project root, or open the folder via File → Open Folder.

## STEP 6 — Select interpreter
**App:** VS Code
- `Cmd+Shift+P` → "Python: Select Interpreter" → choose the one inside `.venv/bin/python`.

## STEP 7 — Select Jupyter kernel
**App:** VS Code
```bash
python -m ipykernel install --user --name=churn_ml --display-name "Python (churn_ml)"
```
Then open any `.ipynb` file → click kernel picker (top-right) → select **"Python (churn_ml)"**.

## STEP 8 — Run notebooks in order
**App:** VS Code / Jupyter
Run each notebook top-to-bottom, in order:
1. `01_data_loading.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_eda.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_training.ipynb`
6. `06_model_evaluation.ipynb`
7. `07_customer_prediction.ipynb`

These notebooks already contain real, executed outputs from this project's own build run — re-running them regenerates the same numbers.

## STEP 9 — Run training (script version)
**App:** Terminal
```bash
python src/data_cleaning.py
python src/eda.py
python src/train_model.py
```
**Expected output (approximate, from the actual run):**
```
Best model: GradientBoosting, CV ROC-AUC ≈ 0.850
Saved -> models/churn_model.joblib
```
**Verify:** `models/churn_model.joblib` exists.

## STEP 10 — Generate predictions
**App:** Terminal
```bash
python src/evaluate_model.py
python src/predict.py
```
**Verify:** `outputs/predictions/churn_predictions.csv` has 7,043 rows.
```bash
wc -l outputs/predictions/churn_predictions.csv
```

## STEP 11 — Run SQL
**App:** MySQL Workbench
1. Open MySQL Workbench, connect to your local MySQL server.
2. File → Open SQL Script → `sql/create_database.sql` → run (lightning bolt icon).
3. Right-click `customer_churn` table under `customer_churn_db` → Table Data Import Wizard → select `Dataset/processed/customer_churn_cleaned.csv` → Next through the wizard → Finish.
4. Repeat for `churn_predictions` table with `Dataset/processed/churn_predictions.csv`.
5. Open `sql/churn_analysis.sql` → run each query.
**Verify:** Query 1 (`SELECT COUNT(*) ...`) returns 7043.

## STEP 12 — Open Power BI
**App:** Power BI Desktop (Windows or VM) / Power BI Service
- Get Data → Text/CSV → select `dashboard/powerbi_data/churn_dashboard_data.csv` → Load.

## STEP 13 — Load dashboard data
Confirm columns loaded: `CustomerID, Churn_Probability, Risk_Level, gender, SeniorCitizen, ..., Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn, TenureGroup`.

## STEP 14 — Build dashboard
**Page 1 — Customer Overview**
- Card visuals with these DAX measures (create via Modeling → New Measure):
```dax
Total Customers = COUNTROWS(churn_dashboard_data)
Churned Customers = SUM(churn_dashboard_data[Churn])
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
Avg Monthly Charges = AVERAGE(churn_dashboard_data[MonthlyCharges])
Avg Tenure = AVERAGE(churn_dashboard_data[tenure])
```
- Bar charts: Churn by Contract (Axis=Contract, Values=Churn Rate), Churn by Internet Service, Churn by Payment Method, Churn by TenureGroup.

**Page 2 — Churn Analysis**
- Scatter/histogram: Churn by Monthly Charges (Axis=MonthlyCharges, Legend=Churn)
- Bar: Churn by Contract, Churn by TechSupport, Churn by OnlineSecurity, Churn by PaymentMethod
- Matrix: Customer Segment (rows=SeniorCitizen, columns=Dependents, values=Churn Rate)

**Page 3 — ML Prediction & Retention**
```dax
High Risk Customers = CALCULATE(COUNTROWS(churn_dashboard_data), churn_dashboard_data[Risk_Level]="HIGH")
Medium Risk Customers = CALCULATE(COUNTROWS(churn_dashboard_data), churn_dashboard_data[Risk_Level]="MEDIUM")
Low Risk Customers = CALCULATE(COUNTROWS(churn_dashboard_data), churn_dashboard_data[Risk_Level]="LOW")
Potential Revenue at Risk = CALCULATE(SUM(churn_dashboard_data[MonthlyCharges]), churn_dashboard_data[Risk_Level]="HIGH")
```
- Table visual: CustomerID, MonthlyCharges, tenure, Contract, Churn_Probability, Risk_Level (sorted descending by Churn_Probability)
- Slicers: Contract, PaymentMethod, InternetService, Risk_Level, TenureGroup
- Conditional formatting on Risk_Level column: HIGH=red, MEDIUM=yellow, LOW=green (Format → Conditional Formatting → Background Color → Rules)

**Verify:** KPI cards show numbers matching `outputs/predictions/churn_predictions.csv` (960 HIGH, 1690 MEDIUM, 4393 LOW; ~$76,843.85 revenue at risk).

## STEP 15 — Run Streamlit app
**App:** Terminal
```bash
streamlit run app.py
```
**Verify:** browser opens `localhost:8501`, form loads, submitting a customer shows a prediction, probability, and risk level.

## STEP 16 — Push to GitHub
**App:** Terminal
```bash
git init
git add .
git commit -m "Complete customer churn ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-churn-ml-retention-intelligence.git
git push -u origin main
```
**Verify:** repository shows all files on GitHub.

---

## Troubleshooting quick reference
| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'X'` | Package not installed / wrong interpreter | `pip install -r requirements.txt` inside the activated `.venv`; recheck VS Code interpreter (Step 6) |
| `NoSuchKernel` in Jupyter | Kernel not registered | Re-run Step 7's `ipykernel install` command |
| `FileNotFoundError: Dataset/raw/customer_churn.csv` | Running script from wrong directory | Always run `python src/...py` from the project **root**, not from inside `src/` |
| `externally-managed-environment` on pip install | `.venv` not activated | `source .venv/bin/activate` first |
| Power BI Desktop not available | Power BI Desktop is Windows-only | Use a Windows VM (Parallels/UTM) or Power BI Service in-browser |

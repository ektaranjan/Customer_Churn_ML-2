"""
eda.py
Generates and saves the required EDA charts into outputs/figures/.

RUN THIS (after data_cleaning.py):
    python src/eda.py
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "processed" / "customer_churn_cleaned.csv"
FIG_DIR = ROOT / "outputs" / "figures"

sns.set_style("whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"Saved {FIG_DIR / name}")


def main():
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: {DATA_PATH} not found. Run src/data_cleaning.py first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})

    # 01 churn distribution (counts)
    plt.figure(figsize=(5, 4))
    sns.countplot(x="ChurnLabel", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"], legend=False)
    plt.title("Churn Distribution (Count)")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    savefig("01_churn_distribution.png")

    # 02 churn percentage (pie)
    plt.figure(figsize=(5, 5))
    df["ChurnLabel"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#2E86AB", "#E63946"])
    plt.title("Churn Percentage")
    plt.ylabel("")
    savefig("02_churn_percentage.png")

    # 03 churn by contract
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Contract", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"])
    plt.title("Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.legend(title="Churn")
    savefig("03_churn_by_contract.png")

    # 04 churn by tenure (distribution)
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="ChurnLabel", multiple="stack",
                 palette=["#2E86AB", "#E63946"], bins=30)
    plt.title("Churn by Tenure (Months)")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    savefig("04_churn_by_tenure.png")

    # 05 churn by monthly charges
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="MonthlyCharges", hue="ChurnLabel", multiple="stack",
                 palette=["#2E86AB", "#E63946"], bins=30)
    plt.title("Churn by Monthly Charges")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Number of Customers")
    savefig("05_churn_by_monthly_charges.png")

    # 06 churn by payment method
    plt.figure(figsize=(7, 4))
    sns.countplot(y="PaymentMethod", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"])
    plt.title("Churn by Payment Method")
    plt.xlabel("Number of Customers")
    plt.ylabel("Payment Method")
    plt.legend(title="Churn")
    savefig("06_churn_by_payment_method.png")

    # 07 churn by internet service
    plt.figure(figsize=(6, 4))
    sns.countplot(x="InternetService", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"])
    plt.title("Churn by Internet Service Type")
    plt.xlabel("Internet Service")
    plt.ylabel("Number of Customers")
    plt.legend(title="Churn")
    savefig("07_churn_by_internet_service.png")

    # 08 churn by tech support
    plt.figure(figsize=(6, 4))
    sns.countplot(x="TechSupport", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"])
    plt.title("Churn by Tech Support Subscription")
    plt.xlabel("Tech Support")
    plt.ylabel("Number of Customers")
    plt.legend(title="Churn")
    savefig("08_churn_by_tech_support.png")

    # 09 churn by online security
    plt.figure(figsize=(6, 4))
    sns.countplot(x="OnlineSecurity", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"])
    plt.title("Churn by Online Security Subscription")
    plt.xlabel("Online Security")
    plt.ylabel("Number of Customers")
    plt.legend(title="Churn")
    savefig("09_churn_by_online_security.png")

    # 10 "age distribution" -> this dataset has SeniorCitizen (0/1), not raw
    # age, so we plot that distribution instead and label it accordingly.
    plt.figure(figsize=(5, 4))
    sns.countplot(x="SeniorCitizen", hue="SeniorCitizen", data=df, palette=["#2E86AB", "#457B9D"], legend=False)
    plt.title("Senior Citizen Distribution\n(dataset has no raw Age column)")
    plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    savefig("10_age_distribution.png")

    # 11 monthly charges boxplot
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="ChurnLabel", y="MonthlyCharges", hue="ChurnLabel", data=df, palette=["#2E86AB", "#E63946"], legend=False)
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges ($)")
    savefig("11_monthly_charges_boxplot.png")

    # 12 correlation heatmap (numeric columns only)
    plt.figure(figsize=(6, 5))
    numeric_df = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    savefig("12_correlation_heatmap.png")

    print("\nAll 12 EDA charts generated successfully.")


if __name__ == "__main__":
    main()

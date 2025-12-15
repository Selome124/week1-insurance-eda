import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ingestion import load_data

def run_eda(data_path, out_dir):
    df = load_data(data_path)

    # Example Plot 1 — Distribution of charges
    plt.figure(figsize=(8,5))
    sns.histplot(df['charges'], kde=True)
    plt.title("Distribution of Charges")
    plt.savefig(f"{out_dir}/charges_distribution.png")
    plt.close()

    # Example Plot 2 — BMI vs Charges
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="bmi", y="charges")
    plt.title("BMI vs Charges")
    plt.savefig(f"{out_dir}/bmi_vs_charges.png")
    plt.close()

    print("EDA plots saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    run_eda(args.data, args.out)

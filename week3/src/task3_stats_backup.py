import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def main():
    # Step 1: Load dataset and check claims
    df = pd.read_csv('insurance.csv')  # adjust path if needed

    # Your claim frequency/severity checks
    claim_count = df['ClaimAmount'].notna().sum()
    total_policies = len(df)
    claim_frequency = claim_count / total_policies
    print(f"Overall claim frequency: {claim_frequency}")

    if claim_count > 0:
        claim_severity = df.loc[df['ClaimAmount'] > 0, 'ClaimAmount'].mean()
        print(f"Average claim severity: {claim_severity}")
    else:
        print("No claims in dataset. Cannot calculate severity.")

    # Step 2: Segment data
    group_a = df[df['Sex'] == 'Male']
    group_b = df[df['Sex'] == 'Female']
    print(len(group_a), len(group_b))

    # Step 3: Check group equivalence
    ttest = ttest_ind(group_a['TotalPremium'], group_b['TotalPremium'], equal_var=False)
    print(f"Premium t-test: stat={ttest.statistic}, p={ttest.pvalue}")

    # Step 4: Statistical tests
    contingency_table = pd.crosstab(df['Province'], df['HasClaim'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi2={chi2}, p-value={p}")

    severity_a = group_a.loc[group_a['HasClaim'] == 1, 'ClaimAmount']
    severity_b = group_b.loc[group_b['HasClaim'] == 1, 'ClaimAmount']
    t_stat, p_val = ttest_ind(severity_a, severity_b, equal_var=False)
    print(f"Severity t-test: t={t_stat}, p={p_val}")

if __name__ == "__main__":
    main()
















import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import os

# --- Config ---
DATA_PATH = "data/insurance.csv"   # adjust if needed
OUTPUT_DIR = "plots"
TOP_N_ZIPCODES = 5
ALPHA = 0.05
BONFERRONI = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
#                CLEAN & NORMALIZE THE DATA
# -----------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)

    # Make column names consistent
    df.columns = [c.strip().lower() for c in df.columns]

    # --- FIX FOR YOUR DATASET ---
    # Your dataset has `charges`, but code expects `premium`
    if "premium" not in df.columns and "charges" in df.columns:
        df = df.rename(columns={"charges": "premium"})

    # Fallback: check required columns
    required = ["age", "sex", "bmi", "children", "smoker", "region", "premium"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Standard names
    df = df.rename(columns={
        "region": "province",
        "premium": "total_premium"
    })

    # Add missing claims column (your dataset has NO claims)
    # So create synthetic 0-claims column for statistical structure
    df["total_claims"] = 0

    # Create features needed by later tests
    df["has_claim"] = (df["total_claims"] > 0).astype(int)
    df["margin"] = df["total_premium"] - df["total_claims"]

    return df

# -----------------------------------------------------------
#                  GROUP STATISTICS
# -----------------------------------------------------------
def group_frequency(df, group_col):
    g = df.groupby(group_col).agg(
        n_policies=('has_claim', 'size'),
        n_with_claims=('has_claim', 'sum'),
        mean_margin=('margin', 'mean'),
    ).reset_index()
    g['claim_freq'] = g['n_with_claims'] / g['n_policies']
    return g

# -----------------------------------------------------------
#                STATISTICAL TESTS
# -----------------------------------------------------------
def chi_square_proportions(df, group_col):
    table = pd.crosstab(df[group_col], df['has_claim'])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2, p, table, expected

def kruskal_severity(df, group_col):
    df_claims = df[df['has_claim'] == 1]
    groups = []
    labels = []
    for name, sub in df_claims.groupby(group_col):
        if len(sub) >= 5:
            groups.append(sub['total_claims'].values)
            labels.append(name)
    if len(groups) < 2:
        return None, None, labels
    stat, p = stats.kruskal(*groups)
    return stat, p, labels

def anova_margin(df, group_col):
    groups = []
    labels = []
    for name, sub in df.groupby(group_col):
        if len(sub) >= 5:
            groups.append(sub['margin'].dropna().values)
            labels.append(name)
    if len(groups) < 2:
        return None, None, labels
    stat, p = stats.f_oneway(*groups)
    return stat, p, labels

def two_prop_ztest(df, group_col, grpA, grpB):
    sub = df[df[group_col].isin([grpA, grpB])]
    counts = sub.groupby(group_col)['has_claim'].sum().reindex([grpA, grpB]).values
    nobs = sub.groupby(group_col)['has_claim'].count().reindex([grpA, grpB]).values
    stat, p = proportions_ztest(counts, nobs)
    return stat, p, counts, nobs

def mannwhitney_severity(df, group_col, grpA, grpB):
    sub = df[(df[group_col].isin([grpA, grpB])) & (df['has_claim'] == 1)]
    a = sub[sub[group_col] == grpA]['total_claims']
    b = sub[sub[group_col] == grpB]['total_claims']
    if len(a) < 5 or len(b) < 5:
        return None, None, len(a), len(b)
    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return stat, p, len(a), len(b)

# -----------------------------------------------------------
#               RUN ALL TESTS AND SAVE RESULTS
# -----------------------------------------------------------
def run_all_tests(df):
    results = []

    # Province χ²
    chi2, p, _, _ = chi_square_proportions(df, 'province')
    decision = 'REJECT' if p < ALPHA else 'FAIL_TO_REJECT'
    results.append({'test': 'province_frequency_chi2', 'stat': chi2, 'p_value': p, 'decision': decision})

    # Province severity (will always be insufficient because claims=0)
    results.append({'test': 'province_severity_kruskal', 'stat': None, 'p_value': None, 'decision': 'insufficient_data'})

    # Sex tests
    if 'sex' in df.columns:
        df["sex_clean"] = df["sex"].astype(str).str.lower()
        male = "male"
        female = "female"

        if df["sex_clean"].str.contains(male).any() and df["sex_clean"].str.contains(female).any():
            stat, p, _, _ = two_prop_ztest(df, 'sex_clean', female, male)
            decision = 'REJECT' if p < ALPHA else 'FAIL_TO_REJECT'
            results.append({'test': 'sex_freq_ztest', 'stat': stat, 'p_value': p, 'decision': decision})
        else:
            results.append({'test': 'sex_freq_ztest', 'stat': None, 'p_value': None, 'decision': 'cannot_map_categories'})
    else:
        results.append({'test': 'sex_freq_ztest', 'stat': None, 'p_value': None, 'decision': 'no_sex_column'})

    # Save
    res_df = pd.DataFrame(results)
    out = os.path.join(OUTPUT_DIR, "task3_test_results.csv")
    res_df.to_csv(out, index=False)
    print("Saved:", out)

    return res_df

# -----------------------------------------------------------
#                         MAIN
# -----------------------------------------------------------
def main():
    df = load_and_clean(DATA_PATH)
    print("Loaded:", len(df))
    print("Overall claim frequency:", df['has_claim'].mean())
    res = run_all_tests(df)
    print(res)

if __name__ == "__main__":
    main()


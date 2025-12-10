import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------
#                 SIMULATE CLAIMS
# -----------------------------------------------------------
def simulate_claims(df, num_claims=50, claim_amount=1000):
    """Randomly assign claims."""
    df = df.copy()
    if num_claims > len(df):
        num_claims = len(df)
    np.random.seed(42)
    claim_indices = np.random.choice(df.index, size=num_claims, replace=False)
    df.loc[claim_indices, 'total_claims'] = claim_amount
    df['has_claim'] = (df['total_claims'] > 0).astype(int)
    df['margin'] = df['total_premium'] - df['total_claims']
    return df

def simulate_biased_claims(df):
    """Simulate claims with bias - smokers have higher claim probability."""
    df = df.copy()
    np.random.seed(42)
    
    smoker_indices = df[df['smoker'] == 'yes'].index
    non_smoker_indices = df[df['smoker'] == 'no'].index
    
    smoker_claims = int(0.3 * len(smoker_indices))
    non_smoker_claims = int(0.1 * len(non_smoker_indices))
    
    smoker_claim_idx = np.random.choice(smoker_indices, size=smoker_claims, replace=False)
    non_smoker_claim_idx = np.random.choice(non_smoker_indices, size=non_smoker_claims, replace=False)
    
    df.loc[list(smoker_claim_idx) + list(non_smoker_claim_idx), 'total_claims'] = 1000
    df['has_claim'] = (df['total_claims'] > 0).astype(int)
    df['margin'] = df['total_premium'] - df['total_claims']
    
    return df

# -----------------------------------------------------------
#                CLEAN & NORMALIZE DATA
# -----------------------------------------------------------
def load_and_clean(path):
    """Load and clean the insurance data with enhanced features."""
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Rename columns for consistency
    if "charges" in df.columns and "premium" not in df.columns:
        df = df.rename(columns={"charges": "premium"})
    
    df = df.rename(columns={"premium": "total_premium", "region": "province"})
    
    # Create synthetic zip codes based on province with some randomness
    np.random.seed(42)
    province_to_zip = {
        'southwest': ['SW001', 'SW002', 'SW003'],
        'southeast': ['SE001', 'SE002', 'SE003'],
        'northwest': ['NW001', 'NW002', 'NW003'],
        'northeast': ['NE001', 'NE002', 'NE003']
    }
    
    def assign_zip(province):
        province_lower = str(province).lower()
        for key in province_to_zip:
            if key in province_lower:
                return np.random.choice(province_to_zip[key])
        return 'UNKNOWN'
    
    df['zip_code'] = df['province'].apply(assign_zip)
    
    # Add synthetic claims if missing
    if "total_claims" not in df.columns:
        df["total_claims"] = 0
    
    df["has_claim"] = (df["total_claims"] > 0).astype(int)
    df["margin"] = df["total_premium"] - df["total_claims"]
    
    # Calculate claim severity (average claim amount for those with claims)
    df['claim_severity'] = df['total_claims'].where(df['has_claim'] == 1, np.nan)
    
    # Create additional demographic features for analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['18-30', '31-50', '51+'])
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                labels=['underweight', 'normal', 'overweight', 'obese'])
    
    return df

# -----------------------------------------------------------
#                STATISTICAL TESTS
# -----------------------------------------------------------
def chi_square_proportions(df, group_col, metric_col='has_claim'):
    """Chi-square test for categorical proportions."""
    table = pd.crosstab(df[group_col], df[metric_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, "insufficient_data"
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2, p, "success"

def two_prop_ztest(df, group_col, grpA, grpB, metric_col='has_claim'):
    """Z-test for two proportions."""
    sub = df[df[group_col].isin([grpA, grpB])]
    
    if len(sub[sub[group_col] == grpA]) < 2 or len(sub[sub[group_col] == grpB]) < 2:
        return None, None, None, None, "insufficient_data"
    
    counts = sub.groupby(group_col)[metric_col].sum().reindex([grpA, grpB]).values
    nobs = sub.groupby(group_col)[metric_col].count().reindex([grpA, grpB]).values
    
    p1 = counts[0] / nobs[0]
    p2 = counts[1] / nobs[1]
    
    p_pool = (counts[0] + counts[1]) / (nobs[0] + nobs[1])
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1]))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value, p1, p2, "success"

def t_test_means(df, group_col, grpA, grpB, metric_col='margin'):
    """T-test for means of two groups."""
    group_a = df[df[group_col] == grpA][metric_col].dropna()
    group_b = df[df[group_col] == grpB][metric_col].dropna()
    
    if len(group_a) < 2 or len(group_b) < 2:
        return None, None, None, None, "insufficient_data"
    
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    
    return t_stat, p_val, mean_a, mean_b, "success"

# -----------------------------------------------------------
#               HYPOTHESIS TESTING
# -----------------------------------------------------------
def test_all_hypotheses(df, ALPHA=0.05):
    """Run all required hypothesis tests."""
    print("\n" + "="*60)
    print("RUNNING REQUIRED HYPOTHESIS TESTS")
    print("="*60)
    
    results = {}
    
    # HYPOTHESIS 1: No risk differences across provinces
    print("\n1. H0: There are no risk differences across provinces")
    print("-" * 50)
    
    # Test Claim Frequency across provinces
    chi2, p, status = chi_square_proportions(df, 'province', 'has_claim')
    if status == "success":
        decision = 'REJECT' if p < ALPHA else 'FAIL_TO_REJECT'
        print(f"   Claim Frequency (Chi-square): chi2 = {chi2:.4f}, p = {p:.4f}, Decision: {decision}")
        results['province_frequency'] = {
            'test': 'chi-square',
            'statistic': chi2,
            'p_value': p,
            'decision': decision
        }
    else:
        print("   Claim Frequency: Insufficient data for test")
        results['province_frequency'] = {'test': 'chi-square', 'statistic': None, 'p_value': None, 'decision': 'insufficient_data'}
    
    # HYPOTHESIS 2: No risk differences between zip codes
    print("\n2. H0: There are no risk differences between zip codes")
    print("-" * 50)
    
    zip_codes = df['zip_code'].value_counts().head(2).index.tolist()
    if len(zip_codes) >= 2:
        zipA, zipB = zip_codes[0], zip_codes[1]
        print(f"   Comparing: {zipA} vs {zipB}")
        
        z_stat, p_val, p1, p2, status = two_prop_ztest(df, 'zip_code', zipA, zipB, 'has_claim')
        if status == "success":
            decision = 'REJECT' if p_val < ALPHA else 'FAIL_TO_REJECT'
            diff_pct = abs(p1 - p2) * 100
            print(f"   Claim Frequency (Z-test): z = {z_stat:.4f}, p = {p_val:.4f}")
            print(f"   Claim Rates: {zipA} = {p1:.2%}, {zipB} = {p2:.2%} (Difference: {diff_pct:.1f}%)")
            print(f"   Decision: {decision}")
            results['zipcode_frequency'] = {
                'test': 'z-test',
                'statistic': z_stat,
                'p_value': p_val,
                'claim_rate_A': p1,
                'claim_rate_B': p2,
                'decision': decision
            }
        else:
            print("   Claim Frequency: Insufficient data for test")
            results['zipcode_frequency'] = {'test': 'z-test', 'statistic': None, 'p_value': None, 'decision': 'insufficient_data'}
    else:
        print("   Insufficient zip codes for comparison")
        results['zipcode_frequency'] = {'test': 'z-test', 'statistic': None, 'p_value': None, 'decision': 'insufficient_zip_codes'}
    
    # HYPOTHESIS 3: No significant margin difference between zip codes
    print("\n3. H0: There is no significant margin difference between zip codes")
    print("-" * 50)
    
    if len(zip_codes) >= 2:
        zipA, zipB = zip_codes[0], zip_codes[1]
        t_stat, p_val, mean_a, mean_b, status = t_test_means(df, 'zip_code', zipA, zipB, 'margin')
        if status == "success":
            decision = 'REJECT' if p_val < ALPHA else 'FAIL_TO_REJECT'
            diff = mean_a - mean_b
            print(f"   Margin Difference (T-test): t = {t_stat:.4f}, p = {p_val:.4f}")
            print(f"   Average Margin: {zipA} = ${mean_a:.2f}, {zipB} = ${mean_b:.2f} (Difference: ${diff:.2f})")
            print(f"   Decision: {decision}")
            results['zipcode_margin'] = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_val,
                'mean_margin_A': mean_a,
                'mean_margin_B': mean_b,
                'margin_difference': diff,
                'decision': decision
            }
        else:
            print("   Margin Analysis: Insufficient data for test")
            results['zipcode_margin'] = {'test': 't-test', 'statistic': None, 'p_value': None, 'decision': 'insufficient_data'}
    
    # HYPOTHESIS 4: No significant risk difference between Women and Men
    print("\n4. H0: There is no significant risk difference between Women and Men")
    print("-" * 50)
    
    if 'sex' in df.columns:
        df['sex_clean'] = df['sex'].astype(str).str.lower().str.strip()
        valid_genders = df['sex_clean'].isin(['male', 'female'])
        
        if valid_genders.sum() > 100:  # Require sufficient data
            gender_data = df[valid_genders]
            
            z_stat, p_val, p_female, p_male, status = two_prop_ztest(gender_data, 'sex_clean', 'female', 'male', 'has_claim')
            if status == "success":
                decision = 'REJECT' if p_val < ALPHA else 'FAIL_TO_REJECT'
                diff_pct = abs(p_female - p_male) * 100
                print(f"   Claim Frequency (Z-test): z = {z_stat:.4f}, p = {p_val:.4f}")
                print(f"   Claim Rates: Female = {p_female:.2%}, Male = {p_male:.2%} (Difference: {diff_pct:.1f}%)")
                print(f"   Decision: {decision}")
                results['gender_frequency'] = {
                    'test': 'z-test',
                    'statistic': z_stat,
                    'p_value': p_val,
                    'claim_rate_female': p_female,
                    'claim_rate_male': p_male,
                    'decision': decision
                }
            else:
                print("   Claim Frequency: Insufficient data for test")
                results['gender_frequency'] = {'test': 'z-test', 'statistic': None, 'p_value': None, 'decision': 'insufficient_data'}
        else:
            print("   Insufficient gender data for reliable test")
            results['gender_frequency'] = {'test': 'z-test', 'statistic': None, 'p_value': None, 'decision': 'insufficient_gender_data'}
    else:
        print("   No gender column in dataset")
        results['gender_frequency'] = {'test': 'z-test', 'statistic': None, 'p_value': None, 'decision': 'no_gender_column'}
    
    return results

# -----------------------------------------------------------
#               ADDITIONAL RISK ANALYSIS
# -----------------------------------------------------------
def additional_risk_analysis(df, ALPHA=0.05):
    """Run additional risk factor analysis."""
    print("\n" + "="*60)
    print("ADDITIONAL RISK FACTOR ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Smoker status analysis
    print("\n* Smoker Status Analysis:")
    if 'smoker' in df.columns:
        smoker_counts = df.groupby('smoker')['has_claim'].agg(['sum', 'count', 'mean'])
        print(f"   Claim Frequency: Smokers = {smoker_counts.loc['yes', 'mean']:.2%}, Non-smokers = {smoker_counts.loc['no', 'mean']:.2%}")
        
        z_stat, p_val, p_yes, p_no, status = two_prop_ztest(df, 'smoker', 'yes', 'no', 'has_claim')
        if status == "success":
            decision = 'REJECT' if p_val < ALPHA else 'FAIL_TO_REJECT'
            results['smoker_frequency'] = {
                'test': 'z-test',
                'statistic': z_stat,
                'p_value': p_val,
                'claim_rate_smoker': p_yes,
                'claim_rate_nonsmoker': p_no,
                'decision': decision
            }
            print(f"   Statistical Test: z = {z_stat:.4f}, p = {p_val:.4f}, Decision: {decision}")
    
    # Age group analysis
    print("\n* Age Group Analysis:")
    age_claims = df.groupby('age_group')['has_claim'].mean()
    for group, rate in age_claims.items():
        print(f"   {group}: {rate:.2%}")
    
    chi2, p, status = chi_square_proportions(df, 'age_group', 'has_claim')
    if status == "success":
        decision = 'REJECT' if p < ALPHA else 'FAIL_TO_REJECT'
        results['age_group_frequency'] = {
            'test': 'chi-square',
            'statistic': chi2,
            'p_value': p,
            'decision': decision
        }
        print(f"   Statistical Test: chi2 = {chi2:.4f}, p = {p:.4f}, Decision: {decision}")
    
    # BMI category analysis
    print("\n* BMI Category Analysis:")
    bmi_claims = df.groupby('bmi_category')['has_claim'].mean()
    for category, rate in bmi_claims.items():
        print(f"   {category}: {rate:.2%}")
    
    chi2, p, status = chi_square_proportions(df, 'bmi_category', 'has_claim')
    if status == "success":
        decision = 'REJECT' if p < ALPHA else 'FAIL_TO_REJECT'
        results['bmi_frequency'] = {
            'test': 'chi-square',
            'statistic': chi2,
            'p_value': p,
            'decision': decision
        }
        print(f"   Statistical Test: chi2 = {chi2:.4f}, p = {p:.4f}, Decision: {decision}")
    
    return results

# -----------------------------------------------------------
#               BUSINESS RECOMMENDATIONS
# -----------------------------------------------------------
def generate_recommendations(main_results, additional_results, df):
    """Generate business recommendations based on statistical findings."""
    print("\n" + "="*60)
    print("BUSINESS RECOMMENDATIONS & SEGMENTATION STRATEGY")
    print("="*60)
    
    recommendations = []
    
    # Check main hypotheses
    if 'zipcode_frequency' in main_results and main_results['zipcode_frequency']['decision'] == 'REJECT':
        p1 = main_results['zipcode_frequency'].get('claim_rate_A', 0)
        p2 = main_results['zipcode_frequency'].get('claim_rate_B', 0)
        diff_pct = abs(p1 - p2) * 100
        recommendations.append(
            f"REJECT H0 for zip code risk differences (p = {main_results['zipcode_frequency']['p_value']:.4f}). "
            f"Implement granular geographic pricing: Adjust premiums by up to {diff_pct:.0f}% based on zip code risk profiles."
        )
    
    if 'zipcode_margin' in main_results and main_results['zipcode_margin']['decision'] == 'REJECT':
        diff = abs(main_results['zipcode_margin'].get('margin_difference', 0))
        recommendations.append(
            f"REJECT H0 for zip code margin differences (p = {main_results['zipcode_margin']['p_value']:.4f}). "
            f"Standardize profitability: Adjust premiums to achieve consistent ${diff:.0f} margin across regions."
        )
    
    if 'province_frequency' in main_results and main_results['province_frequency']['decision'] == 'REJECT':
        recommendations.append(
            f"REJECT H0 for province risk differences (p = {main_results['province_frequency']['p_value']:.4f}). "
            f"Implement regional risk adjustments to account for varying claim patterns across provinces."
        )
    
    if 'gender_frequency' in main_results and main_results['gender_frequency']['decision'] == 'REJECT':
        p_female = main_results['gender_frequency'].get('claim_rate_female', 0)
        p_male = main_results['gender_frequency'].get('claim_rate_male', 0)
        diff_pct = abs(p_female - p_male) * 100
        higher = "women" if p_female > p_male else "men"
        recommendations.append(
            f"REJECT H0 for gender risk differences (p = {main_results['gender_frequency']['p_value']:.4f}). "
            f"Consider gender-based pricing: {higher.capitalize()} show {diff_pct:.1f}% higher claim frequency."
        )
    
    # Check additional risk factors
    if 'smoker_frequency' in additional_results and additional_results['smoker_frequency']['decision'] == 'REJECT':
        p_yes = additional_results['smoker_frequency'].get('claim_rate_smoker', 0)
        p_no = additional_results['smoker_frequency'].get('claim_rate_nonsmoker', 0)
        risk_ratio = p_yes / p_no if p_no > 0 else 0
        recommendations.append(
            f"Smokers are {risk_ratio:.1f}x more likely to file claims ({p_yes:.1%} vs {p_no:.1%}). "
            f"Implement substantial premium surcharge for smokers."
        )
    
    # General recommendations
    if not recommendations:
        recommendations.append(
            "No statistically significant differences found in main hypotheses. "
            "Current premium structure appears appropriate based on available data."
        )
    
    # Always add these recommendations
    recommendations.append("Conduct A/B testing before implementing any premium adjustments.")
    recommendations.append("Review risk models quarterly with updated claim data.")
    recommendations.append("Consider multi-factor segmentation combining geography, demographics, and lifestyle factors.")
    
    # Print recommendations
    print("\nRECOMMENDED ACTIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    # Key metrics
    print("\n" + "-"*40)
    print("KEY PERFORMANCE INDICATORS:")
    print("-"*40)
    print(f"* Overall Claim Frequency: {df['has_claim'].mean():.2%}")
    print(f"* Average Premium: ${df['total_premium'].mean():.2f}")
    print(f"* Average Margin: ${df['margin'].mean():.2f}")
    print(f"* Policies with Claims: {df['has_claim'].sum()} / {len(df)}")
    print(f"* Total Revenue: ${df['total_premium'].sum():,.2f}")
    print(f"* Total Claims Cost: ${df['total_claims'].sum():,.2f}")
    print(f"* Overall Profit Margin: {(df['margin'].sum() / df['total_premium'].sum() * 100):.1f}%")
    
    return recommendations

# -----------------------------------------------------------
#               VISUALIZATION
# -----------------------------------------------------------
def create_visualizations(df, output_dir):
    """Create visualization plots."""
    print(f"\nCreating visualizations in '{output_dir}'...")
    
    # 1. Province claim frequencies
    plt.figure(figsize=(10, 6))
    province_claims = df.groupby('province')['has_claim'].mean().sort_values()
    province_claims.plot(kind='bar', color='skyblue')
    plt.title('Claim Frequency by Province', fontsize=14, fontweight='bold')
    plt.ylabel('Claim Frequency', fontsize=12)
    plt.xlabel('Province', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'province_claim_frequencies.png'))
    plt.close()
    
    # 2. Gender comparison
    if 'sex' in df.columns:
        plt.figure(figsize=(8, 6))
        df['sex_clean'] = df['sex'].astype(str).str.lower().str.strip()
        gender_claims = df[df['sex_clean'].isin(['male', 'female'])].groupby('sex_clean')['has_claim'].mean()
        gender_claims.plot(kind='bar', color=['pink', 'lightblue'])
        plt.title('Claim Frequency by Gender', fontsize=14, fontweight='bold')
        plt.ylabel('Claim Frequency', fontsize=12)
        plt.xlabel('Gender', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gender_claim_frequencies.png'))
        plt.close()
    
    # 3. Age group analysis
    plt.figure(figsize=(8, 6))
    age_claims = df.groupby('age_group')['has_claim'].mean()
    age_claims.plot(kind='bar', color='lightgreen')
    plt.title('Claim Frequency by Age Group', fontsize=14, fontweight='bold')
    plt.ylabel('Claim Frequency', fontsize=12)
    plt.xlabel('Age Group', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_group_claims.png'))
    plt.close()
    
    # 4. Smoker status comparison
    if 'smoker' in df.columns:
        plt.figure(figsize=(8, 6))
        smoker_claims = df.groupby('smoker')['has_claim'].mean()
        smoker_claims.plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Claim Frequency by Smoker Status', fontsize=14, fontweight='bold')
        plt.ylabel('Claim Frequency', fontsize=12)
        plt.xlabel('Smoker Status', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'smoker_claims.png'))
        plt.close()
    
    print(f"Visualizations saved to '{output_dir}'")

# -----------------------------------------------------------
#               MAIN EXECUTION
# -----------------------------------------------------------
def main():
    # Configuration
    DATA_PATH = r"C:\Users\deres\OneDrive\Desktop\week3\data\insurance.csv"
    OUTPUT_DIR = r"plots"
    ALPHA = 0.05
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("TASK 3: STATISTICAL HYPOTHESIS TESTING FOR RISK SEGMENTATION")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[STEP 1] LOADING AND PREPARING DATA...")
    df = load_and_clean(DATA_PATH)
    print(f"   * Records loaded: {len(df):,}")
    print(f"   * Columns available: {len(df.columns)}")
    print(f"   * Initial claim frequency: {df['has_claim'].mean():.2%}")
    
    # Step 2: Simulate claims with smoker bias
    print("\n[STEP 2] SIMULATING CLAIMS DATA...")
    df = simulate_biased_claims(df)
    print(f"   * Final claim frequency: {df['has_claim'].mean():.2%}")
    print(f"   * Total claims simulated: {df['has_claim'].sum():,}")
    print(f"   * Average claim amount: ${df[df['has_claim']==1]['total_claims'].mean():.2f}")
    
    # Step 3: Run required hypothesis tests
    main_results = test_all_hypotheses(df, ALPHA)
    
    # Step 4: Run additional risk analysis
    additional_results = additional_risk_analysis(df, ALPHA)
    
    # Step 5: Generate business recommendations
    recommendations = generate_recommendations(main_results, additional_results, df)
    
    # Step 6: Create visualizations
    create_visualizations(df, OUTPUT_DIR)
    
    # Step 7: Save all results
    print("\n[STEP 7] SAVING RESULTS...")
    
    # Save combined results
    all_results = {**main_results, **additional_results}
    results_list = []
    
    for key, result in all_results.items():
        result_dict = {'analysis': key, **result}
        results_list.append(result_dict)
    
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(OUTPUT_DIR, "task3_comprehensive_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"   * Statistical results saved to: {results_path}")
    
    # Save executive summary
    summary_path = os.path.join(OUTPUT_DIR, "task3_executive_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EXECUTIVE SUMMARY: RISK SEGMENTATION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Dataset: {len(df):,} policies\n")
        f.write(f"Overall Claim Frequency: {df['has_claim'].mean():.2%}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*30 + "\n")
        
        for i, rec in enumerate(recommendations[:5], 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\nRECOMMENDED SEGMENTATION STRATEGY:\n")
        f.write("-"*35 + "\n")
        f.write("1. Geographic Segmentation: Implement zip-code based pricing\n")
        f.write("2. Demographic Segmentation: Consider age and gender factors\n")
        f.write("3. Lifestyle Segmentation: Apply smoker premium adjustments\n")
        f.write("4. Continuous Monitoring: Review quarterly with new data\n\n")
        
        f.write("BUSINESS IMPACT:\n")
        f.write("-"*20 + "\n")
        f.write(f"* Total Premium Revenue: ${df['total_premium'].sum():,.2f}\n")
        f.write(f"* Total Claims Cost: ${df['total_claims'].sum():,.2f}\n")
        f.write(f"* Net Profit Margin: {(df['margin'].sum() / df['total_premium'].sum() * 100):.1f}%\n")
    
    print(f"   * Executive summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review the comprehensive_results.csv for detailed statistics")
    print("2. Check the visualizations in the 'plots' folder")
    print("3. Implement recommended segmentation strategy with A/B testing")
    print("4. Monitor results and adjust models quarterly")

if __name__ == "__main__":
    main()
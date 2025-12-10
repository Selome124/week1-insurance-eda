# src/task3_stats.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import itertools

# config
DATA_PATH = "data/insurance.csv"
OUTPUT_SUMMARY = "reports/task3_summary.csv"
ALPHA = 0.05

def load_data(path):
    df = pd.read_csv(path)
    # ensure numeric claim_amount and premium
    df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce').fillna(0.0)
    df['premium'] = pd.to_numeric(df['premium'], errors='coerce').fillna(0.0)
    return df

def compute_kpis(df, groupcol):
    # per-group metrics
    g = df.groupby(groupcol).agg(
        policies=('policy_id','nunique'),
        policies_with_claim=('claim_amount', lambda x: (x>0).sum()),
        total_claims=('claim_amount','sum'),
        avg_claim_given_claim=('claim_amount', lambda x: x[x>0].mean() if (x>0).any() else np.nan),
        total_premium=('premium','sum')
    ).reset_index()
    g['claim_frequency'] = g['policies_with_claim'] / g['policies']
    g['margin'] = g['total_premium'] - g['total_claims']
    g['avg_margin_per_policy'] = g['margin'] / g['policies']
    return g

def chi2_test_claim_frequency(df, groupcol):
    # contingency table of claim/no-claim by group
    df['has_claim'] = (df['claim_amount']>0).astype(int)
    ct = pd.crosstab(df[groupcol], df['has_claim'])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    return chi2, p, ct

def ttest_severity_between_groups(df, groupcol, groupA, groupB):
    # severity: claim amounts >0
    a = df[(df[groupcol]==groupA) & (df['claim_amount']>0)]['claim_amount']
    b = df[(df[groupcol]==groupB) & (df['claim_amount']>0)]['claim_amount']
    # log transform to reduce skew if desired
    # a_log, b_log = np.log1p(a), np.log1p(b)
    # use Welch's t-test (doesn't assume equal var)
    tstat, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False, nan_policy='omit')
    # also return medians and counts
    return tstat, p, a.count(), b.count(), a.mean(), b.mean()

def anova_severity(df, groupcol):
    groups = [grp[grp['claim_amount']>0]['claim_amount'].values for _,grp in df.groupby(groupcol)]
    fstat, p = stats.f_oneway(*groups)
    return fstat, p

def test_gender(df):
    # frequency
    chi2, p_freq, ct = chi2_test_claim_frequency(df, 'gender')
    # severity (t-test women vs men)
    genders = df['gender'].dropna().unique()
    if len(genders)==2:
        gA, gB = genders[0], genders[1]
        tstat, p_sev, *_ = ttest_severity_between_groups(df, 'gender', gA, gB)
    else:
        tstat, p_sev = None, None
    return {'chi2_freq':chi2, 'p_freq':p_freq, 'chi_table':ct, 'tstat_sev':tstat, 'p_sev':p_sev}

def pairwise_zip_tests(df, max_pairs=50):
    # caution: many zip codes -> many tests
    zips = df['zip'].dropna().unique()
    results = []
    pairs = list(itertools.combinations(zips, 2))
    # optional: limit number of pairs to test to avoid explosion
    for a,b in pairs[:max_pairs]:
        tstat, p, na, nb, ma, mb = ttest_severity_between_groups(df, 'zip', a, b)
        results.append({'zip_a':a,'zip_b':b,'tstat':tstat,'p':p,'n_a':na,'n_b':nb,'mean_a':ma,'mean_b':mb})
    return pd.DataFrame(results)

def main():
    df = load_data(DATA_PATH)

    # H0: no risk differences across provinces (claim frequency & severity)
    print("=== Province KPIs ===")
    prov_kpis = compute_kpis(df, 'province')
    prov_kpis.to_csv('reports/province_kpis.csv', index=False)
    chi2_prov, p_prov, ct_prov = chi2_test_claim_frequency(df, 'province')
    print("Province claim-frequency chi2 p:", p_prov)

    # severity ANOVA across provinces (only on policies with claims)
    f_prov, p_prov_sev = anova_severity(df, 'province')
    print("Province severity ANOVA p:", p_prov_sev)

    # H0: no risk differences between zip codes
    print("=== Zip KPIs ===")
    zip_kpis = compute_kpis(df, 'zip')
    zip_kpis.to_csv('reports/zip_kpis.csv', index=False)
    chi2_zip, p_zip, ct_zip = chi2_test_claim_frequency(df, 'zip')
    print("Zip claim-frequency chi2 p:", p_zip)
    f_zip, p_zip_sev = anova_severity(df, 'zip')
    print("Zip severity ANOVA p:", p_zip_sev)

    # H0: no significant margin difference between zip codes
    # compare avg_margin_per_policy via ANOVA
    margins = [grp['premium'].sum() - grp['claim_amount'].sum() for _,grp in df.groupby('zip')]
    # better: create per-policy average margin per zip
    perzip_margin = df.groupby('zip').apply(lambda g: (g['premium'].sum() - g['claim_amount'].sum())/g['policy_id'].nunique())
    # ANOVA across per-zip averages:
    # (use one-way ANOVA on perzip_margin.values)
    # we'll use kruskal if non-normal
    try:
        f_margin, p_margin = stats.f_oneway(*[grp['avg_margin_per_policy'].dropna().values for name,grp in compute_kpis(df,'zip').groupby('zip')])
    except Exception:
        f_margin, p_margin = None, None

    print("Zip margin ANOVA p:", p_margin)

    # H0: no significant risk difference between Women and Men
    gender_res = test_gender(df)
    print("Gender claim-frequency p:", gender_res['p_freq'])
    print("Gender severity p:", gender_res['p_sev'])

    # Pairwise zip tests (severity) -- limit pairs
    pairwise = pairwise_zip_tests(df, max_pairs=200)
    pairwise.to_csv('reports/zip_pairwise_severity_tests.csv', index=False)

    # Summarize findings
    summary = [
        {'hypothesis':'provinces_claim_freq','p_value':p_prov,'reject': p_prov < ALPHA},
        {'hypothesis':'provinces_severity','p_value':p_prov_sev,'reject': p_prov_sev < ALPHA},
        {'hypothesis':'zip_claim_freq','p_value':p_zip,'reject': p_zip < ALPHA},
        {'hypothesis':'zip_severity','p_value':p_zip_sev,'reject': p_zip_sev < ALPHA},
        {'hypothesis':'zip_margin','p_value':p_margin,'reject': (p_margin is not None and p_margin < ALPHA)},
        {'hypothesis':'gender_claim_freq','p_value':gender_res['p_freq'],'reject': gender_res['p_freq'] < ALPHA},
        {'hypothesis':'gender_severity','p_value':gender_res['p_sev'] if gender_res['p_sev'] is not None else np.nan,'reject': (gender_res['p_sev'] is not None and gender_res['p_sev'] < ALPHA)}
    ]
    pd.DataFrame(summary).to_csv(OUTPUT_SUMMARY, index=False)
    print("Saved summary to", OUTPUT_SUMMARY)

if __name__ == "__main__":
    main()

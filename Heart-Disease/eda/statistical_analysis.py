# ============================================================
# eda/statistical_analysis.py
# ------------------------------------------------------------
# Reusable statistical testing and prevalence functions for
# survey-based epidemiological analysis on CDC BRFSS.
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats

def calculate_prevalence_by_group(df: pd.DataFrame, group_col: str, target_col: str = "_MICHD") -> pd.DataFrame:
    """
    Computes sample respondent counts, cases, and percentage prevalence
    stratified by a demographic or risk factor category.
    """
    summary = df.groupby(group_col)[target_col].agg(
        Total_Respondents="count",
        Positive_Cases="sum",
        Prevalence_Percent=lambda x: round((x.sum() / x.count()) * 100, 2)
    ).reset_index()
    return summary


def compute_chi_square_associations(df: pd.DataFrame, feature_cols: list,
                                    target_col: str = "_MICHD") -> pd.DataFrame:
    """
    Computes Chi-Square tests of independence, p-values, and Cramér's V
    association coefficients for categorical predictors against the target.
    """
    results = []
    for col in feature_cols:
        if col in df.columns:
            valid_df = df[[col, target_col]].dropna()
            contingency = pd.crosstab(valid_df[col], valid_df[target_col])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            k = min(contingency.shape)
            cramers_v = np.sqrt(chi2 / (n * (k - 1))) if (k > 1 and n > 0) else 0

            results.append({
                "Feature": col,
                "Chi2_Statistic": round(chi2, 2),
                "Degrees_of_Freedom": dof,
                "p_value": p_val,
                "Significant_p_001": p_val < 0.001,
                "Cramers_V": round(cramers_v, 4)
            })

    return pd.DataFrame(results).sort_values("Cramers_V", ascending=False).reset_index(drop=True)


def calculate_relative_risk_ratios(df: pd.DataFrame, condition_dict: dict,
                                   target_col: str = "_MICHD") -> pd.DataFrame:
    """
    Calculates empirical Relative Risk (RR) ratios for binary exposures:
      RR = Prevalence(Exposed) / Prevalence(Unexposed)
    """
    rr_data = []
    for col, (label, exposed_val) in condition_dict.items():
        if col in df.columns:
            exposed = df[df[col] == exposed_val]
            unexposed = df[(df[col].notnull()) & (df[col] != exposed_val)]

            p_exposed = exposed[target_col].mean() if len(exposed) > 0 else 0
            p_unexposed = unexposed[target_col].mean() if len(unexposed) > 0 else 0
            rr = p_exposed / p_unexposed if p_unexposed > 0 else np.nan

            rr_data.append({
                "Risk_Factor": label,
                "Prevalence_Exposed_Pct": round(p_exposed * 100, 2),
                "Prevalence_Unexposed_Pct": round(p_unexposed * 100, 2),
                "Relative_Risk_Ratio": round(rr, 2)
            })

    return pd.DataFrame(rr_data).sort_values("Relative_Risk_Ratio", ascending=False).reset_index(drop=True)

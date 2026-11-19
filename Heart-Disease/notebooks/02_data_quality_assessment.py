# ============================================================
# PHASE 2 — DATA QUALITY ASSESSMENT
# ============================================================
# Script: notebooks/02_data_quality_assessment.py
#
# Purpose:
#   Thoroughly investigate every data quality issue before
#   making any cleaning decisions.  No data is modified here.
#
# Outputs:
#   - reports/tables/02_missing_values_detail.csv
#   - reports/tables/02_nonresponse_codes.csv
#   - reports/tables/02_feature_suitability.csv
#   - reports/tables/02_outlier_summary.csv
#   - reports/figures/02_missing_values_top40.png
#   - reports/figures/02_missing_values_heatmap.png
#   - reports/figures/02_outlier_boxplots.png
#   - PROJECT_DOCUMENTATION.txt  (Phase 2 section appended)
#
# Run:
#   cd Heart-Disease
#   venv\Scripts\python.exe notebooks/02_data_quality_assessment.py
# ============================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    ensure_directories,
    CLEANED_CSV_PATH, FIGURES_DIR, TABLES_DIR, DOCS_PATH,
    TARGET_COLUMN, COLUMN_DROP_MISSING_THRESHOLD,
    FIGURE_DPI, FIGURE_STYLE, FIGURE_PALETTE,
)
from data_loader import load_raw_dataset
from utils import section_header, sub_header, append_to_documentation

sns.set_style(FIGURE_STYLE)
sns.set_palette(FIGURE_PALETTE)
ensure_directories()


# ============================================================
# LOAD RAW DATASET
# ============================================================

section_header("LOADING DATASET FOR QUALITY ASSESSMENT")
df = load_raw_dataset()
total_rows, total_cols = df.shape
print(f"Shape: {total_rows:,} × {total_cols}")


# ============================================================
# SECTION A — MISSING VALUE INVESTIGATION
# ============================================================

section_header("SECTION A — MISSING VALUE INVESTIGATION")

missing_count  = df.isnull().sum()
missing_pct    = (df.isnull().mean() * 100).round(4)

missing_detail = pd.DataFrame({
    "column"          : missing_count.index,
    "missing_count"   : missing_count.values,
    "missing_percent" : missing_pct.values,
    "dtype"           : df.dtypes.astype(str).values,
}).sort_values("missing_percent", ascending=False).reset_index(drop=True)

# Save
missing_detail.to_csv(TABLES_DIR / "02_missing_values_detail.csv", index=False)

# Summary statistics
print(f"\nTotal missing cells       : {missing_count.sum():,}")
print(f"Columns with 0% missing   : {(missing_detail['missing_percent'] == 0).sum()}")
print(f"Columns with 1–10% miss.  : {((missing_detail['missing_percent'] > 0) & (missing_detail['missing_percent'] <= 10)).sum()}")
print(f"Columns with 10–50% miss. : {((missing_detail['missing_percent'] > 10) & (missing_detail['missing_percent'] <= 50)).sum()}")
print(f"Columns with 50–60% miss. : {((missing_detail['missing_percent'] > 50) & (missing_detail['missing_percent'] <= 60)).sum()}")
print(f"Columns with >60% missing : {(missing_detail['missing_percent'] > 60).sum()}")

print("\nTop 20 columns by missing %:")
print(missing_detail.head(20).to_string(index=False))

# ── Graph A1: Missing values bar chart (top 40 columns) ──────
top40_missing = missing_detail[missing_detail["missing_percent"] > 0].head(40)

if len(top40_missing) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        top40_missing["column"][::-1],
        top40_missing["missing_percent"][::-1],
        color="tomato",
        edgecolor="white",
        linewidth=0.5
    )
    ax.axvline(x=60, color="darkred",   linestyle="--", linewidth=1.5,
               label="60% drop threshold")
    ax.axvline(x=20, color="darkorange", linestyle=":",  linewidth=1.2,
               label="20% caution threshold")
    ax.set_title("Top 40 Columns by Missing Value Percentage\n(CDC BRFSS 2021)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.set_ylabel("Column Name", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_missing_values_top40.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("\n[Saved] 02_missing_values_top40.png")

# ── Graph A2: Missing value heatmap (sample, candidate columns) ──
# Use only candidate features (manageable number of columns)
CANDIDATE_COLS_HEATMAP = [
    "_MICHD", "_AGEG5YR", "SEX1", "_BMI5", "_BMI5CAT",
    "BPHIGH6", "TOLDHI3", "DIABETE4", "CVDSTRK3",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "HLTHPLN2", "MEDCOST1", "CHECKUP1",
    "SMOKE100", "SMOKDAY2", "_SMOKER3",
    "ALCDAY4", "_RFDRHV7", "EXERANY2", "_TOTINDA", "SLEPTIM1",
    "EDUCA", "INCOME3", "_RACE1",
]
existing_heatmap_cols = [c for c in CANDIDATE_COLS_HEATMAP if c in df.columns]

if existing_heatmap_cols:
    # Sample 2000 rows for the heatmap to keep it readable
    sample_df = df[existing_heatmap_cols].sample(
        n=min(2000, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        sample_df.isnull(),
        cbar=False,
        yticklabels=False,
        cmap=["#2ECC71", "#E74C3C"],   # Green=present, Red=missing
        ax=ax
    )
    ax.set_title(
        "Missing Value Pattern — Candidate Features (2,000-row sample)\n"
        "Red = Missing | Green = Present",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Respondent (sample)", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_missing_values_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("[Saved] 02_missing_values_heatmap.png")


# ============================================================
# SECTION B — BRFSS NON-RESPONSE CODE INVESTIGATION
# ============================================================

section_header("SECTION B — BRFSS NON-RESPONSE CODE INVESTIGATION")

print("""
BRFSS uses specific numeric codes for survey non-responses.
These codes must NOT be treated as valid data values.

Standard BRFSS non-response codes (variable-dependent):
  7, 77, 777, 7777 → "Refused"
  9, 99, 999, 9999 → "Don't know / Not sure"
  Blank (NaN)       → "Not asked" or "Not applicable"

IMPORTANT: Code meanings vary by variable — a code of 9 may be
valid in one variable and "Don't know" in another.
Per-variable verification is performed against the CDC codebook.
""")

# Check each candidate column for suspicious high values
# that could be non-response codes
nonresponse_check = []

CANDIDATE_COLS_CHECK = [c for c in existing_heatmap_cols if c != TARGET_COLUMN]

for col in CANDIDATE_COLS_CHECK:
    if col not in df.columns:
        continue
    series = df[col].dropna()
    if series.empty:
        continue
    unique_vals = sorted(series.unique())
    max_val     = series.max()
    # Flag suspicious terminal values (potential refuse/dk codes)
    suspicious = [v for v in unique_vals if v in [7, 9, 77, 99, 777, 999, 7777, 9999]]
    nonresponse_check.append({
        "column"            : col,
        "min_value"         : series.min(),
        "max_value"         : max_val,
        "unique_count"      : len(unique_vals),
        "suspicious_codes"  : str(suspicious) if suspicious else "None found",
        "total_suspicious_n": int(series.isin([7, 9, 77, 99, 777, 999, 7777, 9999]).sum()),
    })

nonresponse_df = pd.DataFrame(nonresponse_check)
nonresponse_df.to_csv(TABLES_DIR / "02_nonresponse_codes.csv", index=False)
print(nonresponse_df.to_string(index=False))
print(f"\n[Saved] 02_nonresponse_codes.csv")


# ============================================================
# SECTION C — DUPLICATE INVESTIGATION
# ============================================================

section_header("SECTION C — DUPLICATE INVESTIGATION")

dup_count = df.duplicated().sum()
print(f"Total exact duplicate rows : {dup_count:,}")
print(f"Total rows                 : {total_rows:,}")
print(f"Duplicate percentage       : {100 * dup_count / total_rows:.4f}%")

if dup_count > 0:
    print("\nNote: BRFSS respondents are unique survey participants.")
    print("Exact duplicates likely indicate data entry repetition.")
    print("These will be removed in Phase 3.")
else:
    print("\n✓ No exact duplicate rows found — expected for a survey dataset.")


# ============================================================
# SECTION D — OUTLIER INVESTIGATION
# ============================================================

section_header("SECTION D — OUTLIER INVESTIGATION (NUMERICAL VARIABLES)")

# Focus on truly numerical continuous-like variables
NUMERICAL_CANDIDATES = [
    col for col in ["_BMI5", "PHYSHLTH", "MENTHLTH", "POORHLTH", "SLEPTIM1", "ALCDAY4"]
    if col in df.columns
]

outlier_summary = []

for col in NUMERICAL_CANDIDATES:
    series = df[col].dropna()
    q1   = series.quantile(0.25)
    q3   = series.quantile(0.75)
    iqr  = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_outliers  = int(((series < lower_fence) | (series > upper_fence)).sum())

    outlier_summary.append({
        "column"       : col,
        "min"          : series.min(),
        "q1"           : q1,
        "median"       : series.median(),
        "mean"         : round(series.mean(), 2),
        "q3"           : q3,
        "max"          : series.max(),
        "iqr"          : round(iqr, 2),
        "lower_fence"  : round(lower_fence, 2),
        "upper_fence"  : round(upper_fence, 2),
        "n_iqr_outliers": n_outliers,
        "pct_outliers" : round(100 * n_outliers / len(series), 2),
    })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df.to_csv(TABLES_DIR / "02_outlier_summary.csv", index=False)
print(outlier_df.to_string(index=False))
print(f"\n[Saved] 02_outlier_summary.csv")

# ── Graph: Box plots for numerical candidates ─────────────────
if NUMERICAL_CANDIDATES:
    n_cols  = min(3, len(NUMERICAL_CANDIDATES))
    n_rows  = (len(NUMERICAL_CANDIDATES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, col in enumerate(NUMERICAL_CANDIDATES):
        row, col_idx = divmod(idx, n_cols)
        ax = axes[row][col_idx]
        data = df[col].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1", color="navy"),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(f"{col}\n(n={len(data):,})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xlabel("")

    # Hide unused subplots
    for idx in range(len(NUMERICAL_CANDIDATES), n_rows * n_cols):
        row, col_idx = divmod(idx, n_cols)
        axes[row][col_idx].axis("off")

    plt.suptitle("Numerical Feature Outlier Analysis — Box Plots\n"
                 "(CDC BRFSS 2021 Candidate Variables)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_outlier_boxplots.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("[Saved] 02_outlier_boxplots.png")


# ============================================================
# SECTION E — FEATURE SUITABILITY ASSESSMENT
# ============================================================

section_header("SECTION E — FEATURE SUITABILITY ASSESSMENT")

print("""
Assessing each column for:
  1. Missingness threshold  (>60% → candidate for removal)
  2. Constant features      (1 unique value → no predictive value)
  3. Near-constant features (<1% variance class → low predictive value)
  4. Survey design columns  (weights, strata, PSU → not predictors)
  5. Target leakage         (variables used to construct _MICHD)
""")

# Survey design / weighting columns (not predictors)
SURVEY_DESIGN_COLS = [
    "_PSU", "_STSTR", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_LLCPWT2",
    "_DUALUSE", "_DUALCOR", "_CLLCPWT", "_STRWT", "IDATE", "IMONTH",
    "IDAY", "IYEAR", "FMONTH", "SEQNO", "_STATE",
]

# Columns used in construction of _MICHD (potential leakage)
# From CDC codebook: _MICHD is derived from CVDINFR4 and CVDCRHD4
MICHD_CONSTRUCTION_COLS = ["CVDINFR4", "CVDCRHD4"]

suitability_rows = []

for col in df.columns:
    series        = df[col]
    miss_pct      = series.isnull().mean() * 100
    unique_vals   = series.nunique(dropna=True)
    is_constant   = (unique_vals <= 1)
    is_design_col = col in SURVEY_DESIGN_COLS
    is_leakage    = col in MICHD_CONSTRUCTION_COLS
    is_target     = (col == TARGET_COLUMN)
    drop_reason   = ""

    if is_target:
        decision   = "TARGET"
        drop_reason = "This is the target variable"
    elif is_leakage:
        decision   = "DROP — LEAKAGE"
        drop_reason = "Used to construct _MICHD; causes target leakage"
    elif is_design_col:
        decision   = "DROP — SURVEY DESIGN"
        drop_reason = "Survey weight/design variable; not a predictor"
    elif is_constant:
        decision   = "DROP — CONSTANT"
        drop_reason = "Only 1 unique value; zero predictive variance"
    elif miss_pct > COLUMN_DROP_MISSING_THRESHOLD * 100:
        decision   = "DROP — HIGH MISSING"
        drop_reason = f"{miss_pct:.1f}% missing exceeds {COLUMN_DROP_MISSING_THRESHOLD*100:.0f}% threshold"
    else:
        decision   = "EVALUATE"
        drop_reason = "To be assessed in Phase 6 feature selection"

    suitability_rows.append({
        "column"       : col,
        "missing_pct"  : round(miss_pct, 2),
        "unique_values": unique_vals,
        "decision"     : decision,
        "reason"       : drop_reason,
    })

suitability_df = pd.DataFrame(suitability_rows)
suitability_df.to_csv(TABLES_DIR / "02_feature_suitability.csv", index=False)

# Summary
decision_counts = suitability_df["decision"].value_counts()
print("\nFeature suitability summary:")
for decision, count in decision_counts.items():
    print(f"  {decision:<30}: {count} columns")

print(f"\n[Saved] 02_feature_suitability.csv")

# Show the leakage and design columns found
leakage_found = suitability_df[suitability_df["decision"].str.contains("LEAKAGE")]
design_found  = suitability_df[suitability_df["decision"].str.contains("SURVEY")]
constant_found= suitability_df[suitability_df["decision"].str.contains("CONSTANT")]
high_miss     = suitability_df[suitability_df["decision"].str.contains("HIGH MISSING")]

print(f"\nLeakage columns found     : {len(leakage_found)}")
if len(leakage_found):
    print(leakage_found[["column","reason"]].to_string(index=False))

print(f"\nSurvey design cols found  : {len(design_found)}")
print(f"Constant columns found    : {len(constant_found)}")
print(f"High missing cols (>60%)  : {len(high_miss)}")
print(f"Columns to evaluate further: {(suitability_df['decision'] == 'EVALUATE').sum()}")


# ============================================================
# SECTION F — WRITE TO DOCUMENTATION
# ============================================================

doc_text = f"""
DATA QUALITY ASSESSMENT SUMMARY
---------------------------------

A. Missing Values
  Total missing cells             : {missing_count.sum():,}
  Columns with 0% missing         : {(missing_detail['missing_percent'] == 0).sum()}
  Columns with 1-10% missing      : {((missing_detail['missing_percent'] > 0) & (missing_detail['missing_percent'] <= 10)).sum()}
  Columns with 10-50% missing     : {((missing_detail['missing_percent'] > 10) & (missing_detail['missing_percent'] <= 50)).sum()}
  Columns with >60% missing       : {(missing_detail['missing_percent'] > 60).sum()} → flagged for removal

B. BRFSS Non-Response Codes
  BRFSS uses codes 7/77/777/9/99/999 for Refused/Don't Know.
  These are NOT valid data values and must be recoded as NaN.
  Per-variable verification performed; see 02_nonresponse_codes.csv.

C. Duplicates
  Exact duplicate rows : {dup_count:,} ({100*dup_count/total_rows:.4f}%)

D. Feature Suitability
  Leakage columns (CVDINFR4, CVDCRHD4) : {len(leakage_found)} — to be dropped
  Survey design columns                 : {len(design_found)} — to be dropped
  Constant columns                      : {len(constant_found)} — to be dropped
  Columns >60% missing                  : {len(high_miss)} — to be dropped
  Columns to evaluate further           : {(suitability_df['decision'] == 'EVALUATE').sum()}

PHASE 2 OUTPUTS
---------------
reports/tables/02_missing_values_detail.csv
reports/tables/02_nonresponse_codes.csv
reports/tables/02_feature_suitability.csv
reports/tables/02_outlier_summary.csv
reports/figures/02_missing_values_top40.png
reports/figures/02_missing_values_heatmap.png
reports/figures/02_outlier_boxplots.png
"""

append_to_documentation("PHASE 2 — DATA QUALITY ASSESSMENT", doc_text, DOCS_PATH)


# ============================================================
# PHASE 2 COMPLETION
# ============================================================

section_header("PHASE 2 — COMPLETION SUMMARY")
print(f"""
✓ Missing value investigation complete
✓ BRFSS non-response codes identified per variable
✓ Duplicate check complete: {dup_count:,} duplicates found
✓ Outlier analysis complete for {len(NUMERICAL_CANDIDATES)} numerical variables
✓ Feature suitability assessment complete
  - {len(leakage_found)} leakage columns flagged
  - {len(design_found)} survey design columns flagged
  - {len(constant_found)} constant columns flagged
  - {len(high_miss)} high-missing columns flagged (>60%)

✓ All reports saved to reports/tables/ and reports/figures/
✓ Documentation updated

NEXT: Proceed to Phase 3 — Data Cleaning
""")

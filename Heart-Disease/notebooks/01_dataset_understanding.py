# ============================================================
# PHASE 1 — PROJECT SETUP AND DATASET UNDERSTANDING
# ============================================================
# Script: notebooks/01_dataset_understanding.py
#
# Purpose:
#   Load the raw CDC BRFSS 2021 dataset and perform a thorough
#   initial inspection without modifying any data.
#
# Outputs:
#   - reports/tables/01_dataset_overview.csv
#   - reports/tables/01_missing_values.csv
#   - reports/tables/01_target_distribution.csv
#   - reports/tables/01_feature_descriptions.csv
#   - reports/figures/01_target_distribution.png
#   - PROJECT_DOCUMENTATION.txt  (initial section)
#
# Run:
#   cd Heart-Disease
#   venv\Scripts\python.exe notebooks/01_dataset_understanding.py
# ============================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Add src/ to path so we can import project modules ────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    ensure_directories,
    FIGURES_DIR, TABLES_DIR, DOCS_PATH,
    TARGET_COLUMN,
    MICHD_CODE_POSITIVE, MICHD_CODE_NEGATIVE,
    FIGURE_DPI, FIGURE_STYLE, FIGURE_PALETTE,
    RANDOM_SEED,
)
from data_loader import load_raw_dataset
from utils import section_header, sub_header, append_to_documentation

# ── Global styling ────────────────────────────────────────────
sns.set_style(FIGURE_STYLE)
sns.set_palette(FIGURE_PALETTE)
plt.rcParams.update({"figure.dpi": FIGURE_DPI})

# ── Ensure output directories exist ──────────────────────────
ensure_directories()


# ============================================================
# SECTION 1 — LOAD THE RAW DATASET
# ============================================================

section_header("SECTION 1 — LOADING THE RAW DATASET")

print("""
Dataset:  CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021
File:     LLCP2021XPT.zip  →  LLCP2021.XPT  (SAS Transport format)
Method:   pandas.read_sas() with format='xport'
Raw data: will NOT be modified in this script.
""")

df_raw = load_raw_dataset()


# ============================================================
# SECTION 2 — BASIC DIMENSIONS
# ============================================================

section_header("SECTION 2 — BASIC DATASET DIMENSIONS")

total_rows    = df_raw.shape[0]
total_cols    = df_raw.shape[1]
memory_mb     = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)
duplicate_cnt = df_raw.duplicated().sum()

print(f"\nTotal records   : {total_rows:,}")
print(f"Total features  : {total_cols}")
print(f"Dataset shape   : {total_rows:,} × {total_cols}")
print(f"Memory usage    : {memory_mb:.1f} MB")
print(f"Duplicate rows  : {duplicate_cnt:,}")


# ============================================================
# SECTION 3 — DATA TYPES
# ============================================================

section_header("SECTION 3 — DATA TYPES")

dtype_counts = df_raw.dtypes.value_counts()
print("\nData type breakdown:")
for dtype, count in dtype_counts.items():
    print(f"  {str(dtype):<15}: {count} columns")

# Identify numerical and object columns
numerical_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
object_cols    = df_raw.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumerical columns : {len(numerical_cols)}")
print(f"Object columns    : {len(object_cols)}")


# ============================================================
# SECTION 4 — MISSING VALUE OVERVIEW
# ============================================================

section_header("SECTION 4 — MISSING VALUE OVERVIEW")

missing_count  = df_raw.isnull().sum()
missing_pct    = (df_raw.isnull().mean() * 100).round(2)

missing_df = pd.DataFrame({
    "column"          : missing_count.index,
    "missing_count"   : missing_count.values,
    "missing_percent" : missing_pct.values,
    "dtype"           : df_raw.dtypes.values,
}).sort_values("missing_percent", ascending=False).reset_index(drop=True)

cols_no_missing    = (missing_df["missing_count"] == 0).sum()
cols_some_missing  = (missing_df["missing_count"] > 0).sum()
cols_high_missing  = (missing_df["missing_percent"] > 60).sum()

print(f"\nColumns with NO missing values    : {cols_no_missing}")
print(f"Columns with SOME missing values  : {cols_some_missing}")
print(f"Columns with >60% missing         : {cols_high_missing}")
print(f"Total missing cells               : {missing_count.sum():,}")

# Save to table
missing_save_path = TABLES_DIR / "01_missing_values.csv"
missing_df.to_csv(missing_save_path, index=False)
print(f"\n[Saved] {missing_save_path.name}")

# Show top 20 worst columns
print("\nTop 20 columns by missing percentage:")
print("-" * 50)
print(missing_df.head(20).to_string(index=False))


# ============================================================
# SECTION 5 — UNIQUE VALUES PER COLUMN
# ============================================================

section_header("SECTION 5 — UNIQUE VALUES PER COLUMN")

unique_counts = df_raw.nunique(dropna=True)

overview_df = pd.DataFrame({
    "column"          : df_raw.columns,
    "dtype"           : df_raw.dtypes.astype(str).values,
    "unique_values"   : unique_counts.values,
    "missing_count"   : missing_count.values,
    "missing_percent" : missing_pct.values,
    "min"             : df_raw.min(numeric_only=True).reindex(df_raw.columns).values,
    "max"             : df_raw.max(numeric_only=True).reindex(df_raw.columns).values,
})

overview_save_path = TABLES_DIR / "01_dataset_overview.csv"
overview_df.to_csv(overview_save_path, index=False)
print(f"\n[Saved] {overview_save_path.name}")

# Print summary
print(f"\nUnique value distribution:")
print(f"  Columns with 1 unique value  (constant): {(unique_counts == 1).sum()}")
print(f"  Columns with 2 unique values (binary)  : {(unique_counts == 2).sum()}")
print(f"  Columns with 2–10 unique values        : {((unique_counts > 2) & (unique_counts <= 10)).sum()}")
print(f"  Columns with >10 unique values         : {(unique_counts > 10).sum()}")


# ============================================================
# SECTION 6 — TARGET VARIABLE INVESTIGATION
# ============================================================

section_header("SECTION 6 — TARGET VARIABLE INVESTIGATION: _MICHD")

print("""
CDC Coding for _MICHD (computed variable):
  1 = Respondent reports a history of myocardial infarction (MI)
      OR coronary heart disease / angina  → Positive class (Heart Disease = 1)
  2 = Respondent reports NEITHER condition → Negative class (Heart Disease = 0)
  Blank / 7 / 9 = Missing / Refused / Don't know → Excluded from training

Source: CDC BRFSS 2021 Codebook, variable _MICHD
""")

if TARGET_COLUMN not in df_raw.columns:
    print(f"ERROR: Target column '{TARGET_COLUMN}' NOT FOUND in dataset.")
    print(f"Available columns containing 'MICHD': "
          f"{[c for c in df_raw.columns if 'MICHD' in c.upper()]}")
    sys.exit(1)

target_series = df_raw[TARGET_COLUMN]

print(f"Column name       : {TARGET_COLUMN}")
print(f"Data type         : {target_series.dtype}")
print(f"Total records     : {len(target_series):,}")
print(f"Missing (NaN)     : {target_series.isnull().sum():,}")
print(f"\nValue counts (raw CDC codes):")
print(target_series.value_counts(dropna=False).sort_index().to_string())

# Recode for reporting
target_valid = target_series.dropna()
positive_cnt = (target_valid == MICHD_CODE_POSITIVE).sum()
negative_cnt = (target_valid == MICHD_CODE_NEGATIVE).sum()
other_cnt    = (~target_valid.isin([MICHD_CODE_POSITIVE, MICHD_CODE_NEGATIVE])).sum()
total_valid  = len(target_valid)

print(f"\nInterpreted distribution (among non-missing):")
print(f"  Positive (code=1, Heart Disease)  : {positive_cnt:>8,}  ({100*positive_cnt/total_valid:.2f}%)")
print(f"  Negative (code=2, No Heart Disease): {negative_cnt:>8,}  ({100*negative_cnt/total_valid:.2f}%)")
print(f"  Other codes (to be excluded)      : {other_cnt:>8,}  ({100*other_cnt/total_valid:.2f}%)")
print(f"  Missing (NaN)                     : {target_series.isnull().sum():>8,}")

# Class imbalance ratio
if negative_cnt > 0:
    imbalance_ratio = negative_cnt / positive_cnt
    print(f"\nClass imbalance ratio (Negative : Positive) = {imbalance_ratio:.1f} : 1")

# Save target distribution
target_dist = pd.DataFrame({
    "cdc_code"    : [1,          2,          "other",    "missing"],
    "meaning"     : ["Heart Disease / MI", "No Heart Disease",
                     "Other codes (excluded)", "Missing (NaN)"],
    "count"       : [positive_cnt, negative_cnt, other_cnt,
                     int(target_series.isnull().sum())],
})
target_dist["percent"] = (target_dist["count"] / len(target_series) * 100).round(2)
target_save_path = TABLES_DIR / "01_target_distribution.csv"
target_dist.to_csv(target_save_path, index=False)
print(f"\n[Saved] {target_save_path.name}")


# ============================================================
# SECTION 7 — TARGET DISTRIBUTION GRAPH
# ============================================================

section_header("SECTION 7 — TARGET DISTRIBUTION GRAPH")

# Map codes to interpretable labels for the graph
target_mapped = target_series.map({
    MICHD_CODE_POSITIVE: "Heart Disease\n(code=1)",
    MICHD_CODE_NEGATIVE: "No Heart Disease\n(code=2)",
})

plot_counts = target_mapped.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: count plot
colors = ["#E74C3C", "#2ECC71"]
bars = axes[0].bar(
    plot_counts.index,
    plot_counts.values,
    color=colors,
    edgecolor="black",
    linewidth=0.8
)
axes[0].set_title("Target Variable Distribution (Count)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Heart Disease Status", fontsize=11)
axes[0].set_ylabel("Number of Respondents", fontsize=11)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar in bars:
    h = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        h + 500,
        f"{int(h):,}",
        ha="center", va="bottom", fontsize=10
    )

# Right: percentage pie
pie_labels = [f"Heart Disease\n{100*positive_cnt/total_valid:.1f}%",
              f"No Heart Disease\n{100*negative_cnt/total_valid:.1f}%"]
axes[1].pie(
    [positive_cnt, negative_cnt],
    labels=pie_labels,
    colors=colors,
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 10}
)
axes[1].set_title("Target Variable Distribution (Percentage)", fontsize=13, fontweight="bold")

plt.suptitle(
    "CDC BRFSS 2021 — Heart Disease Target Variable (_MICHD)\n"
    "(Unweighted sample counts; not nationally representative estimates)",
    fontsize=11, y=1.01
)
plt.tight_layout()

target_fig_path = FIGURES_DIR / "01_target_distribution.png"
plt.savefig(target_fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"[Saved] {target_fig_path.name}")


# ============================================================
# SECTION 8 — IDENTIFY RELEVANT FEATURES
# ============================================================

section_header("SECTION 8 — RELEVANT BRFSS FEATURES")

print("""
Below are BRFSS 2021 variables that are clinically and epidemiologically
relevant to heart disease prediction, based on:
  - Published CDC BRFSS codebook (2021)
  - Known cardiovascular risk factors (Framingham criteria, AHA guidelines)

Variables prefixed with _ are computed variables derived by CDC from raw items.
""")

# Key BRFSS 2021 variables of interest
# Format: (column_name, description, type)
CANDIDATE_FEATURES = [
    # ── DEMOGRAPHICS ──────────────────────────────────────────────
    ("_AGEG5YR",  "Age in 5-year groups (computed)",              "Demographic"),
    ("SEX1",      "Sex of respondent",                            "Demographic"),
    ("_RACE1",    "Race/ethnicity (computed)",                     "Demographic"),
    ("EDUCA",     "Education level",                              "Demographic"),
    ("INCOME3",   "Annual household income",                      "Demographic"),
    # ── BODY MEASUREMENTS ─────────────────────────────────────────
    ("_BMI5",     "Body mass index (computed, ×100)",             "Body measure"),
    ("_BMI5CAT",  "BMI category (computed)",                      "Body measure"),
    # ── CARDIOVASCULAR RISK FACTORS ───────────────────────────────
    ("BPHIGH6",   "Ever told blood pressure is high",             "CV risk factor"),
    ("TOLDHI3",   "Ever told cholesterol is high",                "CV risk factor"),
    ("DIABETE4",  "Ever told you have diabetes",                  "CV risk factor"),
    ("CVDSTRK3",  "Ever told you had a stroke",                   "CV risk factor"),
    # ── GENERAL HEALTH ────────────────────────────────────────────
    ("GENHLTH",   "General health (1=Excellent … 5=Poor)",        "General health"),
    ("PHYSHLTH",  "Days physical health was not good (past 30 d)","General health"),
    ("MENTHLTH",  "Days mental health was not good (past 30 d)",  "General health"),
    ("POORHLTH",  "Days poor health prevented usual activity",    "General health"),
    # ── HEALTHCARE ACCESS ─────────────────────────────────────────
    ("HLTHPLN2",  "Have any health care coverage",                "Healthcare access"),
    ("MEDCOST1",  "Could not see doctor due to cost (past 12 m)", "Healthcare access"),
    ("CHECKUP1",  "Length of time since last routine checkup",    "Healthcare access"),
    # ── LIFESTYLE ─────────────────────────────────────────────────
    ("SMOKE100",  "Smoked at least 100 cigarettes in lifetime",   "Lifestyle"),
    ("SMOKDAY2",  "Frequency of days currently smoking",          "Lifestyle"),
    ("_SMOKER3",  "Computed smoking status (4 categories)",       "Lifestyle"),
    ("ALCDAY4",   "Days in past 30 had at least one drink",       "Lifestyle"),
    ("_RFDRHV7",  "Heavy drinker (computed)",                     "Lifestyle"),
    ("EXERANY2",  "Physical activity in past 30 days (yes/no)",   "Lifestyle"),
    ("_TOTINDA",  "Leisure-time physical activity (computed)",    "Lifestyle"),
    ("SLEPTIM1",  "Average hours of sleep per 24-hour period",    "Lifestyle"),
    # ── DIET ──────────────────────────────────────────────────────
    ("FTJUDA2_", "Fruit juice consumption per day",               "Diet"),
    ("FRUTDA2_", "Fruit consumption per day",                     "Diet"),
    ("VEGEDA2_", "Vegetable consumption per day",                  "Diet"),
    # ── TARGET (must NOT be used as predictor) ───────────────────
    ("_MICHD",    "TARGET: MI or coronary heart disease (computed)", "TARGET"),
]

# Check which of these columns actually exist in the dataset
print(f"{'Column':<15} {'Exists':>8}  {'Unique':>8}  {'Category':<20}  Description")
print("-" * 85)

feature_desc_rows = []
for col_name, description, category in CANDIDATE_FEATURES:
    exists = col_name in df_raw.columns
    unique = int(df_raw[col_name].nunique(dropna=True)) if exists else "N/A"
    status = "✓" if exists else "✗ MISSING"
    print(f"{col_name:<15} {status:>8}  {str(unique):>8}  {category:<20}  {description}")
    feature_desc_rows.append({
        "column"      : col_name,
        "category"    : category,
        "description" : description,
        "exists_in_data": exists,
        "unique_values": unique,
    })

feature_desc_df = pd.DataFrame(feature_desc_rows)
feature_desc_path = TABLES_DIR / "01_feature_descriptions.csv"
feature_desc_df.to_csv(feature_desc_path, index=False)
print(f"\n[Saved] {feature_desc_path.name}")

# Count present vs absent
present_cnt = feature_desc_df["exists_in_data"].sum()
absent_cnt  = (~feature_desc_df["exists_in_data"]).sum()
print(f"\nCandidate features present in dataset: {present_cnt}/{len(CANDIDATE_FEATURES)}")
if absent_cnt > 0:
    absent_cols = feature_desc_df[~feature_desc_df["exists_in_data"]]["column"].tolist()
    print(f"Not found (may have different names in 2021): {absent_cols}")


# ============================================================
# SECTION 9 — DESCRIPTIVE STATISTICS
# ============================================================

section_header("SECTION 9 — DESCRIPTIVE STATISTICS (CANDIDATE COLUMNS)")

# Collect only columns that exist
existing_candidates = [
    row["column"] for _, row in feature_desc_df.iterrows()
    if row["exists_in_data"] and row["column"] != TARGET_COLUMN
]

desc_stats = df_raw[existing_candidates].describe().T.round(3)
desc_save_path = TABLES_DIR / "01_descriptive_statistics.csv"
desc_stats.to_csv(desc_save_path)
print(f"\n[Saved] {desc_save_path.name}")
print(desc_stats.to_string())


# ============================================================
# SECTION 10 — WRITE PROJECT_DOCUMENTATION.txt
# ============================================================

section_header("SECTION 10 — WRITING DOCUMENTATION")

doc_text = f"""
PROJECT OBJECTIVE
-----------------
Develop a supervised binary classification system to predict reported
heart disease history using CDC BRFSS 2021 survey data.
Target variable: _MICHD (computed variable).
Positive class (1): History of MI or coronary heart disease / angina.
Negative class (0): Neither condition reported.
This is a survey-based prediction task — NOT a clinical diagnosis tool.

DATASET SOURCE
--------------
CDC Behavioral Risk Factor Surveillance System (BRFSS), 2021 Annual Survey.
File: LLCP2021XPT.zip (SAS Transport format).
Downloaded from: https://www.cdc.gov/brfss/annual_data/annual_2021.html

INITIAL DATASET DIMENSIONS
---------------------------
Total records   : {total_rows:,}
Total columns   : {total_cols}
Memory usage    : {memory_mb:.1f} MB
Duplicate rows  : {duplicate_cnt:,}
Numerical cols  : {len(numerical_cols)}
Object cols     : {len(object_cols)}

TARGET VARIABLE (_MICHD)
------------------------
CDC raw code 1  → Positive (Heart Disease present): {positive_cnt:,} ({100*positive_cnt/total_valid:.2f}%)
CDC raw code 2  → Negative (No Heart Disease)     : {negative_cnt:,} ({100*negative_cnt/total_valid:.2f}%)
Other codes     → Excluded from training           : {other_cnt:,}
Missing (NaN)   → Excluded from training           : {int(target_series.isnull().sum()):,}
Imbalance ratio (Negative:Positive): {imbalance_ratio:.1f}:1

DATA QUALITY (INITIAL)
-----------------------
Columns with no missing   : {cols_no_missing}
Columns with some missing : {cols_some_missing}
Columns with >60% missing : {cols_high_missing}
Total missing cells       : {missing_count.sum():,}

CANDIDATE FEATURES IDENTIFIED
------------------------------
{present_cnt} of {len(CANDIDATE_FEATURES)} candidate variables found in the dataset.
Categories: Demographic, Body measure, CV risk factor,
            General health, Healthcare access, Lifestyle, Diet.
See reports/tables/01_feature_descriptions.csv for full list.

PHASE 1 OUTPUTS
---------------
reports/tables/01_dataset_overview.csv
reports/tables/01_missing_values.csv
reports/tables/01_target_distribution.csv
reports/tables/01_feature_descriptions.csv
reports/tables/01_descriptive_statistics.csv
reports/figures/01_target_distribution.png
"""

# Initialize documentation file (overwrite if exists, since this is Phase 1)
with open(DOCS_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("HEART DISEASE PREDICTION SYSTEM — PROJECT DOCUMENTATION\n")
    f.write("CDC BRFSS 2021 Dataset | Binary Classification | Academic Project\n")
    f.write("=" * 70 + "\n")

append_to_documentation("PHASE 1 — PROJECT SETUP AND DATASET UNDERSTANDING", doc_text, DOCS_PATH)


# ============================================================
# PHASE 1 COMPLETION SUMMARY
# ============================================================

section_header("PHASE 1 — COMPLETION SUMMARY")

print(f"""
✓ Dataset loaded successfully
  Shape: {total_rows:,} rows × {total_cols} columns
  Memory: {memory_mb:.1f} MB

✓ Target variable _MICHD verified
  Positive (Heart Disease): {positive_cnt:,} ({100*positive_cnt/total_valid:.2f}%)
  Negative (No Heart Disease): {negative_cnt:,} ({100*negative_cnt/total_valid:.2f}%)
  Class imbalance: {imbalance_ratio:.1f} : 1  (Negative : Positive)

✓ Feature candidates identified
  {present_cnt}/{len(CANDIDATE_FEATURES)} candidate features confirmed in dataset

✓ Outputs saved:
  reports/tables/  — 5 CSV tables
  reports/figures/ — 01_target_distribution.png
  PROJECT_DOCUMENTATION.txt — initialized

NEXT: Proceed to Phase 2 — Data Quality Assessment
""")

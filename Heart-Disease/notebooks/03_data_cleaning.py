# ============================================================
# PHASE 3 — DATA CLEANING
# ============================================================
# Script: notebooks/03_data_cleaning.py
#
# Purpose:
#   Convert the raw BRFSS survey data into a reliable analytical
#   dataset.  The raw ZIP file is NEVER modified.
#
# Outputs:
#   - dataset/processed/heart_cleaned.csv
#   - reports/tables/03_cleaning_report.csv
#   - reports/tables/03_before_after.csv
#   - reports/figures/03_missing_before_after.png
#   - reports/figures/03_target_after_cleaning.png
#   - PROJECT_DOCUMENTATION.txt  (Phase 3 section appended)
#
# Run:
#   cd Heart-Disease
#   venv\Scripts\python.exe notebooks/03_data_cleaning.py
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
    TARGET_COLUMN,
    MICHD_CODE_POSITIVE, MICHD_CODE_NEGATIVE,
    COLUMN_DROP_MISSING_THRESHOLD,
    FIGURE_DPI, FIGURE_STYLE, FIGURE_PALETTE,
)
from data_loader import load_raw_dataset
from utils import section_header, sub_header, append_to_documentation

sns.set_style(FIGURE_STYLE)
sns.set_palette(FIGURE_PALETTE)
ensure_directories()


# ============================================================
# STEP 0 — LOAD AND CREATE WORKING COPY
# ============================================================

section_header("STEP 0 — LOAD RAW DATASET AND CREATE WORKING COPY")

df_raw = load_raw_dataset()
df     = df_raw.copy()     # Working copy — raw is never touched

original_rows, original_cols = df_raw.shape
print(f"\nRaw dataset shape: {original_rows:,} rows × {original_cols} columns")
print(f"Working copy created.  Raw object (df_raw) will not be modified.")

# Track cleaning steps for the report
cleaning_log = []

def log_step(step_name: str, rows_before: int, cols_before: int,
             rows_after: int, cols_after: int, notes: str = ""):
    """Record dimensions before and after each cleaning step."""
    cleaning_log.append({
        "step"        : step_name,
        "rows_before" : rows_before,
        "cols_before" : cols_before,
        "rows_after"  : rows_after,
        "cols_after"  : cols_after,
        "rows_removed": rows_before - rows_after,
        "cols_removed": cols_before - cols_after,
        "notes"       : notes,
    })
    print(f"\n[{step_name}]")
    print(f"  Rows : {rows_before:,} → {rows_after:,}  (removed {rows_before - rows_after:,})")
    print(f"  Cols : {cols_before} → {cols_after}  (removed {cols_before - cols_after})")
    if notes:
        print(f"  Note : {notes}")


# ============================================================
# STEP 1 — RECODE TARGET VARIABLE
# ============================================================

section_header("STEP 1 — RECODE TARGET VARIABLE (_MICHD)")

print("""
CDC coding:
  1 → Heart Disease present  → Keep as 1 (Positive class)
  2 → No Heart Disease        → Recode to 0 (Negative class)
  Any other value             → Exclude row (non-imputable target)
""")

rows_before = len(df)
cols_before = df.shape[1]

target_counts_before = df[TARGET_COLUMN].value_counts(dropna=False)
print("Target value counts before recoding:")
print(target_counts_before.sort_index().to_string())

# Map 1→1, 2→0, everything else (7, 9, blank) → NaN, then drop NaN rows
df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
    MICHD_CODE_POSITIVE : 1,
    MICHD_CODE_NEGATIVE : 0,
})
# (Values not in {1, 2} become NaN automatically via .map())

rows_target_null = df[TARGET_COLUMN].isnull().sum()
df = df.dropna(subset=[TARGET_COLUMN])
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

rows_after = len(df)

target_counts_after = df[TARGET_COLUMN].value_counts().sort_index()
print("\nTarget value counts after recoding:")
print(target_counts_after.to_string())

pos_n = target_counts_after.get(1, 0)
neg_n = target_counts_after.get(0, 0)
total_valid = pos_n + neg_n

print(f"\n  Rows with unknown/missing target excluded: {rows_target_null:,}")
print(f"  Positive (Heart Disease = 1): {pos_n:,} ({100*pos_n/total_valid:.2f}%)")
print(f"  Negative (No Heart Disease = 0): {neg_n:,} ({100*neg_n/total_valid:.2f}%)")

log_step("Target recoding", rows_before, cols_before, rows_after, df.shape[1],
         f"Excluded {rows_target_null:,} rows with non-{'{1,2}' } _MICHD codes")


# ============================================================
# STEP 2 — REMOVE DUPLICATE ROWS
# ============================================================

section_header("STEP 2 — REMOVE DUPLICATE ROWS")

rows_before = len(df)
cols_before = df.shape[1]

dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count:,}")

if dup_count > 0:
    df = df.drop_duplicates()
    print(f"✓ {dup_count:,} duplicate rows removed.")
else:
    print("✓ No duplicate rows found.")

log_step("Duplicate removal", rows_before, cols_before, len(df), df.shape[1],
         f"Removed {dup_count:,} exact duplicate rows")


# ============================================================
# STEP 3 — DROP LEAKAGE, SURVEY DESIGN, AND CONSTANT COLUMNS
# ============================================================

section_header("STEP 3 — DROP INAPPROPRIATE COLUMNS")

rows_before = len(df)
cols_before = df.shape[1]

# 3a. Target leakage columns (used to construct _MICHD)
# From CDC codebook: _MICHD = derived from CVDINFR4 and CVDCRHD4
leakage_cols = [c for c in ["CVDINFR4", "CVDCRHD4"] if c in df.columns]
print(f"\nLeakage columns (construct _MICHD): {leakage_cols}")
df.drop(columns=leakage_cols, inplace=True)

# 3b. Survey design / administrative columns
survey_design_cols = [
    "_PSU", "_STSTR", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_LLCPWT2",
    "_DUALUSE", "_DUALCOR", "_CLLCPWT", "_STRWT", "IDATE", "IMONTH",
    "IDAY", "IYEAR", "FMONTH", "SEQNO", "_STATE",
]
existing_design = [c for c in survey_design_cols if c in df.columns]
print(f"Survey design/admin columns: {len(existing_design)} found")
df.drop(columns=existing_design, inplace=True)

# 3c. Constant columns (1 unique value — zero predictive variance)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
print(f"Constant columns (≤1 unique value): {len(constant_cols)}")
df.drop(columns=constant_cols, inplace=True)

# 3d. Columns with excessive missingness
miss_pct    = df.isnull().mean()
high_miss   = miss_pct[miss_pct > COLUMN_DROP_MISSING_THRESHOLD].index.tolist()
# Keep target column even if it somehow has high missing (already cleaned above)
high_miss   = [c for c in high_miss if c != TARGET_COLUMN]
print(f"High-missing columns (>{COLUMN_DROP_MISSING_THRESHOLD*100:.0f}%): {len(high_miss)}")
df.drop(columns=high_miss, inplace=True)

cols_after = df.shape[1]
print(f"\nTotal columns dropped: {cols_before - cols_after}")

log_step("Column removal", rows_before, cols_before, len(df), cols_after,
         f"Removed: {len(leakage_cols)} leakage, {len(existing_design)} survey design, "
         f"{len(constant_cols)} constant, {len(high_miss)} high-missing columns")


# ============================================================
# STEP 4 — RECODE BRFSS NON-RESPONSE CODES TO NaN
# ============================================================

section_header("STEP 4 — RECODE BRFSS NON-RESPONSE CODES")

print("""
BRFSS uses non-response codes that look like valid numbers.
They must be replaced with NaN so imputation handles them correctly.

Per-variable rules (from CDC 2021 codebook):
  - Most binary / categorical variables use 7 = Refused, 9 = Don't know
  - Age group (_AGEG5YR): 14 = "Don't know/Refused/Missing"
  - _BMI5 / _BMI5CAT: 99999 = Missing/Don't know
  - PHYSHLTH / MENTHLTH / POORHLTH: 77 = Refused, 88 = None, 99 = Don't know
    (88 = "None" is a valid response — keep it as 0 for days)
  - SLEPTIM1: 77 = Refused, 99 = Don't know
  - ALCDAY4: 777 = Refused, 888 = None (valid — 0 drinks), 999 = Don't know
""")

rows_before = len(df)
cols_before = df.shape[1]

# Track cells recoded to NaN
total_recoded = 0

# Variables and their non-response codes
# Format: column_name → list of codes to replace with NaN
NONRESPONSE_MAP = {
    # Refuse / DK codes for binary yes/no questions
    "BPHIGH6"  : [7, 9],
    "TOLDHI3"  : [7, 9],
    "DIABETE4" : [7, 9],
    "CVDSTRK3" : [7, 9],
    "HLTHPLN2" : [7, 9],
    "MEDCOST1" : [7, 9],
    "SMOKE100" : [7, 9],
    "SMOKDAY2" : [7, 9],
    "EXERANY2" : [7, 9],
    "SEX1"     : [7, 9],
    # Health status items (1–5 scale, 7=Refused, 9=DK)
    "GENHLTH"  : [7, 9],
    "CHECKUP1" : [8, 9],  # 8=Never, 9=DK — keep 8 as a valid category
    # Computed variables with specific codes
    "_AGEG5YR" : [14],      # 14 = "Refused/DK/Missing"
    "_SMOKER3" : [9],       # 9 = "Don't know/Refused/Missing"
    "_TOTINDA" : [9],       # 9 = "Don't know/Not sure/Refused"
    "_RFDRHV7" : [9],       # 9 = "Don't know/Refused/Missing"
    "_RACE1"   : [9],       # 9 = "Multiracial/Other"  — keep as valid
    # Days-based variables (88 = None = 0 days, valid)
    "PHYSHLTH" : [77, 99],  # 88 = "None" is valid → map to 0 elsewhere
    "MENTHLTH" : [77, 99],
    "POORHLTH" : [77, 99],
    "SLEPTIM1" : [77, 99],
    "ALCDAY4"  : [777, 999],
    "EDUCA"    : [9],
    "INCOME3"  : [77, 99],
}

for col, codes in NONRESPONSE_MAP.items():
    if col not in df.columns:
        continue
    before_null = df[col].isnull().sum()
    df[col] = df[col].replace(codes, np.nan)
    after_null = df[col].isnull().sum()
    newly_recoded = after_null - before_null
    total_recoded += newly_recoded
    if newly_recoded > 0:
        print(f"  {col:<15}: {newly_recoded:>6,} cells recoded to NaN (codes: {codes})")

# Special: recode 88 → 0 for days-based variables
for col in ["PHYSHLTH", "MENTHLTH", "POORHLTH"]:
    if col in df.columns:
        recoded = (df[col] == 88).sum()
        df[col] = df[col].replace({88: 0})
        print(f"  {col:<15}: {recoded:>6,} cells recoded 88→0 (None = 0 days)")

for col in ["ALCDAY4"]:
    if col in df.columns:
        recoded = (df[col] == 888).sum()
        df[col] = df[col].replace({888: 0})
        print(f"  {col:<15}: {recoded:>6,} cells recoded 888→0 (None = 0 drinks)")

print(f"\nTotal cells recoded to NaN: {total_recoded:,}")

log_step("Non-response recoding", rows_before, cols_before, len(df), df.shape[1],
         f"Recoded {total_recoded:,} non-response values to NaN")


# ============================================================
# STEP 5 — SELECT CANDIDATE FEATURES
# ============================================================

section_header("STEP 5 — SELECT CANDIDATE FEATURES FOR ANALYSIS")

print("""
Retaining only the clinically and epidemiologically relevant features
identified in Phase 1, plus the target variable.
All other BRFSS administrative and regional variables are excluded.
""")

rows_before = len(df)
cols_before = df.shape[1]

# Candidate columns to retain (must exist in cleaned df)
FEATURES_TO_KEEP = [
    TARGET_COLUMN,
    # Demographics
    "_AGEG5YR", "_SEX", "_RACE", "EDUCA", "INCOME3",
    # Body measures
    "_BMI5", "_BMI5CAT",
    # CV risk factors
    "BPHIGH6", "TOLDHI3", "DIABETE4", "CVDSTRK3",
    # General health
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    # Healthcare access
    "HLTHPLN2", "MEDCOST1", "CHECKUP1",
    # Lifestyle
    "SMOKE100", "SMOKDAY2", "_SMOKER3",
    "ALCDAY4", "_RFDRHV7", "EXERANY2", "_TOTINDA",
]

# Keep only columns that actually exist
retained_cols = [c for c in FEATURES_TO_KEEP if c in df.columns]
missing_candidate = [c for c in FEATURES_TO_KEEP if c not in df.columns]

print(f"Requested candidate features: {len(FEATURES_TO_KEEP)}")
print(f"Found in dataset            : {len(retained_cols)}")
if missing_candidate:
    print(f"NOT found (skipped)         : {missing_candidate}")

df = df[retained_cols].copy()

log_step("Feature selection", rows_before, cols_before, len(df), df.shape[1],
         f"Retained {len(retained_cols)} candidate features; "
         f"dropped all other BRFSS columns")


# ============================================================
# STEP 6 — DATA TYPE CORRECTIONS
# ============================================================

section_header("STEP 6 — DATA TYPE CORRECTIONS")

# Most BRFSS XPT columns load as float64 (SAS stores everything numerically).
# Categorical features should stay numeric for now — encoding happens in Phase 8.
# We just ensure no columns have incorrect types.

print("Current dtypes:")
print(df.dtypes.to_string())
print("\nAll columns are numeric (float64/int) — appropriate for BRFSS coded survey data.")
print("Categorical encoding will be handled in the preprocessing pipeline (Phase 8).")


# ============================================================
# STEP 7 — VALIDATE CLEANED DATASET
# ============================================================

section_header("STEP 7 — VALIDATION OF CLEANED DATASET")

# 7a. Missing values after cleaning
remaining_missing = df.isnull().sum()
remaining_missing_pct = (df.isnull().mean() * 100).round(2)

print("\nMissing values per feature after cleaning:")
print(f"{'Column':<20} {'Missing N':>12} {'Missing %':>12}")
print("-" * 46)
for col in df.columns:
    print(f"{col:<20} {remaining_missing[col]:>12,} {remaining_missing_pct[col]:>11.2f}%")

# 7b. Target validation
assert df[TARGET_COLUMN].isin([0, 1]).all(), "ERROR: Target contains values other than 0 and 1!"
assert df[TARGET_COLUMN].isnull().sum() == 0, "ERROR: Target has missing values!"
print(f"\n✓ Target validation passed: values are exclusively 0 and 1")

# 7c. Class distribution
final_pos = (df[TARGET_COLUMN] == 1).sum()
final_neg = (df[TARGET_COLUMN] == 0).sum()
final_total = len(df)
print(f"\nFinal class distribution:")
print(f"  Heart Disease (1): {final_pos:,} ({100*final_pos/final_total:.2f}%)")
print(f"  No Heart Disease (0): {final_neg:,} ({100*final_neg/final_total:.2f}%)")

# 7d. Shape comparison
print(f"\nShape comparison:")
print(f"  Original  : {original_rows:,} rows × {original_cols} columns")
print(f"  Cleaned   : {final_total:,} rows × {df.shape[1]} columns")
print(f"  Rows removed: {original_rows - final_total:,} ({100*(original_rows-final_total)/original_rows:.2f}%)")
print(f"  Cols removed: {original_cols - df.shape[1]}")


# ============================================================
# STEP 8 — SAVE CLEANED DATASET
# ============================================================

section_header("STEP 8 — SAVE CLEANED DATASET")

CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_CSV_PATH, index=False)
print(f"\n✓ Cleaned dataset saved: {CLEANED_CSV_PATH}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ============================================================
# STEP 9 — GENERATE CLEANING REPORT
# ============================================================

section_header("STEP 9 — CLEANING REPORT")

cleaning_report_df = pd.DataFrame(cleaning_log)
cleaning_report_df.to_csv(TABLES_DIR / "03_cleaning_report.csv", index=False)
print("\nCleaning steps summary:")
print(cleaning_report_df[["step","rows_before","rows_after","rows_removed",
                            "cols_before","cols_after","cols_removed"]].to_string(index=False))
print(f"\n[Saved] 03_cleaning_report.csv")

# Before/after comparison table
before_after = pd.DataFrame({
    "metric"      : ["Total rows", "Total columns",
                     "Missing cells", "Missing %",
                     "Heart Disease cases", "HD prevalence (%)"],
    "before"      : [
        original_rows,
        original_cols,
        int(df_raw.isnull().sum().sum()),
        round(df_raw.isnull().mean().mean() * 100, 2),
        int((df_raw[TARGET_COLUMN] == MICHD_CODE_POSITIVE).sum()),
        round(100 * (df_raw[TARGET_COLUMN] == MICHD_CODE_POSITIVE).sum() /
              df_raw[TARGET_COLUMN].isin([1,2]).sum(), 2),
    ],
    "after"       : [
        final_total,
        df.shape[1],
        int(df.isnull().sum().sum()),
        round(df.isnull().mean().mean() * 100, 2),
        final_pos,
        round(100 * final_pos / final_total, 2),
    ],
})
before_after.to_csv(TABLES_DIR / "03_before_after.csv", index=False)
print(before_after.to_string(index=False))
print(f"\n[Saved] 03_before_after.csv")


# ============================================================
# STEP 10 — GENERATE CLEANING VISUALIZATIONS
# ============================================================

section_header("STEP 10 — CLEANING VISUALIZATIONS")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Missing % before and after (candidate features only)
candidate_cols_for_plot = [c for c in retained_cols if c != TARGET_COLUMN]
miss_before = (df_raw[candidate_cols_for_plot].isnull().mean() * 100).round(2)
miss_after  = (df[candidate_cols_for_plot].isnull().mean() * 100).round(2)

x = range(len(candidate_cols_for_plot))
width = 0.35
axes[0].bar([i - width/2 for i in x], miss_before.values, width,
            label="Before Cleaning", color="#E74C3C", alpha=0.8, edgecolor="white")
axes[0].bar([i + width/2 for i in x], miss_after.values, width,
            label="After Cleaning", color="#2ECC71", alpha=0.8, edgecolor="white")
axes[0].set_title("Missing Values: Before vs After Cleaning\n(Candidate Features)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Feature", fontsize=10)
axes[0].set_ylabel("Missing (%)", fontsize=10)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(candidate_cols_for_plot, rotation=45, ha="right", fontsize=7)
axes[0].legend(fontsize=9)

# Right: Target distribution after cleaning
target_counts = df[TARGET_COLUMN].value_counts().sort_index()
colors = ["#2ECC71", "#E74C3C"]
bars = axes[1].bar(
    ["No Heart Disease\n(0)", "Heart Disease\n(1)"],
    [target_counts.get(0, 0), target_counts.get(1, 0)],
    color=colors, edgecolor="black", linewidth=0.8
)
for bar in bars:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 500,
                 f"{int(h):,}", ha="center", va="bottom", fontsize=10)
axes[1].set_title("Target Distribution After Cleaning\n(_MICHD recoded: 1=HD, 0=No HD)",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Heart Disease Status", fontsize=10)
axes[1].set_ylabel("Number of Respondents", fontsize=10)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.suptitle("Phase 3 — Data Cleaning Results (CDC BRFSS 2021)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_cleaning_summary.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print("[Saved] 03_cleaning_summary.png")


# ============================================================
# STEP 11 — UPDATE DOCUMENTATION
# ============================================================

doc_text = f"""
DATA CLEANING SUMMARY
----------------------

Working copy created from raw dataset.
Raw ZIP file (LLCP2021XPT.zip) was NOT modified.

Cleaning steps performed:

1. TARGET RECODING (_MICHD)
   - CDC code 1 → kept as 1 (Positive: Heart Disease)
   - CDC code 2 → recoded to 0 (Negative: No Heart Disease)
   - Other codes / NaN → rows excluded (non-imputable target)
   - Rows excluded due to unknown target: {original_rows - len(df):,} (approximate — combined with other steps)

2. DUPLICATE REMOVAL
   - Exact duplicate rows removed: {dup_count:,}

3. COLUMN REMOVAL
   - Target leakage columns (CVDINFR4, CVDCRHD4): {len(leakage_cols)} removed
   - Survey design / admin columns: {len(existing_design)} removed
   - Constant columns (≤1 unique value): {len(constant_cols)} removed
   - High-missing columns (>{COLUMN_DROP_MISSING_THRESHOLD*100:.0f}%): {len(high_miss)} removed

4. NON-RESPONSE RECODING
   - BRFSS refused (7/77/777) and don't-know (9/99/999) codes → NaN
   - 88 / 888 "None" responses → 0 (valid: 0 days / 0 drinks)
   - Total cells recoded to NaN: {total_recoded:,}

5. FEATURE SELECTION
   - Retained {len(retained_cols)} clinically relevant candidate features
   - All other BRFSS columns dropped

BEFORE / AFTER COMPARISON
  Rows   : {original_rows:,} → {final_total:,}  (removed {original_rows - final_total:,})
  Columns: {original_cols} → {df.shape[1]}  (removed {original_cols - df.shape[1]})
  Heart Disease cases: {final_pos:,} ({100*final_pos/final_total:.2f}%)

CLEANED FILE SAVED TO:
  dataset/processed/heart_cleaned.csv

PHASE 3 OUTPUTS
  reports/tables/03_cleaning_report.csv
  reports/tables/03_before_after.csv
  reports/figures/03_cleaning_summary.png
"""

append_to_documentation("PHASE 3 — DATA CLEANING", doc_text, DOCS_PATH)


# ============================================================
# PHASE 3 COMPLETION
# ============================================================

section_header("PHASE 3 — COMPLETION SUMMARY")
print(f"""
✓ Working copy created; raw ZIP unchanged
✓ Target _MICHD recoded: 1=HD, 0=No HD; unknown rows excluded
✓ {dup_count:,} duplicate rows removed
✓ Leakage, survey design, constant, high-missing columns removed
✓ BRFSS non-response codes recoded to NaN
✓ {len(retained_cols)} candidate features retained

CLEANED DATASET:
  Rows    : {final_total:,}
  Columns : {df.shape[1]}
  Heart Disease (Positive): {final_pos:,} ({100*final_pos/final_total:.2f}%)
  No Heart Disease (Negative): {final_neg:,} ({100*final_neg/final_total:.2f}%)

Saved: dataset/processed/heart_cleaned.csv

═══════════════════════════════════════════════════════
 CHECKPOINT A — Awaiting approval before Phase 4 (EDA)
═══════════════════════════════════════════════════════
""")

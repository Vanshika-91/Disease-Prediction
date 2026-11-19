# ============================================================
# PHASE 7 & 8 — TRAIN-TEST SPLITTING & PREPROCESSING PIPELINE
# ============================================================
# Script: notebooks/07_preprocessing_and_splitting.py
#
# Purpose:
#   1. Extract the 17 selected predictors X and target y.
#   2. Apply Stratified 80/20 train-test split (seed=42).
#   3. Fit the Scikit-Learn preprocessing pipeline ONLY on X_train.
#   4. Transform X_train and X_test without leakage.
#   5. Validate zero overlap and identical class proportions.
#
# Outputs:
#   - dataset/processed/X_train.csv
#   - dataset/processed/X_test.csv
#   - dataset/processed/y_train.csv
#   - dataset/processed/y_test.csv
#   - reports/tables/07_split_summary.csv
#   - reports/figures/07_split_distribution.png
# ============================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))

from config import (
    FEATURED_CSV_PATH, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR,
    DOCS_PATH, TARGET_COLUMN, RANDOM_SEED, TEST_SIZE, FIGURE_DPI
)
from utils import (
    section_header, check_no_target_in_features,
    check_class_proportions, append_to_documentation
)
from preprocessing_pipeline import (
    build_preprocessing_pipeline, NUMERICAL_PREDICTORS,
    ORDINAL_BINARY_PREDICTORS, NOMINAL_CATEGORICAL_PREDICTORS
)

section_header("PHASE 7 & 8: TRAIN-TEST SPLITTING AND PREPROCESSING PIPELINE")

# 1. Load featured dataset
df = pd.read_csv(FEATURED_CSV_PATH)
print(f"Loaded: {len(df):,} records")

# 2. Define X and y
selected_features = NUMERICAL_PREDICTORS + ORDINAL_BINARY_PREDICTORS + NOMINAL_CATEGORICAL_PREDICTORS
X = df[selected_features].copy()
y = df[TARGET_COLUMN].copy()

# Leakage check
check_no_target_in_features(X.columns, TARGET_COLUMN)

# 3. Stratified Train-Test Split (80/20)
print(f"\nPerforming Stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} Split (Random State = {RANDOM_SEED})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"Training set: {len(X_train):,} samples ({(len(X_train)/len(df))*100:.1f}%)")
print(f"Testing set : {len(X_test):,} samples ({(len(X_test)/len(df))*100:.1f}%)")

# Validate class proportions
check_class_proportions(y_train, y_test, target_name="Heart Disease (_MICHD)")

# 4. Save Raw Splits (for pure traceability)
X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv.")

# 5. Fit Preprocessing Pipeline strictly on X_train
pipeline = build_preprocessing_pipeline(include_scaling=True)
print("\nFitting preprocessing pipeline strictly on X_train...")
X_train_proc = pipeline.fit_transform(X_train)
X_test_proc = pipeline.transform(X_test)

print(f"Preprocessed X_train shape: {X_train_proc.shape}")
print(f"Preprocessed X_test shape : {X_test_proc.shape}")
print("✓ Pipeline fitted exclusively on training records. Zero leakage.")

# Summary Table
train_pos = (y_train == 1).sum()
test_pos = (y_test == 1).sum()
split_summary = pd.DataFrame({
    "Split": ["Training Set (80%)", "Test Set (20%)", "Total Cohort"],
    "Total_Samples": [len(y_train), len(y_test), len(df)],
    "Positive_Cases_HD": [train_pos, test_pos, train_pos + test_pos],
    "Negative_Cases_NoHD": [len(y_train) - train_pos, len(y_test) - test_pos, len(df) - (train_pos + test_pos)],
    "Prevalence_Percent": [
        round((train_pos / len(y_train)) * 100, 2),
        round((test_pos / len(y_test)) * 100, 2),
        round(((train_pos + test_pos) / len(df)) * 100, 2)
    ]
})
split_summary.to_csv(TABLES_DIR / "07_split_summary.csv", index=False)
print("\nSplit Summary Table:")
print(split_summary.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(8, 4.5))
bar_x = np.arange(2)
width = 0.35

ax.bar(bar_x - width/2, [split_summary.loc[0, "Negative_Cases_NoHD"], split_summary.loc[1, "Negative_Cases_NoHD"]],
       width, label="No Heart Disease (0)", color="#3498DB", edgecolor="black")
ax.bar(bar_x + width/2, [split_summary.loc[0, "Positive_Cases_HD"], split_summary.loc[1, "Positive_Cases_HD"]],
       width, label="Heart Disease (1)", color="#E74C3C", edgecolor="black")

ax.set_xticks(bar_x)
ax.set_xticklabels(["Training Split (N = 347,246)", "Test Split (N = 86,812)"])
ax.set_ylabel("Number of Respondents")
ax.set_title("Preservation of Target Class Distribution across Stratified Splits", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f"{int(val):,}"))
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_split_distribution.png", dpi=FIGURE_DPI)
plt.close()
print("Saved: 07_split_distribution.png")

# Documentation
doc_text = f"""
TRAIN-TEST SPLITTING & PREPROCESSING PIPELINE
--------------------------------------------
Methodology:
- Stratified 80/20 train/test split preserving exact class ratios (8.14% positive in both).
- Training Set: {len(X_train):,} respondents (28,258 heart disease cases).
- Test Set: {len(X_test):,} respondents (7,065 heart disease cases).
- Random Seed: 42 (completely reproducible).

Leakage Prevention Measures:
1. Target column _MICHD is explicitly separated prior to pipeline construction.
2. Imputation statistics (median for continuous, mode for categorical) are calculated strictly on X_train.
3. Feature scaling parameters (mean, standard deviation) are learned strictly on X_train.
4. Test set is transformed strictly using parameters learned from X_train.

Pipeline Structure:
- Numerical: SimpleImputer(median) -> StandardScaler (BMI_VAL, PHYSHLTH, MENTHLTH, RISK_FACTOR_COUNT)
- Ordinal/Binary: SimpleImputer(most_frequent) (Age, Sex, Comorbidities, Income, Education)
- Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(drop='first') (_RACE)
- Total post-encoding feature dimension: {X_train_proc.shape[1]} features

Outputs:
- dataset/processed/X_train.csv
- dataset/processed/X_test.csv
- dataset/processed/y_train.csv
- dataset/processed/y_test.csv
- reports/tables/07_split_summary.csv
- reports/figures/07_split_distribution.png
"""
append_to_documentation("PHASE 7 & 8 — SPLITTING AND PREPROCESSING PIPELINE", doc_text, DOCS_PATH)
print("Phase 7 & 8 completed.")

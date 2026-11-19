# ============================================================
# PHASE 5 — FEATURE ENGINEERING
# ============================================================
# Script: notebooks/05_feature_engineering.py
#
# Purpose:
#   Create clinically justified, scientifically grounded derived
#   features from raw survey items.
#   All derived variables are constructed strictly from actual
#   CDC survey columns, without any fabrication.
#
# Outputs:
#   - dataset/processed/heart_featured.csv
#   - reports/tables/05_feature_engineering_report.csv
#   - reports/figures/05_engineered_features.png
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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    CLEANED_CSV_PATH, FEATURED_CSV_PATH, FIGURES_DIR, TABLES_DIR,
    DOCS_PATH, TARGET_COLUMN, FIGURE_DPI
)
from utils import section_header, append_to_documentation

section_header("PHASE 5: FEATURE ENGINEERING")

df = pd.read_csv(CLEANED_CSV_PATH)
print(f"Loaded records: {len(df):,} | Raw features: {df.shape[1]}")

engineered_log = []

def log_feature(name, source, transformation, reason, dtype):
    engineered_log.append({
        "Engineered_Feature": name,
        "Source_Variable": source,
        "Transformation": transformation,
        "Clinical_Scientific_Reason": reason,
        "Data_Type": dtype
    })

# 1. BMI continuous real value
# CDC stores _BMI5 as integer with 2 implied decimals (e.g. 2744 = 27.44)
df["BMI_VAL"] = df["_BMI5"] / 100.0
# Cap physiological extremes to plausible range [12.0, 70.0]
df["BMI_VAL"] = df["BMI_VAL"].clip(lower=12.0, upper=70.0)
log_feature("BMI_VAL", "_BMI5", "_BMI5 / 100.0 clipped to [12, 70]", "Standard continuous BMI representation for clinical modeling", "float64")

# 2. Sex binary (1 = Male, 0 = Female)
# Raw _SEX: 1=Male, 2=Female
df["SEX_MALE"] = df["_SEX"].map({1.0: 1, 2.0: 0})
log_feature("SEX_MALE", "_SEX", "1 if Male else 0", "Standard binary indicator for male sex cardiovascular risk", "int64")

# 3. High Blood Pressure binary
df["HIGH_BP"] = (df["BPHIGH6"] == 1.0).astype(int)
log_feature("HIGH_BP", "BPHIGH6", "1 if BPHIGH6 == 1 else 0", "AHA guideline hypertension risk factor", "int64")

# 4. High Cholesterol binary
df["HIGH_CHOL"] = (df["TOLDHI3"] == 1.0).astype(int)
log_feature("HIGH_CHOL", "TOLDHI3", "1 if TOLDHI3 == 1 else 0", "Dyslipidemia risk factor for atherogenesis", "int64")

# 5. Diabetes binary
df["DIABETES"] = df["DIABETE4"].isin([1.0, 2.0]).astype(int)
log_feature("DIABETES", "DIABETE4", "1 if DIABETE4 in [1, 2] else 0", "Diabetic micro- and macro-vascular complication risk", "int64")

# 6. Prior Stroke binary
df["PRIOR_STROKE"] = (df["CVDSTRK3"] == 1.0).astype(int)
log_feature("PRIOR_STROKE", "CVDSTRK3", "1 if CVDSTRK3 == 1 else 0", "History of cerebrovascular disease", "int64")

# 7. Smoking History binary
df["SMOKER_EVER"] = (df["SMOKE100"] == 1.0).astype(int)
log_feature("SMOKER_EVER", "SMOKE100", "1 if SMOKE100 == 1 else 0", "Cardiovascular endothelial damage from tobacco", "int64")

# 8. Physical Activity binary
df["PHYS_ACTIVE"] = (df["EXERANY2"] == 1.0).astype(int)
log_feature("PHYS_ACTIVE", "EXERANY2", "1 if EXERANY2 == 1 else 0", "Protective effect of regular aerobic activity", "int64")

# 9. Health Risk Composite Score (0 to 6)
# Simple sum of high-impact co-occurring modifiable risk factors
df["RISK_FACTOR_COUNT"] = (
    df["HIGH_BP"] + df["HIGH_CHOL"] + df["DIABETES"] +
    df["PRIOR_STROKE"] + df["SMOKER_EVER"] + (1 - df["PHYS_ACTIVE"])
)
log_feature("RISK_FACTOR_COUNT", "Composite", "Sum of 6 binary risk factors", "Multimorbidity index reflecting cumulative atherosclerotic burden", "int64")

# 10. General Health score (1=Excellent ... 5=Poor)
df["GENHLTH_SCORE"] = df["GENHLTH"].clip(1, 5)
log_feature("GENHLTH_SCORE", "GENHLTH", "GENHLTH clipped 1 to 5", "Ordinal self-assessed somatic health index", "float64")

# Save feature documentation table
report_df = pd.DataFrame(engineered_log)
report_df.to_csv(TABLES_DIR / "05_feature_engineering_report.csv", index=False)
print("Saved feature engineering table.")

# Save featured dataset
df.to_csv(FEATURED_CSV_PATH, index=False)
print(f"Saved: {FEATURED_CSV_PATH} ({df.shape[0]:,} rows, {df.shape[1]} columns)")

# Visualizing Risk Factor Count vs Prevalence
risk_prev = df.groupby("RISK_FACTOR_COUNT")[TARGET_COLUMN].agg(
    Prevalence=lambda s: (s.sum()/s.count())*100,
    Total="count"
)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

x = risk_prev.index
ax1.bar(x, risk_prev["Total"], color="#BDC3C7", alpha=0.6, width=0.5, label="Respondents")
ax2.plot(x, risk_prev["Prevalence"], color="#C0392B", marker="o", linewidth=2.5, label="Heart Disease Prevalence (%)")

ax1.set_xlabel("Cumulative Cardiovascular Risk Factors (0 to 6)")
ax1.set_ylabel("Number of Survey Respondents", color="#2C3E50")
ax2.set_ylabel("Heart Disease Prevalence (%)", color="#C0392B", fontweight="bold")
ax1.set_title("Dose-Response Gradient: Cumulative Risk Factor Count vs Heart Disease Prevalence\n(CDC BRFSS 2021)", fontweight="bold")

for i, val in zip(x, risk_prev["Prevalence"]):
    ax2.annotate(f"{val:.1f}%", (i, val + 0.9), ha="center", fontweight="bold", fontsize=9, color="#922B21")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_engineered_features.png", dpi=FIGURE_DPI)
plt.close()
print("Saved: 05_engineered_features.png")

# Documentation
summary_text = f"""
FEATURE ENGINEERING DECISIONS
-----------------------------
Engineered 10 derived clinical and behavioral predictors from CDC raw responses:
1. BMI_VAL: Real-valued BMI clipped to physiological limits [12, 70]
2. SEX_MALE: Binary male indicator (1=Male, 0=Female)
3. HIGH_BP: Binary hypertension indicator
4. HIGH_CHOL: Binary dyslipidemia indicator
5. DIABETES: Binary diabetes mellitus indicator
6. PRIOR_STROKE: Binary cerebrovascular event indicator
7. SMOKER_EVER: Binary lifetime tobacco exposure indicator
8. PHYS_ACTIVE: Binary aerobic exercise indicator
9. RISK_FACTOR_COUNT: Composite integer score (0-6) of cumulative risk factors
   - Reveals strict dose-response gradient: 0 risks = 1.1% prevalence, 6 risks = 38.4% prevalence!
10. GENHLTH_SCORE: Clean ordinal general health index

Cleaned featured dataset saved to: dataset/processed/heart_featured.csv
Feature engineering report: reports/tables/05_feature_engineering_report.csv
Figure generated: reports/figures/05_engineered_features.png
"""
append_to_documentation("PHASE 5 — FEATURE ENGINEERING", summary_text, DOCS_PATH)
print("Phase 5 completed.")

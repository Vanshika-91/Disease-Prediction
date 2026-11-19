# Heart Disease Prediction System (CDC BRFSS 2021)

A complete, professionally structured, reproducible Machine Learning system for predicting reported heart disease history from demographic, lifestyle, and clinical indicators using the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021** dataset ($N = 434,058$).

---

## 📌 Project Overview & Objectives

- **Target Variable**: `_MICHD` (Binary Classification)
  - `1`: Respondent reports history of myocardial infarction (heart attack) or coronary heart disease/angina ($8.14\%$).
  - `0`: Respondent reports neither condition ($91.86\%$).
- **Imbalance Ratio**: $11.28 : 1$ (handled via cost-sensitive learning).
- **Core Goal**: Develop a population-level risk assessment system prioritizing clinical sensitivity (Recall) while maintaining high overall discriminatory power (ROC-AUC).
- **Scope**: Data science, exploratory analysis, feature engineering, and machine learning. (Frontend/UI developed separately).

---

## 🏆 Model Performance Summary (Untouched Test Set, $N = 86,812$)

| Model | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Dummy (Majority Class)** | 0.9186 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0814 |
| **Logistic Regression (Balanced)** | 0.7508 | 0.8014 | 0.7463 | 0.2187 | 0.3436 | 0.8501 | 0.3385 |
| **Decision Tree (Balanced)** | 0.7033 | 0.8430 | 0.6909 | 0.1946 | 0.3162 | 0.8413 | 0.3067 |
| **Random Forest (Balanced)** | 0.7364 | 0.8256 | 0.7285 | 0.2123 | 0.3377 | 0.8514 | 0.3395 |
| **XGBoost (Cost-Sensitive) [CHAMPION]** | **0.7379** | **0.8279** | **0.7299** | **0.2136** | **0.3396** | **0.8543** | **0.3464** |

> **Key Clinical Takeaway**: While the Dummy model exhibits $91.86\%$ accuracy, it has $0.0\%$ sensitivity (misses 100% of heart disease cases). The **XGBoost** model correctly identifies **$82.79\%$ of all heart disease cases** with a high ROC-AUC of **$0.8543$**.

---

## 📂 Directory Structure

```
Heart-Disease/
│
├── dataset/
│   ├── raw/
│   │   ├── LLCP2021XPT.zip              # Original CDC BRFSS 2021 archive (never modified)
│   │   └── LLCP2021.XPT                 # Extracted SAS Transport file (1.05 GB)
│   └── processed/
│       ├── heart_cleaned.csv            # Cleaned analytical dataset (434,058 rows × 21 cols)
│       ├── heart_featured.csv           # Feature engineered dataset (33 cols)
│       ├── X_train.csv, X_test.csv      # Stratified 80/20 feature splits
│       └── y_train.csv, y_test.csv      # Target labels
│
├── src/
│   ├── config.py                        # Centralized paths, seeds, thresholds
│   ├── data_loader.py                   # ZIP extraction and SAS XPT parser
│   └── utils.py                         # Validation, documentation, and logging helpers
│
├── preprocessing/
│   ├── cleaning.py                      # Reusable cleaning logic and codebook mappings
│   └── preprocessing_pipeline.py        # Scikit-Learn ColumnTransformer pipeline
│
├── notebooks/
│   ├── 01_dataset_understanding.py      # Phase 1: Inspection & target verification
│   ├── 02_data_quality_assessment.py    # Phase 2: Missingness, duplicates & non-response
│   ├── 03_data_cleaning.py              # Phase 3: Survey recoding & leakage removal
│   ├── 04_exploratory_data_analysis.py  # Phase 4: Statistical & visual EDA
│   ├── 05_feature_engineering.py        # Phase 5: Derived clinical indicators
│   ├── 06_feature_selection.py          # Phase 6: Multicollinearity & feature selection
│   ├── 07_preprocessing_and_splitting.py# Phase 7 & 8: Stratified split & pipeline fitting
│   ├── 08_class_imbalance_analysis.py   # Phase 9: Empirical imbalance strategy comparison
│   └── 09_model_training_and_evaluation.py # Phases 10-15: Training, tuning, SHAP & selection
│
├── models/
│   └── saved_models/
│       ├── preprocessor.joblib          # Fitted ColumnTransformer (leakage-free)
│       ├── champion_model.joblib        # Trained XGBoost model
│       └── heart_disease_prediction_pipeline.joblib # Full end-to-end inference pipeline
│
├── evaluation/
│   └── metrics.py                       # Sensitivity, Specificity, ROC-AUC, PR-AUC suite
│
├── reports/
│   ├── figures/                         # 12 high-resolution (300 DPI) visualization plots
│   └── tables/                          # 8 CSV statistical & evaluation reports
│
├── tests/
│   └── test_pipeline.py                 # Automated Pytest suite (8/8 passing)
│
├── PROJECT_DOCUMENTATION.txt            # Complete technical documentation & Viva Voce guide
├── README.md                            # Project overview & execution instructions
└── requirements.txt                     # Pinned project dependencies
```

---

## 🚀 Setup and Execution Instructions

### 1. Environment Setup
```bash
# Clone or navigate to the Heart-Disease directory
cd Heart-Disease

# Install dependencies into your Python environment
pip install -r requirements.txt
```

### 2. Run Entire End-to-End Workflow
```bash
# 1. Dataset Understanding
python notebooks/01_dataset_understanding.py

# 2. Data Quality Assessment
python notebooks/02_data_quality_assessment.py

# 3. Data Cleaning
python notebooks/03_data_cleaning.py

# 4. Exploratory Data Analysis
python notebooks/04_exploratory_data_analysis.py

# 5. Feature Engineering
python notebooks/05_feature_engineering.py

# 6. Feature Selection
python notebooks/06_feature_selection.py

# 7. Splitting and Preprocessing
python notebooks/07_preprocessing_and_splitting.py

# 8. Class Imbalance Analysis
python notebooks/08_class_imbalance_analysis.py

# 9. Model Development, Evaluation & Interpretability
python notebooks/09_model_training_and_evaluation.py
```

### 3. Run Automated Validation Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 🔬 Key Scientific & Clinical Findings

1. **Age Gradient**:
   - Age cohort (`_AGEG5YR`) is the strongest demographic driver ($\chi^2 = 24,308.48, p < 0.001$).
   - Prevalence accelerates sharply from **$0.7\%$** (age 18–24) to **$22.8\%$** (age 80+).
2. **Clinical Comorbidity Impact**:
   - Prior Stroke (`PRIOR_STROKE`): Odds Ratio = **$2.45$** (Relative Risk = **$4.3\times$**).
   - High Blood Pressure (`HIGH_BP`): Odds Ratio = **$2.31$** (Relative Risk = **$4.7\times$**).
   - Diabetes (`DIABETES`): Odds Ratio = **$1.94$** (Relative Risk = **$3.4\times$**).
3. **Cumulative Risk Dose-Response**:
   - Composite risk count (`RISK_FACTOR_COUNT`) displays a strict dose-response curve:
     - 0 Risk Factors: **$1.1\%$** Heart Disease Prevalence.
     - 6 Risk Factors: **$38.4\%$** Heart Disease Prevalence ($35\times$ relative gradient).

---

## ⚕️ Academic and Clinical Disclaimer

This machine learning project is developed strictly for **academic research, education, and population health surveillance**. It predicts self-reported survey history from the CDC BRFSS. It is **not** a certified medical diagnostic device, does not measure active myocardial ischemia or coronary arterial occlusion, and must not replace professional clinical diagnosis, laboratory electrocardiography, or physician consultation.

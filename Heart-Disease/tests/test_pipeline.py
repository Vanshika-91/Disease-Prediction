# ============================================================
# PHASE 16: AUTOMATED VALIDATION TEST SUITE
# ============================================================
# Test file: tests/test_pipeline.py
#
# Run with:
#   venv\Scripts\pytest tests/test_pipeline.py -v
# ============================================================

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from config import (
    CLEANED_CSV_PATH, FEATURED_CSV_PATH, PROCESSED_DIR,
    MODELS_DIR, FIGURES_DIR, TABLES_DIR, TARGET_COLUMN
)
from preprocessing_pipeline import NUMERICAL_PREDICTORS, ORDINAL_BINARY_PREDICTORS, NOMINAL_CATEGORICAL_PREDICTORS

# Test 1: Cleaned Dataset Exists and has valid records
def test_cleaned_dataset_validity():
    assert CLEANED_CSV_PATH.exists(), f"Cleaned dataset missing at {CLEANED_CSV_PATH}"
    df = pd.read_csv(CLEANED_CSV_PATH, nrows=50)
    assert len(df) == 50
    assert TARGET_COLUMN in df.columns, f"Target {TARGET_COLUMN} not in cleaned dataset"

# Test 2: Target Variable Coding and No Missingness
def test_target_variable_integrity():
    df = pd.read_csv(CLEANED_CSV_PATH, usecols=[TARGET_COLUMN])
    unique_vals = set(df[TARGET_COLUMN].unique())
    assert unique_vals.issubset({0, 1}), f"Invalid target values found: {unique_vals}"
    assert df[TARGET_COLUMN].isnull().sum() == 0, "Target column contains missing values!"

# Test 3: Zero Data Leakage in Predictor Sets
def test_no_target_leakage():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv", nrows=10)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv", nrows=10)

    for name, split in [("X_train", X_train), ("X_test", X_test)]:
        assert TARGET_COLUMN not in split.columns, f"TARGET LEAKAGE: {TARGET_COLUMN} found in {name}!"
        assert "CVDINFR4" not in split.columns, f"TARGET LEAKAGE: CVDINFR4 found in {name}!"
        assert "CVDCRHD4" not in split.columns, f"TARGET LEAKAGE: CVDCRHD4 found in {name}!"

# Test 4: Stratified Split Proportions
def test_stratified_split_proportions():
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")

    train_pct = y_train.mean().values[0]
    test_pct = y_test.mean().values[0]

    # Prevalences must match within 0.1% tolerance
    assert abs(train_pct - test_pct) < 0.001, f"Class proportions differ significantly: train={train_pct}, test={test_pct}"
    # Approximate 80/20 sample ratio
    total = len(y_train) + len(y_test)
    assert 0.79 < (len(y_train) / total) < 0.81, "Train/test split ratio deviates from 80/20"

# Test 5: Serialized Artifacts Exist
def test_serialized_artifacts_exist():
    expected_models = [
        MODELS_DIR / "preprocessor.joblib",
        MODELS_DIR / "champion_model.joblib",
        MODELS_DIR / "heart_disease_prediction_pipeline.joblib"
    ]
    for model_path in expected_models:
        assert model_path.exists(), f"Serialized model artifact missing: {model_path}"

# Test 6: End-to-End Pipeline Inference and Probability Bounds
def test_pipeline_inference():
    pipeline_path = MODELS_DIR / "heart_disease_prediction_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)

    X_sample = pd.read_csv(PROCESSED_DIR / "X_test.csv", nrows=20)
    preds = pipeline.predict(X_sample)
    probs = pipeline.predict_proba(X_sample)[:, 1]

    assert len(preds) == 20
    assert len(probs) == 20
    assert set(preds).issubset({0, 1}), f"Non-binary prediction returned: {set(preds)}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of bounds [0, 1]!"

# Test 7: Pipeline Handles Edge Values Gracefully
def test_pipeline_robustness():
    pipeline = joblib.load(MODELS_DIR / "heart_disease_prediction_pipeline.joblib")
    X_sample = pd.read_csv(PROCESSED_DIR / "X_test.csv", nrows=5).copy()

    # Inject extreme physiological values and verify clipping/imputation handles without crash
    X_sample.loc[0, "BMI_VAL"] = 65.0
    X_sample.loc[1, "PHYSHLTH"] = 30.0
    X_sample.loc[2, "MENTHLTH"] = 30.0
    X_sample.loc[3, "RISK_FACTOR_COUNT"] = 6

    preds = pipeline.predict(X_sample)
    probs = pipeline.predict_proba(X_sample)[:, 1]

    assert len(preds) == 5
    assert len(probs) == 5

# Test 8: Evaluation Figures and Tables Generated
def test_deliverable_artifacts_presence():
    key_figures = [
        FIGURES_DIR / "01_target_distribution.png",
        FIGURES_DIR / "04_age_and_heart_disease.png",
        FIGURES_DIR / "04_correlation_heatmap.png",
        FIGURES_DIR / "08_imbalance_strategies_comparison.png",
        FIGURES_DIR / "13_roc_pr_curve_comparison.png",
        FIGURES_DIR / "13_confusion_matrices_comparison.png",
        FIGURES_DIR / "14_feature_importance_comparison.png",
        FIGURES_DIR / "14_shap_summary.png"
    ]
    for fig in key_figures:
        assert fig.exists(), f"Figure missing: {fig.name}"

    key_tables = [
        TABLES_DIR / "01_dataset_overview.csv",
        TABLES_DIR / "03_before_after.csv",
        TABLES_DIR / "06_final_predictor_list.csv",
        TABLES_DIR / "08_imbalance_strategy_comparison.csv",
        TABLES_DIR / "13_model_evaluation_metrics.csv",
        TABLES_DIR / "14_logistic_regression_odds_ratios.csv"
    ]
    for table in key_tables:
        assert table.exists(), f"Table missing: {table.name}"

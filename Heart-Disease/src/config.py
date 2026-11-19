# ============================================================
# src/config.py
# ------------------------------------------------------------
# Central configuration file for the Heart Disease ML project.
# All file paths, constants, and settings are defined here.
# Import this module in every script/notebook instead of
# hard-coding paths or values.
# ============================================================

from pathlib import Path

# ============================================================
# ROOT DIRECTORY
# ============================================================

# This file lives inside Heart-Disease/src/
# Two levels up gives us the Heart-Disease project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_DIR        = PROJECT_ROOT / "dataset"
RAW_DIR            = DATASET_DIR / "raw"
PROCESSED_DIR      = DATASET_DIR / "processed"

# Original compressed CDC BRFSS 2021 file — NEVER modified
RAW_ZIP_PATH       = RAW_DIR / "LLCP2021XPT.zip"

# Extracted SAS transport file (created on first run, then cached)
RAW_XPT_PATH       = RAW_DIR / "LLCP2021.XPT"

# Cleaned dataset saved after Phase 3
CLEANED_CSV_PATH   = PROCESSED_DIR / "heart_cleaned.csv"

# Feature-engineered dataset saved after Phase 5
FEATURED_CSV_PATH  = PROCESSED_DIR / "heart_featured.csv"


# ============================================================
# REPORT DIRECTORIES
# ============================================================

REPORTS_DIR        = PROJECT_ROOT / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
TABLES_DIR         = REPORTS_DIR / "tables"
RESULTS_DIR        = REPORTS_DIR / "results"


# ============================================================
# MODEL SAVE DIRECTORY
# ============================================================

MODELS_DIR         = PROJECT_ROOT / "models" / "saved_models"


# ============================================================
# DOCUMENTATION FILE
# ============================================================

DOCS_PATH          = PROJECT_ROOT / "PROJECT_DOCUMENTATION.txt"


# ============================================================
# TARGET VARIABLE
# ============================================================

# CDC BRFSS 2021 — computed variable for heart disease / MI
# Coding in raw data:
#   1 = History of MI or coronary heart disease / angina  → Positive (1)
#   2 = Neither condition reported                        → Negative (0)
#   Missing / refused / other                             → Excluded
TARGET_COLUMN      = "_MICHD"
TARGET_POSITIVE    = 1   # Heart disease present
TARGET_NEGATIVE    = 0   # No heart disease

# Raw BRFSS codes for the target variable
MICHD_CODE_POSITIVE    = 1
MICHD_CODE_NEGATIVE    = 2
MICHD_CODES_EXCLUDE    = [7, 9]   # Refused / Don't know


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED        = 42


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

TEST_SIZE          = 0.20   # 80 % train, 20 % test


# ============================================================
# CROSS-VALIDATION
# ============================================================

CV_FOLDS           = 5     # Stratified K-Fold


# ============================================================
# MISSING VALUE THRESHOLDS
# ============================================================

# Drop a column if more than this fraction of its values are missing.
# Chosen at 60 % — columns beyond this carry too little information
# to justify imputation.
COLUMN_DROP_MISSING_THRESHOLD = 0.60

# Drop a row if the target value is missing (non-imputable).
# Rows with missing predictors are handled via pipeline imputation.


# ============================================================
# GRAPH SETTINGS
# ============================================================

FIGURE_DPI         = 300      # High-resolution output
FIGURE_STYLE       = "whitegrid"
FIGURE_PALETTE     = "Set2"


# ============================================================
# HELPER: ensure output directories exist
# ============================================================

def ensure_directories():
    """
    Create all required output directories if they do not exist.
    Call this at the start of any script or notebook.
    """
    directories = [
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        RESULTS_DIR,
        MODELS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELF-CHECK (run this file directly to verify paths)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Heart Disease Project — Configuration")
    print("=" * 60)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Raw ZIP       : {RAW_ZIP_PATH}  (exists: {RAW_ZIP_PATH.exists()})")
    print(f"Cleaned CSV   : {CLEANED_CSV_PATH}")
    print(f"Figures dir   : {FIGURES_DIR}")
    print(f"Models dir    : {MODELS_DIR}")
    print(f"Random seed   : {RANDOM_SEED}")
    print(f"Test size     : {TEST_SIZE}")
    print(f"CV folds      : {CV_FOLDS}")

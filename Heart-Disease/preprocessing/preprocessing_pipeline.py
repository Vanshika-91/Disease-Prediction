# ============================================================
# preprocessing/preprocessing_pipeline.py
# ------------------------------------------------------------
# Reusable Scikit-Learn Preprocessing Pipeline for CDC BRFSS.
# Follows strict leakage prevention:
#   - Fits transformers ONLY on training splits
#   - Transforms test splits without re-fitting
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_PREDICTORS = [
    "BMI_VAL", "PHYSHLTH", "MENTHLTH", "RISK_FACTOR_COUNT"
]

ORDINAL_BINARY_PREDICTORS = [
    "_AGEG5YR", "SEX_MALE", "GENHLTH_SCORE", "HIGH_BP", "HIGH_CHOL",
    "DIABETES", "PRIOR_STROKE", "SMOKER_EVER", "PHYS_ACTIVE",
    "CHECKUP1", "EDUCA", "INCOME3"
]

NOMINAL_CATEGORICAL_PREDICTORS = [
    "_RACE"
]

def build_preprocessing_pipeline(include_scaling: bool = True):
    """
    Builds a robust ColumnTransformer that:
    1. Imputes numerical variables with median and scales with StandardScaler (if enabled)
    2. Imputes ordinal and binary features with most frequent value
    3. Imputes nominal categorical features and One-Hot Encodes them
    """
    if include_scaling:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_PREDICTORS),
            ("ord", ord_pipeline, ORDINAL_BINARY_PREDICTORS),
            ("cat", cat_pipeline, NOMINAL_CATEGORICAL_PREDICTORS)
        ],
        remainder="drop"
    )
    return preprocessor

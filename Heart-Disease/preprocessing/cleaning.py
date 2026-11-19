# ============================================================
# preprocessing/cleaning.py
# ------------------------------------------------------------
# Reusable cleaning functions for the BRFSS dataset.
# These functions are called from notebook 03_data_cleaning.py
# and can be imported into other scripts as needed.
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# NON-RESPONSE CODE DEFINITIONS
# ============================================================

# BRFSS-specific non-response codes per variable.
# These codes must be replaced with NaN during cleaning.
# Source: CDC BRFSS 2021 Codebook
BRFSS_NONRESPONSE_MAP = {
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
    "GENHLTH"  : [7, 9],
    "CHECKUP1" : [9],
    "_AGEG5YR" : [14],
    "_SMOKER3" : [9],
    "_TOTINDA" : [9],
    "_RFDRHV7" : [9],
    "PHYSHLTH" : [77, 99],
    "MENTHLTH" : [77, 99],
    "POORHLTH" : [77, 99],
    "SLEPTIM1" : [77, 99],
    "ALCDAY4"  : [777, 999],
    "EDUCA"    : [9],
    "INCOME3"  : [77, 99],
}

# Codes that mean "None / Zero" (valid, not missing)
BRFSS_ZERO_RECODE_MAP = {
    "PHYSHLTH" : [88],    # 88 = "None" → 0 days
    "MENTHLTH" : [88],
    "POORHLTH" : [88],
    "ALCDAY4"  : [888],   # 888 = "No drinks" → 0
}

# Columns used to construct _MICHD — target leakage
LEAKAGE_COLUMNS = ["CVDINFR4", "CVDCRHD4"]

# Survey design / administrative columns — not predictors
SURVEY_DESIGN_COLUMNS = [
    "_PSU", "_STSTR", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_LLCPWT2",
    "_DUALUSE", "_DUALCOR", "_CLLCPWT", "_STRWT", "IDATE", "IMONTH",
    "IDAY", "IYEAR", "FMONTH", "SEQNO", "_STATE",
]


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def recode_target(df: pd.DataFrame,
                  target_col: str = "_MICHD",
                  positive_code: int = 1,
                  negative_code: int = 2) -> pd.DataFrame:
    """
    Recode the _MICHD target variable from CDC codes to binary (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the raw target column.
    target_col : str
        Name of the target column.
    positive_code : int
        CDC raw code for positive class (default 1).
    negative_code : int
        CDC raw code for negative class (default 2).

    Returns
    -------
    pd.DataFrame
        DataFrame with target recoded; rows with unknown target dropped.
    """
    df = df.copy()
    df[target_col] = df[target_col].map({positive_code: 1, negative_code: 0})
    rows_before = len(df)
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    rows_dropped = rows_before - len(df)
    print(f"[recode_target] Dropped {rows_dropped:,} rows with unknown _MICHD.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the DataFrame."""
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"[remove_duplicates] Removed {n_before - len(df):,} duplicate rows.")
    return df


def recode_nonresponse(df: pd.DataFrame,
                       nonresponse_map: dict = None,
                       zero_recode_map: dict = None) -> pd.DataFrame:
    """
    Replace BRFSS non-response codes with NaN.
    Replace "None/Zero" codes with 0.

    Parameters
    ----------
    df : pd.DataFrame
    nonresponse_map : dict
        {column: [codes to replace with NaN]}
    zero_recode_map : dict
        {column: [codes to replace with 0]}

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if nonresponse_map is None:
        nonresponse_map = BRFSS_NONRESPONSE_MAP
    if zero_recode_map is None:
        zero_recode_map = BRFSS_ZERO_RECODE_MAP

    total_recoded = 0
    for col, codes in nonresponse_map.items():
        if col not in df.columns:
            continue
        n = df[col].isin(codes).sum()
        df[col] = df[col].replace(codes, np.nan)
        total_recoded += n

    for col, codes in zero_recode_map.items():
        if col not in df.columns:
            continue
        df[col] = df[col].replace(codes, 0)

    print(f"[recode_nonresponse] Recoded {total_recoded:,} cells to NaN.")
    return df


def drop_unsuitable_columns(df: pd.DataFrame,
                             leakage_cols: list = None,
                             design_cols: list = None,
                             missing_threshold: float = 0.60) -> pd.DataFrame:
    """
    Drop columns that are unsuitable for prediction.

    Drops:
    - Leakage columns (used to build _MICHD)
    - Survey design / administrative columns
    - Constant columns (≤1 unique value)
    - Columns with more than missing_threshold fraction missing

    Parameters
    ----------
    df : pd.DataFrame
    leakage_cols : list
        Column names known to cause target leakage.
    design_cols : list
        Survey design column names.
    missing_threshold : float
        Maximum allowed fraction of missing values.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if leakage_cols is None:
        leakage_cols = LEAKAGE_COLUMNS
    if design_cols is None:
        design_cols = SURVEY_DESIGN_COLUMNS

    to_drop = []

    # Leakage
    to_drop += [c for c in leakage_cols if c in df.columns]

    # Survey design
    to_drop += [c for c in design_cols if c in df.columns]

    # Constant
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    to_drop += constant

    # High missing (never drop target)
    high_miss = [
        c for c in df.columns
        if df[c].isnull().mean() > missing_threshold and c != "_MICHD"
    ]
    to_drop += high_miss

    to_drop = list(set(to_drop))
    df.drop(columns=to_drop, inplace=True)

    print(f"[drop_unsuitable_columns] Dropped {len(to_drop)} columns.")
    return df

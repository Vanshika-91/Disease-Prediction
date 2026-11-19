# ============================================================
# src/utils.py
# ------------------------------------------------------------
# Shared helper functions used across multiple notebooks
# and scripts.  Keep functions small and clearly named.
# ============================================================

import textwrap
from pathlib import Path
from datetime import datetime


# ============================================================
# DOCUMENTATION HELPER
# ============================================================

def append_to_documentation(section_title: str, content: str, docs_path: Path):
    """
    Append a titled section to PROJECT_DOCUMENTATION.txt.

    Parameters
    ----------
    section_title : str
        Title printed as a header (e.g., "PHASE 1 — DATASET UNDERSTANDING").
    content : str
        Multi-line text to write under the header.
    docs_path : Path
        Absolute path to PROJECT_DOCUMENTATION.txt.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 70

    block = (
        f"\n{separator}\n"
        f"{section_title}\n"
        f"Updated: {timestamp}\n"
        f"{separator}\n"
        f"{content}\n"
    )

    with open(docs_path, "a", encoding="utf-8") as file:
        file.write(block)

    print(f"[Documentation] Section '{section_title}' appended.")


# ============================================================
# PRINT HELPERS
# ============================================================

def section_header(title: str, width: int = 70):
    """
    Print a clearly visible section header to the console.

    Parameters
    ----------
    title : str
        Section title text.
    width : int
        Total width of the separator line.
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def sub_header(title: str, width: int = 50):
    """Print a sub-section header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def check_no_target_in_features(X_columns, target_column: str):
    """
    Verify that the target column is not accidentally included in
    the predictor set.  Raises ValueError if leakage is detected.

    Parameters
    ----------
    X_columns : list or Index
        Column names of the predictor DataFrame.
    target_column : str
        Name of the target variable to check for.
    """
    if target_column in list(X_columns):
        raise ValueError(
            f"DATA LEAKAGE DETECTED: '{target_column}' is present in X "
            f"(predictor columns). Remove it before training."
        )
    print(f"[Validation] Target '{target_column}' is NOT in X. ✓")


def check_no_missing_values(dataframe, label: str = "DataFrame"):
    """
    Check whether a DataFrame contains any missing values.
    Prints a summary and returns the total missing count.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to inspect.
    label : str
        Name printed in the output for clarity.
    """
    total_missing = dataframe.isnull().sum().sum()
    if total_missing == 0:
        print(f"[Validation] {label}: No missing values found. ✓")
    else:
        print(
            f"[Validation] WARNING — {label}: "
            f"{total_missing} missing values remain."
        )
    return total_missing


def check_class_proportions(y_train, y_test, target_name: str = "target"):
    """
    Compare class proportions in training and test sets to verify
    that stratified splitting preserved the class balance.

    Parameters
    ----------
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.
    target_name : str
        Name of the target variable for display.
    """
    train_counts = y_train.value_counts(normalize=True).sort_index() * 100
    test_counts  = y_test.value_counts(normalize=True).sort_index() * 100

    print(f"\n[Validation] Class proportions for '{target_name}':")
    print(f"{'Class':<10} {'Train %':>10} {'Test %':>10}")
    print("-" * 32)
    for cls in sorted(set(y_train.unique()) | set(y_test.unique())):
        train_pct = train_counts.get(cls, 0.0)
        test_pct  = test_counts.get(cls, 0.0)
        print(f"{cls:<10} {train_pct:>9.2f}% {test_pct:>9.2f}%")

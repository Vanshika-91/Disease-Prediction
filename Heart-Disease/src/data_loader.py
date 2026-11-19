# ============================================================
# src/data_loader.py
# ------------------------------------------------------------
# Responsible for loading the raw CDC BRFSS 2021 dataset.
# The raw ZIP file is NEVER modified.
# On first run the XPT file is extracted to dataset/raw/.
# On subsequent runs the cached XPT is used directly.
# ============================================================

import sys
import zipfile
from pathlib import Path

import pandas as pd

# Allow importing config from the same src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_ZIP_PATH, RAW_XPT_PATH


# ============================================================
# STEP 1 — EXTRACT ZIP (if not already extracted)
# ============================================================

def extract_xpt_from_zip(zip_path: Path = RAW_ZIP_PATH,
                          xpt_path: Path = RAW_XPT_PATH) -> Path:
    """
    Extract the SAS Transport (.XPT) file from the CDC BRFSS ZIP archive.

    Why:  pandas.read_sas() requires an uncompressed file on disk or a
          file-like object.  We extract once and cache for speed.

    Parameters
    ----------
    zip_path : Path
        Location of the downloaded ZIP file.
    xpt_path : Path
        Target extraction path for the .XPT file.

    Returns
    -------
    Path
        Path to the extracted .XPT file.
    """
    if xpt_path.exists():
        print(f"[DataLoader] XPT already extracted at: {xpt_path}")
        return xpt_path

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Raw ZIP file not found: {zip_path}\n"
            f"Please place LLCP2021XPT.zip in {zip_path.parent}"
        )

    print(f"[DataLoader] Extracting ZIP: {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # List contents to find the XPT file
        members = zf.namelist()
        xpt_members = [m for m in members if m.strip().upper().endswith(".XPT")]

        if not xpt_members:
            raise ValueError(
                f"No .XPT file found inside {zip_path.name}.\n"
                f"ZIP contents: {members}"
            )

        # Use the first (and typically only) XPT file
        xpt_member = xpt_members[0]
        print(f"[DataLoader] Found: {xpt_member}")

        # Extract to raw directory
        extracted_name = zf.extract(xpt_member, path=xpt_path.parent)
        extracted = Path(extracted_name)

        # Rename to our expected filename if the extracted name differs
        if extracted.resolve() != xpt_path.resolve():
            if xpt_path.exists():
                xpt_path.unlink()
            extracted.rename(xpt_path)

    print(f"[DataLoader] Extraction complete: {xpt_path}")
    return xpt_path


# ============================================================
# STEP 2 — LOAD RAW DATASET
# ============================================================

def load_raw_dataset(xpt_path: Path = None,
                     zip_path: Path = RAW_ZIP_PATH) -> pd.DataFrame:
    """
    Load the CDC BRFSS 2021 dataset from the SAS Transport file.

    Why:  The BRFSS 2021 data is distributed as a SAS XPT file.
          pandas.read_sas() with format='xport' reads this directly.
          We load every column so that the caller can inspect the full
          dataset before deciding which columns to keep.

    Parameters
    ----------
    xpt_path : Path or None
        Path to the extracted .XPT file.  If None, uses RAW_XPT_PATH
        and extracts the ZIP if needed.
    zip_path : Path
        Path to the ZIP file, used only if extraction is needed.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified CDC BRFSS 2021 dataset.
    """
    # Resolve the XPT path
    if xpt_path is None:
        xpt_path = extract_xpt_from_zip(zip_path)

    print(f"[DataLoader] Loading dataset from: {xpt_path.name}")
    print("[DataLoader] This may take 30–60 seconds for a ~400 K row file ...")

    # pandas.read_sas() with format='xport' reads SAS Transport format.
    # encoding='latin-1' is correct for SAS/CDC XPT exports.
    df = pd.read_sas(
        str(xpt_path),
        format="xport",
        encoding="latin-1"
    )

    print(f"[DataLoader] Dataset loaded successfully.")
    print(f"[DataLoader] Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


# ============================================================
# SELF-CHECK
# ============================================================

if __name__ == "__main__":
    df = load_raw_dataset()
    print("\nFirst 3 rows (selected columns):")
    print(df.head(3).to_string())

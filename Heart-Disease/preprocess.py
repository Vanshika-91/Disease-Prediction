import os
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/processed.cleveland.data"
OUTPUT_DIR = "dataset"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

VALID_TARGETS = [0, 1, 2, 3, 4]

# ============================================================
# START
# ============================================================

print("=" * 70)
print("HEART DISEASE - DATA PREPROCESSING")
print("=" * 70)

# ============================================================
# 1. CHECK DATASET
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    names=COLUMNS,
    na_values="?"
)

df = df.apply(pd.to_numeric, errors="coerce")

print(f"Original shape: {df.shape}")

# ============================================================
# 2. VALIDATE TARGET
# ============================================================

print("\n[2] Validating target...")

# Remove rows where target itself is missing
target_missing = df["target"].isnull().sum()

if target_missing > 0:

    print(
        f"Removing {target_missing} rows "
        "with missing target."
    )

    df = df.dropna(subset=["target"])

# Convert target to integer
df["target"] = df["target"].astype(int)

unexpected_targets = sorted(
    set(df["target"].unique()) - set(VALID_TARGETS)
)

if unexpected_targets:

    raise ValueError(
        f"Unexpected target values found: "
        f"{unexpected_targets}"
    )

print("✓ Target values are valid.")

print("\nTarget meaning:")
print("0 = No heart disease")
print("1 = Severity 1")
print("2 = Severity 2")
print("3 = Severity 3")
print("4 = Severity 4")

# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

print("\n[3] Removing duplicates...")

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()

# Save cleaned dataset
cleaned_path = os.path.join(
    OUTPUT_DIR,
    "heart_cleaned.csv"
)

df.to_csv(
    cleaned_path,
    index=False
)

print(f"✓ Saved: {cleaned_path}")

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n[4] Separating features and target...")

X = df.drop(columns=["target"])
y = df["target"]

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[5] Performing train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\n[6] Handling missing values...")

# IMPORTANT:
# Statistics are calculated ONLY from X_train.

for feature in NUMERICAL_FEATURES:

    median_value = X_train[feature].median()

    X_train[feature] = X_train[feature].fillna(
        median_value
    )

    X_test[feature] = X_test[feature].fillna(
        median_value
    )

    print(
        f"{feature}: missing values → "
        f"training median {median_value}"
    )


for feature in CATEGORICAL_FEATURES:

    mode_values = X_train[feature].mode()

    if len(mode_values) == 0:
        raise ValueError(
            f"Cannot calculate mode for {feature}"
        )

    mode_value = mode_values.iloc[0]

    X_train[feature] = X_train[feature].fillna(
        mode_value
    )

    X_test[feature] = X_test[feature].fillna(
        mode_value
    )

    print(
        f"{feature}: missing values → "
        f"training mode {mode_value}"
    )

# ============================================================
# 7. ONE-HOT ENCODING
# ============================================================

print("\n[7] Encoding categorical features...")

X_train = pd.get_dummies(
    X_train,
    columns=CATEGORICAL_FEATURES,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=CATEGORICAL_FEATURES,
    dtype=int
)

# Ensure test has exactly the same columns as train
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print(
    f"Features after encoding: {X_train.shape[1]}"
)

# ============================================================
# 8. SCALE NUMERICAL FEATURES
# ============================================================

print("\n[8] Scaling numerical features...")

scaler = StandardScaler()

# Fit ONLY on training data
X_train[NUMERICAL_FEATURES] = scaler.fit_transform(
    X_train[NUMERICAL_FEATURES]
)

# Apply the same transformation to test data
X_test[NUMERICAL_FEATURES] = scaler.transform(
    X_test[NUMERICAL_FEATURES]
)

print(
    f"Scaled features: {NUMERICAL_FEATURES}"
)

# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\n[9] Final preprocessing checks...")

train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

print(
    f"Training missing values: {train_missing}"
)

print(
    f"Testing missing values: {test_missing}"
)

if train_missing != 0 or test_missing != 0:
    raise ValueError(
        "Missing values still remain after preprocessing."
    )

# Check train/test feature consistency
if list(X_train.columns) != list(X_test.columns):

    raise ValueError(
        "Training and testing features do not match."
    )

print("✓ No missing feature values remain.")
print("✓ Train/test feature columns match.")

# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print("\n[10] Target distribution")

print("\nTraining:")
print(y_train.value_counts().sort_index())

print("\nTesting:")
print(y_test.value_counts().sort_index())

# ============================================================
# 11. SAVE PROCESSED DATASETS
# ============================================================

print("\n[11] Saving processed datasets...")

# X + y combined for convenient inspection
train_processed = X_train.copy()
train_processed["target"] = y_train.values

test_processed = X_test.copy()
test_processed["target"] = y_test.values

processed = pd.concat(
    [train_processed, test_processed],
    ignore_index=True
)

processed_path = os.path.join(
    OUTPUT_DIR,
    "heart_processed.csv"
)

processed.to_csv(
    processed_path,
    index=False
)

# Training features
X_train.to_csv(
    os.path.join(OUTPUT_DIR, "X_train.csv"),
    index=False
)

# Testing features
X_test.to_csv(
    os.path.join(OUTPUT_DIR, "X_test.csv"),
    index=False
)

# Training target
y_train.to_csv(
    os.path.join(OUTPUT_DIR, "y_train.csv"),
    index=False,
    header=["target"]
)

# Testing target
y_test.to_csv(
    os.path.join(OUTPUT_DIR, "y_test.csv"),
    index=False,
    header=["target"]
)

# ============================================================
# 12. SAVE PREPROCESSING SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "preprocessing_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HEART DISEASE DATA PREPROCESSING SUMMARY\n"
    )

    file.write("=" * 55 + "\n\n")

    file.write(
        f"Original dataset shape: {df.shape}\n"
    )

    file.write(
        f"Training samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing samples: {len(X_test)}\n"
    )

    file.write(
        "Train/test split: 80/20\n"
    )

    file.write(
        "Random state: 42\n"
    )

    file.write(
        "Stratification: Yes\n\n"
    )

    file.write(
        "Missing value handling:\n"
    )

    file.write(
        "- Numerical features: training-set median\n"
    )

    file.write(
        "- Categorical features: training-set mode\n\n"
    )

    file.write(
        "Categorical encoding:\n"
    )

    file.write(
        "- One-hot encoding\n\n"
    )

    file.write(
        "Scaling:\n"
    )

    file.write(
        "- StandardScaler\n"
    )

    file.write(
        "- Fitted only on training data\n\n"
    )

    file.write(
        "Target mapping:\n"
    )

    file.write(
        "0 = No heart disease\n"
    )

    file.write(
        "1 = Severity 1\n"
    )

    file.write(
        "2 = Severity 2\n"
    )

    file.write(
        "3 = Severity 3\n"
    )

    file.write(
        "4 = Severity 4\n\n"
    )

    file.write(
        f"Final number of features: {X_train.shape[1]}\n"
    )

    file.write(
        "Machine learning model trained: NO\n"
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")
print("✓ dataset/heart_cleaned.csv")
print("✓ dataset/heart_processed.csv")
print("✓ dataset/X_train.csv")
print("✓ dataset/X_test.csv")
print("✓ dataset/y_train.csv")
print("✓ dataset/y_test.csv")
print("✓ dataset/preprocessing_summary.txt")

print("\nFinal training shape:")
print(X_train.shape)

print("\nFinal testing shape:")
print(X_test.shape)

print("\nTarget values:")
print(sorted(y.unique()))

print("\nTarget meaning:")
print("0 → No disease")
print("1 → Severity 1")
print("2 → Severity 2")
print("3 → Severity 3")
print("4 → Severity 4")

print("\nNO MACHINE LEARNING MODEL WAS TRAINED.")

print("=" * 70)
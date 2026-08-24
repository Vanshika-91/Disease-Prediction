
import pandas as pd
import os


# ============================================================
# 1. Column names
# ============================================================

columns = [
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


# ============================================================
# 2. Load the UCI Cleveland dataset
# ============================================================

data_path = "dataset/processed.cleveland.data"

df = pd.read_csv(
    data_path,
    names=columns,
    na_values="?"
)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ============================================================
# 3. Convert all columns to numeric
# ============================================================

df = df.apply(pd.to_numeric, errors="coerce")

print("\nData types:")
print(df.dtypes)


# ============================================================
# 4. Check missing values
# ============================================================

print("\nMissing values before handling:")
print(df.isnull().sum())


# ============================================================
# 5. Handle missing values
# ============================================================

# Fill missing numerical values with the median
df = df.fillna(df.median(numeric_only=True))

print("\nMissing values after handling:")
print(df.isnull().sum())


# ============================================================
# 6. Convert target into binary
# ============================================================

# Original UCI target:
# 0 = No disease
# 1, 2, 3, 4 = Disease

df["target"] = (df["target"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nTarget distribution (%):")
print(df["target"].value_counts(normalize=True) * 100)


# ============================================================
# 7. Separate features and target
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 8. Create cleaned dataset
# ============================================================

# Combine the cleaned features and target
cleaned_df = X.copy()
cleaned_df["target"] = y


# ============================================================
# 9. Save cleaned dataset
# ============================================================

os.makedirs("dataset", exist_ok=True)

output_path = "dataset/heart_cleaned.csv"

cleaned_df.to_csv(
    output_path,
    index=False
)

print("\n=============================")
print("PREPROCESSING COMPLETED")
print("=============================")

print("Cleaned dataset shape:", cleaned_df.shape)
print("Saved to:", output_path)

print("\nFinal missing values:")
print(cleaned_df.isnull().sum())

print("\nFirst 5 cleaned rows:")
print(cleaned_df.head())


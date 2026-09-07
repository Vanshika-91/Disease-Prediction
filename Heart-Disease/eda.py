import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/processed.cleveland.data"
EDA_DIR = "eda"
PLOTS_DIR = os.path.join(EDA_DIR, "plots")

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

# ============================================================
# 1. CLEAN OLD EDA OUTPUTS
# ============================================================

if os.path.exists(EDA_DIR):
    shutil.rmtree(EDA_DIR)

os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("HEART DISEASE - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ============================================================
# 2. LOAD DATA
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

print(f"Dataset shape: {df.shape}")

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n[2] Dataset information")

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")

print("\nData types:")
print(df.dtypes)

# Save data types
dtype_df = pd.DataFrame({
    "feature": df.columns,
    "data_type": df.dtypes.astype(str).values
})

dtype_df.to_csv(
    os.path.join(EDA_DIR, "data_types.csv"),
    index=False
)

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\n[3] Descriptive statistics")

statistics = df.describe().T
statistics.to_csv(
    os.path.join(EDA_DIR, "descriptive_statistics.csv")
)

print(statistics)

# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n[4] Missing value analysis")

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (df.isnull().mean() * 100).round(2)
})

missing.to_csv(
    os.path.join(EDA_DIR, "missing_values.csv")
)

print(missing)

# ============================================================
# 6. DUPLICATES
# ============================================================

print("\n[5] Duplicate analysis")

duplicate_count = df.duplicated().sum()

duplicate_analysis = pd.DataFrame({
    "duplicate_rows": [duplicate_count],
    "total_rows": [len(df)],
    "duplicate_percentage": [
        round((duplicate_count / len(df)) * 100, 2)
    ]
})

duplicate_analysis.to_csv(
    os.path.join(EDA_DIR, "duplicate_analysis.csv"),
    index=False
)

print(f"Duplicate rows: {duplicate_count}")

# ============================================================
# 7. TARGET ANALYSIS
# ============================================================

print("\n[6] Target analysis")

VALID_TARGETS = [0, 1, 2, 3, 4]

target_counts = df["target"].value_counts().sort_index()

print("\nTarget distribution:")
print(target_counts)

# Check unexpected target values
unexpected_targets = sorted(
    set(df["target"].dropna().unique()) - set(VALID_TARGETS)
)

if unexpected_targets:
    print(
        f"WARNING: Unexpected target values found: "
        f"{unexpected_targets}"
    )
else:
    print("\n✓ Target values are valid: 0, 1, 2, 3, 4")

target_mapping = {
    0: "No heart disease",
    1: "Severity 1",
    2: "Severity 2",
    3: "Severity 3",
    4: "Severity 4"
}

target_table = pd.DataFrame({
    "target": VALID_TARGETS,
    "meaning": [target_mapping[x] for x in VALID_TARGETS],
    "count": [target_counts.get(x, 0) for x in VALID_TARGETS]
})

target_table["percentage"] = (
    target_table["count"] / len(df) * 100
).round(2)

target_table.to_csv(
    os.path.join(EDA_DIR, "target_distribution.csv"),
    index=False
)

# ============================================================
# 8. TARGET GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="target",
    order=VALID_TARGETS
)

plt.title("Heart Disease Severity Distribution")
plt.xlabel("Target / Severity")
plt.ylabel("Number of Patients")

plt.xticks(
    VALID_TARGETS,
    [
        "0 - No Disease",
        "1 - Severity 1",
        "2 - Severity 2",
        "3 - Severity 3",
        "4 - Severity 4"
    ],
    rotation=20
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "01_target_distribution.png"),
    dpi=300
)

plt.close()

# ============================================================
# 9. NUMERICAL DISTRIBUTIONS
# ============================================================

print("\n[7] Numerical feature analysis")

# One combined figure instead of many separate graphs
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8)
)

axes = axes.flatten()

for i, feature in enumerate(NUMERICAL_FEATURES):

    sns.histplot(
        df[feature].dropna(),
        kde=True,
        ax=axes[i]
    )

    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

# Hide unused sixth subplot
axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "02_numerical_distributions.png"),
    dpi=300
)

plt.close()

# ============================================================
# 10. OUTLIER ANALYSIS
# ============================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df[NUMERICAL_FEATURES]
)

plt.title("Numerical Features - Outlier Analysis")
plt.xlabel("Features")
plt.ylabel("Values")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "03_numerical_boxplots.png"),
    dpi=300
)

plt.close()

# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print("\n[8] Correlation analysis")

correlation = df.corr(numeric_only=True)

correlation.to_csv(
    os.path.join(EDA_DIR, "correlation_matrix.csv")
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "04_correlation_heatmap.png"),
    dpi=300
)

plt.close()

# ============================================================
# 12. FEATURE VS TARGET
# ============================================================

print("\n[9] Feature vs target analysis")

# Numerical features vs severity
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8)
)

axes = axes.flatten()

for i, feature in enumerate(NUMERICAL_FEATURES):

    sns.boxplot(
        data=df,
        x="target",
        y=feature,
        order=VALID_TARGETS,
        ax=axes[i]
    )

    axes[i].set_title(f"{feature} vs Disease Severity")
    axes[i].set_xlabel("Target / Severity")
    axes[i].set_ylabel(feature)

axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "05_numerical_vs_target.png"),
    dpi=300
)

plt.close()

# ============================================================
# 13. SIMPLE CATEGORICAL VS TARGET ANALYSIS
# ============================================================

# Combine categorical relationships into one figure
fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 9)
)

axes = axes.flatten()

for i, feature in enumerate(CATEGORICAL_FEATURES):

    cross_table = pd.crosstab(
        df[feature],
        df["target"]
    )

    cross_table.plot(
        kind="bar",
        ax=axes[i]
    )

    axes[i].set_title(f"{feature} vs Target")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")
    axes[i].legend(
        title="Severity",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "06_categorical_vs_target.png"),
    dpi=300
)

plt.close()

# ============================================================
# 14. FEATURE SUMMARY
# ============================================================

feature_summary = []

for feature in df.columns:

    feature_summary.append({
        "feature": feature,
        "type": (
            "numerical"
            if feature in NUMERICAL_FEATURES
            else "categorical"
            if feature in CATEGORICAL_FEATURES
            else "target"
        ),
        "missing_values": int(df[feature].isnull().sum()),
        "unique_values": int(df[feature].nunique())
    })

feature_summary = pd.DataFrame(feature_summary)

feature_summary.to_csv(
    os.path.join(EDA_DIR, "feature_summary.csv"),
    index=False
)

# ============================================================
# 15. AUTOMATIC OBSERVATIONS
# ============================================================

print("\n[10] Generating observations...")

observations = []

observations.append(
    f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
)

observations.append(
    f"Duplicate rows found: {duplicate_count}."
)

missing_features = missing[
    missing["missing_count"] > 0
].index.tolist()

if missing_features:
    observations.append(
        "Missing values are present in: "
        + ", ".join(missing_features)
        + "."
    )
else:
    observations.append(
        "No missing values were found."
    )

observations.append(
    "The target variable contains five classes: "
    "0, 1, 2, 3 and 4."
)

observations.append(
    "Target interpretation: "
    "0 = No heart disease; "
    "1–4 = increasing disease severity."
)

# Highest absolute correlations with target
target_correlations = (
    correlation["target"]
    .drop("target")
    .abs()
    .sort_values(ascending=False)
)

if len(target_correlations) > 0:

    strongest_feature = target_correlations.index[0]

    observations.append(
        f"The feature with the strongest absolute correlation "
        f"with target is '{strongest_feature}'."
    )

# ============================================================
# SAVE OBSERVATIONS
# ============================================================

with open(
    os.path.join(EDA_DIR, "eda_observations.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HEART DISEASE DATASET - EDA OBSERVATIONS\n"
    )

    file.write("=" * 50 + "\n\n")

    for i, observation in enumerate(observations, 1):

        file.write(
            f"{i}. {observation}\n"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated folder:")
print("eda/")

print("\nGenerated reports:")
print("✓ data_types.csv")
print("✓ descriptive_statistics.csv")
print("✓ missing_values.csv")
print("✓ duplicate_analysis.csv")
print("✓ target_distribution.csv")
print("✓ correlation_matrix.csv")
print("✓ feature_summary.csv")
print("✓ eda_observations.txt")

print("\nGenerated graphs:")
print("✓ 01_target_distribution.png")
print("✓ 02_numerical_distributions.png")
print("✓ 03_numerical_boxplots.png")
print("✓ 04_correlation_heatmap.png")
print("✓ 05_numerical_vs_target.png")
print("✓ 06_categorical_vs_target.png")

print("\nNo machine learning model was trained.")
print("=" * 70)
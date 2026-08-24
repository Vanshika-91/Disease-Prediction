import pandas as pd

file_path = "dataset/processed.cleveland.data"

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

df = pd.read_csv(file_path, header=None, names=columns)

# Replace ? with missing values
df = df.replace("?", pd.NA)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Remove rows containing missing values
df = df.dropna()

# Convert target: 0 = No Disease, 1-4 = Disease
df["target"] = (df["target"] > 0).astype(int)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "heart_disease_model.pkl")

print("Model saved successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Column names
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# 2. Load dataset
data_path = "dataset/processed.cleveland.data"

df = pd.read_csv(
    data_path,
    names=columns,
    na_values="?"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# 3. Convert columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# 4. Handle missing values
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# 5. Convert target into binary
# 0 = No disease
# 1,2,3,4 = Disease
df["target"] = (df["target"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

# 6. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL RESULTS")
print("-----------------------------")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Save model and scaler
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/heart_disease_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")
print("model/heart_disease_model.pkl")
print("model/scaler.pkl")
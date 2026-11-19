"""
Interactive Heart Disease Prediction CLI
Loads the trained champion pipeline and prompts the user for patient indicators.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "saved_models" / "heart_disease_prediction_pipeline.joblib"

def ask_binary(question):
    while True:
        resp = input(f"{question} (y/n): ").strip().lower()
        if resp in ['y', 'yes', '1']:
            return 1
        elif resp in ['n', 'no', '0']:
            return 0
        print("Please enter 'y' or 'n'.")

def ask_int(question, min_val, max_val, default=None):
    while True:
        prompt = f"{question} [{min_val}-{max_val}]"
        if default is not None:
            prompt += f" (default: {default})"
        prompt += ": "
        resp = input(prompt).strip()
        if not resp and default is not None:
            return default
        try:
            val = int(resp)
            if min_val <= val <= max_val:
                return val
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid integer.")

def ask_float(question, min_val, max_val, default=None):
    while True:
        prompt = f"{question} [{min_val}-{max_val}]"
        if default is not None:
            prompt += f" (default: {default})"
        prompt += ": "
        resp = input(prompt).strip()
        if not resp and default is not None:
            return default
        try:
            val = float(resp)
            if min_val <= val <= max_val:
                return val
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

def map_age_to_code(age):
    if age < 25: return 1   # 18-24
    elif age < 30: return 2  # 25-29
    elif age < 35: return 3  # 30-34
    elif age < 40: return 4  # 35-39
    elif age < 45: return 5  # 40-44
    elif age < 50: return 6  # 45-49
    elif age < 55: return 7  # 50-54
    elif age < 60: return 8  # 55-59
    elif age < 65: return 9  # 60-64
    elif age < 70: return 10 # 65-69
    elif age < 75: return 11 # 70-74
    elif age < 80: return 12 # 75-79
    else: return 13          # 80+

def main():
    print("=" * 60)
    print("   HEART DISEASE PREDICTION SYSTEM - INTERACTIVE SCREENING")
    print("   CDC BRFSS 2021 Model | Supervised Binary Classification")
    print("=" * 60)
    print("Please enter the patient details below:\n")

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)

    # 1. Demographics
    sex = ask_binary("1. Is the patient Male?")
    age = ask_int("2. Patient Age in years", 18, 120, default=45)
    age_code = map_age_to_code(age)

    # 2. General Health
    print("\n3. How would the patient rate their general health?")
    print("   1: Excellent | 2: Very Good | 3: Good | 4: Fair | 5: Poor")
    genhlth = ask_int("   Select rating", 1, 5, default=3)

    # 3. Clinical Indicators
    print("\n4. Medical Conditions:")
    high_bp = ask_binary("   - Diagnosed with High Blood Pressure (Hypertension)?")
    high_chol = ask_binary("   - Diagnosed with High Blood Cholesterol?")
    diabetes = ask_binary("   - Diagnosed with Diabetes?")
    prior_stroke = ask_binary("   - Ever diagnosed with a Stroke?")

    # 4. Lifestyle & Body Measures
    print("\n5. Lifestyle & Body Measures:")
    bmi = ask_float("   - Body Mass Index (BMI)", 12.0, 70.0, default=26.5)
    smoker = ask_binary("   - Smoked at least 100 cigarettes in lifetime?")
    phys_active = ask_binary("   - Participated in physical activities/exercise in past 30 days?")
    phys_hlth = ask_int("   - Days of poor physical health in past 30 days", 0, 30, default=0)
    ment_hlth = ask_int("   - Days of poor mental health in past 30 days", 0, 30, default=0)
    checkup = ask_binary("   - Visited a doctor for routine checkup within the past year?")

    # Calculate engineered risk factor count
    # (High BP, High Chol, Diabetes, Prior Stroke, Ever Smoker, Inactive)
    risk_factor_count = high_bp + high_chol + diabetes + prior_stroke + smoker + (1 - phys_active)

    # Prepare DataFrame
    patient_data = pd.DataFrame([{
        '_AGEG5YR': age_code,
        'SEX_MALE': sex,
        'GENHLTH_SCORE': genhlth,
        'HIGH_BP': high_bp,
        'HIGH_CHOL': high_chol,
        'DIABETES': diabetes,
        'PRIOR_STROKE': prior_stroke,
        'BMI_VAL': bmi,
        'SMOKER_EVER': smoker,
        'PHYS_ACTIVE': phys_active,
        'PHYSHLTH': float(phys_hlth),
        'MENTHLTH': float(ment_hlth),
        'RISK_FACTOR_COUNT': float(risk_factor_count),
        'CHECKUP1': 1 if checkup == 1 else 2,
        '_RACE': 1,      # Default standard category
        'EDUCA': 5,      # Default standard category
        'INCOME3': 7     # Default standard category
    }])

    # Predict
    prob = pipeline.predict_proba(patient_data)[0][1]
    pred = pipeline.predict(patient_data)[0]

    print("\n" + "=" * 60)
    print("                    PREDICTION RESULT")
    print("=" * 60)
    print(f" Estimated Heart Disease Risk: {prob * 100:.1f}%")
    if pred == 1:
        print(" Classification:  ⚠️  HIGH RISK / POSITIVE SCREEN")
        print(" Interpretation: Patient shows key cardiovascular indicators.")
        print(" Recommendation: Suggest comprehensive clinical cardiovascular evaluation.")
    else:
        print(" Classification:  ✅  LOW RISK / NEGATIVE SCREEN")
        print(" Interpretation: No high-probability signals for heart disease detected.")
    print("=" * 60)
    print(f" Cumulative Risk Factors Present: {risk_factor_count} out of 6")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

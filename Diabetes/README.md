# Diabetes Risk Analysis 🩺

A machine learning project focused on analyzing health indicators and identifying patterns associated with diabetes risk using large-scale CDC Behavioral Risk Factor Surveillance System (BRFSS) data.

## 📌 Overview

Diabetes is a major public health concern, and identifying associated health and lifestyle indicators can help improve awareness and support early risk assessment.

This project uses the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset to explore relationships between demographic characteristics, health conditions, lifestyle habits, and diabetes.

The goal is to perform exploratory data analysis, preprocess health-related data, identify important predictors, and develop machine learning models for diabetes risk classification.

## 📊 Dataset

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS), distributed through the UCI Machine Learning Repository.
- **Dataset size:** 253,680 records.
- **Data type:** Health indicators and lifestyle-related survey responses.
- **Target variable:** `Diabetes_binary`

### Target Variable

| Value | Meaning |
|---|---|
| 0 | No diabetes |
| 1 | Prediabetes or diabetes |

### Features

The dataset includes health and demographic indicators such as:

- High Blood Pressure
- High Cholesterol
- Body Mass Index (BMI)
- Smoking History
- History of Stroke
- Heart Disease or Heart Attack
- Physical Activity
- Fruit and Vegetable Consumption
- Age
- Sex
- Education
- Income

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and machine learning |
| Jupyter Notebook | Experimentation and analysis |

## 🔍 Project Workflow

### 1. Data Understanding
- Explore the dataset structure and dimensions.
- Understand feature descriptions and target distribution.
- Inspect data types and statistical summaries.

### 2. Exploratory Data Analysis (EDA)
- Analyze diabetes class distribution.
- Study relationships between health indicators and diabetes.
- Identify patterns, trends, and potential correlations.
- Visualize important features.

### 3. Data Preprocessing
- Check data quality and consistency.
- Prepare features for machine learning.
- Apply appropriate encoding and scaling where required.
- Split data into training and testing sets.

### 4. Feature Analysis and Selection
- Investigate important health indicators.
- Analyze feature relationships.
- Identify relevant predictors for classification.

### 5. Machine Learning
- Develop classification models.
- Compare model performance.
- Evaluate generalization using appropriate validation techniques.

### 6. Model Evaluation
Models will be evaluated using relevant classification metrics, including:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

### 7. Explainability
Analyze feature importance to understand which health indicators contribute most to model predictions.

## 📁 Project Structure

```text
diabetes-risk-prediction-ml/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── diabetes_analysis.ipynb
│
├── models/
│
├── src/
│
├── README.md
└── requirements.txt
```

*The structure above is a suggested organization; adjust it to match the actual repository.*

## ⚙️ Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Bhumika-78/diabetes-risk-prediction-ml.git
```

2. Navigate to the project directory:

```bash
cd diabetes-risk-prediction-ml
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Launch Jupyter Notebook:

```bash
jupyter notebook
```

## 🎯 Project Objectives

- Analyze large-scale public health survey data.
- Understand relationships between lifestyle, demographic, and health indicators.
- Apply data preprocessing and feature selection techniques.
- Build and compare machine learning classification models.
- Understand the factors associated with diabetes risk.

# ============================================================
# evaluation/metrics.py
# ------------------------------------------------------------
# Evaluates classification performance across multiple clinical
# and statistical metrics.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    """
    Computes a comprehensive suite of binary classification metrics.
    Includes clinical sensitivity (recall) and specificity.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan

    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Sensitivity_Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "Precision": round(prec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "True_Positives": int(tp),
        "False_Negatives": int(fn),
        "True_Negatives": int(tn),
        "False_Positives": int(fp)
    }
    return metrics, (tn, fp, fn, tp), (y_prob, y_pred)

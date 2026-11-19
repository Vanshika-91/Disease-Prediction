# ============================================================
# eda/visualizations.py
# ------------------------------------------------------------
# Reusable visualization functions for exploratory data analysis
# on the CDC BRFSS 2021 dataset.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df: pd.DataFrame, target_col: str = "_MICHD",
                             save_path: Path = None, dpi: int = 300):
    """
    Plots the target variable distribution displaying counts and percentages.
    """
    counts = df[target_col].value_counts().sort_index()
    labels = ["No Heart Disease (0)", "Heart Disease (1)"]
    colors = ["#3498DB", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.55)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2, h + 5000,
                f"{h:,}\n({pct:.2f}%)", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("Number of Respondents")
    ax.set_title("Target Distribution: Reported History of Heart Disease (_MICHD)\nCDC BRFSS 2021", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    return fig


def plot_demographic_prevalence(df: pd.DataFrame, age_col: str = "_AGEG5YR",
                                target_col: str = "_MICHD", save_path: Path = None, dpi: int = 300):
    """
    Plots heart disease prevalence across CDC age cohorts.
    """
    age_labels = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
    }
    df_age = df[df[age_col].isin(age_labels.keys())].copy()
    df_age["Age_Group"] = df_age[age_col].map(age_labels)

    age_prev = df_age.groupby("Age_Group")[target_col].agg(
        Total="count",
        Prevalence=lambda x: (x.sum() / x.count()) * 100
    ).reindex(list(age_labels.values()))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(age_prev))
    ax1.bar(x, age_prev["Total"], width=0.45, color="#BDC3C7", alpha=0.7, label="Total Sample")
    ax2.plot(x, age_prev["Prevalence"], color="#C0392B", marker="o", linewidth=2.5, label="Prevalence (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(age_prev.index, rotation=35)
    ax1.set_xlabel("Age Category")
    ax1.set_ylabel("Respondent Count")
    ax2.set_ylabel("Prevalence (%)", color="#C0392B", fontweight="bold")
    ax1.set_title("Heart Disease Prevalence Across Age Cohorts (CDC BRFSS 2021)", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f"{int(val):,}"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    return fig


def plot_clinical_comorbidities(df: pd.DataFrame, target_col: str = "_MICHD",
                                save_path: Path = None, dpi: int = 300):
    """
    Compares heart disease prevalence among individuals with vs without major clinical comorbidities.
    """
    clinical_vars = {
        "BPHIGH6": ("High BP", 1.0),
        "TOLDHI3": ("High Chol", 1.0),
        "DIABETE4": ("Diabetes", 1.0),
        "CVDSTRK3": ("Stroke", 1.0)
    }
    rows = []
    overall_prev = df[target_col].mean() * 100

    for col, (label, val) in clinical_vars.items():
        if col in df.columns:
            present = df[df[col] == val]
            absent = df[(df[col].notnull()) & (df[col] != val)]
            prev_p = (present[target_col].sum() / len(present)) * 100 if len(present) > 0 else 0
            prev_a = (absent[target_col].sum() / len(absent)) * 100 if len(absent) > 0 else 0
            rows.append({"Condition": label, "Present": prev_p, "Absent": prev_a})

    df_p = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df_p))
    w = 0.35

    ax.bar(x - w/2, df_p["Present"], w, label="Condition Present", color="#E74C3C", edgecolor="black")
    ax.bar(x + w/2, df_p["Absent"], w, label="Condition Absent", color="#2ECC71", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(df_p["Condition"])
    ax.set_ylabel("Heart Disease Prevalence (%)")
    ax.axhline(overall_prev, color="black", linestyle=":", label=f"Average ({overall_prev:.1f}%)")
    ax.set_title("Prevalence by Major Comorbid Risk Factor", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    return fig


def plot_lifestyle_factors(df: pd.DataFrame, target_col: str = "_MICHD",
                           save_path: Path = None, dpi: int = 300):
    """
    Plots the impact of modifiable lifestyle choices (smoking, exercise, general health).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    if "SMOKE100" in df.columns:
        sp = df.groupby("SMOKE100")[target_col].mean() * 100
        axes[0].bar(["<100 Cigs", "≥100 Cigs"], [sp.get(2.0, 0), sp.get(1.0, 0)],
                    color=["#27AE60", "#D35400"], width=0.45, edgecolor="black")
        axes[0].set_title("Lifetime Smoking", fontweight="bold")
        axes[0].set_ylabel("Prevalence (%)")

    if "EXERANY2" in df.columns:
        ep = df.groupby("EXERANY2")[target_col].mean() * 100
        axes[1].bar(["Active", "Inactive"], [ep.get(1.0, 0), ep.get(2.0, 0)],
                    color=["#2980B9", "#E74C3C"], width=0.45, edgecolor="black")
        axes[1].set_title("Physical Activity", fontweight="bold")

    if "GENHLTH" in df.columns:
        labels = {1: "Exc", 2: "V.Good", 3: "Good", 4: "Fair", 5: "Poor"}
        gp = df[df["GENHLTH"].isin(labels.keys())].groupby("GENHLTH")[target_col].mean() * 100
        axes[2].plot(list(labels.values()), [gp.get(k, 0) for k in labels.keys()],
                     marker="s", color="#8E44AD", linewidth=2.5)
        axes[2].set_title("General Health Rating", fontweight="bold")

    plt.suptitle("Modifiable Lifestyle and Subjective Health Impact", fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list, save_path: Path = None, dpi: int = 300):
    """
    Renders a Spearman rank correlation heatmap.
    """
    corr = df[columns].corr(method="spearman").round(3)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-0.3, vmax=0.5, cbar_kws={"label": "Spearman Correlation"}, ax=ax)
    ax.set_title("Spearman Rank Correlation Heatmap", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    return fig

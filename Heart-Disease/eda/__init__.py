# eda/__init__.py
"""
Exploratory Data Analysis package for Heart Disease Prediction System (CDC BRFSS 2021).
Provides modular visualization generators and statistical analysis routines.
"""
from .visualizations import (
    plot_target_distribution,
    plot_demographic_prevalence,
    plot_clinical_comorbidities,
    plot_lifestyle_factors,
    plot_correlation_heatmap
)
from .statistical_analysis import (
    calculate_prevalence_by_group,
    compute_chi_square_associations,
    calculate_relative_risk_ratios
)

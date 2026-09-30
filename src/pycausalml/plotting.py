"""Plots for CATE, SHAP, uplift, and causal trees."""

from .interpretation import (
    plot_cate_hist,
    plot_shap_dependence,
    plot_shap_importance,
    plot_shap_waterfall,
)
from .tree_based_models import plot_causal_tree
from .uplift_learners import plot_uplift_curve

__all__ = [
    "plot_cate_hist",
    "plot_shap_importance",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "plot_causal_tree",
    "plot_uplift_curve",
]

"""Bayesian linear CATE and quasi-experimental designs."""

from .estimators import (
    BayesianLinearCATE,
    DifferenceInDifferences,
    InstrumentalVariable,
    InterruptedTimeSeries,
    RegressionDiscontinuity,
    SyntheticControl,
    linear_shap,
    rope_probability,
)

__all__ = [
    "BayesianLinearCATE",
    "DifferenceInDifferences",
    "RegressionDiscontinuity",
    "InterruptedTimeSeries",
    "InstrumentalVariable",
    "SyntheticControl",
    "linear_shap",
    "rope_probability",
]

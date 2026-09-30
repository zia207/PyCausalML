"""PyCausalML: classical and deep causal inference in one package.

PyCausalML extends `PyDeepCausalML <https://github.com/zia207/PyDeepCausalML>`_.
Every public name from that package is re-exported here. The modules added
in this package are:

- :mod:`pycausalml.tree_based_models` — honest trees and DR / IV forests
- :mod:`pycausalml.uplift_learners` — uplift trees, forests, and Qini curves
- :mod:`pycausalml.meta_learners` — S, T, X, R, DR, domain adaptation
- :mod:`pycausalml.double_machine_learning` — partially linear and interactive DML
- :mod:`pycausalml.policy_learners` — doubly robust treatment policies
- :mod:`pycausalml.bayesian_causal_inference` — Bayesian CATE and quasi-experiments
- :mod:`pycausalml.bayesian_causalML` — Bayesian quasi-experiments, causal networks, structural time series

Every CATE estimator implements ``explain`` and ``plot_shap``. Optional
adapters for DoWhy, EconML, causal-learn, Uber CausalML, DoubleML, CausalPy,
and skgrf live in :mod:`pycausalml.integrations`.
"""

from __future__ import annotations

import pydeepcausalml as _deep

from .bayesian_causal_inference import (
    BayesianLinearCATE,
    DifferenceInDifferences,
    InstrumentalVariable,
    InterruptedTimeSeries,
    RegressionDiscontinuity,
    SyntheticControl,
)
from .bayesian_causalML import (
    BayesianDifferenceInDifferences,
    BayesianInstrumentalVariable,
    BayesianInterruptedTimeSeries,
    BayesianRegressionDiscontinuity,
    BayesianStructuralSCM,
    BayesianSyntheticControl,
    BayesianTimeSeriesSCM,
    CausalGraph,
    DiscreteBayesianNetwork,
    GaussianBayesianNetwork,
    NaiveAdjustmentRegressor,
    NaiveIVRegressor,
)
from .data import synthetic_data
from .double_machine_learning import (
    CausalForestDML,
    InteractiveRegressionDML,
    KernelDML,
    LinearDML,
    NonParamDML,
    PartiallyLinearDML,
    SparseLinearDML,
)
from .integrations import available_backends
from .interpretation import explain_cate
from .meta_learners import (
    DomainAdaptationLearner,
    DRLearner,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)
from .policy_learners import DRPolicyForest, DRPolicyTree, PolicyLearner
from .tree_based_models import (
    BoostedIVForest,
    CausalForest,
    CausalSurvivalForest,
    CausalTree,
    CXGBoost,
    InstrumentalForest,
    MultiArmCausalBoost,
    MultiArmCausalForest,
)
from .uplift_learners import InteractionTree, UpliftRandomForest, UpliftTree, UpliftXGBoost

# Re-export the PyDeepCausalML public API without dropping any name.
for _name in _deep.__all__:
    globals()[_name] = getattr(_deep, _name)

__version__ = "0.2.0"

__all__ = list(_deep.__all__) + [
    "CausalTree",
    "CausalForest",
    "CausalSurvivalForest",
    "InstrumentalForest",
    "MultiArmCausalForest",
    "CXGBoost",
    "BoostedIVForest",
    "MultiArmCausalBoost",
    "UpliftTree",
    "UpliftRandomForest",
    "UpliftXGBoost",
    "InteractionTree",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "DRLearner",
    "DomainAdaptationLearner",
    "PartiallyLinearDML",
    "LinearDML",
    "SparseLinearDML",
    "NonParamDML",
    "CausalForestDML",
    "KernelDML",
    "InteractiveRegressionDML",
    "PolicyLearner",
    "DRPolicyTree",
    "DRPolicyForest",
    "BayesianLinearCATE",
    "DifferenceInDifferences",
    "RegressionDiscontinuity",
    "InterruptedTimeSeries",
    "InstrumentalVariable",
    "SyntheticControl",
    "BayesianDifferenceInDifferences",
    "BayesianRegressionDiscontinuity",
    "BayesianInterruptedTimeSeries",
    "BayesianInstrumentalVariable",
    "BayesianSyntheticControl",
    "CausalGraph",
    "GaussianBayesianNetwork",
    "DiscreteBayesianNetwork",
    "NaiveAdjustmentRegressor",
    "NaiveIVRegressor",
    "BayesianStructuralSCM",
    "BayesianTimeSeriesSCM",
    "synthetic_data",
    "explain_cate",
    "available_backends",
    "__version__",
]

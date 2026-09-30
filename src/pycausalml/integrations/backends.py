"""Optional adapters for the causal Python ecosystem.

Native estimators in this package do not import these libraries. Call the
functions below when you want the upstream implementation. Each function
raises ``ImportError`` with an install extra if the library is missing.

Libraries
---------
- `DoWhy <https://github.com/py-why/dowhy>`_
- `EconML <https://github.com/py-why/econml>`_
- `causal-learn <https://github.com/py-why/causal-learn>`_
- `Uber CausalML <https://github.com/uber/causalml>`_
- `DoubleML <https://docs.doubleml.org/stable/api/api.html>`_
- `CausalPy <https://github.com/pymc-labs/CausalPy>`_
- ``skgrf`` (generalized random forests)
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "available_backends",
    "causallearn_direct_lingam",
    "causallearn_ges",
    "causallearn_pc",
    "causalml_slearner",
    "causalml_tlearner",
    "causalpy_difference_in_differences",
    "doubleml_irm",
    "doubleml_plr",
    "dowhy_ate",
    "econml_causal_forest",
    "econml_dml",
    "econml_drlearner",
    "require_module",
]


_EXTRAS = {
    "dowhy": "dowhy",
    "econml": "econml",
    "causallearn": "causallearn",
    "causalml": "uber",
    "doubleml": "doubleml",
    "causalpy": "causalpy",
    "skgrf": "skgrf",
    "shap": "shap",
    "xgboost": "boost",
}


def available_backends() -> Dict[str, bool]:
    """Whether each optional library can be imported."""
    found = {}
    for name in _EXTRAS:
        found[name] = importlib.util.find_spec(name) is not None
    return found


def require_module(module: str, extra: str = None):
    extra = extra or _EXTRAS.get(module, module)
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{module}' is not installed. "
            f"Install it with: pip install 'pycausalml[{extra}]'"
        ) from exc


def _frame(X, treatment, y):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    columns = [f"x{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    frame["treatment"] = np.asarray(treatment).ravel()
    frame["outcome"] = np.asarray(y, dtype=float).ravel()
    return frame, columns


def dowhy_ate(X, treatment, y, method_name: str = "backdoor.linear_regression"):
    """Average treatment effect from DoWhy's backdoor estimator.

    Returns the DoWhy estimate object. The numeric effect is ``result.value``.
    """
    require_module("dowhy")
    from dowhy import CausalModel

    frame, columns = _frame(X, treatment, y)
    model = CausalModel(
        data=frame,
        treatment="treatment",
        outcome="outcome",
        common_causes=columns,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    return model.estimate_effect(identified, method_name=method_name)


def econml_dml(X, treatment, y, discrete_treatment: bool = True):
    """Fit EconML ``LinearDML`` and return the fitted estimator."""
    require_module("econml")
    from econml.dml import LinearDML
    from sklearn.linear_model import LinearRegression, LogisticRegression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(treatment).ravel()
    model_t = LogisticRegression(max_iter=400) if discrete_treatment else LinearRegression()
    est = LinearDML(
        model_y=LinearRegression(),
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        cv=2,
        random_state=0,
    )
    est.fit(y, t, X=X)
    return est


def econml_causal_forest(X, treatment, y):
    """Fit EconML ``CausalForestDML``."""
    require_module("econml")
    from econml.dml import CausalForestDML

    est = CausalForestDML(discrete_treatment=True, n_estimators=50, cv=2, random_state=0)
    est.fit(np.asarray(y, dtype=float).ravel(), np.asarray(treatment).ravel(), X=np.asarray(X, dtype=float))
    return est


def econml_drlearner(X, treatment, y):
    """Fit EconML ``DRLearner``."""
    require_module("econml")
    from econml.dr import DRLearner
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    est = DRLearner(
        model_propensity=LogisticRegression(max_iter=400),
        model_regression=GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0),
        model_final=GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0),
        cv=2,
    )
    est.fit(np.asarray(y, dtype=float).ravel(), np.asarray(treatment).ravel(), X=np.asarray(X, dtype=float))
    return est


def _graph_to_adjacency(graph) -> np.ndarray:
    """Convert a causal-learn ``GeneralGraph`` to ``A[effect, cause] = 1``."""
    G = np.asarray(graph)
    p = G.shape[0]
    adjacency = np.zeros((p, p), dtype=int)
    for cause in range(p):
        for effect in range(p):
            if cause == effect:
                continue
            # causal-learn: G[i, j] == 1 and G[j, i] == -1 means i -> j.
            if G[cause, effect] == 1 and G[effect, cause] == -1:
                adjacency[effect, cause] = 1
    return adjacency


def causallearn_pc(data, alpha: float = 0.05):
    """PC algorithm. Returns a binary adjacency with ``A[i, j] = 1`` if ``j -> i``."""
    require_module("causallearn")
    from causallearn.search.ConstraintBased.PC import pc

    data = np.asarray(data, dtype=float)
    graph = pc(data, alpha=alpha, show_progress=False, verbose=False)
    return _graph_to_adjacency(graph.G.graph)


def causallearn_ges(data):
    """GES score-based search. Same adjacency orientation as :func:`causallearn_pc`."""
    require_module("causallearn")
    from causallearn.search.ScoreBased.GES import ges

    data = np.asarray(data, dtype=float)
    record = ges(data)
    graph = record["G"]
    return _graph_to_adjacency(graph.graph)


def causallearn_direct_lingam(data):
    """DirectLiNGAM adjacency, oriented as ``A[effect, cause] = 1`` for nonzero entries."""
    require_module("causallearn")
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    data = np.asarray(data, dtype=float)
    model = DirectLiNGAM()
    model.fit(data)
    raw = np.asarray(model.adjacency_matrix_, dtype=float)
    # LiNGAM stores B with x = B x + e, so B[i, j] is the effect of j on i.
    return (np.abs(raw) > 1e-8).astype(int)


def causalml_slearner(X, treatment, y):
    """Uber CausalML ``BaseSRegressor`` predictions (n, 1)."""
    require_module("causalml")
    from causalml.inference.meta import BaseSRegressor
    from sklearn.linear_model import LinearRegression

    learner = BaseSRegressor(learner=LinearRegression())
    return np.asarray(
        learner.fit_predict(np.asarray(X, dtype=float), np.asarray(treatment).ravel(), np.asarray(y, dtype=float).ravel()),
        dtype=float,
    )


def causalml_tlearner(X, treatment, y):
    """Uber CausalML ``BaseTRegressor`` predictions."""
    require_module("causalml")
    from causalml.inference.meta import BaseTRegressor
    from sklearn.linear_model import LinearRegression

    learner = BaseTRegressor(learner=LinearRegression())
    return np.asarray(
        learner.fit_predict(np.asarray(X, dtype=float), np.asarray(treatment).ravel(), np.asarray(y, dtype=float).ravel()),
        dtype=float,
    )


def doubleml_irm(X, treatment, y, n_folds: int = 2):
    """Fit ``doubleml.DoubleMLIRM`` and return the fitted object."""
    require_module("doubleml")
    import doubleml as dml
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = np.asarray(X, dtype=float)
    data = dml.DoubleMLData.from_arrays(
        X,
        np.asarray(y, dtype=float).ravel(),
        np.asarray(treatment).ravel(),
    )
    obj = dml.DoubleMLIRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=40, max_depth=3, random_state=0),
        ml_m=RandomForestClassifier(n_estimators=40, max_depth=3, random_state=0),
        n_folds=n_folds,
        score="ATE",
    )
    obj.fit()
    return obj


def doubleml_plr(X, treatment, y, n_folds: int = 2):
    """Fit ``doubleml.DoubleMLPLR`` for a partially linear model."""
    require_module("doubleml")
    import doubleml as dml
    from sklearn.ensemble import RandomForestRegressor

    X = np.asarray(X, dtype=float)
    data = dml.DoubleMLData.from_arrays(
        X,
        np.asarray(y, dtype=float).ravel(),
        np.asarray(treatment, dtype=float).ravel(),
    )
    learner = RandomForestRegressor(n_estimators=40, max_depth=3, random_state=0)
    obj = dml.DoubleMLPLR(data, ml_l=learner, ml_m=learner, n_folds=n_folds)
    obj.fit()
    return obj


def causalpy_difference_in_differences(y, treated, post, group_time=None):
    """Fit CausalPy's difference-in-differences model and return the result."""
    require_module("causalpy")
    import causalpy as cp

    frame = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float).ravel(),
            "treated": np.asarray(treated).ravel(),
            "post": np.asarray(post).ravel(),
        }
    )
    frame["unit"] = np.where(frame["treated"] == 1, "treated", "control")
    frame["time"] = frame["post"]
    return cp.DifferenceInDifferences(
        frame,
        formula="y ~ 1 + post + treated + post:treated",
        time_variable_name="time",
        group_variable_name="unit",
        model=cp.skl_models.LinearRegression(),
    )

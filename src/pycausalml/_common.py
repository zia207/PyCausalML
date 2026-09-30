"""Shared array checks, nuisance fitting, and the SHAP mixin.

Estimators take ``fit(X, treatment, y)``. That matches the RCausalML
meta-learner order. The tree tutorial file ``causal_tree_methods.py`` uses
``fit(X, Y, W)``; pass those arrays as ``y`` and ``treatment``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold

__all__ = [
    "ShapMixin",
    "as_float_matrix",
    "check_lengths",
    "clip_propensity",
    "cross_fit_mu_by_arm",
    "cross_fit_propensity",
    "cross_fit_regression",
    "dr_scores",
    "make_classifier",
    "make_regressor",
    "r_loss_pseudo",
]


def as_float_matrix(X, feature_names: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Return a float array of shape ``(n, p)`` and column names."""
    names = None
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if names is None:
        names = [f"X{i}" for i in range(1, X.shape[1] + 1)]
    if feature_names is not None:
        names = [str(n) for n in feature_names]
        if len(names) != X.shape[1]:
            raise ValueError("feature_names length does not match X.shape[1].")
    return X, names


def as_1d(values, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return out


def check_lengths(n: int, **arrays) -> None:
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"{name} has length {len(arr)}; expected {n}.")


def clip_propensity(e, eps: float = 0.01) -> np.ndarray:
    return np.clip(np.asarray(e, dtype=float), eps, 1.0 - eps)


def make_regressor(kind: str = "gbr", seed: int = 0, n_estimators: int = 40):
    kind = kind.lower()
    if kind in {"ols", "lm", "linear"}:
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
    if kind == "xgb":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=2,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
        except ImportError:
            kind = "gbr"
    if kind != "gbr":
        raise ValueError(f"Unknown regressor '{kind}'.")
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.08,
        random_state=seed,
    )


def make_classifier(kind: str = "gbc", seed: int = 0, n_estimators: int = 40):
    kind = kind.lower()
    if kind in {"logit", "logistic", "glm"}:
        return LogisticRegression(max_iter=400, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=2,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                eval_metric="logloss",
            )
        except ImportError:
            kind = "gbc"
    if kind != "gbc":
        raise ValueError(f"Unknown classifier '{kind}'.")
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.08,
        random_state=seed,
    )


def _folds(n: int, n_splits: int, seed: int):
    splits = min(n_splits, n)
    if splits < 2:
        raise ValueError("Need at least 2 rows to cross-fit.")
    return KFold(n_splits=splits, shuffle=True, random_state=seed).split(np.arange(n))


def _positive_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes and 1.0 not in classes:
        return np.zeros(len(X))
    index = classes.index(1 if 1 in classes else 1.0)
    return proba[:, index]


def cross_fit_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 2,
    seed: int = 0,
    kind: str = "gbr",
    n_estimators: int = 40,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Out-of-fold predictions of ``E[y | X]``."""
    pred = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(_folds(len(y), n_splits, seed)):
        model = make_regressor(kind, seed + fold, n_estimators)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, dtype=float)[tr]
        model.fit(X[tr], y[tr], **kw)
        pred[te] = model.predict(X[te])
    return pred


def cross_fit_propensity(
    X: np.ndarray,
    treatment: np.ndarray,
    n_splits: int = 2,
    seed: int = 0,
    kind: str = "gbc",
    n_estimators: int = 40,
    eps: float = 0.01,
) -> np.ndarray:
    """Out-of-fold ``P(treatment = 1 | X)``."""
    w = np.asarray(treatment).astype(int).ravel()
    pred = np.zeros(len(w), dtype=float)
    for fold, (tr, te) in enumerate(_folds(len(w), n_splits, seed)):
        if len(np.unique(w[tr])) < 2:
            pred[te] = float(np.mean(w[tr]))
            continue
        model = make_classifier(kind, seed + fold, n_estimators)
        model.fit(X[tr], w[tr])
        pred[te] = _positive_proba(model, X[te])
    return clip_propensity(pred, eps)


def cross_fit_mu_by_arm(
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    n_splits: int = 2,
    seed: int = 0,
    kind: str = "gbr",
    n_estimators: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold outcome regressions ``mu0(X)`` and ``mu1(X)``."""
    w = np.asarray(treatment).astype(int).ravel()
    mu0 = np.zeros(len(y), dtype=float)
    mu1 = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(_folds(len(y), n_splits, seed)):
        for arm, store in ((0, mu0), (1, mu1)):
            rows = tr[w[tr] == arm]
            fallback = float(y[rows].mean()) if len(rows) else float(y[tr].mean())
            if len(rows) < 8:
                store[te] = fallback
                continue
            model = make_regressor(kind, seed + fold + 17 * arm, n_estimators)
            model.fit(X[rows], y[rows])
            store[te] = model.predict(X[te])
    return mu0, mu1


def dr_scores(y, treatment, mu0, mu1, propensity, eps: float = 0.01) -> np.ndarray:
    """Doubly robust (AIPW) pseudo-outcome for the CATE."""
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(treatment, dtype=float).ravel()
    e = clip_propensity(propensity, eps)
    mu0 = np.asarray(mu0, dtype=float).ravel()
    mu1 = np.asarray(mu1, dtype=float).ravel()
    return (mu1 - mu0) + w * (y - mu1) / e - (1.0 - w) * (y - mu0) / (1.0 - e)


def r_loss_pseudo(y, treatment, m_hat, e_hat, floor: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """R-learner pseudo-outcome and weights (Nie and Wager)."""
    y_res = np.asarray(y, dtype=float).ravel() - np.asarray(m_hat, dtype=float).ravel()
    t_res = np.asarray(treatment, dtype=float).ravel() - np.asarray(e_hat, dtype=float).ravel()
    small = np.abs(t_res) < floor
    denom = t_res.copy()
    denom[small] = np.sign(denom[small]) * floor
    denom[denom == 0] = floor
    target = np.clip(y_res / denom, -25.0, 25.0)
    weights = t_res ** 2
    return target, weights


def fit_weighted(model, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
    return model


class ShapMixin:
    """SHAP interpretation for any estimator with ``predict`` or ``predict_cate``."""

    feature_names_: List[str]

    def explain(self, X, **kwargs):
        """Permutation SHAP values for CATE predictions.

        Uses :func:`pycausalml.interpretation.explain_cate`. Pass
        ``method='shap'`` to call the ``shap`` package when it is installed.
        """
        from .interpretation import explain_cate

        names = kwargs.pop("feature_names", getattr(self, "feature_names_", None))
        return explain_cate(self, X, feature_names=names, **kwargs)

    def plot_shap(self, X, kind: str = "importance", row: int = 0, feature=0, **kwargs):
        """Draw a SHAP importance, dependence, or waterfall plot."""
        explanation = self.explain(X, **kwargs)
        if kind == "importance":
            return explanation.plot_importance()
        if kind == "dependence":
            return explanation.plot_dependence(feature)
        if kind == "waterfall":
            return explanation.plot_waterfall(row)
        raise ValueError("kind must be 'importance', 'dependence', or 'waterfall'.")


def mean_se(scores) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=float).ravel()
    estimate = float(np.mean(scores))
    if len(scores) < 2:
        return estimate, float("nan")
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return estimate, se

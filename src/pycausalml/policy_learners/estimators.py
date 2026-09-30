"""Treatment-assignment policies from doubly robust scores.

The weighted-classification policy follows Athey and Wager: label a unit
by the sign of its doubly robust score and weight the classifier by the
absolute score. ``DRPolicyTree`` and ``DRPolicyForest`` are the tree and
forest versions of that rule, in the spirit of EconML's policy learners
and ``R/policy_learner.R``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .._common import (
    ShapMixin,
    as_1d,
    as_float_matrix,
    check_lengths,
    clip_propensity,
    cross_fit_mu_by_arm,
    cross_fit_propensity,
    dr_scores,
    make_regressor,
)

__all__ = [
    "DRPolicyForest",
    "DRPolicyTree",
    "PolicyLearner",
    "policy_value",
]


def policy_value(y, treatment, policy, propensity) -> float:
    """Horvitz–Thompson value of a deterministic 0/1 policy."""
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(treatment, dtype=float).ravel()
    pi = np.asarray(policy, dtype=float).ravel()
    e = clip_propensity(propensity)
    prob = np.where(w == 1, e, 1.0 - e)
    match = (w == pi).astype(float)
    return float(np.mean(match * y / prob))


class _PolicyBase(ShapMixin):
    def __init__(
        self,
        n_splits: int = 2,
        nuisance: str = "gbr",
        n_estimators: int = 30,
        random_state: int = 0,
    ):
        self.n_splits = int(n_splits)
        self.nuisance = nuisance
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def _scores(self, X, treatment, y, feature_names):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        mu0, mu1 = cross_fit_mu_by_arm(
            X, y, w, self.n_splits, self.random_state, self.nuisance, self.n_estimators
        )
        e_hat = cross_fit_propensity(
            X, w, self.n_splits, self.random_state + 3, "gbc", self.n_estimators
        )
        gamma = dr_scores(y, w, mu0, mu1, e_hat)
        self.feature_names_ = names
        self.dr_scores_ = gamma
        self.e_hat_ = e_hat
        self._y = y
        self._w = w
        self.cate_model_ = make_regressor(self.nuisance, self.random_state, self.n_estimators)
        self.cate_model_.fit(X, gamma)
        return X, w, y, gamma

    def predict_cate(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.cate_model_.predict(X), dtype=float)

    def predict_value(self, X):
        """Incremental value of assigning the recommended treatment versus control."""
        cate = self.predict_cate(X)
        return self.predict(X).astype(float) * cate

    def training_policy_value(self) -> float:
        """IPW value of the learned policy on the training sample."""
        return policy_value(self._y, self._w, self.predict(self._X), self.e_hat_)

    def feature_importance(self) -> np.ndarray:
        if hasattr(self, "policy_") and hasattr(self.policy_, "feature_importances_"):
            return np.asarray(self.policy_.feature_importances_, dtype=float)
        return np.zeros(len(self.feature_names_))


class PolicyLearner(_PolicyBase):
    """Doubly robust policy as a weighted classifier on the sign of the DR score."""

    def __init__(self, policy: str = "tree", max_depth: int = 3, n_policy_trees: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.max_depth = int(max_depth)
        self.n_policy_trees = int(n_policy_trees)

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y, gamma = self._scores(X, treatment, y, feature_names)
        self._X = X
        labels = (gamma > 0).astype(int)
        weights = np.abs(gamma)
        if self.policy == "forest":
            self.policy_ = RandomForestClassifier(
                n_estimators=self.n_policy_trees,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=1,
            )
        else:
            self.policy_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=max(5, len(X) // 20),
                random_state=self.random_state,
            )
        # A constant label has no classification problem; assign everyone that arm.
        if len(np.unique(labels)) < 2:
            self.constant_ = int(labels[0])
            self.policy_ = None
        else:
            self.constant_ = None
            self.policy_.fit(X, labels, sample_weight=weights)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        if self.constant_ is not None:
            return np.full(len(X), self.constant_, dtype=int)
        return np.asarray(self.policy_.predict(X), dtype=int)

    def predict_proba(self, X):
        X, _ = as_float_matrix(X)
        if self.constant_ is not None:
            proba = np.zeros((len(X), 2))
            proba[:, self.constant_] = 1.0
            return proba
        return np.asarray(self.policy_.predict_proba(X), dtype=float)


class DRPolicyTree(PolicyLearner):
    """Single tree policy. Same rule as :class:`PolicyLearner` with ``policy='tree'``."""

    def __init__(self, max_depth: int = 3, **kwargs):
        super().__init__(policy="tree", max_depth=max_depth, **kwargs)


class DRPolicyForest(PolicyLearner):
    """Forest policy. Same rule as :class:`PolicyLearner` with ``policy='forest'``.

    ``n_policy_trees`` sets the number of trees in the ensemble.
    """

    def __init__(self, max_depth: int = 3, **kwargs):
        super().__init__(policy="forest", max_depth=max_depth, **kwargs)

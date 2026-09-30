"""S, T, X, R, DR, and domain-adaptation learners.

The constructions follow Künzel et al. (metalearners) and Nie and Wager
(R-learner), and the fitting steps in ``R/meta_learners.R``. First-stage
regressions are cross-fitted for the R-learner and DR-learner. Final
nuisance models for S, T, and X are fit on the full sample so that
``predict`` does not depend on fold assignment.
"""

from __future__ import annotations

import numpy as np

from .._common import (
    ShapMixin,
    as_1d,
    as_float_matrix,
    check_lengths,
    clip_propensity,
    cross_fit_mu_by_arm,
    cross_fit_propensity,
    cross_fit_regression,
    dr_scores,
    fit_weighted,
    make_classifier,
    make_regressor,
    mean_se,
    r_loss_pseudo,
)

__all__ = [
    "DRLearner",
    "DomainAdaptationLearner",
    "RLearner",
    "SLearner",
    "TLearner",
    "XLearner",
]


class _MetaBase(ShapMixin):
    def __init__(
        self,
        learner: str = "gbr",
        n_estimators: int = 40,
        n_splits: int = 2,
        random_state: int = 0,
    ):
        self.learner = learner
        self.n_estimators = int(n_estimators)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _xy(self, X, treatment, y, feature_names):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        if len(np.unique(w)) < 2:
            raise ValueError("treatment must contain both arms.")
        self.feature_names_ = names
        return X, w, y

    def predict_ate(self, X=None):
        if hasattr(self, "dr_scores_"):
            estimate, _ = mean_se(self.dr_scores_)
            return estimate
        if X is None:
            raise ValueError("Pass X to predict_ate for this learner.")
        return float(np.mean(self.predict(X)))

    def average_treatment_effect(self, X=None):
        if hasattr(self, "dr_scores_"):
            estimate, se = mean_se(self.dr_scores_)
            return {"estimate": estimate, "std.err": se}
        estimate = self.predict_ate(X)
        return {"estimate": estimate, "std.err": float("nan")}


class SLearner(_MetaBase):
    """Single learner: ``mu(X, T)``, CATE = ``mu(X, 1) - mu(X, 0)``."""

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        self.model_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.model_.fit(np.column_stack([X, w]), y)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        ones = np.ones((len(X), 1))
        zeros = np.zeros((len(X), 1))
        return self.model_.predict(np.column_stack([X, ones])) - self.model_.predict(
            np.column_stack([X, zeros])
        )

    predict_cate = predict


class TLearner(_MetaBase):
    """Two learners: ``mu1(X) - mu0(X)``."""

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        self.mu0_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.mu1_ = make_regressor(self.learner, self.random_state + 1, self.n_estimators)
        self.mu0_.fit(X[w == 0], y[w == 0])
        self.mu1_.fit(X[w == 1], y[w == 1])
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self.mu1_.predict(X) - self.mu0_.predict(X)

    predict_cate = predict


class XLearner(_MetaBase):
    """X-learner: imputed treatment effects, weighted by the propensity."""

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        self.mu0_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.mu1_ = make_regressor(self.learner, self.random_state + 1, self.n_estimators)
        self.mu0_.fit(X[w == 0], y[w == 0])
        self.mu1_.fit(X[w == 1], y[w == 1])
        d1 = y[w == 1] - self.mu0_.predict(X[w == 1])
        d0 = self.mu1_.predict(X[w == 0]) - y[w == 0]
        self.tau1_ = make_regressor(self.learner, self.random_state + 2, self.n_estimators)
        self.tau0_ = make_regressor(self.learner, self.random_state + 3, self.n_estimators)
        self.tau1_.fit(X[w == 1], d1)
        self.tau0_.fit(X[w == 0], d0)
        self.propensity_ = make_classifier("logit", self.random_state)
        self.propensity_.fit(X, w)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        e = clip_propensity(self.propensity_.predict_proba(X)[:, 1])
        return e * self.tau0_.predict(X) + (1.0 - e) * self.tau1_.predict(X)

    predict_cate = predict


class RLearner(_MetaBase):
    """R-learner with a cross-fitted Robinson residual and a weighted second stage."""

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        m_hat = cross_fit_regression(
            X, y, self.n_splits, self.random_state, self.learner, self.n_estimators
        )
        e_hat = cross_fit_propensity(
            X, w, self.n_splits, self.random_state + 5, "gbc", self.n_estimators
        )
        target, weights = r_loss_pseudo(y, w, m_hat, e_hat)
        self.final_ = make_regressor(self.learner, self.random_state + 7, self.n_estimators)
        fit_weighted(self.final_, X, target, weights)
        self.dr_scores_ = target
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.final_.predict(X), dtype=float)

    predict_cate = predict


class DRLearner(_MetaBase):
    """DR-learner: second-stage regression of the augmented-IPW pseudo-outcome."""

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        mu0, mu1 = cross_fit_mu_by_arm(
            X, y, w, self.n_splits, self.random_state, self.learner, self.n_estimators
        )
        e_hat = cross_fit_propensity(
            X, w, self.n_splits, self.random_state + 9, "gbc", self.n_estimators
        )
        gamma = dr_scores(y, w, mu0, mu1, e_hat)
        self.dr_scores_ = gamma
        self.final_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.final_.fit(X, gamma)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.final_.predict(X), dtype=float)

    predict_cate = predict


class DomainAdaptationLearner(_MetaBase):
    """Propensity-weighted X-style learner from ``R/meta_learners.R``.

    Control outcomes are weighted by ``e / (1 - e)`` and treated outcomes by
    ``(1 - e) / e``. Imputed effects from the opposite arm are then fit by
    the final learner.
    """

    def fit(self, X, treatment, y, feature_names=None):
        X, w, y = self._xy(X, treatment, y, feature_names)
        self.propensity_ = make_classifier("logit", self.random_state)
        self.propensity_.fit(X, w)
        e = clip_propensity(self.propensity_.predict_proba(X)[:, 1], 1e-3)
        w_ctrl = e[w == 0] / (1.0 - e[w == 0])
        w_trt = (1.0 - e[w == 1]) / e[w == 1]
        self.mu0_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.mu1_ = make_regressor(self.learner, self.random_state + 1, self.n_estimators)
        fit_weighted(self.mu0_, X[w == 0], y[w == 0], w_ctrl)
        fit_weighted(self.mu1_, X[w == 1], y[w == 1], w_trt)
        imp_ctrl = self.mu1_.predict(X[w == 0]) - y[w == 0]
        imp_trt = y[w == 1] - self.mu0_.predict(X[w == 1])
        X_imp = np.vstack([X[w == 0], X[w == 1]])
        y_imp = np.concatenate([imp_ctrl, imp_trt])
        self.final_ = make_regressor(self.learner, self.random_state + 2, self.n_estimators)
        self.final_.fit(X_imp, y_imp)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.final_.predict(X), dtype=float)

    predict_cate = predict

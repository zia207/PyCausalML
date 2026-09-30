"""Double / debiased machine learning for the ATE and the CATE.

The residual-on-residual construction follows Chernozhukov et al. and the
EconML ``DML`` family implemented in ``R/DMLearner.R``. Cross-fitting is
used for every first stage. :class:`PartiallyLinearDML` estimates a constant
effect. :class:`InteractiveRegressionDML` is the doubly robust score used by
DoubleML's IRM model for a binary treatment.

Optional wrappers that call the EconML and DoubleML packages live in
:mod:`pycausalml.integrations`.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from .._common import (
    ShapMixin,
    as_1d,
    as_float_matrix,
    check_lengths,
    cross_fit_mu_by_arm,
    cross_fit_propensity,
    cross_fit_regression,
    dr_scores,
    fit_weighted,
    make_regressor,
    mean_se,
    r_loss_pseudo,
)

__all__ = [
    "CausalForestDML",
    "InteractiveRegressionDML",
    "KernelDML",
    "LinearDML",
    "NonParamDML",
    "PartiallyLinearDML",
    "SparseLinearDML",
]


def _residualize(X, treatment, y, n_splits, seed, kind, n_estimators, discrete: bool):
    y = as_1d(y, "y")
    t = as_1d(treatment, "treatment")
    m_y = cross_fit_regression(X, y, n_splits, seed, kind, n_estimators)
    if discrete:
        m_t = cross_fit_propensity(X, t.astype(int), n_splits, seed + 4, "gbc", n_estimators)
    else:
        m_t = cross_fit_regression(X, t, n_splits, seed + 4, kind, n_estimators)
    return y - m_y, t - m_t


class PartiallyLinearDML(ShapMixin):
    """Partially linear DML: ``Y = theta * T + g(X) + e``, constant ``theta``."""

    def __init__(
        self,
        discrete_treatment: bool = True,
        n_splits: int = 2,
        model_y: str = "ridge",
        n_estimators: int = 40,
        random_state: int = 0,
    ):
        self.discrete_treatment = bool(discrete_treatment)
        self.n_splits = int(n_splits)
        self.model_y = model_y
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_res, t_res = _residualize(
            X,
            treatment,
            y,
            self.n_splits,
            self.random_state,
            self.model_y,
            self.n_estimators,
            self.discrete_treatment,
        )
        check_lengths(len(X), y=y_res, treatment=t_res)
        denom = float(np.dot(t_res, t_res))
        if abs(denom) < 1e-10:
            raise ValueError("Treatment residuals have no variation.")
        theta = float(np.dot(t_res, y_res) / denom)
        psi = t_res * (y_res - theta * t_res) / np.mean(t_res ** 2)
        self.theta_ = theta
        self.influence_ = psi
        self.feature_names_ = names
        self._n = len(X)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.full(len(X), self.theta_)

    predict_cate = predict

    def effect(self):
        return self.theta_

    def average_treatment_effect(self):
        estimate, se = mean_se(self.influence_)
        z = 1.959963984540054
        return {
            "estimate": self.theta_,
            "std.err": se,
            "ci_lower": self.theta_ - z * se,
            "ci_upper": self.theta_ + z * se,
        }

    def confidence_interval(self, level: float = 0.95):
        from scipy import stats

        summary = self.average_treatment_effect()
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return (self.theta_ - z * summary["std.err"], self.theta_ + z * summary["std.err"])

    def feature_importance(self):
        return np.zeros(len(self.feature_names_))


class _LinearCateDML(ShapMixin):
    """Cross-fit residuals, then a linear model for ``theta(X)``."""

    final_kind = "ols"

    def __init__(
        self,
        discrete_treatment: bool = True,
        n_splits: int = 2,
        model_y: str = "gbr",
        n_estimators: int = 40,
        fit_cate_intercept: bool = True,
        random_state: int = 0,
        alpha: float = 0.01,
    ):
        self.discrete_treatment = bool(discrete_treatment)
        self.n_splits = int(n_splits)
        self.model_y = model_y
        self.n_estimators = int(n_estimators)
        self.fit_cate_intercept = bool(fit_cate_intercept)
        self.random_state = int(random_state)
        self.alpha = float(alpha)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_res, t_res = _residualize(
            X,
            treatment,
            y,
            self.n_splits,
            self.random_state,
            self.model_y,
            self.n_estimators,
            self.discrete_treatment,
        )
        self.feature_names_ = names
        phi = np.column_stack([np.ones(len(X)), X]) if self.fit_cate_intercept else X
        # Residual-on-residual design: columns are treatment_residual * phi(X).
        design = t_res[:, None] * phi
        if self.final_kind == "lasso":
            model = Lasso(
                alpha=self.alpha,
                max_iter=5000,
                fit_intercept=False,
                random_state=self.random_state,
            )
        elif self.final_kind == "ridge":
            model = Ridge(alpha=self.alpha, fit_intercept=False)
        else:
            model = LinearRegression(fit_intercept=False)
        model.fit(design, y_res)
        coef = np.asarray(model.coef_, dtype=float).ravel()
        self.coef_ = coef
        self.intercept_ = float(coef[0]) if self.fit_cate_intercept else 0.0
        self.feature_coef_ = coef[1:] if self.fit_cate_intercept else coef
        cate = self._cate_from(X)
        psi = t_res * (y_res - cate * t_res)
        scale = float(np.mean(t_res ** 2)) or 1.0
        self.influence_ = psi / scale
        return self

    def _cate_from(self, X):
        return self.intercept_ + X @ self.feature_coef_

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self._cate_from(X)

    predict_cate = predict

    def feature_importance(self) -> np.ndarray:
        coef = np.abs(self.feature_coef_)
        total = coef.sum()
        return coef / total if total > 0 else coef


class LinearDML(_LinearCateDML):
    """Linear CATE: ``theta(X) = a + X beta``, fit on cross-fit residuals.

    ``average_treatment_effect`` reports the orthogonal score for the
    average effect (mean of the doubly robust residual scale). The returned
    point estimate is the average of ``predict`` on the training covariates
    only when those covariates are passed back in; the stored influence
    function standard error is the partially linear standard error.
    """

    final_kind = "ols"

    def fit(self, X, treatment, y, feature_names=None):
        super().fit(X, treatment, y, feature_names)
        self._X = as_float_matrix(X, feature_names)[0]
        return self

    def average_treatment_effect(self):
        cate = self.predict(self._X)
        estimate = float(np.mean(cate))
        _, se = mean_se(self.influence_)
        z = 1.959963984540054
        return {
            "estimate": estimate,
            "std.err": se,
            "ci_lower": estimate - z * se,
            "ci_upper": estimate + z * se,
        }

    def confidence_interval(self, level: float = 0.95):
        from scipy import stats

        summary = self.average_treatment_effect()
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return (summary["estimate"] - z * summary["std.err"], summary["estimate"] + z * summary["std.err"])


class SparseLinearDML(LinearDML):
    """Linear CATE with an L1 penalty on the residual-on-residual design."""

    final_kind = "lasso"


class NonParamDML(ShapMixin):
    """Final-stage regression of the R-learner pseudo-outcome."""

    def __init__(
        self,
        discrete_treatment: bool = True,
        n_splits: int = 2,
        model_y: str = "gbr",
        model_final: str = "gbr",
        n_estimators: int = 40,
        random_state: int = 0,
    ):
        self.discrete_treatment = bool(discrete_treatment)
        self.n_splits = int(n_splits)
        self.model_y = model_y
        self.model_final = model_final
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_arr = as_1d(y, "y")
        t_arr = as_1d(treatment, "treatment")
        check_lengths(len(X), y=y_arr, treatment=t_arr)
        if self.discrete_treatment:
            mu0, mu1 = cross_fit_mu_by_arm(
                X, y_arr, t_arr.astype(int), self.n_splits, self.random_state, self.model_y, self.n_estimators
            )
            e_hat = cross_fit_propensity(
                X, t_arr.astype(int), self.n_splits, self.random_state + 3, "gbc", self.n_estimators
            )
            gamma = dr_scores(y_arr, t_arr, mu0, mu1, e_hat)
            self.final_ = make_regressor(self.model_final, self.random_state, self.n_estimators)
            self.final_.fit(X, gamma)
            self.influence_ = gamma
        else:
            m_hat = cross_fit_regression(
                X, y_arr, self.n_splits, self.random_state, self.model_y, self.n_estimators
            )
            e_hat = cross_fit_regression(
                X, t_arr, self.n_splits, self.random_state + 5, self.model_y, self.n_estimators
            )
            target, weights = r_loss_pseudo(y_arr, t_arr, m_hat, e_hat)
            self.final_ = make_regressor(self.model_final, self.random_state + 7, self.n_estimators)
            fit_weighted(self.final_, X, target, weights)
            cate = np.asarray(self.final_.predict(X), dtype=float)
            scale = float(np.mean((t_arr - e_hat) ** 2)) or 1.0
            self.influence_ = (t_arr - e_hat) * (y_arr - m_hat - cate * (t_arr - e_hat)) / scale
        self.feature_names_ = names
        self._X = X
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.final_.predict(X), dtype=float)

    predict_cate = predict

    def average_treatment_effect(self):
        if not self.discrete_treatment and hasattr(self, "_X"):
            estimate = float(np.mean(self.predict(self._X)))
            _, se = mean_se(self.influence_)
            return {"estimate": estimate, "std.err": se}
        estimate, se = mean_se(self.influence_)
        return {"estimate": estimate, "std.err": se}

    def feature_importance(self) -> np.ndarray:
        if hasattr(self.final_, "feature_importances_"):
            return np.asarray(self.final_.feature_importances_, dtype=float)
        return np.zeros(len(self.feature_names_))


class CausalForestDML(NonParamDML):
    """DML whose final stage is a random forest of the DR pseudo-outcome."""

    def __init__(self, num_trees: int = 80, min_node_size: int = 5, n_splits: int = 2, random_state: int = 0):
        super().__init__(
            discrete_treatment=True,
            n_splits=n_splits,
            model_y="gbr",
            model_final="rf",
            n_estimators=num_trees,
            random_state=random_state,
        )
        self.min_node_size = int(min_node_size)


class KernelDML(ShapMixin):
    """Random-Fourier-feature approximation of a kernel CATE (EconML KernelDML)."""

    def __init__(
        self,
        n_components: int = 30,
        bandwidth: float = 1.0,
        alpha: float = 1.0,
        n_splits: int = 2,
        random_state: int = 0,
    ):
        self.n_components = int(n_components)
        self.bandwidth = float(bandwidth)
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _features(self, X):
        proj = X @ self.omega_ + self.bias_
        return np.column_stack([np.ones(len(X)), np.sqrt(2.0 / self.n_components) * np.cos(proj)])

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_res, t_res = _residualize(
            X, treatment, y, self.n_splits, self.random_state, "gbr", 30, discrete=True
        )
        rng = np.random.default_rng(self.random_state)
        self.omega_ = rng.normal(scale=1.0 / self.bandwidth, size=(X.shape[1], self.n_components))
        self.bias_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        phi = self._features(X)
        design = t_res[:, None] * phi
        gram = design.T @ design + self.alpha * np.eye(design.shape[1])
        self.coef_ = np.linalg.solve(gram, design.T @ y_res)
        self.feature_names_ = names
        cate = phi @ self.coef_
        scale = float(np.mean(t_res ** 2)) or 1.0
        self.influence_ = t_res * (y_res - cate * t_res) / scale
        self._X = X
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self._features(X) @ self.coef_

    predict_cate = predict

    def average_treatment_effect(self):
        estimate = float(np.mean(self.predict(self._X)))
        _, se = mean_se(self.influence_)
        return {"estimate": estimate, "std.err": se}

    def feature_importance(self) -> np.ndarray:
        # Sensitivity of the random features to each input, averaged.
        scale = np.mean(np.abs(self.omega_), axis=1)
        total = scale.sum()
        return scale / total if total > 0 else scale


class InteractiveRegressionDML(ShapMixin):
    """Doubly robust ATE for a binary treatment (DoubleML IRM score)."""

    def __init__(self, n_splits: int = 2, model_y: str = "gbr", n_estimators: int = 40, random_state: int = 0):
        self.n_splits = int(n_splits)
        self.model_y = model_y
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_arr = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y_arr, treatment=w)
        mu0, mu1 = cross_fit_mu_by_arm(
            X, y_arr, w, self.n_splits, self.random_state, self.model_y, self.n_estimators
        )
        e_hat = cross_fit_propensity(
            X, w, self.n_splits, self.random_state + 2, "gbc", self.n_estimators
        )
        gamma = dr_scores(y_arr, w, mu0, mu1, e_hat)
        self.influence_ = gamma
        self.theta_ = float(np.mean(gamma))
        self.feature_names_ = names
        self.mu_model_ = make_regressor(self.model_y, self.random_state, self.n_estimators)
        self.mu_model_.fit(X, gamma)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.mu_model_.predict(X), dtype=float)

    predict_cate = predict

    def effect(self):
        return self.theta_

    def average_treatment_effect(self):
        estimate, se = mean_se(self.influence_)
        z = 1.959963984540054
        return {
            "estimate": estimate,
            "std.err": se,
            "ci_lower": estimate - z * se,
            "ci_upper": estimate + z * se,
        }

    def confidence_interval(self, level: float = 0.95):
        from scipy import stats

        summary = self.average_treatment_effect()
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return (summary["estimate"] - z * summary["std.err"], summary["estimate"] + z * summary["std.err"])

    def feature_importance(self):
        if hasattr(self.mu_model_, "feature_importances_"):
            return np.asarray(self.mu_model_.feature_importances_, dtype=float)
        return np.zeros(len(self.feature_names_))

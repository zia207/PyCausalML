"""Bayesian and quasi-experimental estimators.

These are the designs covered by CausalPy (difference-in-differences,
synthetic control, regression discontinuity, interrupted time series,
instrumental variables), with uncertainty from a Bayesian bootstrap or,
for the linear CATE, from a normal posterior. PyMC is not required.
When CausalPy is installed, the wrappers in :mod:`pycausalml.integrations`
call that package directly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .._common import (
    ShapMixin,
    as_1d,
    as_float_matrix,
    check_lengths,
    cross_fit_mu_by_arm,
    cross_fit_propensity,
    dr_scores,
)

__all__ = [
    "BayesianLinearCATE",
    "DifferenceInDifferences",
    "InstrumentalVariable",
    "InterruptedTimeSeries",
    "RegressionDiscontinuity",
    "SyntheticControl",
    "linear_shap",
    "rope_probability",
]


def rope_probability(draws, low: float, high: float) -> float:
    """Share of posterior draws inside a region of practical equivalence."""
    draws = np.asarray(draws, dtype=float).ravel()
    return float(np.mean((draws >= low) & (draws <= high)))


def _weighted_lstsq(design, y, weights):
    sw = np.sqrt(np.asarray(weights, dtype=float))
    beta, _, _, _ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    return beta


def _bayesian_bootstrap(n, stat, n_boot, seed):
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        weights = rng.dirichlet(np.ones(n))
        draws[b] = stat(weights)
    return draws


def _summary(effect, draws, rope=None):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out = {
        "estimate": float(effect),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "posterior_mean": float(np.mean(draws)),
        "draws": draws,
    }
    if rope is not None:
        out["rope_probability"] = rope_probability(draws, rope[0], rope[1])
    return out


def linear_shap(coef, X, feature_names, intercept: float = 0.0):
    """Exact SHAP values of a linear predictor ``intercept + X @ coef``."""
    from ..interpretation import ShapExplanation

    X, names = as_float_matrix(X, feature_names)
    coef = np.asarray(coef, dtype=float).ravel()
    center = X.mean(axis=0)
    base = intercept + float(center @ coef)
    values = (X - center) * coef
    predictions = intercept + X @ coef
    return ShapExplanation(
        values=values,
        base_values=np.full(len(X), base),
        data=X,
        feature_names=names,
        predictions=predictions,
    )


class BayesianLinearCATE(ShapMixin):
    """Normal posterior for a linear CATE on doubly robust pseudo-outcomes.

    ``theta(X) = a + X beta`` with a ridge prior. ``credible_interval``
    returns marginal posterior intervals for each row.
    """

    def __init__(self, prior_precision: float = 1.0, n_splits: int = 2, random_state: int = 0):
        self.prior_precision = float(prior_precision)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y_arr = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y_arr, treatment=w)
        mu0, mu1 = cross_fit_mu_by_arm(X, y_arr, w, self.n_splits, self.random_state, "ridge", 20)
        e_hat = cross_fit_propensity(X, w, self.n_splits, self.random_state + 1, "logit", 20)
        gamma = dr_scores(y_arr, w, mu0, mu1, e_hat)
        phi = np.column_stack([np.ones(len(X)), X])
        p = phi.shape[1]
        penalty = self.prior_precision * np.eye(p)
        penalty[0, 0] = 1e-4
        precision = phi.T @ phi + penalty
        self.cov_ = np.linalg.inv(precision)
        self.coef_ = self.cov_ @ (phi.T @ gamma)
        resid = gamma - phi @ self.coef_
        self.sigma2_ = float(np.sum(resid ** 2) / max(len(X) - p, 1))
        self.posterior_cov_ = self.sigma2_ * self.cov_
        self.feature_names_ = names
        self.dr_scores_ = gamma
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        phi = np.column_stack([np.ones(len(X)), X])
        return phi @ self.coef_

    predict_cate = predict

    def credible_interval(self, X, level: float = 0.95):
        from scipy import stats

        X, _ = as_float_matrix(X)
        phi = np.column_stack([np.ones(len(X)), X])
        mean = phi @ self.coef_
        var = np.sum(phi * (phi @ self.posterior_cov_), axis=1)
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        se = np.sqrt(np.clip(var, 0, None))
        return mean - z * se, mean + z * se

    def explain(self, X, **kwargs):
        X, _ = as_float_matrix(X)
        return linear_shap(self.coef_[1:], X, self.feature_names_, intercept=float(self.coef_[0]))

    def average_treatment_effect(self):
        estimate = float(np.mean(self.dr_scores_))
        se = float(np.std(self.dr_scores_, ddof=1) / np.sqrt(len(self.dr_scores_)))
        return {"estimate": estimate, "std.err": se}


class _DesignEffect(ShapMixin):
    """OLS effect on a user-built design, with a Bayesian bootstrap interval."""

    effect_index: int = 1

    def __init__(self, n_boot: int = 200, rope=None, random_state: int = 0):
        self.n_boot = int(n_boot)
        self.rope = rope
        self.random_state = int(random_state)

    def _fit_design(self, design, y, names, effect_index):
        y = as_1d(y, "y")
        design = np.asarray(design, dtype=float)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        check_lengths(len(design), y=y)
        beta = _weighted_lstsq(design, y, np.ones(len(y)))
        self.coef_ = beta
        self.feature_names_ = list(names)
        self._design = design
        self._y = y
        index = effect_index

        def stat(weights):
            return float(_weighted_lstsq(design, y, weights)[index])

        self.draws_ = _bayesian_bootstrap(len(y), stat, self.n_boot, self.random_state)
        self.effect_ = float(beta[index])
        self.summary_ = _summary(self.effect_, self.draws_, self.rope)
        return self

    def predict(self, X=None):
        design = self._design if X is None else np.asarray(X, dtype=float)
        return design @ self.coef_

    def explain(self, X=None, **kwargs):
        design = self._design if X is None else np.asarray(X, dtype=float)
        return linear_shap(self.coef_, design, self.feature_names_, intercept=0.0)

    def summary(self):
        public = {k: v for k, v in self.summary_.items() if k != "draws"}
        return public


class DifferenceInDifferences(_DesignEffect):
    """Two-way difference-in-differences.

    ``Y = a + b * treated + c * post + tau * treated * post``. ``tau`` is
    the effect. ``treated`` and ``post`` are 0/1 indicators.
    """

    def fit(self, y, treated, post):
        y = as_1d(y, "y")
        treated = as_1d(treated, "treated")
        post = as_1d(post, "post")
        check_lengths(len(y), treated=treated, post=post)
        design = np.column_stack([
            np.ones(len(y)),
            treated,
            post,
            treated * post,
        ])
        return self._fit_design(
            design,
            y,
            ["intercept", "treated", "post", "treated_x_post"],
            effect_index=3,
        )


class RegressionDiscontinuity(_DesignEffect):
    """Local linear regression discontinuity at ``cutoff``.

    The design is ``1, (R-c), D, (R-c)*D``. The effect is the jump in ``D``.
    Rows outside ``bandwidth`` are dropped when ``bandwidth`` is set.
    """

    def __init__(self, cutoff: float = 0.0, bandwidth: float = None, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth

    def fit(self, running, y, treatment=None):
        running = as_1d(running, "running")
        y = as_1d(y, "y")
        check_lengths(len(running), y=y)
        if treatment is None:
            treatment = (running >= self.cutoff).astype(float)
        else:
            treatment = as_1d(treatment, "treatment")
        x = running - self.cutoff
        mask = np.ones(len(y), dtype=bool)
        if self.bandwidth is not None:
            mask = np.abs(x) <= float(self.bandwidth)
        x, y, d = x[mask], y[mask], treatment[mask]
        design = np.column_stack([np.ones(len(y)), x, d, x * d])
        self.running_ = running
        return self._fit_design(design, y, ["intercept", "slope", "treatment", "slope_x_treatment"], 2)

    def explain(self, X=None, **kwargs):
        return super().explain(X, **kwargs)


class InterruptedTimeSeries(_DesignEffect):
    """Level and slope change after ``intervention_time``.

    The reported effect is the immediate level shift.
    """

    def fit(self, time, y, intervention_time: float):
        time = as_1d(time, "time")
        y = as_1d(y, "y")
        check_lengths(len(time), y=y)
        post = (time >= intervention_time).astype(float)
        t_since = np.maximum(time - intervention_time, 0.0)
        design = np.column_stack([np.ones(len(y)), time, post, t_since])
        return self._fit_design(design, y, ["intercept", "time", "level_shift", "slope_change"], 2)


class InstrumentalVariable(ShapMixin):
    """Two-stage least squares.

    First stage ``W ~ Z, X``. Second stage ``Y ~ W_hat, X``. The treatment
    coefficient is the effect. Uncertainty is a Bayesian bootstrap of both
    stages.
    """

    def __init__(self, n_boot: int = 200, random_state: int = 0):
        self.n_boot = int(n_boot)
        self.random_state = int(random_state)

    def fit(self, y, treatment, instrument, X=None):
        y = as_1d(y, "y")
        w = as_1d(treatment, "treatment")
        z = as_1d(instrument, "instrument")
        check_lengths(len(y), treatment=w, instrument=z)
        if X is None:
            covariates = np.zeros((len(y), 0))
            names = []
        else:
            covariates, names = as_float_matrix(X)
            check_lengths(len(y), X=covariates)
        self.feature_names_ = ["treatment", *names]
        self._y = y
        self._w = w
        self._z = z
        self._X = covariates

        def _theta(weights):
            coef, _ = self._stages(y, w, z, covariates, weights)
            return float(np.asarray(coef).ravel()[0])

        coef, _ = self._stages(y, w, z, covariates, np.full(len(y), 1.0 / len(y)))
        self.coef_ = np.asarray(coef, dtype=float).ravel()
        self.effect_ = float(self.coef_[0])
        self.draws_ = _bayesian_bootstrap(len(y), _theta, self.n_boot, self.random_state)
        self.summary_ = _summary(self.effect_, self.draws_)
        return self

    @staticmethod
    def _stages(y, w, z, covariates, weights):
        first_x = np.column_stack([np.ones(len(y)), z, covariates]) if covariates.size else np.column_stack(
            [np.ones(len(y)), z]
        )
        pi = _weighted_lstsq(first_x, w, weights)
        w_hat = first_x @ pi
        second_x = np.column_stack([np.ones(len(y)), w_hat, covariates]) if covariates.size else np.column_stack(
            [np.ones(len(y)), w_hat]
        )
        beta = _weighted_lstsq(second_x, y, weights)
        return beta[1:], w_hat

    def predict(self, treatment=None, X=None):
        """Structural prediction using the second-stage coefficients.

        ``treatment`` is the exposure level to plug in (not the first-stage
        fitted value). ``X`` are the same covariates used at fit time.
        """
        if treatment is None:
            treatment = self._w
        treatment = as_1d(treatment, "treatment")
        if X is None:
            covariates = np.zeros((len(treatment), 0)) if self._X.size == 0 else self._X
            if len(covariates) != len(treatment):
                covariates = np.zeros((len(treatment), self._X.shape[1]))
        else:
            covariates, _ = as_float_matrix(X)
        # coef_ is [treatment, covariates...] from beta[1:]
        return self.coef_[0] * treatment + (
            covariates @ self.coef_[1:] if covariates.size else 0.0
        )

    def summary(self):
        return {k: v for k, v in self.summary_.items() if k != "draws"}

    def explain(self, X=None, **kwargs):
        if self._X.size == 0:
            design = np.asarray(self._w, dtype=float).reshape(-1, 1)
            return linear_shap(np.array([self.effect_]), design, ["treatment"])
        covariates = self._X if X is None else as_float_matrix(X)[0]
        names = self.feature_names_[1:]
        coef = np.asarray(self.coef_[1:], dtype=float).ravel()
        return linear_shap(coef, covariates, names)


class SyntheticControl:
    """Donor weights for a single treated unit.

    ``control_pre`` has shape ``(n_pre, n_donors)`` and ``treated_pre`` has
    shape ``(n_pre,)``. Weights are non-negative and sum to one.
    """

    def __init__(self, n_boot: int = 100, random_state: int = 0):
        self.n_boot = int(n_boot)
        self.random_state = int(random_state)

    def fit(self, control_pre, treated_pre, control_post, treated_post):
        control_pre = np.asarray(control_pre, dtype=float)
        treated_pre = np.asarray(treated_pre, dtype=float).ravel()
        control_post = np.asarray(control_post, dtype=float)
        treated_post = np.asarray(treated_post, dtype=float).ravel()
        if control_pre.shape[0] != len(treated_pre):
            raise ValueError("control_pre and treated_pre must share the pre-period length.")
        if control_post.shape[0] != len(treated_post):
            raise ValueError("control_post and treated_post must share the post-period length.")
        self.weights_ = self._weights(control_pre, treated_pre)
        self.synth_post_ = control_post @ self.weights_
        self.effect_path_ = treated_post - self.synth_post_
        self.effect_ = float(np.mean(self.effect_path_))
        rng = np.random.default_rng(self.random_state)
        draws = np.empty(self.n_boot)
        n_pre = len(treated_pre)
        for b in range(self.n_boot):
            take = rng.integers(0, n_pre, size=n_pre)
            w = self._weights(control_pre[take], treated_pre[take])
            draws[b] = float(np.mean(treated_post - control_post @ w))
        self.draws_ = draws
        self.summary_ = _summary(self.effect_, draws)
        self.feature_names_ = [f"donor_{i}" for i in range(control_pre.shape[1])]
        return self

    @staticmethod
    def _weights(control_pre, treated_pre):
        n_donors = control_pre.shape[1]

        def loss(w):
            return np.sum((control_pre @ w - treated_pre) ** 2)

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_donors
        start = np.full(n_donors, 1.0 / n_donors)
        result = minimize(loss, start, method="SLSQP", bounds=bounds, constraints=cons)
        if not result.success:
            w = np.clip(result.x, 0, None)
            total = w.sum()
            return w / total if total > 0 else start
        return np.asarray(result.x, dtype=float)

    def predict(self, control_post=None):
        if control_post is None:
            return self.synth_post_
        return np.asarray(control_post, dtype=float) @ self.weights_

    def summary(self):
        return {k: v for k, v in self.summary_.items() if k != "draws"}

    def explain(self, X=None, **kwargs):
        """SHAP-style donor contributions on the pre-period fit.

        Each donor's attribution is its weight times the gap between that
        donor's pre-period mean and the average donor.
        """
        from ..interpretation import ShapExplanation

        if X is None:
            raise ValueError("Pass control_pre as X to explain donor contributions.")
        donors, _ = as_float_matrix(X)
        # donors: (n_pre, n_donors). Attribution of donor j is constant over time:
        # weight_j * (donor_j - mean donor), broadcast to each pre period.
        centered = donors - donors.mean(axis=1, keepdims=True)
        values = centered * self.weights_
        base = np.full(len(donors), float(donors.mean()))
        predictions = donors @ self.weights_
        return ShapExplanation(values, base, donors, self.feature_names_, predictions)

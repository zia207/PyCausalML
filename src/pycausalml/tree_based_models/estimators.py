"""Honest causal trees and generalized-random-forest style estimators.

The identifying formulas follow Athey and Imbens (honest causal trees),
Athey, Tibshirani and Wager (generalized random forests), and the Python
stand-ins in ``causal_tree_methods.py``. Leaf search uses a causal splitting
rule. Forest estimators regress a cross-fitted doubly robust or Wald score,
which is the scikit-learn counterpart of ``grf`` used in that tutorial file.
``skgrf`` is used by :func:`fit_skgrf_causal_forest` when that package is
installed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .._common import (
    ShapMixin,
    as_1d,
    as_float_matrix,
    check_lengths,
    cross_fit_mu_by_arm,
    cross_fit_propensity,
    cross_fit_regression,
    dr_scores,
    make_regressor,
    mean_se,
)
from ..data import synthetic_data

__all__ = [
    "BoostedIVForest",
    "CXGBoost",
    "CausalForest",
    "CausalSurvivalForest",
    "CausalTree",
    "InstrumentalForest",
    "MultiArmCausalBoost",
    "MultiArmCausalForest",
    "average_treatment_effect",
    "best_linear_projection",
    "fit_skgrf_causal_forest",
    "plot_causal_tree",
    "rank_average_treatment_effect",
    "synthetic_data",
    "variable_importance",
]


def _safe_cate(y, w) -> float:
    treated = y[np.asarray(w) == 1]
    control = y[np.asarray(w) == 0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def _leaf():
    return {"is_leaf": True, "n": 0, "cate": float("nan"), "gain": 0.0}


class CausalTree(ShapMixin):
    """Honest causal tree (Athey and Imbens, 2016).

    Splits maximize the squared difference in treatment effects. When
    ``honest=True``, leaf effects are estimated on a held-out sample.
    """

    def __init__(
        self,
        min_samples_leaf: int = 20,
        max_depth: int = 3,
        honest: bool = True,
        n_thresholds: int = 6,
        random_state: int = 0,
    ):
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = int(max_depth)
        self.honest = bool(honest)
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        self.feature_names_ = names
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        if self.honest:
            perm = rng.permutation(len(X))
            n_est = max(self.min_samples_leaf * 2, len(X) // 2)
            est_idx = np.sort(perm[:n_est])
            split_mask = np.ones(len(X), dtype=bool)
            split_mask[est_idx] = False
            split_idx = np.flatnonzero(split_mask)
            if len(split_idx) < self.min_samples_leaf:
                split_idx = np.arange(len(X))
        else:
            split_idx = np.arange(len(X))
            est_idx = split_idx
        self.split_idx_ = split_idx
        self.est_idx_ = est_idx
        self.tree_ = self._grow(X[split_idx], y[split_idx], w[split_idx], 0)
        self._estimate_leaves(self.tree_, X[est_idx], y[est_idx], w[est_idx])
        self._X = X
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.array([self._predict_row(self.tree_, row) for row in X])

    predict_cate = predict

    def feature_importance(self) -> np.ndarray:
        gains = np.zeros(self.n_features_)

        def walk(node):
            if node["is_leaf"]:
                return
            gains[node["feature"]] += node.get("gain", 0.0)
            walk(node["left"])
            walk(node["right"])

        walk(self.tree_)
        total = gains.sum()
        return gains / total if total > 0 else gains

    def rules(self):
        found = []

        def walk(node, conds):
            if node["is_leaf"]:
                found.append(
                    {
                        "cate": node.get("cate", np.nan),
                        "n": node.get("n_est", node.get("n", 0)),
                        "conditions": " AND ".join(conds) if conds else "(root)",
                    }
                )
                return
            name = self.feature_names_[node["feature"]]
            thr = node["threshold"]
            walk(node["left"], conds + [f"{name} <= {thr:.3f}"])
            walk(node["right"], conds + [f"{name} > {thr:.3f}"])

        walk(self.tree_, [])
        return found

    def _grow(self, X, y, w, depth):
        node = {"is_leaf": True, "n": int(len(y)), "cate": _safe_cate(y, w), "gain": 0.0}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(X, y, w)
        if split is None:
            return node
        feature, threshold, gain = split
        left = X[:, feature] <= threshold
        node.update(
            is_leaf=False,
            feature=int(feature),
            threshold=float(threshold),
            gain=float(gain),
            left=self._grow(X[left], y[left], w[left], depth + 1),
            right=self._grow(X[~left], y[~left], w[~left], depth + 1),
        )
        return node

    def _best_split(self, X, y, w):
        n, p = X.shape
        best = None
        best_score = 0.0
        min_arm = max(2, self.min_samples_leaf // 5)
        for j in range(p):
            col = X[:, j]
            if np.nanstd(col) < 1e-8:
                continue
            cuts = np.unique(np.quantile(col, np.linspace(0.2, 0.8, self.n_thresholds)))
            for thr in cuts:
                left = col <= thr
                n_left = int(left.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                if w[left].sum() < min_arm or (n_left - w[left].sum()) < min_arm:
                    continue
                if w[~left].sum() < min_arm or (n_right - w[~left].sum()) < min_arm:
                    continue
                tau_l = _safe_cate(y[left], w[left])
                tau_r = _safe_cate(y[~left], w[~left])
                if not np.isfinite(tau_l) or not np.isfinite(tau_r):
                    continue
                score = (n_left * n_right / n) * (tau_l - tau_r) ** 2
                if score > best_score:
                    best_score = score
                    best = (j, float(thr), float(score))
        return best

    def _estimate_leaves(self, node, X, y, w):
        if node["is_leaf"]:
            node["cate"] = _safe_cate(y, w) if len(y) else float("nan")
            node["n_est"] = int(len(y))
            return
        left = X[:, node["feature"]] <= node["threshold"]
        self._estimate_leaves(node["left"], X[left], y[left], w[left])
        self._estimate_leaves(node["right"], X[~left], y[~left], w[~left])

    def _predict_row(self, node, row):
        while not node["is_leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        cate = node.get("cate", np.nan)
        return float(cate) if np.isfinite(cate) else 0.0


def plot_causal_tree(model: CausalTree, title: str = "Causal tree: leaf treatment effects"):
    """Draw an honest causal tree. Leaves are labeled with the CATE."""
    import matplotlib.pyplot as plt

    positions = {}
    labels = {}
    edges = []

    def layout(node, node_id, depth, x0, x1):
        x = 0.5 * (x0 + x1)
        positions[node_id] = (x, -depth)
        if node["is_leaf"]:
            labels[node_id] = f"CATE {node.get('cate', np.nan):.2f}\nn={node.get('n_est', 0)}"
            return
        name = model.feature_names_[node["feature"]]
        labels[node_id] = f"{name}\n<= {node['threshold']:.2f}"
        edges.append((node_id, node_id * 2))
        edges.append((node_id, node_id * 2 + 1))
        layout(node["left"], node_id * 2, depth + 1, x0, x)
        layout(node["right"], node_id * 2 + 1, depth + 1, x, x1)

    layout(model.tree_, 1, 0, 0.0, 1.0)
    depth = max(-pos[1] for pos in positions.values())
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * (2 ** min(depth, 4))), 1.5 * (depth + 2)))
    for parent, child in edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#888888", lw=1)
    for node_id, (x, y) in positions.items():
        is_leaf = "CATE" in labels[node_id]
        ax.text(
            x,
            y,
            labels[node_id],
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#F5C6CB" if is_leaf else "#D6EAF8",
                edgecolor="#333333",
            ),
        )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig


class CausalForest(ShapMixin):
    """Doubly robust causal forest.

    Cross-fitted outcome and propensity models form an augmented-IPW score.
    A random forest regresses that score on covariates. This is the same
    construction as ``CausalForest`` in ``causal_tree_methods.py``.
    """

    def __init__(
        self,
        num_trees: int = 100,
        min_node_size: int = 5,
        sample_fraction: float = 0.5,
        n_splits: int = 2,
        nuisance: str = "gbr",
        nuisance_estimators: int = 40,
        random_state: int = 0,
    ):
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.sample_fraction = float(sample_fraction)
        self.n_splits = int(n_splits)
        self.nuisance = nuisance
        self.nuisance_estimators = int(nuisance_estimators)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        self.feature_names_ = names
        mu0, mu1 = cross_fit_mu_by_arm(
            X, y, w, self.n_splits, self.random_state, self.nuisance, self.nuisance_estimators
        )
        e_hat = cross_fit_propensity(
            X, w, self.n_splits, self.random_state + 11, "gbc", self.nuisance_estimators
        )
        gamma = dr_scores(y, w, mu0, mu1, e_hat)
        self.dr_scores_ = gamma
        self.forest_ = RandomForestRegressor(
            n_estimators=self.num_trees,
            min_samples_leaf=self.min_node_size,
            max_samples=self.sample_fraction,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.forest_.fit(X, gamma)
        self._X = X
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.asarray(self.forest_.predict(X), dtype=float)

    predict_cate = predict

    def predict_interval(self, X):
        """Tree-to-tree standard error of the forest prediction."""
        X, _ = as_float_matrix(X)
        tree_preds = np.column_stack([est.predict(X) for est in self.forest_.estimators_])
        se = tree_preds.std(axis=1) / np.sqrt(self.num_trees)
        mean = tree_preds.mean(axis=1)
        return mean, se

    def average_treatment_effect(self):
        estimate, se = mean_se(self.dr_scores_)
        return {"estimate": estimate, "std.err": se}

    def feature_importance(self) -> np.ndarray:
        return np.asarray(self.forest_.feature_importances_, dtype=float)

    def rank_average_treatment_effect(self, priorities):
        return rank_average_treatment_effect(self.dr_scores_, priorities)


def average_treatment_effect(model) -> dict:
    """ATE and standard error from a forest that stores doubly robust scores."""
    if hasattr(model, "average_treatment_effect"):
        return model.average_treatment_effect()
    raise TypeError("This estimator does not store an ATE influence function.")


def variable_importance(model) -> np.ndarray:
    return np.asarray(model.feature_importance(), dtype=float)


def rank_average_treatment_effect(scores, priorities) -> dict:
    """Targeting operator characteristic of scores sorted by a priority."""
    priorities = np.asarray(priorities, dtype=float).ravel()
    gamma = np.asarray(scores, dtype=float).ravel()
    if len(priorities) != len(gamma):
        raise ValueError("priorities must have one score per training row.")
    order = np.argsort(-priorities)
    ranked = gamma[order]
    n = len(ranked)
    q = np.arange(1, n + 1) / n
    toc = np.cumsum(ranked) / np.arange(1, n + 1) - ranked.mean()
    autoc = float(np.trapezoid(toc, q))
    return {"q": q, "toc": toc, "autoc": autoc}


def best_linear_projection(scores, covariate):
    """OLS of a doubly robust score on a covariate, with HC1 standard errors."""
    a = np.asarray(covariate, dtype=float).ravel()
    gamma = np.asarray(scores, dtype=float).ravel()
    design = np.column_stack([np.ones(len(a)), a])
    beta, _, _, _ = np.linalg.lstsq(design, gamma, rcond=None)
    resid = gamma - design @ beta
    n, k = design.shape
    meat = design.T @ (design * (resid ** 2)[:, None])
    xtx_inv = np.linalg.pinv(design.T @ design)
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    try:
        from scipy import stats

        pvalues = 2 * stats.norm.sf(np.abs(beta / np.maximum(se, 1e-12)))
    except Exception:
        pvalues = np.full(k, np.nan)
    return {"params": beta, "bse": se, "pvalues": pvalues}


def _km_censoring_survival(time, event) -> np.ndarray:
    """Kaplan-Meier estimate of ``P(C >= t)`` evaluated at each row's time."""
    time = np.asarray(time, dtype=float)
    censored = (1 - np.asarray(event).astype(int)).astype(int)
    order = np.argsort(time)
    t_sorted = time[order]
    c_sorted = censored[order]
    n = len(time)
    surv_at_sorted = np.ones(n)
    at_risk = n
    surv = 1.0
    i = 0
    while i < n:
        t = t_sorted[i]
        d = 0
        leave = 0
        while i < n and t_sorted[i] == t:
            d += int(c_sorted[i] == 1)
            leave += 1
            i += 1
        surv_at_sorted[i - leave : i] = surv
        if at_risk > 0 and d > 0:
            surv *= 1.0 - d / at_risk
        at_risk -= leave
    out = np.empty(n)
    out[order] = np.clip(surv_at_sorted, 0.05, 1.0)
    return out


class CausalSurvivalForest(ShapMixin):
    """IPCW causal forest for a restricted-mean survival contrast.

    Censoring weights come from a Kaplan-Meier estimate of the censoring
    distribution. The weighted doubly robust forest then estimates the
    treatment contrast in truncated event time, the target used by
    ``grf::causal_survival_forest``.
    """

    def __init__(self, horizon: float, num_trees: int = 80, min_node_size: int = 5, random_state: int = 0):
        self.horizon = float(horizon)
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.random_state = int(random_state)

    def fit(self, X, treatment, time, event, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        time = as_1d(time, "time")
        event = np.asarray(event).astype(int).ravel()
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), time=time, event=event, treatment=w)
        g_hat = _km_censoring_survival(time, event)
        t_star = np.minimum(time, self.horizon)
        observed = ((event == 1) & (time <= self.horizon)) | (time >= self.horizon)
        pseudo = np.where(observed, t_star / g_hat, 0.0)
        self.feature_names_ = names
        self.forest_ = CausalForest(
            num_trees=self.num_trees,
            min_node_size=self.min_node_size,
            random_state=self.random_state,
        )
        self.forest_.fit(X, w, pseudo, feature_names=names)
        self.dr_scores_ = self.forest_.dr_scores_
        return self

    def predict(self, X):
        return self.forest_.predict(X)

    predict_cate = predict

    def average_treatment_effect(self):
        return self.forest_.average_treatment_effect()

    def feature_importance(self):
        return self.forest_.feature_importance()


def _safe_denominator(den, floor=0.05):
    den = np.asarray(den, dtype=float).copy()
    small = np.abs(den) < floor
    den[small] = np.sign(den[small]) * floor
    den[den == 0] = floor
    return den


class InstrumentalForest(ShapMixin):
    """Forest Wald estimator for a conditional local average treatment effect.

    Nuisance functions ``E[Y|X]``, ``E[W|X]``, and ``E[Z|X]`` are
    cross-fitted. Two forests estimate the residual covariances with the
    instrument, and their ratio is the conditional Wald estimand used by
    ``grf::instrumental_forest``.
    """

    def __init__(self, num_trees: int = 80, min_node_size: int = 5, n_splits: int = 2, random_state: int = 0):
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, instrument, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = as_1d(treatment, "treatment")
        z = as_1d(instrument, "instrument")
        check_lengths(len(X), y=y, treatment=w, instrument=z)
        self.feature_names_ = names
        m_y = cross_fit_regression(X, y, self.n_splits, self.random_state)
        m_w = cross_fit_regression(X, w, self.n_splits, self.random_state + 3)
        if set(np.unique(np.round(z, 6))).issubset({0.0, 1.0}):
            m_z = cross_fit_propensity(X, z.astype(int), self.n_splits, self.random_state + 5)
        else:
            m_z = cross_fit_regression(X, z, self.n_splits, self.random_state + 5)
        y_res = y - m_y
        w_res = w - m_w
        z_res = z - m_z
        num = y_res * z_res
        den = w_res * z_res
        kw = dict(
            n_estimators=self.num_trees,
            min_samples_leaf=self.min_node_size,
            max_features="sqrt",
            n_jobs=1,
        )
        self.num_forest_ = RandomForestRegressor(random_state=self.random_state, **kw)
        self.den_forest_ = RandomForestRegressor(random_state=self.random_state + 1, **kw)
        self.num_forest_.fit(X, num)
        self.den_forest_.fit(X, den)
        self.tau_hat_ = self.num_forest_.predict(X) / _safe_denominator(self.den_forest_.predict(X))
        den_mean = float(np.mean(den)) or 1e-8
        self.psi_ = z_res * (y_res - np.mean(self.tau_hat_) * w_res) / den_mean
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self.num_forest_.predict(X) / _safe_denominator(self.den_forest_.predict(X))

    predict_cate = predict

    def average_treatment_effect(self):
        estimate = float(np.mean(self.tau_hat_))
        _, se = mean_se(self.psi_)
        return {"estimate": estimate, "std.err": se}

    def feature_importance(self) -> np.ndarray:
        return 0.5 * (
            np.asarray(self.num_forest_.feature_importances_)
            + np.asarray(self.den_forest_.feature_importances_)
        )


class MultiArmCausalForest(ShapMixin):
    """One doubly robust forest per contrast against a baseline arm."""

    def __init__(self, baseline: int = 0, num_trees: int = 60, min_node_size: int = 5, random_state: int = 0):
        self.baseline = baseline
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        arms = [arm for arm in sorted(np.unique(w)) if arm != self.baseline]
        if not arms:
            raise ValueError("treatment has no arm other than the baseline.")
        self.arms_ = arms
        self.feature_names_ = names
        self.forests_ = []
        for i, arm in enumerate(arms):
            mask = (w == self.baseline) | (w == arm)
            forest = CausalForest(
                num_trees=self.num_trees,
                min_node_size=self.min_node_size,
                random_state=self.random_state + i,
            )
            treated = (w[mask] == arm).astype(int)
            forest.fit(X[mask], treated, y[mask], feature_names=names)
            self.forests_.append(forest)
        return self

    def predict(self, X):
        cols = [forest.predict(X) for forest in self.forests_]
        return np.column_stack(cols)

    def predict_cate(self, X):
        return self.predict(X)

    def feature_importance(self) -> np.ndarray:
        imps = [forest.feature_importance() for forest in self.forests_]
        return np.mean(np.vstack(imps), axis=0)


class _BoostedContrast(ShapMixin):
    """X-learner with gradient boosting, shared by CXGBoost and multi-arm boost."""

    def __init__(self, n_estimators: int = 60, max_depth: int = 2, learner: str = "gbr", random_state: int = 0):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learner = learner
        self.random_state = int(random_state)

    def _fit_binary(self, X, treatment, y, names):
        w = np.asarray(treatment).astype(int).ravel()
        y = as_1d(y, "y")
        self.feature_names_ = names
        kind = self.learner
        self.mu0_ = make_regressor(kind, self.random_state, self.n_estimators)
        self.mu1_ = make_regressor(kind, self.random_state + 1, self.n_estimators)
        self.mu0_.fit(X[w == 0], y[w == 0])
        self.mu1_.fit(X[w == 1], y[w == 1])
        d1 = y[w == 1] - self.mu0_.predict(X[w == 1])
        d0 = self.mu1_.predict(X[w == 0]) - y[w == 0]
        self.tau1_ = make_regressor(kind, self.random_state + 2, self.n_estimators)
        self.tau0_ = make_regressor(kind, self.random_state + 3, self.n_estimators)
        self.tau1_.fit(X[w == 1], d1)
        self.tau0_.fit(X[w == 0], d0)
        from .._common import make_classifier

        self.propensity_ = make_classifier("logit", self.random_state)
        self.propensity_.fit(X, w)
        return self

    def _predict_binary(self, X):
        from .._common import clip_propensity

        e = clip_propensity(self.propensity_.predict_proba(X)[:, 1])
        tau1 = self.tau1_.predict(X)
        tau0 = self.tau0_.predict(X)
        return e * tau0 + (1.0 - e) * tau1

    def feature_importance(self) -> np.ndarray:
        imps = []
        for model in (self.tau0_, self.tau1_):
            if hasattr(model, "feature_importances_"):
                imps.append(np.asarray(model.feature_importances_, dtype=float))
        if not imps:
            return np.zeros(len(self.feature_names_))
        return np.mean(np.vstack(imps), axis=0)


class CXGBoost(_BoostedContrast):
    """Boosted X-learner (RCausalML ``CXGBoost``).

    Uses XGBoost when ``learner='xgb'`` and XGBoost is installed, otherwise
    scikit-learn gradient boosting.
    """

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        return self._fit_binary(X, treatment, y, names)

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self._predict_binary(X)

    predict_cate = predict


class BoostedIVForest(ShapMixin):
    """Boosted residual Wald estimator for a conditional LATE."""

    def __init__(self, n_estimators: int = 60, n_splits: int = 2, learner: str = "gbr", random_state: int = 0):
        self.n_estimators = int(n_estimators)
        self.n_splits = int(n_splits)
        self.learner = learner
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, instrument, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = as_1d(treatment, "treatment")
        z = as_1d(instrument, "instrument")
        check_lengths(len(X), y=y, treatment=w, instrument=z)
        self.feature_names_ = names
        m_y = cross_fit_regression(X, y, self.n_splits, self.random_state, self.learner, self.n_estimators)
        m_w = cross_fit_regression(X, w, self.n_splits, self.random_state + 1, self.learner, self.n_estimators)
        m_z = cross_fit_regression(X, z, self.n_splits, self.random_state + 2, self.learner, self.n_estimators)
        self.num_ = make_regressor(self.learner, self.random_state, self.n_estimators)
        self.den_ = make_regressor(self.learner, self.random_state + 1, self.n_estimators)
        self.num_.fit(X, (y - m_y) * (z - m_z))
        self.den_.fit(X, (w - m_w) * (z - m_z))
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self.num_.predict(X) / _safe_denominator(self.den_.predict(X))

    predict_cate = predict

    def feature_importance(self) -> np.ndarray:
        parts = []
        for model in (self.num_, self.den_):
            if hasattr(model, "feature_importances_"):
                parts.append(np.asarray(model.feature_importances_, dtype=float))
        if not parts:
            return np.zeros(len(self.feature_names_))
        return np.mean(np.vstack(parts), axis=0)


class MultiArmCausalBoost(ShapMixin):
    """One boosted X-learner per contrast against the baseline arm."""

    def __init__(self, baseline: int = 0, n_estimators: int = 40, learner: str = "gbr", random_state: int = 0):
        self.baseline = baseline
        self.n_estimators = int(n_estimators)
        self.learner = learner
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        arms = [arm for arm in sorted(np.unique(w)) if arm != self.baseline]
        self.arms_ = arms
        self.feature_names_ = names
        self.models_ = []
        for i, arm in enumerate(arms):
            mask = (w == self.baseline) | (w == arm)
            model = CXGBoost(
                n_estimators=self.n_estimators,
                learner=self.learner,
                random_state=self.random_state + i,
            )
            model.fit(X[mask], (w[mask] == arm).astype(int), y[mask], feature_names=names)
            self.models_.append(model)
        return self

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models_])

    def predict_cate(self, X):
        return self.predict(X)

    def feature_importance(self) -> np.ndarray:
        return np.mean(np.vstack([m.feature_importance() for m in self.models_]), axis=0)


def fit_skgrf_causal_forest(X, y, treatment, **kwargs):
    """Fit ``skgrf.ensemble.causal_forest.GRFCausalForest`` when installed.

    The result is the upstream estimator. Call ``predict`` on it for CATE.
    Install with ``pip install 'pycausalml[skgrf]'``.
    """
    from ..integrations.backends import require_module

    require_module("skgrf", "skgrf")
    from skgrf.ensemble.causal_forest import GRFCausalForest

    X, _ = as_float_matrix(X)
    est = GRFCausalForest(**kwargs)
    est.fit(X, np.asarray(y, dtype=float).ravel(), np.asarray(treatment).ravel())
    return est

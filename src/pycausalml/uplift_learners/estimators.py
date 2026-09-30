"""Uplift trees, uplift forests, and uplift curves.

Split criteria follow the divergence gains used by uplift random forests
(KL, Euclidean, chi-squared, contextual treatment selection, and
delta-delta-p). Continuous outcomes use an honest difference-of-means
split, the same rule as an interaction tree. The XGBoost uplift model is
a T-learner, matching ``R/uplift_xgboost.R``.
"""

from __future__ import annotations

import numpy as np

from .._common import ShapMixin, as_1d, as_float_matrix, check_lengths, make_regressor

__all__ = [
    "InteractionTree",
    "UpliftRandomForest",
    "UpliftTree",
    "UpliftXGBoost",
    "format_uplift_tree",
    "plot_uplift_curve",
    "plot_uplift_tree",
    "qini_curve",
]


def _clip_prob(p):
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _divergence(p_t, p_c, criterion: str) -> float:
    p_t = _clip_prob(p_t)
    p_c = _clip_prob(p_c)
    if criterion == "kl":
        return p_t * np.log(p_t / p_c) + (1.0 - p_t) * np.log((1.0 - p_t) / (1.0 - p_c))
    if criterion == "ed":
        return (p_t - p_c) ** 2
    if criterion == "chi":
        return (p_t - p_c) ** 2 / p_c
    if criterion == "cts":
        return max(p_t, p_c)
    if criterion == "ddp":
        return p_t - p_c
    raise ValueError(f"Unknown uplift criterion '{criterion}'.")


def _is_binary(y) -> bool:
    values = set(np.unique(np.asarray(y)))
    return values.issubset({0, 1, 0.0, 1.0})


class UpliftTree(ShapMixin):
    """Single uplift tree.

    Parameters
    ----------
    criterion : {'kl', 'ed', 'chi', 'cts', 'ddp'}
        Divergence used to score a split when the outcome is binary.
        Continuous outcomes always split on the squared CATE difference.
    """

    def __init__(
        self,
        criterion: str = "kl",
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        min_samples_treatment: int = 5,
        n_thresholds: int = 5,
        max_features: int = None,
        random_state: int = 0,
    ):
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_treatment = int(min_samples_treatment)
        self.n_thresholds = int(n_thresholds)
        self.max_features = max_features
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        self.feature_names_ = names
        self.n_features_ = X.shape[1]
        self.binary_ = _is_binary(y)
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow(X, y, w, 0)
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return np.array([self._predict_row(self.tree_, row) for row in X])

    predict_cate = predict

    def predict_rates(self, X):
        """Leaf control and treated outcome means for each row.

        For a binary outcome these are response probabilities. For a
        continuous outcome they are sample means.
        """
        X, _ = as_float_matrix(X)
        rows = [self._rate_row(self.tree_, row) for row in X]
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    def feature_importance(self) -> np.ndarray:
        gains = np.zeros(self.n_features_)

        def walk(node):
            if node.get("is_leaf", True):
                return
            gains[node["feature"]] += max(node.get("gain", 0.0), 0.0)
            walk(node["left"])
            walk(node["right"])

        walk(self.tree_)
        total = gains.sum()
        return gains / total if total > 0 else gains

    def _rates(self, y, w):
        treated = y[w == 1]
        control = y[w == 0]
        p_t = float(treated.mean()) if len(treated) else float("nan")
        p_c = float(control.mean()) if len(control) else float("nan")
        return p_t, p_c

    def _grow(self, X, y, w, depth):
        p_t, p_c = self._rates(y, w)
        node = {
            "is_leaf": True,
            "cate": (p_t - p_c) if np.isfinite(p_t) and np.isfinite(p_c) else 0.0,
            "p_treat": p_t if np.isfinite(p_t) else 0.0,
            "p_control": p_c if np.isfinite(p_c) else 0.0,
            "n": int(len(y)),
        }
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
        p_t, p_c = self._rates(y, w)
        if not np.isfinite(p_t) or not np.isfinite(p_c):
            return None
        parent = _divergence(p_t, p_c, self.criterion) if self.binary_ else (p_t - p_c)
        n_features = p if self.max_features is None else min(p, int(self.max_features))
        features = self._rng.choice(p, size=n_features, replace=False)
        best = None
        best_gain = 0.0
        for j in features:
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
                if min(w[left].sum(), n_left - w[left].sum()) < self.min_samples_treatment:
                    continue
                if min(w[~left].sum(), n_right - w[~left].sum()) < self.min_samples_treatment:
                    continue
                pt_l, pc_l = self._rates(y[left], w[left])
                pt_r, pc_r = self._rates(y[~left], w[~left])
                if not all(np.isfinite(v) for v in (pt_l, pc_l, pt_r, pc_r)):
                    continue
                if self.binary_ and self.criterion == "ddp":
                    gain = abs((pt_l - pc_l) - (pt_r - pc_r))
                elif self.binary_:
                    d_l = _divergence(pt_l, pc_l, self.criterion)
                    d_r = _divergence(pt_r, pc_r, self.criterion)
                    gain = (n_left / n) * d_l + (n_right / n) * d_r - parent
                else:
                    gain = (n_left * n_right / n) * ((pt_l - pc_l) - (pt_r - pc_r)) ** 2
                if gain > best_gain:
                    best_gain = gain
                    best = (int(j), float(thr), float(gain))
        return best

    def _predict_row(self, node, row):
        while not node.get("is_leaf", True):
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return float(node["cate"])

    def _rate_row(self, node, row):
        while not node.get("is_leaf", True):
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return float(node.get("p_control", 0.0)), float(node.get("p_treat", 0.0))


class InteractionTree(UpliftTree):
    """Interaction tree (Su et al.): splits on the treatment-effect difference."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20, random_state: int = 0):
        super().__init__(
            criterion="ed",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_treatment=5,
            random_state=random_state,
        )


class UpliftRandomForest(ShapMixin):
    """Bagged uplift trees.

    Parameters
    ----------
    criterion : {'kl', 'ed', 'chi', 'cts', 'ddp'}
        Split criterion passed to each :class:`UpliftTree`.
    """

    def __init__(
        self,
        criterion: str = "kl",
        n_trees: int = 25,
        max_depth: int = 3,
        min_samples_leaf: int = 15,
        max_features=None,
        random_state: int = 0,
    ):
        self.criterion = criterion
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        self.feature_names_ = names
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n = len(X)
        n_features = X.shape[1] if self.max_features is None else self.max_features
        if n_features == "sqrt":
            n_features = max(1, int(np.sqrt(X.shape[1])))
        for i in range(self.n_trees):
            take = rng.integers(0, n, size=n)
            tree = UpliftTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=int(n_features),
                random_state=int(rng.integers(0, 10_000_000)),
            )
            tree.fit(X[take], w[take], y[take], feature_names=names)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        preds = np.column_stack([tree.predict(X) for tree in self.trees_])
        return preds.mean(axis=1)

    predict_cate = predict

    def predict_rates(self, X):
        """Average leaf control and treated means across trees."""
        stacked = [tree.predict_rates(X) for tree in self.trees_]
        p_control = np.mean([pair[0] for pair in stacked], axis=0)
        p_treat = np.mean([pair[1] for pair in stacked], axis=0)
        return p_control, p_treat

    def feature_importance(self) -> np.ndarray:
        imps = np.vstack([tree.feature_importance() for tree in self.trees_])
        return imps.mean(axis=0)


class UpliftXGBoost(ShapMixin):
    """T-learner uplift model.

    ``learner='xgb'`` uses XGBoost when it is installed. Any other learner
    name is passed to the shared gradient-boosting factory (``gbr``).
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learner: str = "xgb",
        random_state: int = 0,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
    ):
        self.n_estimators = int(n_estimators)
        self.learner = learner
        self.random_state = int(random_state)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)

    def _booster(self, seed: int):
        if str(self.learner).lower() == "xgb":
            try:
                from xgboost import XGBRegressor

                return XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    random_state=seed,
                    n_jobs=1,
                    verbosity=0,
                )
            except ImportError:
                pass
        return make_regressor(self.learner, seed, self.n_estimators)

    def fit(self, X, treatment, y, feature_names=None):
        X, names = as_float_matrix(X, feature_names)
        y = as_1d(y, "y")
        w = np.asarray(treatment).astype(int).ravel()
        check_lengths(len(X), y=y, treatment=w)
        if w.sum() < 5 or (len(w) - w.sum()) < 5:
            raise ValueError("UpliftXGBoost needs observations in both treatment arms.")
        self.feature_names_ = names
        self.mu0_ = self._booster(self.random_state)
        self.mu1_ = self._booster(self.random_state + 1)
        self.mu0_.fit(X[w == 0], y[w == 0])
        self.mu1_.fit(X[w == 1], y[w == 1])
        return self

    def predict(self, X):
        X, _ = as_float_matrix(X)
        return self.mu1_.predict(X) - self.mu0_.predict(X)

    predict_cate = predict

    def predict_potential_outcomes(self, X):
        X, _ = as_float_matrix(X)
        return self.mu0_.predict(X), self.mu1_.predict(X)

    def feature_importance(self) -> np.ndarray:
        parts = []
        for model in (self.mu0_, self.mu1_):
            if hasattr(model, "feature_importances_"):
                parts.append(np.asarray(model.feature_importances_, dtype=float))
        if not parts:
            return np.zeros(len(self.feature_names_))
        return np.mean(np.vstack(parts), axis=0)


def qini_curve(y, treatment, uplift):
    """Cumulative incremental gains, ordered by predicted uplift.

    Returns ``fraction`` (share of the sample targeted) and ``qini``
    (cumulative treated-minus-control outcome, scaled to the treated count).
    """
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(treatment, dtype=float).ravel()
    score = np.asarray(uplift, dtype=float).ravel()
    order = np.argsort(-score)
    y, w = y[order], w[order]
    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1.0 - w)
    cum_yt = np.cumsum(y * w)
    cum_yc = np.cumsum(y * (1.0 - w))
    with np.errstate(divide="ignore", invalid="ignore"):
        incremental = cum_yt - cum_yc * cum_t / np.maximum(cum_c, 1.0)
    incremental = np.nan_to_num(incremental)
    fraction = np.arange(1, len(y) + 1) / len(y)
    return {"fraction": fraction, "qini": incremental}


def qini_coefficient(y, treatment, uplift) -> float:
    """Area between the model Qini curve and a random-targeting line."""
    curve = qini_curve(y, treatment, uplift)
    q = curve["qini"]
    fraction = curve["fraction"]
    random_line = fraction * q[-1]
    return float(np.trapezoid(q - random_line, fraction))


def format_uplift_tree(model) -> str:
    """Plain-text split rules for one uplift or interaction tree."""
    names = list(getattr(model, "feature_names_", []))
    lines = []

    def walk(node, depth):
        pad = "  " * depth
        tau = float(node.get("cate", 0.0))
        n = int(node.get("n", 0))
        if node.get("is_leaf", True):
            lines.append(f"{pad}* leaf  tau={tau:.3f}  n={n}")
            return
        fname = names[node["feature"]] if names else f"x{node['feature']}"
        gain = float(node.get("gain", 0.0))
        lines.append(
            f"{pad}[{fname} <= {node['threshold']:.3f}]  gain={gain:.4f}  tau={tau:.3f}  n={n}"
        )
        walk(node["left"], depth + 1)
        walk(node["right"], depth + 1)

    walk(model.tree_, 0)
    return "\n".join(lines)


def plot_uplift_tree(model, title: str = "Uplift tree", ax=None):
    """Draw one uplift or interaction tree. Leaves show the estimated uplift."""
    import matplotlib.pyplot as plt

    positions = {}
    labels = {}
    edges = []

    def layout(node, node_id, depth, x0, x1):
        x = 0.5 * (x0 + x1)
        positions[node_id] = (x, -depth)
        if node.get("is_leaf", True):
            labels[node_id] = f"uplift {node.get('cate', 0.0):.2f}\nn={node.get('n', 0)}"
            return
        names = model.feature_names_
        name = names[node["feature"]]
        labels[node_id] = f"{name}\n<= {node['threshold']:.2f}"
        edges.append((node_id, node_id * 2))
        edges.append((node_id, node_id * 2 + 1))
        layout(node["left"], node_id * 2, depth + 1, x0, x)
        layout(node["right"], node_id * 2 + 1, depth + 1, x, x1)

    layout(model.tree_, 1, 0, 0.0, 1.0)
    depth = max(-pos[1] for pos in positions.values())
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * (2 ** min(depth, 4))), 1.6 * (depth + 2)))
    else:
        fig = ax.figure
    for parent, child in edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#888888", lw=1)
    for node_id, (x, y) in positions.items():
        is_leaf = labels[node_id].startswith("uplift")
        if is_leaf:
            positive = "uplift -" not in labels[node_id]
            face = "#D5F5E3" if positive else "#FADBD8"
        else:
            face = "#D6EAF8"
        ax.text(
            x,
            y,
            labels[node_id],
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor="#333333"),
        )
    ax.set_axis_off()
    ax.set_title(title)
    if own_fig:
        fig.tight_layout()
    return fig


def plot_uplift_curve(y, treatment, uplift, ax=None, title: str = "Qini curve"):
    """Plot the Qini curve against a random-targeting diagonal."""
    import matplotlib.pyplot as plt

    curve = qini_curve(y, treatment, uplift)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(curve["fraction"], curve["qini"], color="#1F4E79", label="model")
    ax.plot(curve["fraction"], curve["fraction"] * curve["qini"][-1], color="#888888", ls="--", label="random")
    ax.set_xlabel("fraction targeted")
    ax.set_ylabel("cumulative incremental gain")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    return ax.figure

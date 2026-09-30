"""SHAP values and plots for CATE predictions.

The default explainer is an interventional permutation SHAP (Štrumbelj and
Kononenko). It does not require the ``shap`` package. When ``shap`` is
installed, ``method='shap'`` uses ``shap.PermutationExplainer``.

This is the Python counterpart of ``RCausalML::explain_cate`` in
``R/shapviz_integration.R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ._common import as_float_matrix

__all__ = [
    "ShapExplanation",
    "explain_cate",
    "plot_cate_hist",
    "plot_shap_dependence",
    "plot_shap_importance",
    "plot_shap_waterfall",
    "predict_cate_values",
]


def predict_cate_values(model, X) -> np.ndarray:
    """Numeric CATE vector from ``predict_cate``, ``predict``, or a dict."""
    X_arr, _ = as_float_matrix(X)
    if hasattr(model, "predict_cate"):
        out = model.predict_cate(X_arr)
    else:
        out = model.predict(X_arr)
    if isinstance(out, dict):
        if "predictions" not in out:
            raise TypeError("Dict predictions must contain a 'predictions' entry.")
        out = out["predictions"]
    values = np.asarray(out, dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    return values.ravel()


def _native_shap(predict, X, background, n_permutations, seed) -> tuple:
    rng = np.random.default_rng(seed)
    bg = np.median(np.asarray(background, dtype=float), axis=0)
    n, p = X.shape
    current = np.repeat(bg.reshape(1, -1), n, axis=0)
    base = np.asarray(predict(current), dtype=float).ravel()
    values = np.zeros((n, p), dtype=float)
    for _ in range(n_permutations):
        order = rng.permutation(p)
        current = np.repeat(bg.reshape(1, -1), n, axis=0)
        prev = base.copy()
        for j in order:
            current[:, j] = X[:, j]
            cur = np.asarray(predict(current), dtype=float).ravel()
            values[:, j] += cur - prev
            prev = cur
    values /= float(n_permutations)
    # Finite-permutation correction so attributions add up to the prediction.
    full = np.asarray(predict(X), dtype=float).ravel()
    residual = full - (base + values.sum(axis=1))
    values += residual[:, None] / p
    return values, base, full


def _library_shap(predict, X, background, n_permutations, seed):
    import shap

    explainer = shap.PermutationExplainer(predict, background, seed=seed)
    raw = explainer(X, max_evals=max(2 * X.shape[1] + 1, n_permutations * X.shape[1]))
    values = np.asarray(raw.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    base = np.asarray(raw.base_values, dtype=float).ravel()
    if base.size == 1:
        base = np.full(len(X), float(base[0]))
    full = np.asarray(predict(X), dtype=float).ravel()
    return values, base, full


@dataclass
class ShapExplanation:
    """CATE SHAP values for a set of rows.

    Attributes
    ----------
    values : ndarray of shape (n, p)
        Feature attributions.
    base_values : ndarray of shape (n,)
        Expected prediction when features are at the background.
    data : ndarray of shape (n, p)
        Rows that were explained.
    feature_names : list of str
    predictions : ndarray of shape (n,)
        Model CATE on ``data``.
    """

    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    feature_names: list
    predictions: np.ndarray

    def importance(self) -> np.ndarray:
        """Mean absolute SHAP value of each feature."""
        return np.mean(np.abs(self.values), axis=0)

    def plot_importance(self, max_display: int = 15, ax=None, title: Optional[str] = None):
        return plot_shap_importance(self, max_display=max_display, ax=ax, title=title)

    def plot_dependence(self, feature=0, ax=None, title: Optional[str] = None):
        return plot_shap_dependence(self, feature, ax=ax, title=title)

    def plot_waterfall(self, row: int = 0, ax=None, title: Optional[str] = None):
        return plot_shap_waterfall(self, row=row, ax=ax, title=title)


def explain_cate(
    model,
    X,
    background=None,
    n_permutations: int = 12,
    max_samples: Optional[int] = 48,
    feature_names: Optional[Sequence[str]] = None,
    method: str = "native",
    random_state: int = 0,
):
    """SHAP values for a fitted CATE model.

    Parameters
    ----------
    model
        Any object with ``predict`` or ``predict_cate`` returning a numeric
        vector. PyDeepCausalML effect estimators are accepted.
    X
        Rows to explain.
    background
        Reference sample. Defaults to ``X``.
    method
        ``"native"`` (default) or ``"shap"`` (the ``shap`` package).
    """
    X_arr, names = as_float_matrix(X, feature_names)
    if max_samples is not None and len(X_arr) > max_samples:
        rng = np.random.default_rng(random_state)
        take = np.sort(rng.choice(len(X_arr), size=max_samples, replace=False))
        X_arr = X_arr[take]
    if background is None:
        background = X_arr
    else:
        background, _ = as_float_matrix(background)
    if len(background) > 80:
        rng = np.random.default_rng(random_state)
        background = background[rng.choice(len(background), size=80, replace=False)]

    def predict(batch):
        return predict_cate_values(model, batch)

    if method == "shap":
        values, base, full = _library_shap(predict, X_arr, background, n_permutations, random_state)
    elif method == "native":
        values, base, full = _native_shap(predict, X_arr, background, n_permutations, random_state)
    else:
        raise ValueError("method must be 'native' or 'shap'.")
    return ShapExplanation(values, base, X_arr, names, full)


def _mpl():
    import matplotlib.pyplot as plt

    return plt


def plot_shap_importance(explanation: ShapExplanation, max_display: int = 15, ax=None, title=None):
    """Horizontal bar chart of mean absolute SHAP values."""
    plt = _mpl()
    imp = explanation.importance()
    order = np.argsort(imp)[-max_display:]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 0.38 * len(order) + 1.4))
    labels = np.asarray(explanation.feature_names, dtype=object)[order]
    ax.barh(labels, imp[order], color="#1F4E79")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title or "SHAP importance for the CATE")
    ax.figure.tight_layout()
    return ax.figure


def plot_shap_dependence(explanation: ShapExplanation, feature=0, ax=None, title=None):
    """SHAP dependence plot for one feature."""
    plt = _mpl()
    if isinstance(feature, str):
        index = explanation.feature_names.index(feature)
    else:
        index = int(feature)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    x = explanation.data[:, index]
    y = explanation.values[:, index]
    ax.scatter(x, y, s=18, c="#C0392B", alpha=0.85)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xlabel(explanation.feature_names[index])
    ax.set_ylabel("SHAP value")
    ax.set_title(title or f"SHAP dependence: {explanation.feature_names[index]}")
    ax.figure.tight_layout()
    return ax.figure


def plot_shap_waterfall(explanation: ShapExplanation, row: int = 0, ax=None, title=None):
    """Single-row waterfall of SHAP contributions."""
    plt = _mpl()
    phi = explanation.values[row]
    order = np.argsort(np.abs(phi))
    labels = [explanation.feature_names[j] for j in order]
    contrib = phi[order]
    base = float(explanation.base_values[row])
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 0.38 * len(order) + 1.6))
    cumulative = base
    for i, value in enumerate(contrib):
        color = "#1E8449" if value >= 0 else "#A93226"
        ax.barh(i, value, left=cumulative, color=color)
        cumulative += value
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(base, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("CATE contribution")
    pred = float(explanation.predictions[row])
    ax.set_title(title or f"SHAP waterfall (row {row}, CATE {pred:.3f})")
    ax.figure.tight_layout()
    return ax.figure


def plot_cate_hist(cate, ax=None, title: str = "Estimated CATE"):
    """Histogram of conditional average treatment effects."""
    plt = _mpl()
    cate = np.asarray(cate, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(cate, bins=20, color="#2E86AB", edgecolor="white")
    ax.axvline(np.mean(cate), color="#C0392B", lw=1.2, label=f"mean {np.mean(cate):.2f}")
    ax.set_xlabel("CATE")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    return ax.figure

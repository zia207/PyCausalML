"""Synthetic designs used by the classical causal modules.

``synthetic_data`` follows the heterogeneous-treatment design in
``grf::synthetic_data`` and ``causal_tree_methods.synthetic_data``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_binary_uplift",
    "make_iv_data",
    "make_linear_treatment",
    "make_multiarm_data",
    "make_survival_data",
    "synthetic_data",
]


def synthetic_data(n=2000, p=10, sigma=1.0, mode=2, random_state=None):
    """Return ``X``, ``y``, ``w``, and the true CATE ``tau``.

    Mode 1 is a zero effect. Mode 2 is ``1{X1 > 0} * (1 + X2)``. Mode 3 is
    ``X1 + log(1 + exp(X2))``. Columns are 0-indexed in the returned array;
    the formulas above use the grf names ``X1`` and ``X2``.
    """
    if p < 3:
        raise ValueError("synthetic_data requires p >= 3")
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    if mode == 1:
        tau = np.zeros(n)
    elif mode == 2:
        tau = (X[:, 0] > 0).astype(float) * (1.0 + X[:, 1])
    elif mode == 3:
        tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))
    else:
        raise ValueError("mode must be 1, 2, or 3")
    propensity = 1.0 / (1.0 + np.exp(-0.5 * X[:, 2]))
    w = rng.binomial(1, propensity)
    y = np.maximum(X[:, 0], 0.0) + 0.5 * X[:, 1] + w * tau + sigma * rng.normal(size=n)
    return {"X": X, "y": y, "w": w.astype(int), "tau": tau}


def make_linear_treatment(n=800, p=4, tau=2.0, sigma=1.0, random_state=0):
    """Partially linear model with a constant treatment effect."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    propensity = 1.0 / (1.0 + np.exp(-0.4 * X[:, 0]))
    w = rng.binomial(1, propensity)
    beta = np.linspace(0.4, -0.2, p)
    y = X @ beta + tau * w + sigma * rng.normal(size=n)
    return {"X": X, "y": y, "w": w.astype(int), "tau": np.full(n, tau)}


def make_binary_uplift(n=800, p=4, random_state=0):
    """Binary outcome whose uplift is larger when the first covariate is positive."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    tau = np.where(X[:, 0] > 0, 0.35, 0.02)
    baseline = 1.0 / (1.0 + np.exp(-(0.2 * X[:, 1])))
    propensity = 1.0 / (1.0 + np.exp(-0.3 * X[:, 2]))
    w = rng.binomial(1, propensity)
    prob = np.clip(baseline + w * tau, 0.02, 0.98)
    y = rng.binomial(1, prob)
    return {"X": X, "y": y.astype(float), "w": w.astype(int), "tau": tau}


def make_iv_data(n=900, random_state=0):
    """Linear IV design. The structural effect of ``W`` on ``Y`` is 1.5."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, 3))
    Z = rng.binomial(1, 0.5, size=n).astype(float)
    U = rng.normal(size=n)
    W = 0.9 * Z + 0.6 * U + 0.2 * X[:, 0] + 0.1 * rng.normal(size=n)
    tau = np.full(n, 1.5)
    Y = tau * W + 0.8 * U + 0.3 * X[:, 0] + 0.2 * rng.normal(size=n)
    return {"X": X, "Y": Y, "W": W, "Z": Z, "tau": tau}


def make_multiarm_data(n=700, random_state=0):
    """Three arms. Contrasts are arm 1 and arm 2 against control 0."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, 3))
    logits = np.column_stack([
        np.zeros(n),
        0.2 * X[:, 0],
        -0.2 * X[:, 0],
    ])
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    w = np.array([rng.choice(3, p=prob[i]) for i in range(n)])
    tau1 = 1.0 + 0.5 * X[:, 1]
    tau2 = -0.4 + 0.2 * X[:, 0]
    y = 0.3 * X[:, 0] + np.where(w == 1, tau1, 0.0) + np.where(w == 2, tau2, 0.0)
    y = y + rng.normal(scale=0.5, size=n)
    return {"X": X, "y": y, "w": w.astype(int), "tau1": tau1, "tau2": tau2}


def make_survival_data(n=500, horizon=4.0, random_state=0):
    """Exponential survival times. Treatment lengthens the event time."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, 3))
    w = rng.binomial(1, 0.5, size=n)
    scale = np.exp(0.15 * X[:, 0]) * np.where(w == 1, 1.8, 1.0)
    time = rng.exponential(scale)
    censor = rng.exponential(10.0, size=n)
    event = (time <= censor).astype(int)
    observed = np.minimum(time, censor)
    return {
        "X": X,
        "time": observed,
        "event": event,
        "w": w.astype(int),
        "horizon": horizon,
    }

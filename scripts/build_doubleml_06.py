#!/usr/bin/env python3
"""Notebook 4.6."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_doubleml_notebooks import BANNER, C, IMPORTS, M, SETUP, write


def nb06():
    write(
        "02_08_05_04_06_double_ml_weighted_double_ml.ipynb",
        [
            M(BANNER),
            M("""
# 4.6 Weighted Double Machine Learning

Weighted DML puts a weight $\\omega$ into the orthogonal score so the target is not the average effect in the whole sample. $\\omega = 1$ is the ordinary ATE. $\\omega = 1\\{G = k\\}$ is the effect in group $k$. Inverse-propensity weights target the effect on the treated.

The same idea shows up when raw rows have been collapsed to segments with frequency weights. Fitting on the raw sample is the expanded version of that table. Multiplying the doubly robust scores by $\\omega$ answers the weighted estimand directly. This notebook generates a binary design with a known segment effect, fits `LinearDML`, and reweights the scores from `InteractiveRegressionDML`.
"""),
            M("""
## Overview

For a binary treatment the doubly robust score $\\psi_i$ has mean equal to the ATE. A weighted estimand is

$$\\hat\\theta_\\omega = \\frac{\\sum_i \\omega_i \\psi_i}{\\sum_i \\omega_i}.$$

Cross-fitting does not change. The weight only changes which rows the final average emphasizes.

| Weight | Estimand |
|--------|----------|
| $\\omega = 1$ | ATE |
| $\\omega = 1\\{A = 1\\}$ | Effect in the $A = 1$ segment |
| $\\omega = 1\\{A = 0\\}$ | Effect in the $A = 0$ segment |
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS),
            M("""
### Generating the sample

Binary features. Column $A$ shifts the propensity and sets the true effect $\\theta = -1 + 2A$: the effect is $-1$ when $A = 0$ and $+1$ when $A = 1$. The other columns are controls. Errors are heteroskedastic in $A$.
"""),
            C('''
rng = np.random.default_rng(123)
n, p = 6000, 10
X_raw = rng.binomial(1, 0.5, size=(n, p)).astype(float)
propensity = 1.0 / (1.0 + np.exp(-(X_raw[:, 1] - 0.5)))
treatment = rng.binomial(1, propensity).astype(float)
theta = -1.0 + 2.0 * X_raw[:, 1]
noise = rng.normal(scale=1.0 + X_raw[:, 1], size=n)
outcome = theta * treatment + X_raw[:, 1] + noise
names = ["A", "B", "C", "D"] + [f"W{j}" for j in range(1, 6)]
X = X_raw[:, 1:]
print("True effect when A=1: 1 ; when A=0: -1")
pd.DataFrame(X, columns=names).head()
'''),
            M("""
## Applying LinearDML

`LinearDML` is fit on the raw rows. That is the frequency-weighted segment table written out one row per observation.
"""),
            C('''
est = LinearDML(discrete_treatment=True, n_splits=3, model_y="rf", n_estimators=40, random_state=123)
est.fit(X, treatment, outcome, feature_names=names)
print(est.average_treatment_effect())
'''),
            M("""
### Treatment effect in one segment

The segment $A=1, B=0, C=0, D=0$ has true effect $+1$. Controls are set to their sample means. `confidence_interval` is the interval for the average effect, not a pointwise band for that one row.
"""),
            C('''
means = X.mean(axis=0)
segment = means.copy()
segment[:4] = np.array([1.0, 0.0, 0.0, 0.0])
print("CATE at A=1, B=C=D=0:", round(float(est.predict(segment.reshape(1, -1))[0]), 4))
print("ATE interval:", est.confidence_interval())
'''),
            M("### Coefficients\n"),
            C('''
print("Intercept:", round(float(est.intercept_), 4))
pd.DataFrame({"feature": names, "coef": est.feature_coef_}).round(4)
'''),
            M("""
## Weighted scores

`InteractiveRegressionDML` stores the doubly robust score on `influence_`. The ATE is its mean. The segment effects are the means inside $A=1$ and $A=0$.
"""),
            C('''
irm = InteractiveRegressionDML(n_splits=3, model_y="rf", n_estimators=40, random_state=123)
irm.fit(X, treatment, outcome, feature_names=names)
psi = irm.influence_
print("Doubly robust ATE:", round(float(np.mean(psi)), 4))
print("A=1 segment:", round(float(np.mean(psi[X[:, 0] == 1])), 4), "(true 1)")
print("A=0 segment:", round(float(np.mean(psi[X[:, 0] == 0])), 4), "(true -1)")
'''),
            M("""
## Polynomial CATE

`SparseLinearDML` on a degree-2 expansion of $A,B,C,D$. The true effect is linear in $A$, so the extra terms should stay small, and the segment prediction should stay near $+1$.
"""),
            C('''
poly = PolynomialFeatures(degree=2, include_bias=False)
modifiers = poly.fit_transform(X[:, :4])
poly_names = [f"p{j}" for j in range(modifiers.shape[1])] + names[4:]
X_poly = np.column_stack([modifiers, X[:, 4:]])
sparse = SparseLinearDML(
    discrete_treatment=True, n_splits=2, model_y="rf", n_estimators=30, alpha=0.01, random_state=7,
)
sparse.fit(X_poly, treatment, outcome, feature_names=poly_names)
seg_row = np.column_stack([poly.transform(np.array([[1.0, 0.0, 0.0, 0.0]])), means[4:].reshape(1, -1)])
print(sparse.average_treatment_effect())
print("Polynomial CATE at the segment:", round(float(sparse.predict(seg_row)[0]), 4))
print("Intercept", round(float(sparse.intercept_), 4))
'''),
            M("""
## Random-forest final stage

`CausalForestDML` lets the CATE be a step function. The two modes should sit near $-1$ and $+1$.
"""),
            C('''
forest = CausalForestDML(num_trees=60, n_splits=2, random_state=9)
forest.fit(X, treatment, outcome, feature_names=names)
cate = forest.predict(X)
print(forest.average_treatment_effect())
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.hist(cate[X[:, 0] == 0], bins=20, alpha=0.55, label="A = 0 (true -1)", color="#1F4E79")
ax.hist(cate[X[:, 0] == 1], bins=20, alpha=0.55, label="A = 1 (true +1)", color="#C0392B")
ax.set_xlabel("Predicted CATE")
ax.set_ylabel("Count")
ax.set_title("CausalForestDML by segment")
ax.legend(frameon=False)
fig.tight_layout()
'''),
            M("""
## Summary

The orthogonal score, averaged with a weight, is the weighted estimand. On this design $1\\{A=1\\}$ recovers $+1$ and $1\\{A=0\\}$ recovers $-1$. `LinearDML` reports the same structure as coefficients. `CausalForestDML` recovers it as two clumps in the CATE distribution.

## Resources

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- PyCausalML: `LinearDML`, `SparseLinearDML`, `InteractiveRegressionDML`, `CausalForestDML`.
"""),
        ],
    )


if __name__ == "__main__":
    nb06()

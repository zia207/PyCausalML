#!/usr/bin/env python3
"""Write the Double / debiased machine learning tutorial notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Tutorials" / "05_04_DoubleML"

BANNER = "![Python Banner](../../Image/05_04_doubleML.png)\n"

SETUP = '''
import subprocess
import sys
from pathlib import Path

def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "src" / "pycausalml").is_dir():
            return cand
    raise FileNotFoundError(
        "PyCausalML repository was not found. Run this notebook from the repository."
    )

root = _repo_root()
data_dir = root / "Data"
if not data_dir.is_dir():
    raise FileNotFoundError(
        "The Data directory was not found. It should sit next to pyproject.toml."
    )

try:
    import pycausalml  # noqa: F401
    print("PyCausalML ready.")
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{root}[plot]"],
    )
    print("PyCausalML installed.")
print("Data directory:", data_dir)
'''.strip() + "\n"


def M(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def C(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def write(name: str, cells) -> None:
    path = OUT / name
    nbf.write(notebook(cells), path)
    print("wrote", path.name, "cells", len(cells))


def nb00():
    write(
        "02_08_05_04_00_double_ml_introduction.ipynb",
        [
            M(BANNER),
            M("""
# 4 Double/Debiased Machine Learning

Two powerful methodologies have historically talked past each other:

**Machine learning** predicts outcomes with remarkable accuracy, but its regularization techniques — shrinking or zeroing out coefficients to prevent overfitting — introduce *systematic bias* into any coefficient it produces. You can't use a Lasso regression to estimate a causal effect and trust the result.

**Classical econometrics** (OLS, IV, etc.) produces unbiased, inferentially valid estimates, but breaks down under high dimensionality. With hundreds or thousands of control variables, standard regression is either infeasible or hopelessly overfit.

**Double Machine Learning** solves both problems simultaneously. It uses ML's predictive power to handle high-dimensional confounders, then recovers valid causal inference through a clever two-step residualization procedure.
"""),
            M("""
## How it works

### The model

DML assumes a *partially linear* data-generating process:

$$
\\begin{align}
Y &= \\theta D + g(X) + \\varepsilon \\\\
D &= m(X) + V
\\end{align}
$$

where $Y$ is the outcome, $D$ is the treatment, $X$ is a (potentially very large) vector of confounders, and $\\theta$ is the causal effect you want. The functions $g(X)$ and $m(X)$ are called *nuisance functions* — they govern how confounders affect the outcome and treatment, but you don't care about them directly.
"""),
            M("""
### The five-step pipeline

**Step 1 — Residualize the outcome.** Train any ML model to predict $Y$ from $X$. Compute the residual $\\tilde{Y} = Y - \\hat{g}(X)$: the part of $Y$ that confounders cannot explain.

**Step 2 — Residualize the treatment.** Train a second ML model to predict $D$ from $X$. Compute $\\tilde{D} = D - \\hat{m}(X)$: the part of $D$ that confounders cannot explain.

**Step 3 — Cross-fit to prevent overfitting.** If you train the nuisance models and compute residuals on the *same* data, the residuals will be artificially small (overfitting). Instead, split the data into $K$ folds, train on $K-1$ folds, and predict residuals on the held-out fold. Rotate through all folds and aggregate.

**Step 4 — Final OLS.** Regress $\\tilde{Y}$ on $\\tilde{D}$: $$\\tilde{Y} = \\theta\\,\\tilde{D} + \\varepsilon$$

The coefficient $\\hat{\\theta}$ is your causal estimate. By construction, you've *partialled out* all confounding variation captured in $X$.

**Step 5 — Inference.** Because of the orthogonality property (below), $\\hat{\\theta}$ is $\\sqrt{n}$-consistent and asymptotically normal, giving you valid standard errors, confidence intervals, and p-values.

![Double ML flowchart](../../Image/doubl_ML_flowchart.png)
"""),
            M("""
### Why this works: Neyman orthogonality

The critical insight is that the score function used to estimate $\\theta$ is *orthogonal* to the nuisance parameters. In plain terms: if your ML models for $g(X)$ and $m(X)$ are slightly wrong — which they always are — those errors have only a *second-order* (negligible) impact on $\\hat{\\theta}$. This is the "debiasing" step that makes valid inference possible despite using regularized ML.
"""),
            M("""
## Supported models and causal structures

PyCausalML is *learner-agnostic* for the nuisance stages — you can plug in virtually any supervised learning algorithm:

| Category         | Examples                                   |
|------------------|--------------------------------------------|
| Penalized linear | Lasso, Ridge, Elastic Net                  |
| Tree-based       | Random Forest, gradient boosting, XGBoost  |
| Neural networks  | MLPs via scikit-learn                      |
| Other            | SVM, k-NN                                  |

The notebooks in this chapter use these structural models:

| Model | When to use | PyCausalML estimator |
|------------------------------------|------------------------------------|------------------------------------|
| **PLR** — Partially Linear Regression | Standard observational setting; a constant effect | `PartiallyLinearDML` |
| **Linear / sparse CATE** | The effect varies linearly (or sparsely) with covariates | `LinearDML`, `SparseLinearDML` |
| **IRM** — Interactive Regression Model | Binary treatment; doubly robust score for the ATE and a flexible CATE | `InteractiveRegressionDML` |
| **PLIV** — Partially Linear IV | Endogenous treatment; a valid instrument | Cross-fitted residual-on-residual IV, plus `InstrumentalVariable` |
| **Nonparametric CATE** | A flexible final stage on the orthogonal pseudo-outcome | `NonParamDML`, `CausalForestDML`, `KernelDML` |
"""),
            M("""
## Advantages

| Capability | What it enables |
|------------------------------------|------------------------------------|
| High dimensionality | Works when the number of controls $p \\gg n$ |
| Nonlinearity | Captures complex confounder–outcome relationships OLS cannot |
| Valid inference | Produces standard errors, CIs, and hypothesis tests |
| Robustness | Resilient to moderate ML estimation error (orthogonality) |
| Modularity | Swap learners freely; gradient boosting today, random forests tomorrow |
"""),
            M("""
## Limitations

**No hidden confounders.** Like all observational methods, standard DML (PLR) assumes *unconfoundedness*: every variable affecting both $D$ and $Y$ is in $X$. An unmeasured confounder still biases $\\hat{\\theta}$. The PLIV variant handles endogeneity, but requires valid instruments.

**Sample size still matters.** DML handles high *dimensionality* but not small *n*. The final regression step requires enough observations for asymptotic normality to hold.

**Nuisance functions remain black boxes.** $\\hat{\\theta}$ is interpretable; $\\hat{g}(X)$ and $\\hat{m}(X)$ are not. You learn the effect of treatment but not *how* each confounder influences the outcome.

**Computational cost.** Cross-fitting means training $K$ nuisance models per variable instead of one. Expect runtime roughly $K\\times$ that of a comparable OLS workflow.

**Weak instruments.** If you use PLIV, DML inherits the weak-instrument problem from classical 2SLS — a weak instrument inflates standard errors and can bias estimates.
"""),
            M("""
## Python ecosystem

### Primary package: PyCausalML

The estimators in this chapter live in `pycausalml.double_machine_learning`.

- Partially linear and interactive models: `PartiallyLinearDML`, `InteractiveRegressionDML`
- Heterogeneous effects: `LinearDML`, `SparseLinearDML`, `NonParamDML`, `CausalForestDML`, `KernelDML`
- Cross-fitted first stages for every estimator
- Influence-function standard errors and confidence intervals on the average effect
- SHAP explanations of a fitted CATE through `explain` and `pycausalml.interpretation`

Nuisance learners are scikit-learn models (random forests, gradient boosting, ridge, lasso). Later notebooks also show pipelines, averaged ensembles, and a small hyperparameter grid around the same orthogonal score.

### A related estimator: causal forests

If the question is *who* benefits from treatment rather than a single average effect, `CausalForest` (honest trees) and `CausalForestDML` (a forest on the doubly robust pseudo-outcome) are the natural next tools. They use a related but distinct splitting criterion and are built for effect heterogeneity.
"""),
            M("""
## Applications

DML is the natural choice wherever causal questions meet high-dimensional data:

- **Economics** — price elasticity controlling for thousands of product and market characteristics
- **Marketing** — true ad-campaign lift after controlling for browsing history and demographics
- **Healthcare** — drug efficacy controlling for genomic or electronic health record data
- **Policy evaluation** — job-training program impact controlling for employment history, education, and local conditions

**Double Machine Learning** is a state-of-the-art method for estimating causal effects when confounders are high-dimensional. It lets you use the predictive power of random forests, gradient boosting, or penalized regression to control for confounders, without giving up the standard errors required to make a causal claim. PyCausalML implements that workflow for partially linear, interactive, instrumental-variable, panel, and clustered designs. Like any observational method, it still relies on the design assumptions of the model you choose.
"""),
            M("""
## Resources

**Key References**

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal, 21(1), C1–C68.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2020). *DoubleML – Double/debiased machine learning in R and Python*. Journal of Statistical Software.

**Documentation**

- [DoubleML user guide](https://docs.doubleml.org/stable/guide/index.html) — the method, independent of this package
- PyCausalML `double_machine_learning` module: `PartiallyLinearDML`, `LinearDML`, `SparseLinearDML`, `NonParamDML`, `CausalForestDML`, `KernelDML`, `InteractiveRegressionDML`
"""),
        ],
    )


IMPORTS = '''
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pycausalml import (
    CausalForestDML,
    InteractiveRegressionDML,
    KernelDML,
    LinearDML,
    NonParamDML,
    PartiallyLinearDML,
    SparseLinearDML,
    explain_cate,
)
from pycausalml.interpretation import plot_cate_hist

plt.rcParams["figure.dpi"] = 110
print("Imported PyCausalML")
'''.strip() + "\n"


def _hist_cell():
    return C('''
def plot_t_hist(values, title, color, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(values, bins=30, density=True, alpha=0.35, color=color, edgecolor=color, label=label)
    ax.plot(grid, stats.norm.pdf(grid), color="black", lw=1.2, label="N(0,1)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("Rescaled estimate")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
''')


def nb01():
    write(
        "02_08_05_04_01_double_ml_from_basics_to_application.ipynb",
        [
            M(BANNER),
            M("""
# 4.1. Double ML: Bias, Orthogonalization

**Double/debiased machine learning (DML)** provides valid causal inference when flexible ML methods are used to control for high-dimensional confounding. Standard ML estimators are ill-suited to causal questions: regularization shrinks coefficients toward zero (introducing bias in $\\hat{\\theta}$), and fitting nuisance functions on the same data used for causal estimation causes overfitting bias. DML corrects both problems by combining **Neyman-orthogonal scores**, **sample splitting**, and **cross-fitting**, yielding $\\sqrt{n}$-consistent, asymptotically normal estimates of causal parameters.

This notebook is organized in two parts:

- **Part A — Synthetic data:** builds intuition by simulating the partially linear model and comparing three estimators side-by-side — naive ML (biased), orthogonalized ML without sample splitting (overfitting bias), and full DML with cross-fitting (valid inference).
- **Part B — Real-world application:** applies DML to the 401(k) eligibility dataset in `Data/data_401k.csv`, covering a train–test split, causal estimation with `PartiallyLinearDML`, holdout validation of nuisance predictions, and SHAP interpretation of heterogeneous treatment effects with `LinearDML.explain`.
"""),
            M("""
## Overview

ML models predict well but produce biased causal estimates because regularization (for example, a shallow random forest or a penalized regression) distorts the coefficients we care about. The table below summarizes the three challenges DML addresses and how it addresses each one.

| Challenge | Why it matters | DML's response |
|-----------|---------------|----------------|
| Regularization bias | ML shrinks nuisance fits, which contaminates $\\hat{\\theta}$ | Neyman-orthogonal scores (partialling-out or IV-type) make $\\hat{\\theta}$ insensitive to first-order nuisance errors |
| Overfitting bias | Using the same data for ML nuisances and for estimating $\\theta$ inflates apparent fit | Sample splitting: train nuisances on one fold, estimate $\\theta$ on another |
| High-dimensional confounders | Classical OLS breaks down when $p \\gg n$ or relationships are non-linear | Flexible learners (random forests, gradient boosting, lasso) estimate $g_0(X)$ and $m_0(X)$ |

Cross-fitting combines both halves of the sample split by rotating roles across $K$ folds, so no observations are wasted.
"""),
            M("""
## The Partially Linear Regression (PLR) Model

Both parts of this notebook use the **partially linear regression (PLR)** model, which separates the linear treatment effect $\\theta_0$ from the nonparametric confounding functions $g_0$ and $m_0$:

$$Y = D\\,\\theta_0 + g_0(X) + \\zeta, \\quad \\mathbb{E}(\\zeta \\mid D, X) = 0$$

$$D = m_0(X) + V, \\quad \\mathbb{E}(V \\mid X) = 0$$

| Symbol | Role |
|--------|------|
| $Y$ | Outcome variable |
| $D$ | Treatment variable |
| $X$ | High-dimensional confounders |
| $\\theta_0$ | Causal parameter (ATE) |
| $g_0(X),\\, m_0(X)$ | Nuisance functions estimated by ML |
| $\\zeta,\\, V$ | Stochastic errors |
"""),
            M("""
## Implementation in Python

Throughout this notebook the causal estimators come from **PyCausalML**. Part A writes the three scores explicitly — non-orthogonal, IV-type, and partialling-out — so the source of the bias is visible. The cross-fitted partialling-out estimator is the same construction as `PartiallyLinearDML`. Part B calls `PartiallyLinearDML` and `LinearDML` directly on `Data/data_401k.csv`.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS),
            M("""
## Part A: Diagnosing Bias with Synthetic Data

Part A uses Monte Carlo simulation to make the bias problems visible. We draw replications of the partially linear model from Chernozhukov et al. (2018) and compare the distribution of the rescaled estimator $(\\hat{\\theta} - \\theta_0) / \\hat{\\text{se}}$ against $\\mathcal{N}(0,1)$ under three approaches. A valid estimator produces a histogram that matches the standard normal; deviations expose regularization or overfitting bias.

The data-generating process is:

$$y_i = \\theta_0\\, d_i + g_0(x_i) + \\zeta_i, \\quad \\zeta_i \\sim \\mathcal{N}(0,1)$$

$$d_i = m_0(x_i) + v_i, \\quad v_i \\sim \\mathcal{N}(0,1)$$

Covariates $x_i \\sim \\mathcal{N}(0, \\Sigma)$ with $\\Sigma_{kj} = 0.7^{|j-k|}$. The true causal parameter is $\\theta_0 = 0.5$. Nuisance functions:

$$m_0(x_i) = x_{i,1} + \\frac{1}{4}\\frac{\\exp(x_{i,3})}{1 + \\exp(x_{i,3})}$$

$$g_0(x_i) = \\frac{\\exp(x_{i,1})}{1 + \\exp(x_{i,1})} + \\frac{1}{4}x_{i,3}$$

Each replication draws $n = 500$ observations with $p = 20$ covariates. The extra coordinates are pure noise, which is what makes a regularized forest shrink $g_0$. The Monte Carlo uses 30 replications so the notebook finishes in one pass; increase `n_rep` for a tighter histogram.

### A.1 Generate Simulated Data
"""),
            C('''
def make_plr_ccddhnr2018(n_obs, dim_x, alpha, rng):
    """Partially linear DGP from Chernozhukov et al. (2018)."""
    index = np.arange(dim_x)
    cov = 0.7 ** np.abs(index[:, None] - index[None, :])
    x = rng.multivariate_normal(np.zeros(dim_x), cov, size=n_obs)
    m0 = x[:, 0] + 0.25 / (1.0 + np.exp(-x[:, 2]))
    g0 = 1.0 / (1.0 + np.exp(-x[:, 0])) + 0.25 * x[:, 2]
    d = m0 + rng.normal(size=n_obs)
    y = alpha * d + g0 + rng.normal(size=n_obs)
    columns = [f"X{j + 1}" for j in range(dim_x)]
    frame = pd.DataFrame(x, columns=columns)
    frame["d"] = d
    frame["y"] = y
    return frame


def forest(seed):
    return RandomForestRegressor(
        n_estimators=40,
        max_depth=4,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )


def _predict_nuisance(X, target, n_splits, seed, cross_fit):
    if not cross_fit:
        model = forest(seed)
        model.fit(X, target)
        return model.predict(X)
    pred = np.zeros(len(target))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(X)):
        model = forest(seed + fold)
        model.fit(X[tr], target[tr])
        pred[te] = model.predict(X[te])
    return pred


def plr_estimate(X, d, y, score, cross_fit, seed, n_splits=2):
    """Return (theta, se) for one PLR score.

    score='nonorth' uses only the outcome nuisance and a single sample split.
    score='iv' residualizes Y with g(X) and instruments D with D - m(X).
    score='partial' residualizes both Y and D (the PartiallyLinearDML score).
    """
    if score == "nonorth":
        splitter = KFold(n_splits=2, shuffle=True, random_state=seed)
        tr, te = next(splitter.split(X))
        model = forest(seed)
        model.fit(X[tr], y[tr])
        g_hat = model.predict(X[te])
        d_te, y_te = d[te], y[te]
        theta = float(np.dot(d_te, y_te - g_hat) / np.dot(d_te, d_te))
        resid = d_te * (y_te - g_hat - theta * d_te)
        se = float(np.std(resid, ddof=1) / (np.sqrt(len(te)) * np.mean(d_te ** 2)))
        return theta, se

    g_hat = _predict_nuisance(X, y, n_splits, seed, cross_fit)
    m_hat = _predict_nuisance(X, d, n_splits, seed + 17, cross_fit)
    if score == "iv":
        v_hat = d - m_hat
        u_hat = y - g_hat
        denom = float(np.dot(v_hat, d))
        theta = float(np.dot(v_hat, u_hat) / denom)
        psi = v_hat * (u_hat - theta * d) / np.mean(v_hat * d)
    elif score == "partial":
        y_res = y - g_hat
        d_res = d - m_hat
        denom = float(np.dot(d_res, d_res))
        theta = float(np.dot(d_res, y_res) / denom)
        psi = d_res * (y_res - theta * d_res) / np.mean(d_res ** 2)
    else:
        raise ValueError(score)
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return theta, se


def replicate(datasets, score, cross_fit, seed):
    theta = np.empty(len(datasets))
    se = np.empty(len(datasets))
    for i, frame in enumerate(datasets):
        x_cols = [c for c in frame.columns if c.startswith("X")]
        theta[i], se[i] = plr_estimate(
            frame[x_cols].to_numpy(),
            frame["d"].to_numpy(),
            frame["y"].to_numpy(),
            score=score,
            cross_fit=cross_fit,
            seed=seed + i,
        )
    return theta, se


rng = np.random.default_rng(1234)
n_rep = 30
n_obs = 500
n_vars = 20
alpha = 0.5
datasets = [make_plr_ccddhnr2018(n_obs, n_vars, alpha, rng) for _ in range(n_rep)]
datasets[0].head()
'''),
            M("""
## A.2 Define Shared Learners

Both nuisance functions ($g_0$ and $m_0$) are estimated with a random forest (40 trees, max depth 4). The same specification is rebuilt for every replication so the hyperparameters stay fixed. `PartiallyLinearDML` uses the same forest family through `model_y="rf"`.
"""),
            C('''
print(forest(0))
'''),
            M("""
## A.3 Approach 1 — Naive ML (Non-Orthogonal Score)

A **naive** estimator splits the sample in two, fits $\\hat{g}_0(X)$ with ML on one half, then uses the other half to compute:

$$\\hat{\\theta}_0 = \\Bigl(\\tfrac{1}{n}\\sum_{i \\in I} D_i^2\\Bigr)^{-1}
  \\tfrac{1}{n}\\sum_{i \\in I} D_i\\bigl(Y_i - \\hat{g}_0(X_i)\\bigr)$$

This uses a **non-orthogonal score**. Regularization bias from estimating $g_0$ with ML does not vanish; $\\sqrt{n}(\\hat{\\theta}_0 - \\theta_0)$ does not stay bounded.

### A.3.1 Define Non-Orthogonal Score

The score below uses only $\\hat{g}(X)$ to residualize $Y$, ignoring $D$'s dependence on $X$. The moment condition is therefore not orthogonal.

### A.3.2 Fit Naive ML Over Replications
"""),
            C('''
theta_nonorth, se_nonorth = replicate(datasets, score="nonorth", cross_fit=False, seed=1111)
print("Mean theta:", round(float(np.mean(theta_nonorth)), 4))
'''),
            M("""
### A.3.3 Plot Distribution — Non-Orthogonal

Under valid inference, $(\\hat{\\theta} - \\theta_0) / \\hat{\\text{se}}$ should follow $\\mathcal{N}(0,1)$. For the naive estimator it does not: the histogram is shifted and spread, revealing **regularization bias** from the forest's shrinkage of $g_0(X)$.
"""),
            _hist_cell(),
            C('''
t_nonorth = (theta_nonorth - alpha) / se_nonorth
plot_t_hist(t_nonorth, "Naive ML: Regularization Bias", "darkblue", "Non-orthogonal ML")
'''),
            M("""
## A.4 Approach 2 — Orthogonalized ML, No Sample Splitting

Orthogonalization removes the leading regularization bias by partialling $X$ out of $D$ as well. Let $\\hat{V} = D - \\hat{m}(X)$ and use the IV-type score:

$$\\check{\\theta}_0 = \\Bigl(\\tfrac{1}{n}\\sum_{i \\in I} \\hat{V}_i D_i\\Bigr)^{-1}
  \\tfrac{1}{n}\\sum_{i \\in I} \\hat{V}_i\\bigl(Y_i - \\hat{g}_0(X_i)\\bigr)$$

With nuisances and the final estimate computed on the **same data**, overfitting bias is not eliminated by sample splitting.

### A.4.1 Fit IV-Type Score, No Sample Splitting
"""),
            C('''
theta_orth_nosplit, se_orth_nosplit = replicate(datasets, score="iv", cross_fit=False, seed=2222)
print("Mean theta:", round(float(np.nanmean(theta_orth_nosplit)), 4))
'''),
            M("### A.4.2 Plot Distribution — Orthogonal, No Sample Splitting\n"),
            C('''
t_orth_nosplit = (theta_orth_nosplit - alpha) / se_orth_nosplit
plot_t_hist(
    t_orth_nosplit,
    "Orthogonal Score, No Sample Splitting: Overfitting Bias",
    "darkorange",
    "DML (no sample splitting)",
)
'''),
            M("""
Even with an orthogonal score, omitting sample splitting allows overfitting bias to persist. Extreme or non-finite t-values (filtered in the histogram) are a characteristic symptom.

## A.5 Approach 3 — Full DML with Cross-Fitting (IV-Type Score)

Cross-fitting completes the fix. With two folds, nuisances are trained on one fold and $\\theta$ estimated on the held-out fold; the roles are then swapped and the estimates combined. This eliminates both regularization bias (orthogonal score) and overfitting bias (sample splitting) while using all observations.

### A.5.1 Fit DML with Cross-Fitting (IV-Type)
"""),
            C('''
theta_dml, se_dml = replicate(datasets, score="iv", cross_fit=True, seed=3333)
print("Mean theta:", round(float(np.mean(theta_dml)), 4))
'''),
            M("### A.5.2 Plot Distribution — DML with Cross-Fitting\n"),
            C('''
t_dml = (theta_dml - alpha) / se_dml
plot_t_hist(t_dml, "Full DML with Cross-Fitting: Valid Inference", "darkgreen", "DML with cross-fitting")
'''),
            M("""
The rescaled estimates now sit much closer to $\\mathcal{N}(0,1)$, confirming that combining an orthogonal score with cross-fitting restores valid inference.

## A.6 Comparison: All Three IV-Type Approaches
"""),
            C('''
fig, ax = plt.subplots(figsize=(7.4, 4.4))
grid = np.linspace(-6, 6, 400)
specs = [
    (t_nonorth, "darkblue", "Non-orthogonal ML"),
    (t_orth_nosplit, "darkorange", "DML (no sample splitting)"),
    (t_dml, "darkgreen", "DML with cross-fitting"),
]
for values, color, label in specs:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.hist(finite, bins=30, density=True, alpha=0.28, color=color, label=label)
ax.plot(grid, stats.norm.pdf(grid), color="black", lw=1.2, label="N(0,1)")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-6, 6)
ax.set_xlabel("Rescaled estimate")
ax.set_ylabel("Density")
ax.set_title("Comparison of All Three Approaches")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
'''),
            M("""
The three histograms make the progression concrete: naive ML (blue) is shifted and spread; orthogonal without splitting (orange) is better but still distorted; full DML with cross-fitting (green) is the one that lines up with $\\mathcal{N}(0,1)$.

## A.7 Partialling-Out Score

The **partialling-out** score is the most common choice in practice. It uses $\\ell_0(X) = \\mathbb{E}[Y|X]$ and $m_0(X) = \\mathbb{E}[D|X]$ directly. The orthogonal moment condition is:

$$\\psi(W;\\,\\theta,\\eta) := \\bigl[Y - \\ell(X) - \\theta(D - m(X))\\bigr](D - m(X))$$

`PartiallyLinearDML` solves this score. We repeat the no-splitting versus cross-fitting comparison to confirm that the result holds independently of which orthogonal score is used.

### A.7.1 Partialling-Out, No Sample Splitting
"""),
            C('''
theta_po_nosplit, se_po_nosplit = replicate(datasets, score="partial", cross_fit=False, seed=4444)
print("Mean theta:", round(float(np.mean(theta_po_nosplit)), 4))
'''),
            M("### A.7.2 Partialling-Out with Cross-Fitting\n"),
            C('''
theta_dml_po, se_dml_po = replicate(datasets, score="partial", cross_fit=True, seed=5555)
# The same score, called through PyCausalML on the first replication.
frame0 = datasets[0]
x_cols = [c for c in frame0.columns if c.startswith("X")]
package_fit = PartiallyLinearDML(
    discrete_treatment=False, n_splits=2, model_y="rf", n_estimators=40, random_state=5555,
)
package_fit.fit(frame0[x_cols], frame0["d"], frame0["y"], feature_names=x_cols)
print("Manual cross-fit mean theta:", round(float(np.mean(theta_dml_po)), 4))
print("PartiallyLinearDML on replication 1:", package_fit.average_treatment_effect())
'''),
            M("### A.7.3 Plot Distribution — Partialling-Out with Cross-Fitting\n"),
            C('''
t_dml_po = (theta_dml_po - alpha) / se_dml_po
plot_t_hist(
    t_dml_po,
    "Partialling-Out Score with Cross-Fitting: Valid Inference",
    "darkgreen",
    "DML (partialling out)",
)
'''),
            M("### A.7.4 Comparison: Partialling-Out vs IV-Type (with Cross-Fitting)\n"),
            C('''
fig, ax = plt.subplots(figsize=(7.4, 4.4))
grid = np.linspace(-6, 6, 400)
for values, color, label in [
    (t_dml, "darkgreen", "DML — IV-type"),
    (t_dml_po, "steelblue", "DML — partialling out"),
]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.hist(finite, bins=30, density=True, alpha=0.3, color=color, label=label)
ax.plot(grid, stats.norm.pdf(grid), color="black", lw=1.2, label="N(0,1)")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-6, 6)
ax.set_xlabel("Rescaled estimate")
ax.set_ylabel("Density")
ax.set_title("IV-Type vs Partialling-Out Score (both with cross-fitting)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
'''),
            M("""
Both scores achieve approximately valid inference when combined with cross-fitting. The two distributions are close in practice, which supports using whichever score is computationally simpler for a given problem.

### A.7.5 Summary Table — Simulation Results
"""),
            C('''
names = [
    "Non-orthogonal ML",
    "IV-type, no sample splitting",
    "IV-type with cross-fitting",
    "Partialling-out, no sample splitting",
    "Partialling-out with cross-fitting",
]
estimates = [theta_nonorth, theta_orth_nosplit, theta_dml, theta_po_nosplit, theta_dml_po]
summary_sim = pd.DataFrame([
    {
        "Approach": name,
        "Mean estimate": round(float(np.nanmean(est)), 4),
        "SD": round(float(np.nanstd(est, ddof=1)), 4),
        "Bias": round(float(np.nanmean(est) - alpha), 4),
    }
    for name, est in zip(names, estimates)
])
print(f"Simulation results across {n_rep} replications (true theta_0 = {alpha})")
summary_sim
'''),
            M("""
## Part B: Real-World Application — 401(k) Eligibility and Net Financial Assets

Part B applies DML to the **401(k) dataset** (Chernozhukov et al., 2020), stored as `Data/data_401k.csv`, to estimate the average causal effect of employer-sponsored 401(k) eligibility (`e401`) on net total financial assets (`net_tfa`), controlling for age, income, education, family structure, and other background variables.

The workflow has four stages: (1) an 80/20 stratified train–test split; (2) causal estimation with `PartiallyLinearDML` on the training fold; (3) out-of-sample validation of the nuisance models on the held-out test fold; and (4) SHAP interpretation of treatment-effect heterogeneity with `LinearDML`. Formal PLR inference applies to the training sample. The test fold is used for predictive checks of the nuisance regressions.

## B.1 Load and Inspect the 401(k) Data
"""),
            C('''
df_401k = pd.read_csv(data_dir / "data_401k.csv")
print("Dimensions:", df_401k.shape)
df_401k.head()
'''),
            M("""
## B.2 Outcome, Treatment, Covariates, and Train–Test Split

The split is stratified on `e401` (80% train / 20% test) so both folds retain similar treatment prevalence.
"""),
            C('''
x_cols_401k = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]
X_401k = df_401k[x_cols_401k].to_numpy(dtype=float)
y_401k = df_401k["net_tfa"].to_numpy(dtype=float)
t_401k = df_401k["e401"].to_numpy(dtype=int)

rng_split = np.random.default_rng(3141)
train_parts = []
for arm in (0, 1):
    pool = np.flatnonzero(t_401k == arm)
    take = max(1, int(np.floor(0.8 * len(pool))))
    train_parts.append(rng_split.choice(pool, size=take, replace=False))
train_idx = np.sort(np.concatenate(train_parts))
test_idx = np.setdiff1d(np.arange(len(t_401k)), train_idx)
X_tr, X_te = X_401k[train_idx], X_401k[test_idx]
y_tr, y_te = y_401k[train_idx], y_401k[test_idx]
t_tr, t_te = t_401k[train_idx], t_401k[test_idx]
print(
    "Train n =", len(train_idx),
    "| Test n =", len(test_idx),
    "| mean(e401) train =", round(float(t_tr.mean()), 3),
    "test =", round(float(t_te.mean()), 3),
)
'''),
            M("""
## B.3 Specify Learners

Both nuisance models use a **random forest** (40 trees, max depth 4). Random forests suit this sample: the covariate set is moderate and several relationships are unlikely to be linear. `model_y="rf"` selects that learner inside PyCausalML.
"""),
            C('''
print("Nuisance learner: random forest via PartiallyLinearDML(model_y='rf')")
'''),
            M("""
## B.4 Fit DML on the Training Set

We fit the PLR with the partialling-out score and cross-fitting on the training data `(X_tr, t_tr, y_tr)`.
"""),
            C('''
fit_401k = PartiallyLinearDML(
    discrete_treatment=True,
    n_splits=3,
    model_y="rf",
    n_estimators=40,
    random_state=3142,
)
fit_401k.fit(X_tr, t_tr, y_tr, feature_names=x_cols_401k)
fit_401k.average_treatment_effect()
'''),
            M("## B.5 Statistical Inference (training sample)\n\n### B.5.1 Coefficient Summary\n"),
            C('''
summary_401k = fit_401k.average_treatment_effect()
print("Effect of 401(k) eligibility on net financial assets (training fit):")
for key, value in summary_401k.items():
    print(f"  {key}: {value:,.2f}")
'''),
            M("### B.5.2 Confidence Interval\n"),
            C('''
ci_lo, ci_hi = fit_401k.confidence_interval(level=0.95)
print("95% confidence interval (training sample):")
print(f"  [{ci_lo:,.2f}, {ci_hi:,.2f}]")
'''),
            M("### B.5.3 Visualize the Estimate\n"),
            C('''
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.errorbar(
    [0], [summary_401k["estimate"]],
    yerr=[[summary_401k["estimate"] - ci_lo], [ci_hi - summary_401k["estimate"]]],
    fmt="o", color="darkgreen", ms=8, capsize=6, lw=1.4,
)
ax.axhline(0, color="grey", ls="--")
ax.set_xticks([0])
ax.set_xticklabels(["e401 → net_tfa (train)"])
ax.set_ylabel("Estimated effect (USD)")
ax.set_title("DML Estimate: 401(k) Eligibility and Net Financial Assets\\nTraining sample; partialling-out, random-forest nuisances")
fig.tight_layout()
'''),
            M("""
### B.5.4 Interpretation

The estimated coefficient on `e401` is the PLR average treatment effect for the training subsample, under the standard PLR assumptions (unconfoundedness, overlap, and nuisance fits that converge fast enough). A positive estimate is consistent with 401(k) eligibility raising net financial assets, in line with Chernozhukov et al. (2020).

## B.6 Holdout Validation of Nuisance Models

To check that the nuisance models generalize beyond the training fold, we refit random forests on the training data only and evaluate them on the held-out test set. We report RMSE for the outcome regression $\\hat{\\ell}(X) \\approx \\mathbb{E}[Y|X]$ and the Brier score for the treatment propensity $\\hat{m}(X) \\approx \\mathbb{E}[D|X]$. Lower values indicate better out-of-sample nuisance fit. This is a predictive sanity check — it does not replace the PLR assumptions, but a severely poor fit would signal that the nuisance models are misspecified.
"""),
            C('''
rf_y = RandomForestRegressor(n_estimators=80, max_depth=5, min_samples_leaf=2, random_state=1, n_jobs=1)
rf_d = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=2, random_state=1, n_jobs=1)
rf_y.fit(X_tr, y_tr)
rf_d.fit(X_tr, t_tr)
y_hat = rf_y.predict(X_te)
p_hat = rf_d.predict_proba(X_te)[:, 1]
rmse_y = float(np.sqrt(mean_squared_error(y_te, y_hat)))
brier_d = float(brier_score_loss(t_te, p_hat))
validation_tbl = pd.DataFrame({
    "Metric": ["RMSE of Y (net_tfa)", "Brier score for D (e401)"],
    "Value": [round(rmse_y, 2), round(brier_d, 4)],
})
validation_tbl
'''),
            M("""
## B.7 SHAP Interpretation of Treatment Effect Heterogeneity

`PartiallyLinearDML` estimates a single scalar $\\hat{\\theta}$, which is not a function of $X$ and therefore cannot be explained by SHAP. To explore which covariates drive heterogeneity, we fit `LinearDML` on the training fold (random-forest first stage, OLS on the residual-on-residual design). This produces a CATE prediction $\\hat{\\tau}(X)$ that varies with $X$. Permutation SHAP is computed with `explain_cate` on a subsample of the test rows.
"""),
            C('''
ldml_401k = LinearDML(
    discrete_treatment=True, n_splits=3, model_y="rf", n_estimators=40, random_state=3143,
)
ldml_401k.fit(X_tr, t_tr, y_tr, feature_names=x_cols_401k)
explanation = explain_cate(
    ldml_401k, X_te, feature_names=x_cols_401k,
    max_samples=48, n_permutations=8, random_state=3143,
)
print("ATE from the linear CATE:", ldml_401k.average_treatment_effect())
explanation.plot_importance(title="SHAP importance for predicted CATE (LinearDML)")
'''),
            C('''
explanation.plot_dependence("age", title="SHAP dependence on age")
'''),
            M("""
**Reading the plots.** SHAP values here are contributions to $\\hat{\\tau}(X)$ from `LinearDML` — a descriptive heterogeneity model, not the same object as the PLR scalar $\\hat{\\theta}$ in B.4–B.5. Use them to explore which covariates are associated with larger or smaller estimated treatment effects under this specification.
"""),
            C('''
explanation.plot_waterfall(row=0, title="SHAP waterfall for one test row")
'''),
            M("""
---

# Summary

| Step | Lesson |
|------|--------|
| **Naive ML** | Non-orthogonal score + ML → regularization bias, invalid inference |
| **Orthogonalization** | IV-type or partialling-out score removes leading bias |
| **No sample splitting** | Same data for ML and $\\hat{\\theta}$ → overfitting bias |
| **Cross-fitting** | Sample splitting + cross-fitting → valid $\\sqrt{n}$-consistent estimates |
| **Real data** | Train–test split, `PartiallyLinearDML` on train; validate nuisances on test; SHAP for the `LinearDML` CATE |

**Takeaway.** Double ML combines orthogonal scores, sample splitting, and cross-fitting so that flexible ML algorithms can handle high-dimensional confounders without biasing the final causal estimate. The simulation in Part A demonstrates each component's contribution; Part B shows the method applied end-to-end on observational economic data.

---

# Resources

**Key References**

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal, 21(1), C1–C68.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2020). *DoubleML – Double/debiased machine learning in R and Python*. Journal of Statistical Software.

**Documentation**

- [DoubleML user guide](https://docs.doubleml.org/stable/guide/index.html)
- PyCausalML: `PartiallyLinearDML`, `LinearDML`, `explain_cate`
"""),
        ],
    )


BUILDERS = [nb00, nb01]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for builder in BUILDERS:
        builder()


if __name__ == "__main__":
    main()

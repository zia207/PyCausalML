#!/usr/bin/env python3
"""Notebooks 4.2, 4.3, and 4.4."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_doubleml_notebooks import BANNER, C, IMPORTS, M, SETUP, write

HELPERS = '''
def cross_fit_predict(X, y, n_splits, seed, kind="ridge"):
    """Out-of-fold predictions. kind is 'ridge', 'rf', or 'gbr'."""
    pred = np.zeros(len(y), dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = np.asarray(y, dtype=float)
    for fold, (tr, te) in enumerate(splitter.split(X)):
        model = _nuisance(kind, seed + fold)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def _nuisance(kind, seed):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "lasso":
        return Lasso(alpha=0.02, max_iter=8000)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=40, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=1,
        )
    if kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=40, max_depth=2, learning_rate=0.08, random_state=seed,
        )
    raise ValueError(kind)


def plot_beeswarm(explanation, title, max_display=8):
    imp = explanation.importance()
    order = np.argsort(imp)[-max_display:]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.3))
    rng = np.random.default_rng(0)
    for i, j in enumerate(order):
        vals = explanation.values[:, j]
        raw = explanation.data[:, j]
        span = float(np.ptp(raw)) or 1.0
        color = (raw - np.min(raw)) / span
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter, c=color, cmap="coolwarm", s=18, alpha=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([explanation.feature_names[j] for j in order])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    return fig


print("Helpers ready")
'''.strip() + "\n"


def nb02():
    write(
        "02_08_05_04_02_double_ml_single_multiple_treatment.ipynb",
        [
            M(BANNER),
            M("""
# 4.2 Double ML - Single/multiple Treatment

Double Machine Learning (DML) fits outcome and treatment with flexible ML, then regresses outcome residuals on treatment residuals to estimate the conditional average treatment effect (CATE). **PyCausalML** provides:

- **LinearDML**: ordinary least squares final stage for heterogeneous treatment effects.
- **SparseLinearDML**: Lasso final stage (for a high-dimensional set of effect modifiers).
- **NonParamDML**: a flexible final stage on the R-learner pseudo-outcome.
- **CausalForestDML**: a random forest on the doubly robust pseudo-outcome for a binary treatment.
- **KernelDML**: a random-Fourier approximation of a kernel CATE for a binary treatment.

This notebook follows the structure of the [EconML DML examples](https://github.com/py-why/EconML/blob/main/notebooks/Double%20Machine%20Learning%20Examples.ipynb). Section 6 applies the same estimators to the NMES sample in `Data/nmes_data.csv`.
"""),
            M("""
## Overview

`LinearDML`, `SparseLinearDML`, and `NonParamDML` / `CausalForestDML` are variants of DML for heterogeneous treatment effects: $\\theta(X)$ may vary across individuals, rather than being a single constant $\\theta_0$.

The shared backbone is the same: residualize $Y$ and $D$ with ML nuisance models, then estimate the treatment effect from the residuals. What differs is how the final stage models $\\theta(X)$.

### The shared setup

$$Y = \\theta(X) \\cdot D + g_0(X) + \\varepsilon$$

After partialling out nuisances:

$$\\tilde{Y} = Y - \\hat{\\ell}(X), \\qquad \\tilde{D} = D - \\hat{m}(X)$$

$$\\tilde{Y} = \\theta(X) \\cdot \\tilde{D} + \\text{noise}$$

### LinearDML

$\\theta(X) = \\beta_0 + \\beta_1 Z_1 + \\beta_2 Z_2 + \\cdots$. The final stage is OLS of $\\tilde{Y}$ on $\\tilde{D}$ interacted with the effect modifiers. Use it when you want interpretable coefficients and can defend a linear specification.

### SparseLinearDML

The same linear CATE, with an L1 penalty, when many candidate modifiers are zero. Use it when the modifier list is long.

### NonParamDML and CausalForestDML

The final stage is a flexible regression of the orthogonal pseudo-outcome. `CausalForestDML` is the binary-treatment version whose final stage is a random forest. `NonParamDML` accepts a continuous treatment.
"""),
            M("""
## Implementation in Python

The sections below use PyCausalML for a single continuous treatment, a single binary treatment, polynomial features, observational NMES data, and two continuous treatments at once.

**Contents**

1. Single continuous treatment (synthetic) and model selection
2. Single binary treatment (synthetic) and ATE inference
3. Single continuous treatment with polynomial features
4. Single continuous treatment on NMES
5. Multiple continuous treatments on NMES
6. Use case: compare `LinearDML`, `SparseLinearDML`, and `NonParamDML` on NMES, with SHAP
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS),
            C(HELPERS),
            M("""
## 1. Single Continuous Treatment (Synthetic) and Model Selection

### 1.1 Data-generating process

The DGP follows Chernozhukov et al.:

$$T = \\langle W, \\beta\\rangle + \\eta$$

$$Y = T\\cdot \\theta(X) + \\langle W, \\gamma\\rangle + \\epsilon$$

with $\\theta(x) = \\exp(2 x_1)$. Confounders $W$ and the effect modifier $X$ are independent. The treatment stays continuous; `LinearDML(..., discrete_treatment=False)` residualizes it with a regression nuisance.
"""),
            C('''
def dgp_continuous(n=1800, n_w=6, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    x = rng.uniform(0, 1, size=(n, 1))
    w = rng.normal(size=(n, n_w))
    beta = np.array([0.5, 0.35, 0.25, -0.2, 0.15, -0.1])[:n_w]
    gamma = np.array([0.4, -0.3, 0.25, 0.15, -0.2, 0.1])[:n_w]
    theta = np.exp(2.0 * x[:, 0])
    treatment = w @ beta + rng.normal(scale=0.6, size=n)
    outcome = treatment * theta + w @ gamma + rng.normal(size=n)
    return x, w, treatment, outcome, theta


def design(x, w):
    names = ["x1"] + [f"W{j + 1}" for j in range(w.shape[1])]
    return np.column_stack([x, w]), names


rng = np.random.default_rng(7)
x, w, treatment, outcome, theta_true = dgp_continuous(rng=rng)
X, x_names = design(x, w)
hold = rng.choice(len(X), size=int(0.3 * len(X)), replace=False)
train = np.setdiff1d(np.arange(len(X)), hold)
print("Train", len(train), "Test", len(hold), "true CATE mean", round(float(theta_true.mean()), 3))
'''),
            M("""
### 1.2 Train estimators

We compare **LinearDML**, **SparseLinearDML** on a degree-3 polynomial of $x_1$ plus $W$, **SparseLinearDML** on a degree-6 polynomial, and **NonParamDML** with a random-forest final stage.
"""),
            C('''
def poly_design(x, w, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    xp = poly.fit_transform(x)
    names = [f"x1^{p}" for p in range(1, degree + 1)] + [f"W{j + 1}" for j in range(w.shape[1])]
    return np.column_stack([xp, w]), names


est = LinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=1)
est.fit(X[train], treatment[train], outcome[train], feature_names=x_names)

X3, names3 = poly_design(x, w, 3)
est1 = SparseLinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", alpha=0.05, random_state=1)
est1.fit(X3[train], treatment[train], outcome[train], feature_names=names3)

X6, names6 = poly_design(x, w, 6)
est2 = SparseLinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", alpha=0.2, random_state=1)
est2.fit(X6[train], treatment[train], outcome[train], feature_names=names6)

est_np = NonParamDML(
    discrete_treatment=False, n_splits=2, model_y="gbr", model_final="rf", n_estimators=40, random_state=1,
)
est_np.fit(X[train], treatment[train], outcome[train], feature_names=x_names)
print("Fitted LinearDML, two SparseLinearDML specifications, and NonParamDML")
'''),
            M("### 1.3 Performance visualization\n"),
            C('''
w_bar = w[train].mean(axis=0)
grid = np.linspace(0.02, 0.98, 60).reshape(-1, 1)
w_grid = np.repeat(w_bar.reshape(1, -1), len(grid), axis=0)
Xg, _ = design(grid, w_grid)
Xg3, _ = poly_design(grid, w_grid, 3)
Xg6, _ = poly_design(grid, w_grid, 6)
truth = np.exp(2.0 * grid[:, 0])

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(grid[:, 0], truth, color="black", lw=2, label="True CATE")
ax.plot(grid[:, 0], est.predict(Xg), label="LinearDML")
ax.plot(grid[:, 0], est1.predict(Xg3), label="SparseLinearDML (degree 3)")
ax.plot(grid[:, 0], est2.predict(Xg6), label="SparseLinearDML (degree 6)")
ax.plot(grid[:, 0], est_np.predict(Xg), label="NonParamDML")
ax.set_xlabel("x1")
ax.set_ylabel("CATE")
ax.set_title("Continuous treatment: CATE along x1 (W at the training mean)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
'''),
            M("""
### 1.4 Model selection

Mean squared error of the predicted CATE against the true CATE on the test sample. Lower is better.
"""),
            C('''
scores = {
    "LinearDML": np.mean((est.predict(X[hold]) - theta_true[hold]) ** 2),
    "SparseLinearDML degree 3": np.mean((est1.predict(X3[hold]) - theta_true[hold]) ** 2),
    "SparseLinearDML degree 6": np.mean((est2.predict(X6[hold]) - theta_true[hold]) ** 2),
    "NonParamDML": np.mean((est_np.predict(X[hold]) - theta_true[hold]) ** 2),
}
pd.DataFrame({"Test CATE MSE": scores}).round(4).sort_values("Test CATE MSE")
'''),
            M("""
## 2. Single Binary Treatment (Synthetic) and ATE

### 2.1 Data-generating process

$$T \\sim \\text{Bernoulli}(\\sigma(\\langle W, \\beta\\rangle))$$

$$Y = T\\cdot \\theta(X) + \\langle W, \\gamma\\rangle + \\epsilon$$

with the same $\\theta(x) = \\exp(2 x_1)$.
"""),
            C('''
def dgp_binary(n=1800, n_w=6, rng=None):
    rng = np.random.default_rng(1 if rng is None else rng)
    x = rng.uniform(0, 1, size=(n, 1))
    w = rng.normal(size=(n, n_w))
    beta = np.array([0.4, 0.3, -0.2, 0.2, 0.1, -0.1])[:n_w]
    gamma = np.array([0.3, -0.25, 0.2, 0.1, -0.15, 0.05])[:n_w]
    theta = np.exp(2.0 * x[:, 0])
    propensity = 1.0 / (1.0 + np.exp(-(w @ beta)))
    treatment = rng.binomial(1, propensity).astype(float)
    outcome = treatment * theta + w @ gamma + rng.normal(size=n)
    return x, w, treatment, outcome, theta


rng_b = np.random.default_rng(11)
xb, wb, tb, yb, thetab = dgp_binary(rng=rng_b)
Xb, names_b = design(xb, wb)
hold_b = rng_b.choice(len(Xb), size=int(0.3 * len(Xb)), replace=False)
train_b = np.setdiff1d(np.arange(len(Xb)), hold_b)
print("Treated share", round(float(tb.mean()), 3))
'''),
            M("### 2.2 Train estimators (binary treatment)\n"),
            C('''
lin_b = LinearDML(discrete_treatment=True, n_splits=2, model_y="rf", n_estimators=40, random_state=2)
lin_b.fit(Xb[train_b], tb[train_b], yb[train_b], feature_names=names_b)
cf_b = CausalForestDML(num_trees=60, n_splits=2, random_state=2)
cf_b.fit(Xb[train_b], tb[train_b], yb[train_b], feature_names=names_b)
irm_b = InteractiveRegressionDML(n_splits=2, model_y="rf", n_estimators=40, random_state=2)
irm_b.fit(Xb[train_b], tb[train_b], yb[train_b], feature_names=names_b)
ker_b = KernelDML(n_components=25, n_splits=2, random_state=2)
ker_b.fit(Xb[train_b], tb[train_b], yb[train_b], feature_names=names_b)
print("Binary-treatment models fitted")
'''),
            M("""
### 2.3 Visualization and ATE

Pointwise intervals are not part of these estimators. The plot shows the CATE along $x_1$. The table is the average effect, with the influence-function interval from each model, next to the mean of the true CATE on the test sample.
"""),
            C('''
wg = np.repeat(wb[train_b].mean(axis=0).reshape(1, -1), len(grid), axis=0)
Xbg, _ = design(grid, wg)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(grid[:, 0], np.exp(2 * grid[:, 0]), color="black", lw=2, label="True CATE")
ax.plot(grid[:, 0], lin_b.predict(Xbg), label="LinearDML")
ax.plot(grid[:, 0], cf_b.predict(Xbg), label="CausalForestDML")
ax.plot(grid[:, 0], ker_b.predict(Xbg), label="KernelDML")
ax.set_xlabel("x1")
ax.set_ylabel("CATE")
ax.set_title("Binary treatment: CATE along x1")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
'''),
            C('''
rows = []
for name, model in [("LinearDML", lin_b), ("CausalForestDML", cf_b), ("InteractiveRegressionDML", irm_b), ("KernelDML", ker_b)]:
    summary = model.average_treatment_effect()
    rows.append({"Model": name, **{k: round(float(summary[k]), 3) for k in summary}})
rows.append({"Model": "True mean CATE (test)", "estimate": round(float(thetab[hold_b].mean()), 3)})
pd.DataFrame(rows)
'''),
            M("""
## 3. Single Continuous Treatment with Polynomial Features

Section 1 already fit three linear final stages. Here $\\hat{\\theta}(z) = \\hat{\\beta}_0 + z^\\top \\hat{\\beta}$, where $z$ is a row of the design passed to `fit`. `intercept_` is $\\hat{\\beta}_0$ and `feature_coef_` is $\\hat{\\beta}$.

### 3.1 Coefficients and intercept
"""),
            C('''
coef_table = pd.DataFrame({"feature": x_names, "LinearDML": est.feature_coef_})
print("LinearDML intercept:", round(float(est.intercept_), 4))
coef_table.round(4)
'''),
            M("""
### 3.2 Worked example: CATE from coefficients vs `predict`

For `LinearDML` the reconstruction $\\hat{\\beta}_0 + z^\\top\\hat{\\beta}$ matches `predict`. For `SparseLinearDML` the same identity holds: the Lasso is fit without an extra intercept on top of the column of ones inside the residual design.
"""),
            C('''
recon = est.intercept_ + X[hold] @ est.feature_coef_
direct = est.predict(X[hold])
recon_s = est1.intercept_ + X3[hold] @ est1.feature_coef_
direct_s = est1.predict(X3[hold])
print("LinearDML max |reconstructed - predict|:", float(np.max(np.abs(recon - direct))))
print("SparseLinearDML max |reconstructed - predict|:", float(np.max(np.abs(recon_s - direct_s))))
'''),
            M("""
## 4. Single Continuous Treatment (Observational Data)

The observational example uses the 1987 National Medical Expenditure Survey extract in `Data/nmes_data.csv`. The outcome is `TOTALEXP` and the continuous treatment is `packyears`. The remaining columns are covariates.
"""),
            C('''
nmes = pd.read_csv(data_dir / "nmes_data.csv").dropna().reset_index(drop=True)
if len(nmes) > 3000:
    nmes = nmes.sample(3000, random_state=21).reset_index(drop=True)
y_nmes = nmes["TOTALEXP"].to_numpy(dtype=float)
t_pack = nmes["packyears"].to_numpy(dtype=float)
x_nmes_cols = [c for c in nmes.columns if c not in {"TOTALEXP", "packyears"}]
X_nmes = nmes[x_nmes_cols].to_numpy(dtype=float)
lin_nmes = LinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=4)
lin_nmes.fit(X_nmes, t_pack, y_nmes, feature_names=x_nmes_cols)
print(nmes.shape)
print(lin_nmes.average_treatment_effect())
'''),
            M("""
## 5. Multiple Continuous Treatments (Observational)

Two continuous exposures, `packyears` and `AGESMOKE`, enter a joint residual-on-residual regression. Cross-fitted residuals of $Y$ are regressed on the cross-fitted residuals of both treatments, which is the constant-effect version of a multi-treatment final stage. Each column also gets its own `LinearDML` so the marginal CATE can vary with $X$.
"""),
            C('''
t_age = nmes["AGESMOKE"].to_numpy(dtype=float)
x_multi_cols = [c for c in nmes.columns if c not in {"TOTALEXP", "packyears", "AGESMOKE"}]
X_multi = nmes[x_multi_cols].to_numpy(dtype=float)
y_res = y_nmes - cross_fit_predict(X_multi, y_nmes, 2, 5, "ridge")
t1_res = t_pack - cross_fit_predict(X_multi, t_pack, 2, 6, "ridge")
t2_res = t_age - cross_fit_predict(X_multi, t_age, 2, 7, "ridge")
joint, *_ = np.linalg.lstsq(np.column_stack([t1_res, t2_res]), y_res, rcond=None)
print("Joint constant effects [packyears, AGESMOKE]:", np.round(joint, 4))

marg_pack = LinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=8)
marg_age = LinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=9)
marg_pack.fit(X_multi, t_pack, y_nmes, feature_names=x_multi_cols)
marg_age.fit(X_multi, t_age, y_nmes, feature_names=x_multi_cols)
pd.DataFrame({
    "Treatment": ["packyears", "AGESMOKE"],
    "LinearDML ATE": [
        marg_pack.average_treatment_effect()["estimate"],
        marg_age.average_treatment_effect()["estimate"],
    ],
    "std.err": [
        marg_pack.average_treatment_effect()["std.err"],
        marg_age.average_treatment_effect()["std.err"],
    ],
}).round(3)
'''),
            M("""
## 6. Use Case: Double Machine Learning on NMES

Outcome `TOTALEXP`, continuous treatment `packyears`, covariates as in Section 4. We compare **LinearDML**, **SparseLinearDML**, and **NonParamDML**, then the binary high-versus-low dose version with **CausalForestDML**. Metrics: the average effect and its interval, Spearman agreement of $\\hat{\\theta}(X)$, and holdout RMSE of a refit outcome nuisance. SHAP attributes the `LinearDML` CATE to covariates.

### 6.1 Data Preparation
"""),
            C('''
rng_n = np.random.default_rng(30)
perm = rng_n.permutation(len(nmes))
cut = int(0.7 * len(nmes))
tr_n, te_n = perm[:cut], perm[cut:]
print("Train", len(tr_n), "Test", len(te_n))
'''),
            M("### 6.2 Fit DML Variants and Compare\n"),
            C('''
models = {
    "LinearDML": LinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=12),
    "SparseLinearDML": SparseLinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", alpha=0.05, random_state=12),
    "NonParamDML": NonParamDML(
        discrete_treatment=False, n_splits=2, model_y="gbr", model_final="rf", n_estimators=40, random_state=12,
    ),
}
for model in models.values():
    model.fit(X_nmes[tr_n], t_pack[tr_n], y_nmes[tr_n], feature_names=x_nmes_cols)
high = (t_pack >= np.median(t_pack)).astype(int)
cf_nmes = CausalForestDML(num_trees=50, n_splits=2, random_state=12)
cf_nmes.fit(X_nmes[tr_n], high[tr_n], y_nmes[tr_n], feature_names=x_nmes_cols)
print("Fitted continuous-dose models and a binary-dose CausalForestDML")
'''),
            M("### 6.3 ATE Inference\n"),
            C('''
ate_rows = []
for name, model in models.items():
    summary = model.average_treatment_effect()
    ate_rows.append({"Model": name, "estimate": summary["estimate"], "std.err": summary["std.err"]})
cf_summary = cf_nmes.average_treatment_effect()
ate_rows.append({
    "Model": "CausalForestDML (packyears above median)",
    "estimate": cf_summary["estimate"],
    "std.err": cf_summary["std.err"],
})
pd.DataFrame(ate_rows).round(2)
'''),
            M("### 6.4 Predicted CATE and Spearman Agreement\n"),
            C('''
cate = {name: model.predict(X_nmes[te_n]) for name, model in models.items()}
pairs = []
names = list(cate)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        rho = stats.spearmanr(cate[a], cate[b]).statistic
        pairs.append({"Pair": f"{a} vs {b}", "Spearman": round(float(rho), 3)})
pd.DataFrame(pairs)
'''),
            M("""
### 6.5 Holdout RMSE of Outcome Nuisance

A standalone random forest is fit on the training split and scored on the held-out 30%. This checks $\\hat{E}[Y \\mid X]$ independently of the DML final stage.
"""),
            C('''
nuisance = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=5, random_state=3, n_jobs=1)
nuisance.fit(X_nmes[tr_n], y_nmes[tr_n])
rmse = float(np.sqrt(mean_squared_error(y_nmes[te_n], nuisance.predict(X_nmes[te_n]))))
print("Holdout RMSE of E[Y | X]:", round(rmse, 2))
'''),
            M("### 6.6 CATE Distribution Plot\n"),
            C('''
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for name, values in cate.items():
    ax.hist(values, bins=30, alpha=0.35, label=name)
ax.set_xlabel("Predicted CATE")
ax.set_ylabel("Count")
ax.set_title("NMES: distribution of predicted packyears effects")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
'''),
            M("""
### 6.7 SHAP for Predicted CATE

`explain_cate` computes permutation SHAP for the `LinearDML` surface. Values are contributions to $\\hat{\\theta}(X)$, not to the scalar ATE.
"""),
            C('''
explanation = explain_cate(
    models["LinearDML"], X_nmes[te_n], feature_names=x_nmes_cols,
    max_samples=40, n_permutations=6, random_state=14,
)
plot_beeswarm(explanation, "SHAP beeswarm for the LinearDML CATE")
'''),
            M("#### Beeswarm and bar importance\n"),
            C('''
explanation.plot_importance(title="Mean |SHAP| for the LinearDML CATE")
'''),
            M("#### SHAP dependence plots\n"),
            C('''
explanation.plot_dependence(x_nmes_cols[0], title=f"SHAP dependence on {x_nmes_cols[0]}")
'''),
            M("#### SHAP for a single prediction\n"),
            C('''
explanation.plot_waterfall(row=0, title="Waterfall for one NMES row")
'''),
            M("##### Subgroup-specific SHAP\n"),
            C('''
if "MALE" in explanation.feature_names:
    column = explanation.feature_names.index("MALE")
    hits = np.flatnonzero(np.asarray(explanation.data[:, column]) == 1)
    male_idx = int(hits[0]) if len(hits) else 0
else:
    male_idx = 0
explanation.plot_waterfall(row=male_idx, title="Waterfall for a row with MALE = 1")
'''),
            M("""
## Summary and Conclusions

This notebook uses PyCausalML to estimate heterogeneous treatment effects for continuous and binary treatments, on synthetic data and on the NMES extract.

- **Section 1.** Continuous treatment. `LinearDML`, polynomial `SparseLinearDML`, and `NonParamDML`, selected by test-set CATE MSE.
- **Section 2.** Binary treatment. CATE curves and average effects with influence-function intervals, including `CausalForestDML`, `KernelDML`, and `InteractiveRegressionDML`.
- **Section 3.** `intercept_` and `feature_coef_` rebuild `predict`.
- **Sections 4 and 5.** NMES. One continuous exposure, then a joint residual regression for two exposures.
- **Section 6.** The same NMES sample, with agreement across final stages and SHAP on `LinearDML`.

## Resources

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Battocchi, K., et al. EconML: a Python package for ML-based heterogeneous treatment effect estimation.
- The NMES extract is `Data/nmes_data.csv` (1987 National Medical Expenditure Survey).
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
"""),
        ],
    )


def nb03():
    write(
        "02_08_05_04_03_double_ml_did.ipynb",
        [
            M(BANNER),
            M("""
# 4.3 Double ML for Difference-in-Differences: Panel Data

**Double machine learning for difference-in-differences** keeps the identification logic of DiD and lets the nuisance functions be flexible. Classic two-way fixed effects assume

$$Y_{it} = \\alpha_i + \\gamma_t + \\theta (D_{it} \\times \\text{Post}_t) + \\varepsilon_{it}.$$

With a first-differenced outcome the partially linear version is

$$\\Delta Y_{it} = \\theta D_{it} + g(X_{it}) + \\zeta_{it},$$

where $g$ is cross-fit rather than forced to be linear.
"""),
            M("""
## Overview

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | Nuisance estimation | Propensity $\\hat{e}(X) = P(D=1\\mid X)$ and control regression $\\hat{\\ell}(X) = E[\\Delta Y \\mid D=0, X]$ |
| 2 | Cross-fitting | Train on $K-1$ folds, predict on the held-out fold |
| 3 | Orthogonal score | $\\psi_i = \\big(\\frac{D_i}{\\bar{p}} - \\frac{(1-D_i)\\hat{e}(X_i)}{\\bar{p}(1-\\hat{e}(X_i))}\\big)(\\Delta Y_i - \\hat{\\ell}(X_i))$ |
| 4 | Final estimation | $\\hat{\\theta} = n^{-1}\\sum_i \\psi_i$, with an influence-function standard error |

| Challenge | Linear DiD | DML-DiD |
|-----------|------------|---------|
| Many controls | OLS is noisy or infeasible when $p$ is large | ML fits the nuisances |
| Nonlinear confounding | A linear $g(X)$ is misspecified | Forests and boosting approximate $g$ |
| Inference | Valid under correct linear specification | Valid if the nuisances converge and the score is orthogonal |

The group-time estimand $\\mathrm{ATT}(g,t)$ is the effect at calendar time $t$ for the cohort first treated at $g$, using not-yet-treated and never-treated units as controls (Callaway and Sant'Anna, 2021). The score above is the doubly robust DiD score of Sant'Anna and Zhao (2020) applied to $\\Delta Y$.
"""),
            M("""
## Implementation in Python

Each group-time cell is a 2×2 difference in differences on the first-differenced outcome. Three nuisance pairs are compared:

| Wrapper | Outcome nuisance | Propensity |
|---------|------------------|------------|
| linear | ridge | logistic regression |
| random forest | random forest | random forest |
| gradient boosting | gradient boosting | gradient boosting |

`DifferenceInDifferences` from PyCausalML is the unadjusted two-period benchmark on one cohort. Aggregation is a simple average of the group-time estimates. SHAP is applied to a `LinearDML` fit on a single first-differenced slice, which describes CATE heterogeneity inside that slice.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS + "\nfrom pycausalml import DifferenceInDifferences\n"),
            C('''
def _classifier(kind, seed):
    if kind == "logit":
        return LogisticRegression(max_iter=400)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=40, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=1,
        )
    return GradientBoostingClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.08, random_state=seed,
    )


def _regressor(kind, seed):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=40, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=1,
        )
    return GradientBoostingRegressor(
        n_estimators=30, max_depth=2, learning_rate=0.08, random_state=seed,
    )


def dr_att(X, d, dy, learner="linear", seed=0, n_splits=2):
    """Doubly robust ATT for a first-differenced 2×2 comparison."""
    d = np.asarray(d, dtype=int)
    dy = np.asarray(dy, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(d)
    e_hat = np.full(n, float(d.mean()))
    mu0 = np.full(n, float(dy[d == 0].mean()) if np.any(d == 0) else 0.0)
    kind_c = {"linear": "logit", "rf": "rf", "gbr": "gbr"}[learner]
    kind_r = {"linear": "ridge", "rf": "rf", "gbr": "gbr"}[learner]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(X)):
        if len(np.unique(d[tr])) < 2 or np.sum(d[tr] == 0) < 8:
            continue
        clf = _classifier(kind_c, seed + fold)
        clf.fit(X[tr], d[tr])
        e_hat[te] = clf.predict_proba(X[te])[:, 1]
        reg = _regressor(kind_r, seed + fold)
        ctrl = tr[d[tr] == 0]
        reg.fit(X[ctrl], dy[ctrl])
        mu0[te] = reg.predict(X[te])
    e_hat = np.clip(e_hat, 0.05, 0.95)
    p_hat = float(np.mean(d))
    psi = (d / p_hat - (1 - d) * e_hat / (p_hat * (1 - e_hat))) * (dy - mu0)
    theta = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    rmse = float(np.sqrt(np.mean((dy[d == 0] - mu0[d == 0]) ** 2)))
    acc = float(np.mean((e_hat >= 0.5).astype(int) == d))
    return {"att": theta, "se": se, "rmse_mu0": rmse, "propensity_acc": acc}


def simulate_panel(n_units=700, seed=0):
    rng = np.random.default_rng(seed)
    cohort = rng.choice([0, 2, 3, 4], size=n_units, p=[0.45, 0.2, 0.2, 0.15])
    x = rng.normal(size=(n_units, 2))
    rows = []
    for period in (1, 2, 3, 4):
        treated_now = ((cohort > 0) & (period >= cohort)).astype(float)
        exposure = np.where(treated_now == 1, period - cohort, 0)
        outcome = (
            0.4 * x[:, 0]
            + 0.15 * period
            + treated_now * (1.0 + 0.4 * exposure)
            + 0.25 * x[:, 1] * treated_now
            + rng.normal(scale=0.45, size=n_units)
        )
        rows.append(pd.DataFrame({
            "id": np.arange(n_units),
            "period": period,
            "G": cohort,
            "Y": outcome,
            "treat": treated_now.astype(int),
            "x1": x[:, 0],
            "x2": x[:, 1],
        }))
    return pd.concat(rows, ignore_index=True)


print("DiD helpers ready")
'''),
            M("""
## Part A: Simulated Data

Four periods and staggered adoption. The effect equals $1 + 0.4 \\times \\text{exposure}$ plus a small $x_2$ interaction, so the cohort-time target, averaging over $x_2$, is about $1 + 0.4(t-g)$. Units are stored in long format.
"""),
            C('''
panel = simulate_panel()
print(panel.groupby(["G", "period"]).size().unstack(fill_value=0))
panel.head()
'''),
            M("""
| Variable | Role |
|---|---|
| `G` | First period treated (0 = never) |
| `period` | Time |
| `Y` | Outcome |
| `treat` | 1 if the unit is treated in this period |
| `x1`, `x2` | Covariates |
| `id` | Unit identifier |

### Benchmark: two-period DiD

`DifferenceInDifferences` fits $Y = a + b\\,\\text{treated} + c\\,\\text{post} + \\tau\\,\\text{treated}\\times\\text{post}$ on the never-treated units and the cohort first treated in period 2, using periods 1 and 2. The target for that cell is about 1.
"""),
            C('''
sub = panel[panel["G"].isin([0, 2]) & panel["period"].isin([1, 2])].copy()
did = DifferenceInDifferences(n_boot=80, random_state=0)
did.fit(
    sub["Y"].to_numpy(),
    (sub["G"] == 2).astype(float).to_numpy(),
    (sub["period"] == 2).astype(float).to_numpy(),
)
print(did.summary())
'''),
            M("""
### Group-time ATT with three nuisance learners

For each cohort $g \\in \\{2,3,4\\}$ and time $t \\ge g$, the pre-period is $g-1$ and the control group is never-treated units. The outcome in the score is $\\Delta Y = Y_t - Y_{g-1}$.
"""),
            C('''
def group_time_table(frame, id_col, time_col, cohort_col, y_col, x_cols, learner):
    cohorts = sorted(g for g in frame[cohort_col].unique() if g > 0)
    rows = []
    for g in cohorts:
        pre = int(g) - 1
        for t in sorted(frame[time_col].unique()):
            if t < g or pre not in set(frame[time_col]):
                continue
            post = frame[frame[time_col] == t][[id_col, y_col, cohort_col, *x_cols]]
            base = frame[frame[time_col] == pre][[id_col, y_col]].rename(columns={y_col: "y_pre"})
            merged = post.merge(base, on=id_col)
            treated = merged[cohort_col] == g
            control = merged[cohort_col] == 0
            sample = merged[treated | control]
            if sample[cohort_col].eq(g).sum() < 15 or sample[cohort_col].eq(0).sum() < 15:
                continue
            result = dr_att(
                sample[x_cols].to_numpy(),
                (sample[cohort_col] == g).astype(int).to_numpy(),
                (sample[y_col] - sample["y_pre"]).to_numpy(),
                learner=learner,
                seed=int(g) * 10 + int(t),
            )
            result.update({"G": int(g), "time": int(t), "learner": learner, "target": 1.0 + 0.4 * (int(t) - int(g))})
            rows.append(result)
    return pd.DataFrame(rows)


tables = [group_time_table(panel, "id", "period", "G", "Y", ["x1", "x2"], learner) for learner in ("linear", "rf", "gbr")]
att_a = pd.concat(tables, ignore_index=True)
att_a.round(3)
'''),
            M("""
On this near-linear design the linear nuisance usually tracks the target closely. Random forests and gradient boosting add flexibility; their standard errors are often a little larger.
"""),
            C('''
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for learner, color in (("linear", "#1F4E79"), ("rf", "#C0392B"), ("gbr", "#1E8449")):
    part = att_a[att_a["learner"] == learner]
    ax.errorbar(
        part["time"] + {"linear": -0.08, "rf": 0.0, "gbr": 0.08}[learner],
        part["att"],
        yerr=1.96 * part["se"],
        fmt="o", label=learner, color=color, capsize=3,
    )
ax.set_xlabel("Calendar time (cohorts overplotted)")
ax.set_ylabel("ATT(g, t)")
ax.set_title("Simulated group-time effects")
ax.legend(frameon=False)
fig.tight_layout()
'''),
            M("""
### Aggregate effects

The simple aggregate is the unweighted mean of $\\widehat{\\mathrm{ATT}}(g,t)$ across cells with $t \\ge g$. Cell-by-cell RMSE is the root mean squared difference between two learners.
"""),
            C('''
agg = att_a.groupby("learner")["att"].mean().rename("simple_aggregate")
wide = att_a.pivot_table(index=["G", "time"], columns="learner", values="att")
rmse_rows = []
for a, b in (("linear", "rf"), ("linear", "gbr"), ("rf", "gbr")):
    rmse_rows.append({"comparison": f"{a} vs {b}", "RMSE": float(np.sqrt(np.mean((wide[a] - wide[b]) ** 2)))})
print(agg.round(3))
pd.DataFrame(rmse_rows).round(4)
'''),
            M("""
### Nuisance diagnostics (Part A)

`rmse_mu0` is the error of the control outcome regression on control units. `propensity_acc` is the accuracy of a 0.5 threshold on the cross-fitted propensity. On a nearly linear design the linear nuisance often has the smallest outcome RMSE. Flexible learners can still recover a similar ATT because the score is orthogonal.
"""),
            C('''
att_a.groupby("learner")[["rmse_mu0", "propensity_acc"]].mean().round(4)
'''),
            M("""
## Part B: Minimum Wages and Teen Employment

`Data/mpdta.csv` is the county panel used by Callaway and Sant'Anna (2003–2007): log teen employment `lemp`, log population `lpop`, and the first year the county's minimum wage rose above the federal level (`first.treat`, with 0 for never treated).

### Load and inspect
"""),
            C('''
mpdta = pd.read_csv(data_dir / "mpdta.csv")
print(mpdta.shape)
print(mpdta["first.treat"].value_counts())
mpdta.head()
'''),
            M("### Benchmark and DML wrappers on mpdta\n"),
            C('''
# Cohort 2004 versus never-treated, years 2003 and 2004, classic two-period DiD.
slice_b = mpdta[mpdta["first.treat"].isin([0, 2004]) & mpdta["year"].isin([2003, 2004])].copy()
did_b = DifferenceInDifferences(n_boot=80, random_state=1)
did_b.fit(
    slice_b["lemp"].to_numpy(),
    (slice_b["first.treat"] == 2004).astype(float).to_numpy(),
    (slice_b["year"] == 2004).astype(float).to_numpy(),
)
print("Two-period DiD, cohort 2004:", did_b.summary())

att_b = pd.concat([
    group_time_table(mpdta, "countyreal", "year", "first.treat", "lemp", ["lpop"], learner)
    for learner in ("linear", "rf", "gbr")
], ignore_index=True)
att_b.round(4)
'''),
            M("### Compare wrapper performance (Part B)\n"),
            C('''
print(att_b.groupby("learner")["att"].agg(["mean", "std"]).round(4))
wide_b = att_b.pivot_table(index=["G", "time"], columns="learner", values="att")
pd.DataFrame([
    {"comparison": f"{a} vs {b}", "RMSE": float(np.sqrt(np.mean((wide_b[a] - wide_b[b]) ** 2)))}
    for a, b in (("linear", "rf"), ("linear", "gbr"), ("rf", "gbr"))
]).round(4)
'''),
            M("### Nuisance diagnostics and the aggregate effect\n"),
            C('''
print(att_b.groupby("learner")[["rmse_mu0", "propensity_acc"]].mean().round(4))
print("Simple aggregate ATT by learner:")
print(att_b.groupby("learner")["att"].mean().round(4))
'''),
            C('''
ref = att_b[att_b["learner"] == "linear"].copy()
ref["rel"] = ref["time"] - ref["G"]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.errorbar(ref["rel"], ref["att"], yerr=1.96 * ref["se"], fmt="o", color="#1F4E79", capsize=3)
ax.axhline(0, color="grey", ls="--")
ax.set_xlabel("Years since first treatment")
ax.set_ylabel("ATT(g, t)")
ax.set_title("Minimum wage and log teen employment (linear nuisances)")
fig.tight_layout()
'''),
            M("""
The aggregate is one number for the average post-treatment cell, with `lpop` as the covariate. Forests and boosting answer the same cells with different nuisances; small differences are expected.

### SHAP on one DiD slice

Group-time fits do not have a single `predict` surface. To show SHAP we take the 2004 cohort against never-treated counties, build $\\Delta Y = \\text{lemp}_{2004} - \\text{lemp}_{2003}$, and fit `LinearDML` on `lpop` plus a second copy of the covariate so the CATE can vary. These values describe that slice, not the group-time table.
"""),
            C('''
post = mpdta[mpdta["year"] == 2004][["countyreal", "lemp", "lpop", "first.treat"]]
pre = mpdta[mpdta["year"] == 2003][["countyreal", "lemp"]].rename(columns={"lemp": "lemp0"})
slice_s = post.merge(pre, on="countyreal")
slice_s = slice_s[slice_s["first.treat"].isin([0, 2004])].copy()
dy = (slice_s["lemp"] - slice_s["lemp0"]).to_numpy()
treat = (slice_s["first.treat"] == 2004).astype(int).to_numpy()
# Two columns so a linear CATE is not forced to a constant.
xs = np.column_stack([
    slice_s["lpop"].to_numpy(),
    (slice_s["lpop"] - slice_s["lpop"].mean()) ** 2,
])
ldml = LinearDML(discrete_treatment=True, n_splits=2, model_y="ridge", random_state=5)
ldml.fit(xs, treat, dy, feature_names=["lpop", "lpop_sq"])
explanation = explain_cate(ldml, xs, feature_names=["lpop", "lpop_sq"], max_samples=40, n_permutations=8, random_state=5)
explanation.plot_importance(title="SHAP importance on one minimum-wage slice")
'''),
            C('''
explanation.plot_dependence("lpop", title="SHAP dependence on lpop")
'''),
            C('''
explanation.plot_waterfall(row=0, title="One county in the 2004 cohort slice")
'''),
            C('''
row = int(np.argmax(explanation.data[:, 0]))
explanation.plot_waterfall(row=row, title="County with the largest lpop in the explained sample")
'''),
            M("""
## Summary

| Component | What we did |
|---|---|
| Data | Long panels with a cohort, a clock, an outcome, and covariates |
| Benchmark | `DifferenceInDifferences` on one 2×2 block |
| DML-DiD | Doubly robust score with linear, forest, and boosting nuisances |
| Diagnostics | Simple aggregates, pairwise ATT RMSE, nuisance RMSE and propensity accuracy |
| Real data | `Data/mpdta.csv`, minimum wages and log teen employment |
| SHAP | `LinearDML` on one first-differenced slice |

The orthogonal score lets the nuisances be nonlinear without handing the ATT a first-order bias from those fits. The linear nuisance is a good default; forests and boosting matter when the propensity or the outcome surface is clearly nonlinear.

## Resources

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*, 219(1), 101–122.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""),
        ],
    )


def nb04():
    write(
        "02_08_05_04_04_double_ml_iv.ipynb",
        [
            M(BANNER),
            M("""
# 4.4 Double ML with Instrumental Variables

**DML-IV** combines an instrument with cross-fitted nuisance models. The instrument $Z$ addresses endogeneity of the treatment $D$. Machine learning partials observed covariates out of $Y$, $D$, and $Z$. The final regression of outcome residuals on treatment residuals, using the instrument residuals as the instrument, is a cross-fitted cousin of two-stage least squares.
"""),
            M("""
## Overview

| Node | Role |
|---|---|
| $Z$ | Instrument: affects $D$, and affects $Y$ only through $D$ |
| $C$ | Unobserved confounder |
| $D$ | Endogenous treatment |
| $Y$ | Outcome |

A regression of $Y$ on $D$ is biased when $C$ moves both. Partial linear DML without an instrument does not remove that bias. DML-IV does, under the usual IV assumptions (relevance, exclusion, and independence of $Z$ given the observed covariates).

With covariates $W$, the partially linear IV score residualizes all three variables:

$$\\tilde Y = Y - \\hat E[Y\\mid W], \\quad \\tilde D = D - \\hat E[D\\mid W], \\quad \\tilde Z = Z - \\hat E[Z\\mid W]$$

$$\\hat\\theta = \\frac{\\sum_i \\tilde Z_i \\tilde Y_i}{\\sum_i \\tilde Z_i \\tilde D_i}.$$

`InstrumentalVariable` in PyCausalML is the linear two-stage least squares version of the same idea, with a Bayesian bootstrap interval. `PartiallyLinearDML` is the no-instrument comparison.
"""),
            M("""
## Implementation in Python

1. Simulate (Part A) or load (Part B) the sample.
2. Pass observed covariates into the nuisance regressions so a forest has something to split on.
3. Compute the naive mean difference.
4. Fit DML-IV and `PartiallyLinearDML` for ridge, random-forest, and gradient-boosting nuisances.
5. Compare estimates with the known effect in Part A, and across learners in Part B.

`Data/data_401k.csv` records **eligibility** (`e401`) and net financial assets, which is the partially linear 401(k) design. It does not contain a separate participation indicator, so the instrumental-variables contrast — eligibility as an instrument for an endogenous decision — is identified in the simulation, where $Z$, $D$, and $C$ are all generated. Part B estimates the eligibility effect on the real file and explains a linear CATE with SHAP.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS + "\nfrom pycausalml import InstrumentalVariable\n"),
            C(HELPERS),
            C('''
def dml_iv(W, d, z, y, kind="ridge", n_splits=3, seed=0):
    y_res = y - cross_fit_predict(W, y, n_splits, seed, kind)
    d_res = d - cross_fit_predict(W, d, n_splits, seed + 3, kind)
    z_res = z - cross_fit_predict(W, z, n_splits, seed + 6, kind)
    denom = float(np.dot(z_res, d_res))
    theta = float(np.dot(z_res, y_res) / denom)
    psi = z_res * (y_res - theta * d_res) / np.mean(z_res * d_res)
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return theta, se


class IVLinearCATE:
    """Linear CATE from a cross-fitted IV score, for SHAP."""

    def __init__(self, kind="ridge", n_splits=2, seed=0):
        self.kind = kind
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, d, z, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y_res = y - cross_fit_predict(X, y, self.n_splits, self.seed, self.kind)
        d_res = d - cross_fit_predict(X, d, self.n_splits, self.seed + 3, self.kind)
        z_res = z - cross_fit_predict(X, z, self.n_splits, self.seed + 6, self.kind)
        phi = np.column_stack([np.ones(len(X)), X])
        design = z_res[:, None] * phi
        # Project y_res onto the instrumented treatment interacted with phi.
        # Use d_res on the left of the normal equations via 2SLS-style scaling:
        # solve (Zphi' Dphi) beta = Zphi' y, with Dphi = d_res * phi.
        d_design = d_res[:, None] * phi
        beta, *_ = np.linalg.lstsq(design.T @ d_design, design.T @ y_res, rcond=None)
        self.intercept_ = float(beta[0])
        self.feature_coef_ = np.asarray(beta[1:], dtype=float)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.feature_coef_

    predict_cate = predict


print("IV helpers ready")
'''),
            M("""
## Part A: DML-IV with Synthetic Data

### Data-generating process

- $C$: binary unobserved confounder
- $Z$: binary instrument
- $D$: binary decision, increasing in $Z$ and in $C$
- $Y$: continuous outcome. The structural effect of $D$ is $-2$
- $W$: three noise covariates passed to the nuisance models

The naive difference in means is pulled upward by $C$.
"""),
            C('''
def make_iv_sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    confounder = rng.binomial(1, 0.5, size=n).astype(float)
    instrument = rng.binomial(1, 0.5, size=n).astype(float)
    covariates = rng.normal(size=(n, 3))
    logits = -0.4 + 1.6 * instrument + 1.7 * confounder
    propensity = 1.0 / (1.0 + np.exp(-logits))
    decision = rng.binomial(1, propensity).astype(float)
    outcome = -2.0 * decision + 4.0 * confounder + 0.15 * covariates[:, 0] + rng.normal(size=n)
    return covariates, instrument, decision, outcome


W, Z, D, Y = make_iv_sample()
naive = float(Y[D == 1].mean() - Y[D == 0].mean())
print("Naive mean difference:", round(naive, 3), "(true effect is -2)")
'''),
            M("""
The naive estimate is far from $-2$. Units with $C = 1$ are both more likely to take the decision and have a higher baseline outcome.

### DML-IV versus standard DML
"""),
            C('''
rows = [{"Estimator": "Naive mean difference", "Learner": "—", "Estimate": naive, "Std. err": np.nan}]
for kind in ("ridge", "rf", "gbr"):
    theta, se = dml_iv(W, D, Z, Y, kind=kind, seed=20)
    rows.append({"Estimator": "DML-IV", "Learner": kind, "Estimate": theta, "Std. err": se})
    plr = PartiallyLinearDML(
        discrete_treatment=True, n_splits=3, model_y=kind if kind != "gbr" else "gbr",
        n_estimators=40, random_state=20,
    )
    plr.fit(W, D, Y, feature_names=["W1", "W2", "W3"])
    summary = plr.average_treatment_effect()
    rows.append({
        "Estimator": "PartiallyLinearDML (no instrument)",
        "Learner": kind,
        "Estimate": summary["estimate"],
        "Std. err": summary["std.err"],
    })
comparison = pd.DataFrame(rows)
comparison.round(3)
'''),
            M("""
**Takeaway.** For every first-stage learner, DML-IV recovers a value near the structural effect of $-2$. `PartiallyLinearDML` has no instrument, so it stays near the confounded naive contrast.

### Minimal reference fit

Linear nuisances, and `InstrumentalVariable` (two-stage least squares) on the same sample.
"""),
            C('''
theta_ref, se_ref = dml_iv(W, D, Z, Y, kind="ridge", n_splits=5, seed=1)
iv_linear = InstrumentalVariable(n_boot=80, random_state=1)
iv_linear.fit(Y, D, Z, X=W)
print("DML-IV ridge:", round(theta_ref, 3), "se", round(se_ref, 3))
print("InstrumentalVariable:", iv_linear.summary())
'''),
            M("""
### Second-stage coefficients

With a constant second stage the only coefficient is $\\hat\\theta$. The heterogeneous fit below lets the effect depend on $W$, which is what SHAP needs. The DGP's structural effect does not actually vary with $W$, so those slopes should be small and the average prediction should stay near $-2$.
"""),
            C('''
het = IVLinearCATE(kind="ridge", n_splits=2, seed=3)
het.fit(W, D, Z, Y, feature_names=["W1", "W2", "W3"])
print("Intercept", round(het.intercept_, 3))
print(pd.DataFrame({"feature": het.feature_names_, "coef": het.feature_coef_}).round(3))
print("Mean predicted CATE", round(float(het.predict(W).mean()), 3))
'''),
            M("""
### Summary of Part A

| Estimator | What it recovers |
|---|---|
| Naive mean difference | Confounded contrast, pulled up by $C$ |
| DML-IV | Structural effect of the decision, near $-2$ |
| `PartiallyLinearDML` | The same confounded contrast, with covariates partialled out of the *observed* $W$ only |

With binary $Z$ and $D$ this is a LATE-style estimand for compliers. A constant second stage collapses it to one number, analogous to cross-fitted 2SLS.
"""),
            M("""
## Part B: 401(k) Eligibility and Net Financial Assets

| Variable | Role in this extract |
|---|---|
| `net_tfa` | Outcome |
| `e401` | Eligibility, the treatment in the partially linear specification |
| `age`, `inc`, `fsize`, `educ`, `db`, `marr`, `twoearn`, `pira`, `hown` | Covariates |

Eligibility is the policy variable in the file. The simulation in Part A is where an unobserved confounder and a separate instrument are both present. Part B compares a naive mean difference with `PartiallyLinearDML` and `LinearDML`, then explains the linear CATE.
"""),
            C('''
pension = pd.read_csv(data_dir / "data_401k.csv")
x_cols = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]
Xp = pension[x_cols].to_numpy(dtype=float)
yp = pension["net_tfa"].to_numpy(dtype=float)
dp = pension["e401"].to_numpy(dtype=int)
rng = np.random.default_rng(3141)
train_parts = []
for arm in (0, 1):
    pool = np.flatnonzero(dp == arm)
    train_parts.append(rng.choice(pool, size=int(0.8 * len(pool)), replace=False))
tr = np.concatenate(train_parts)
te = np.setdiff1d(np.arange(len(dp)), tr)
naive_p = float(yp[dp == 1].mean() - yp[dp == 0].mean())
print("Train", len(tr), "Test", len(te), "naive difference", round(naive_p, 1))
'''),
            M("### Train–test comparison\n"),
            C('''
results = [{"Estimator": "Naive mean difference", "Learner": "—", "Estimate": naive_p, "Std. err": np.nan}]
for kind, label in (("ridge", "ridge"), ("rf", "rf"), ("gbr", "gbr")):
    model = PartiallyLinearDML(discrete_treatment=True, n_splits=3, model_y=kind, n_estimators=30, random_state=8)
    model.fit(Xp[tr], dp[tr], yp[tr], feature_names=x_cols)
    summary = model.average_treatment_effect()
    results.append({
        "Estimator": "PartiallyLinearDML",
        "Learner": label,
        "Estimate": summary["estimate"],
        "Std. err": summary["std.err"],
    })
result_table = pd.DataFrame(results)
result_table.round(1)
'''),
            M("### Visualize estimates\n"),
            C('''
plot_df = result_table.dropna().reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.errorbar(
    np.arange(len(plot_df)),
    plot_df["Estimate"],
    yerr=1.96 * plot_df["Std. err"],
    fmt="o", color="#1F4E79", capsize=4,
)
ax.axhline(0, color="grey", ls="--")
ax.set_xticks(np.arange(len(plot_df)))
ax.set_xticklabels(plot_df["Learner"])
ax.set_ylabel("Effect on net financial assets (USD)")
ax.set_title("401(k) eligibility: partially linear DML")
fig.tight_layout()
'''),
            M("""
**Reading the table.** Every `PartiallyLinearDML` row is the effect of eligibility after partialling out the observed covariates. The naive row is the raw mean difference. There is no simulation truth on this file; stability across learners is the check.

The DML-IV rows from Part A are the right comparison when participation is endogenous and eligibility is only an instrument. That second endogenous regressor is not in `data_401k.csv`.

### SHAP

`LinearDML` supplies $\\hat\\tau(X)$ on the training fold. Permutation SHAP uses a subsample of the test fold.
"""),
            C('''
cate_model = LinearDML(discrete_treatment=True, n_splits=2, model_y="rf", n_estimators=30, random_state=9)
cate_model.fit(Xp[tr], dp[tr], yp[tr], feature_names=x_cols)
# Heterogeneous IV surface on the synthetic design, for a side-by-side reading of SHAP.
iv_surface = IVLinearCATE(kind="rf", n_splits=2, seed=4)
iv_surface.fit(W, D, Z, Y, feature_names=["W1", "W2", "W3"])
explanation = explain_cate(
    cate_model, Xp[te], feature_names=x_cols, max_samples=40, n_permutations=6, random_state=9,
)
iv_explanation = explain_cate(
    iv_surface, W, feature_names=["W1", "W2", "W3"], max_samples=40, n_permutations=6, random_state=4,
)
plot_beeswarm(explanation, "401(k) LinearDML CATE")
'''),
            C('''
explanation.plot_importance(title="Mean |SHAP| for 401(k) eligibility CATE")
'''),
            C('''
plot_beeswarm(iv_explanation, "Synthetic DML-IV CATE (effect should not vary with W)")
'''),
            C('''
explanation.plot_dependence("age", title="SHAP dependence on age")
'''),
            C('''
explanation.plot_waterfall(row=0, title="One explained 401(k) row")
'''),
            C('''
married = explanation.feature_names.index("marr")
hits = np.flatnonzero(explanation.data[:, married] == 1)
row = int(hits[0]) if len(hits) else 0
explanation.plot_waterfall(row=row, title="A row with marr = 1")
'''),
            M("""
DML-IV and partially linear CATE surfaces answer different questions. On the simulation they should disagree in level, because only the IV score removes $C$. On the 401(k) extract the plotted SHAP values describe heterogeneity in the eligibility effect.

## Summary

| Component | Description |
|---|---|
| IV structure | $Z$ shifts $D$; $C$ otherwise confounds $D$ and $Y$ |
| DML-IV | Cross-fitted residuals of $Y$, $D$, and $Z$, then $\\tilde Z$ instruments $\\tilde D$ |
| `InstrumentalVariable` | Linear 2SLS with a bootstrap interval |
| `PartiallyLinearDML` | No instrument; biased when an unobserved confounder remains |
| 401(k) file | Eligibility effect via `PartiallyLinearDML` and `LinearDML` |
| SHAP | `explain_cate` on the linear CATE and on the simulated IV surface |

## References

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2020). DoubleML. *Journal of Statistical Software*.
- [DoubleML user guide](https://docs.doubleml.org/stable/guide/index.html)
"""),
        ],
    )


if __name__ == "__main__":
    nb02()
    nb03()
    nb04()

#!/usr/bin/env python3
"""Notebooks 4.7 and 4.8."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_doubleml_notebooks import BANNER, C, IMPORTS, M, SETUP, write


def nb07():
    write(
        "02_08_05_04_07_double_ml_cluster_robust.ipynb",
        [
            M(BANNER),
            M("""
# 4.7 Cluster Robust Double Machine Learning

When rows are clustered — students in classrooms, products in markets, years of the same firm — errors inside a cluster move together. Two pieces of DML have to change.

1. **Sample splitting.** A random row split puts two rows from the same cluster on opposite sides of a fold, so the nuisance model memorizes the cluster. Assign folds by cluster id.
2. **Standard errors.** Sum the orthogonal scores inside each cluster, then take the variance of those totals (Cameron, Gelbach, and Miller, 2011; Chiang, Kato, Ma, and Sasaki, 2021). Two-way clustering adds the two one-way variances and subtracts the intersection.

An interval that ignores the clusters is too narrow.
"""),
            M("""
## Overview

The score is the partially linear IV moment. After cross-fitting,

$$\\hat\\theta = \\frac{\\sum_i \\tilde Z_i \\tilde Y_i}{\\sum_i \\tilde Z_i \\tilde D_i}, \\qquad
\\psi_i = \\frac{\\tilde Z_i(\\tilde Y_i - \\hat\\theta \\tilde D_i)}{\\overline{\\tilde Z\\tilde D}}.$$

The cluster-robust standard error replaces $\\psi_i$ with $\\Psi_c = \\sum_{i \\in c} \\psi_i$. The asymptotics need many clusters, not just many rows. A working rule is 30 or more clusters in each dimension you cluster on.

| Design | Correlation | Variance |
|--------|-------------|----------|
| One-way | Rows that share one id | Variance of that id's score totals |
| Two-way | Rows that share either of two ids | Sum of the two one-way variances, minus the cell variance |
"""),
            M("""
## Implementation in Python

Part 1 simulates a two-way grid in the spirit of Chiang et al. (2021). Part 2 turns the second shock off. Part 3 uses `Data/BLP.csv` (Berry, Levinsohn, and Pakes): demand residual `y`, log price as the treatment, an instrument built from other firms' `hpwt` in the same market, and clusters `model.id` and `cdid`.

Nuisances are cross-fitted lasso fits. Folds are assignments of cluster ids.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS),
            C('''
def fold_ids(labels, n_folds, seed):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    mapping = {label: i % n_folds for i, label in enumerate(order)}
    return np.array([mapping[label] for label in labels])


def cross_fit_lasso(X, y, folds, alpha=0.05):
    pred = np.zeros(len(y), dtype=float)
    y = np.asarray(y, dtype=float)
    for fold in np.unique(folds):
        te = folds == fold
        tr = ~te
        if tr.sum() < 10 or te.sum() == 0:
            pred[te] = np.mean(y[tr]) if tr.any() else np.mean(y)
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        model = Lasso(alpha=alpha, max_iter=8000)
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(scaler.transform(X[te]))
    return pred


def cluster_variance(psi, clusters):
    totals = pd.DataFrame({"psi": psi, "c": clusters}).groupby("c")["psi"].sum().to_numpy()
    g = len(totals)
    return (g / (g - 1.0)) * np.sum(totals ** 2) / (len(psi) ** 2)


def twoway_se(psi, cluster_i, cluster_j):
    pair = pd.Series(np.asarray(cluster_i).astype(str)) + ":" + pd.Series(np.asarray(cluster_j).astype(str))
    variance = cluster_variance(psi, cluster_i) + cluster_variance(psi, cluster_j) - cluster_variance(psi, pair.to_numpy())
    return float(np.sqrt(max(variance, 0.0)))


def pliv(X, y, d, z, folds, alpha=0.05):
    y_res = y - cross_fit_lasso(X, y, folds, alpha)
    d_res = d - cross_fit_lasso(X, d, folds, alpha)
    z_res = z - cross_fit_lasso(X, z, folds, alpha)
    denom = float(np.mean(z_res * d_res))
    theta = float(np.dot(z_res, y_res) / np.dot(z_res, d_res))
    psi = z_res * (y_res - theta * d_res) / denom
    iid_se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return theta, psi, iid_se


def make_cluster_pliv(n_i=20, n_j=20, p=25, theta=1.0, seed=0, one_way=False):
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing="ij")
    cluster_i, cluster_j = ii.ravel(), jj.ravel()
    n = len(cluster_i)
    shock_i = rng.normal(size=n_i)
    shock_j = np.zeros(n_j) if one_way else rng.normal(size=n_j)
    X = rng.normal(scale=0.4, size=(n, p))
    X[:, 0] += shock_i[cluster_i] + shock_j[cluster_j]
    eps = rng.normal(size=n) + 0.8 * (shock_i[cluster_i] + shock_j[cluster_j])
    instrument = 0.8 * X[:, 1] + rng.normal(size=n)
    treatment = 0.7 * instrument + 0.3 * X[:, 2] + 0.5 * eps + rng.normal(scale=0.3, size=n)
    outcome = theta * treatment + 0.8 * X[:, 0] + eps
    return X, outcome, treatment, instrument, cluster_i, cluster_j


print("Cluster helpers ready")
'''),
            M("""
## Part 1: Two-way clusters

A $20 \\times 20$ grid. The true effect is 1. Most covariates are noise, so the nuisance is a lasso. Each cluster dimension is split into 3 folds; the product is 9 folds.
"""),
            C('''
X, y, d, z, ci, cj = make_cluster_pliv(seed=7)
folds = fold_ids(ci, 3, 1) * 3 + fold_ids(cj, 3, 2)
theta, psi, iid_se = pliv(X, y, d, z, folds, alpha=0.08)
print("Folds", len(np.unique(folds)))
print("Estimate", round(theta, 3), "(true value 1)")
print("i.i.d. SE", round(iid_se, 3))
print("Two-way SE", round(twoway_se(psi, ci, cj), 3))
'''),
            M("""
## Part 2: One-way clusters

The second shock is off. Folds follow only the first cluster id, so there are $K$ folds rather than $K^2$.
"""),
            C('''
X1, y1, d1, z1, ci1, _cj1 = make_cluster_pliv(seed=8, one_way=True)
folds1 = fold_ids(ci1, 3, 3)
theta1, psi1, iid1 = pliv(X1, y1, d1, z1, folds1, alpha=0.08)
print("Estimate", round(theta1, 3))
print("i.i.d. SE", round(iid1, 3), "| one-way SE", round(float(np.sqrt(cluster_variance(psi1, ci1))), 3))
'''),
            M("""
## Part 3: Automobile demand (BLP)

`Data/BLP.csv`. Outcome `y`, treatment log price (price shifted by 11.761 before the log), instrument equal to the sum of `hpwt` over other firms in the same market. Covariates `hpwt`, `air`, `mpd`, and `space` are expanded to squares and interactions.
"""),
            C('''
blp = pd.read_csv(data_dir / "BLP.csv")
blp["log_price"] = np.log(blp["price"] + 11.761)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_blp = poly.fit_transform(blp[["hpwt", "air", "mpd", "space"]].to_numpy(dtype=float))
market_sum = blp.groupby("cdid")["hpwt"].transform("sum")
firm_sum = blp.groupby(["cdid", "firm.id"])["hpwt"].transform("sum")
instrument = (market_sum - firm_sum).to_numpy(dtype=float)
y_blp = blp["y"].to_numpy(dtype=float)
d_blp = blp["log_price"].to_numpy(dtype=float)
product = blp["model.id"].to_numpy()
market = blp["cdid"].to_numpy()
print(blp.shape, "markets", blp["cdid"].nunique(), "models", pd.Series(product).nunique())
'''),
            C('''
specs = {
    "No clustering": (fold_ids(np.arange(len(blp)), 4, 4), "iid"),
    "Product": (fold_ids(product, 4, 5), "product"),
    "Market": (fold_ids(market, 4, 6), "market"),
    "Product and market": (fold_ids(product, 2, 7) * 2 + fold_ids(market, 2, 8), "two"),
}
rows = []
for name, (folds_b, kind) in specs.items():
    theta_b, psi_b, iid_b = pliv(X_blp, y_blp, d_blp, instrument, folds_b, alpha=0.02)
    if kind == "iid":
        se = iid_b
    elif kind == "product":
        se = float(np.sqrt(cluster_variance(psi_b, product)))
    elif kind == "market":
        se = float(np.sqrt(cluster_variance(psi_b, market)))
    else:
        se = twoway_se(psi_b, product, market)
    rows.append({"Specification": name, "Estimate": theta_b, "Std. err": se, "Folds": int(len(np.unique(folds_b)))})
blp_table = pd.DataFrame(rows)
blp_table.round(4)
'''),
            C('''
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.errorbar(np.arange(len(blp_table)), blp_table["Estimate"], yerr=1.96 * blp_table["Std. err"], fmt="o", color="#1F4E79", capsize=4)
ax.axhline(0, color="grey", ls="--")
ax.set_xticks(np.arange(len(blp_table)))
ax.set_xticklabels(blp_table["Specification"], rotation=15)
ax.set_ylabel("Effect of log price on demand")
ax.set_title("BLP: the interval widens once clusters are acknowledged")
fig.tight_layout()
'''),
            M("""
The price coefficient should be similar across splits. The standard error grows once the variance estimator allows correlation within products, within markets, or both.

## Summary

| Step | What it does |
|------|----------------|
| Clustered folds | Every row of a cluster stays in one fold |
| Two-way SE | One-way variances along each dimension, minus the cell variance |
| One-way SE | Sum the scores inside the single cluster id |
| BLP | `Data/BLP.csv`, instrumented log price |

## Resources

- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile prices in market equilibrium. *Econometrica*, 63(4), 841–890.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238–249.
- Chiang, H. D., Kato, K., Ma, Y., & Sasaki, Y. (2021). Multiway cluster-robust double/debiased machine learning. *Journal of Business & Economic Statistics*.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""),
        ],
    )


def nb08():
    write(
        "02_08_05_04_08_double_ml_ensemble_learners.ipynb",
        [
            M(BANNER),
            M("""
# 4.8 Double ML with Ensemble Learners

The orthogonal score does not care which algorithm fits $E[Y\\mid X]$ and $E[D\\mid X]$, as long as those fits are cross-fitted and converge fast enough. A pipeline, an average of several models, or a stack is just another function from $X$ to a prediction.

This notebook builds those learners with scikit-learn and plugs them into the partialling-out score. `PartiallyLinearDML` is that same score with the package's ridge and random-forest nuisances.
"""),
            M("""
## Overview

| Learner | Nuisance |
|---------|----------|
| Lasso or a forest | One algorithm |
| Pipeline | Scale the columns, then fit |
| Averaging ensemble | Mean of lasso, a forest, and gradient boosting |
| Stack | First-stage predictions joined back to $X$, then a second model |
| Small grid | Choose the lasso penalty by cross-validated outcome error, then refit the score |

Changing a decent nuisance should move $\\hat\\theta$ only within sampling error.
"""),
            M("""
## Implementation in Python

Two samples. A simulated partially linear model with a continuous treatment and true effect 0.8. And `Data/data_401k.csv`: binary eligibility `e401`, outcome `net_tfa`, capped at 4,000 rows. `fit_plr` cross-fits any scikit-learn estimator. A classifier is used for the propensity when the treatment is binary.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS + '''
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
'''),
            C('''
def fit_plr(X, d, y, learner_y, learner_d, n_splits=2, seed=0, discrete=False):
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    y_hat = np.zeros(len(y))
    d_hat = np.zeros(len(y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X):
        model_y, model_d = clone(learner_y), clone(learner_d)
        model_y.fit(X[tr], y[tr])
        y_hat[te] = model_y.predict(X[te])
        if discrete:
            model_d.fit(X[tr], d[tr].astype(int))
            d_hat[te] = model_d.predict_proba(X[te])[:, 1]
        else:
            model_d.fit(X[tr], d[tr])
            d_hat[te] = model_d.predict(X[te])
    y_res, d_res = y - y_hat, d - d_hat
    theta = float(np.dot(d_res, y_res) / np.dot(d_res, d_res))
    psi = d_res * (y_res - theta * d_res) / np.mean(d_res ** 2)
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return theta, se


def simulate_plr(n=900, p=12, theta=0.8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:4] = np.array([0.8, -0.5, 0.4, 0.3])
    treatment = X @ beta + rng.normal(size=n)
    outcome = theta * treatment + np.sin(X[:, 0]) + 0.3 * X[:, 1] + rng.normal(size=n)
    return X, treatment, outcome


print("Ensemble helpers ready")
'''),
            M("## Part 1: One learner on the simulation\n"),
            C('''
X, d, y = simulate_plr()
lasso = Lasso(alpha=0.02, max_iter=5000)
forest = RandomForestRegressor(n_estimators=40, max_depth=4, min_samples_leaf=5, random_state=0, n_jobs=1)
theta_lasso, se_lasso = fit_plr(X, d, y, lasso, lasso, seed=1)
theta_forest, se_forest = fit_plr(X, d, y, forest, forest, seed=1)
package = PartiallyLinearDML(discrete_treatment=False, n_splits=2, model_y="ridge", random_state=1)
package.fit(X, d, y)
print("Lasso", round(theta_lasso, 3), "se", round(se_lasso, 3))
print("Forest", round(theta_forest, 3), "se", round(se_forest, 3))
print("PartiallyLinearDML", package.average_treatment_effect())
print("True theta 0.8")
'''),
            M("## Part 2: 401(k) eligibility\n"),
            C('''
pension = pd.read_csv(data_dir / "data_401k.csv")
if len(pension) > 4000:
    pension = pension.sample(4000, random_state=4).reset_index(drop=True)
x_cols = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]
Xp = pension[x_cols].to_numpy(dtype=float)
yp = pension["net_tfa"].to_numpy(dtype=float)
dp = pension["e401"].to_numpy(dtype=float)
rf_y = RandomForestRegressor(n_estimators=40, max_depth=5, min_samples_leaf=5, random_state=2, n_jobs=1)
rf_d = RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_leaf=5, random_state=2, n_jobs=1)
theta_rf, se_rf = fit_plr(Xp, dp, yp, rf_y, rf_d, discrete=True, seed=2)
pkg = PartiallyLinearDML(discrete_treatment=True, n_splits=2, model_y="rf", n_estimators=40, random_state=2)
pkg.fit(Xp, dp, yp, feature_names=x_cols)
print("Forest score", round(theta_rf, 1), "se", round(se_rf, 1))
print("PartiallyLinearDML", pkg.average_treatment_effect())
'''),
            M("## Part 3: A pipeline\n"),
            C('''
pipe = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=0.02, max_iter=5000))])
theta_pipe, se_pipe = fit_plr(X, d, y, pipe, pipe, seed=3)
print("Scaled lasso", round(theta_pipe, 3), "se", round(se_pipe, 3))
print("Unscaled lasso", round(theta_lasso, 3))
'''),
            M("## Part 4: An average of learners\n"),
            C('''
average_y = VotingRegressor([
    ("lasso", Lasso(alpha=0.02, max_iter=5000)),
    ("forest", RandomForestRegressor(n_estimators=30, max_depth=4, random_state=0, n_jobs=1)),
    ("gbr", GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=0)),
])
theta_avg, se_avg = fit_plr(X, d, y, average_y, clone(average_y), seed=4)
print("Averaged nuisances, simulation", round(theta_avg, 3), "se", round(se_avg, 3))
average_d = VotingClassifier([
    ("logit", Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])),
    ("forest", RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0, n_jobs=1)),
    ("gbr", GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=0)),
], voting="soft")
theta_vote, se_vote = fit_plr(Xp, dp, yp, clone(average_y), average_d, discrete=True, seed=4)
print("Averaged nuisances, 401(k)", round(theta_vote, 1), "se", round(se_vote, 1))
'''),
            M("## Part 5: A stack\n"),
            C('''
stack = StackingRegressor(
    estimators=[
        ("tree", DecisionTreeRegressor(max_depth=3, random_state=0)),
        ("ridge", Ridge(alpha=1.0)),
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,
)
theta_stack, se_stack = fit_plr(X, d, y, stack, Ridge(alpha=1.0), seed=5)
print("Stacked outcome nuisance", round(theta_stack, 3), "se", round(se_stack, 3))
'''),
            M("""
## Part 6: A small penalty grid

Three lasso penalties. The column `outcome CV-MSE` is how well each penalty predicts $Y$ from $X$. The partialling-out estimate should stay near 0.8 across the grid.
"""),
            C('''
rows = []
best_alpha, best_mse = None, np.inf
splitter = KFold(n_splits=3, shuffle=True, random_state=6)
for alpha in (0.001, 0.02, 0.2):
    mse = []
    for tr, te in splitter.split(X):
        model = Lasso(alpha=alpha, max_iter=6000)
        model.fit(X[tr], y[tr])
        mse.append(mean_squared_error(y[te], model.predict(X[te])))
    mse = float(np.mean(mse))
    learner = Lasso(alpha=alpha, max_iter=6000)
    theta, se = fit_plr(X, d, y, learner, clone(learner), seed=6)
    rows.append({"alpha": alpha, "outcome CV-MSE": round(mse, 4), "theta": round(theta, 3), "se": round(se, 3)})
    if mse < best_mse:
        best_alpha, best_mse = alpha, mse
print("Lowest outcome CV-MSE at alpha", best_alpha)
pd.DataFrame(rows)
'''),
            M("""
## Summary

Pipelines, averages, and stacks are nuisance functions. The causal step is still the partialling-out score, which `PartiallyLinearDML` implements for ridge and random forests. On the simulation the estimate stays near the true value 0.8. On the 401(k) extract the same score estimates the eligibility effect.

## Resources

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
"""),
        ],
    )


if __name__ == "__main__":
    nb07()
    nb08()

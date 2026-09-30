"""Second half of the meta-learner notebook builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "meta_nb",
    Path(__file__).with_name("build_meta_learner_notebooks.py"),
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

INSTALL = mod.INSTALL_CODE
IMPORTS = mod.IMPORTS
METRICS = mod.METRICS
ABORTION = mod.ABORTION_LOAD
ABORTION_COLUMNS = mod.ABORTION_COLUMNS


def _chunks(name, extra=None):
    chunks = mod.load_markdown(name)
    if extra:
        chunks = [extra(c) for c in chunks]
    return chunks


def build_multi():
    name = "02_08_05_02_02_meta_learners_multiple_discrete_treatment.ipynb"
    part0 = INSTALL + "\n" + IMPORTS + "\n" + METRICS + r'''
base = synthetic_data(mode=2, n=900, p=6, sigma=1.0, random_state=1)
X0, y0, tau0 = base["X"], base["y"], base["tau"]
w0 = base["w"].copy()
treated = np.flatnonzero(w0 == 1)
pick = np.random.default_rng(1).choice(treated, size=int(0.2 * len(treated)), replace=False)
w0[pick] = 2
names0 = [f"X{i+1}" for i in range(X0.shape[1])]
print("Arm counts", {int(a): int((w0 == a).sum()) for a in np.unique(w0)})
print("Latent tau is shared across arms in Part 0; mean", round(float(tau0.mean()), 3))
'''
    summary = r'''
def contrast(X, w, y, arm, control=0):
    mask = (w == control) | (w == arm)
    return X[mask], (w[mask] == arm).astype(int), y[mask]

rows = []
fitted_part0 = {}
for arm in (1, 2):
    Xc, wc, yc = contrast(X0, w0, y0, arm)
    for label, cls in (("S", SLearner), ("T", TLearner), ("X", XLearner), ("R", RLearner)):
        model = cls(learner="gbr", n_estimators=N_EST, random_state=arm)
        model.fit(Xc, wc, yc, feature_names=names0)
        fitted_part0[(label, arm)] = (model, Xc)
        row = ate_row(f"{label}-Learner arm {arm}", model, Xc)
        row["true_ate"] = float(tau0[(w0 == 0) | (w0 == arm)].mean())
        rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, arm in zip(axes, (1, 2)):
    for label in ("T", "X", "R"):
        model, Xc = fitted_part0[(label, arm)]
        ax.hist(model.predict(Xc), bins=25, histtype="step", density=True, label=label)
    ax.hist(tau0[(w0 == 0) | (w0 == arm)], bins=25, histtype="step", density=True, color="black", label="Actual")
    ax.set_title(f"Arm {arm} vs control")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
'''
    multi = r'''
draw = make_multiarm_data(n=900, random_state=11)
Xa, ya, wa = draw["X"], draw["y"], draw["w"]
tau1, tau2 = draw["tau1"], draw["tau2"]
names_a = [f"X{i+1}" for i in range(Xa.shape[1])]
print("Arm counts", {int(a): int((wa == a).sum()) for a in np.unique(wa)})
print("True mean contrasts", round(float(tau1.mean()), 3), round(float(tau2.mean()), 3))
'''
    dr = r'''
arm_models = {}
cmp_rows = []
truth = {1: tau1, 2: tau2}
for arm in (1, 2):
    Xc, wc, yc = contrast(Xa, wa, ya, arm)
    true = truth[arm][(wa == 0) | (wa == arm)]
    for label, cls in (("X", XLearner), ("DR", DRLearner), ("T", TLearner), ("R", RLearner)):
        model = cls(learner="gbr", n_estimators=N_EST, random_state=10 + arm)
        model.fit(Xc, wc, yc, feature_names=names_a)
        pred = model.predict(Xc)
        arm_models[(label, arm)] = model
        cmp_rows.append({
            "learner": label,
            "arm": arm,
            "ate": float(np.mean(pred)),
            "true_ate": float(np.mean(true)),
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        })
print(pd.DataFrame(cmp_rows).round(3).to_string(index=False))
'''
    viz = r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
order = np.argsort(Xa[:, 0])
for ax, arm, truth_arm in ((axes[0], 1, tau1), (axes[1], 2, tau2)):
    model = arm_models[("X", arm)]
    pred = model.predict(Xa)
    ax.scatter(Xa[order, 0], truth_arm[order], s=8, alpha=0.35, label="True")
    ax.scatter(Xa[order, 0], pred[order], s=8, alpha=0.35, label="X-Learner")
    ax.set_xlabel("X1")
    ax.set_title(f"Arm {arm} vs control")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
'''
    holdout = r'''
rng = np.random.default_rng(24)
hold = rng.choice(len(ya), size=int(0.3 * len(ya)), replace=False)
train = np.setdiff1d(np.arange(len(ya)), hold)
hold_rows = []
for arm in (1, 2):
    mask = (wa == 0) | (wa == arm)
    tr = train[mask[train]]
    te = hold[mask[hold]]
    Xtr, wtr, ytr = Xa[tr], (wa[tr] == arm).astype(int), ya[tr]
    Xte = Xa[te]
    true = truth[arm][te]
    for label, cls in (("X", XLearner), ("DR", DRLearner)):
        model = cls(learner="gbr", n_estimators=N_EST, random_state=24)
        model.fit(Xtr, wtr, ytr, feature_names=names_a)
        pred = model.predict(Xte)
        hold_rows.append({
            "learner": label, "arm": arm,
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        })
print(pd.DataFrame(hold_rows).round(3).to_string(index=False))
'''
    overlap = r'''
from sklearn.linear_model import LogisticRegression
prop_model = LogisticRegression(max_iter=400)
# one-vs-rest propensities on the full sample
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, arm in zip(axes, (0, 1, 2)):
    clf = LogisticRegression(max_iter=400)
    clf.fit(Xa, (wa == arm).astype(int))
    score = clf.predict_proba(Xa)[:, 1]
    ax.hist(score, bins=25, color="#2E86AB")
    ax.set_title(f"P(T={arm} | X)")
fig.tight_layout()
plt.show()
print("Share of scores outside (0.05, 0.95):")
for arm in (0, 1, 2):
    clf = LogisticRegression(max_iter=400).fit(Xa, (wa == arm).astype(int))
    score = clf.predict_proba(Xa)[:, 1]
    print(arm, round(float(((score < 0.05) | (score > 0.95)).mean()), 3))
'''
    balance = r'''
def smd(x, mask_t, mask_c, weights=None):
    xt, xc = x[mask_t], x[mask_c]
    if weights is None:
        wt, wc = np.ones(len(xt)), np.ones(len(xc))
    else:
        wt, wc = weights[mask_t], weights[mask_c]
    mt = np.average(xt, weights=wt)
    mc = np.average(xc, weights=wc)
    vt = np.average((xt - mt) ** 2, weights=wt)
    vc = np.average((xc - mc) ** 2, weights=wc)
    return (mt - mc) / np.sqrt(0.5 * (vt + vc) + 1e-8)

# multinomial logit weights
logits = np.column_stack([
    np.zeros(len(Xa)),
    Xa[:, 0],
    -Xa[:, 0],
])
logits -= logits.max(axis=1, keepdims=True)
prob = np.exp(logits)
prob /= prob.sum(axis=1, keepdims=True)
ipw = np.ones(len(Xa))
for arm in (1, 2):
    ipw[wa == arm] = prob[wa == arm, 0] / np.clip(prob[wa == arm, arm], 1e-3, None)
rows = []
for arm in (1, 2):
    for j, name in enumerate(names_a):
        rows.append({
            "arm": arm, "covariate": name,
            "smd": smd(Xa[:, j], wa == arm, wa == 0),
            "smd_ipw": smd(Xa[:, j], wa == arm, wa == 0, ipw),
        })
smd_df = pd.DataFrame(rows)
print(smd_df.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
plot_s = smd_df[smd_df.arm == 1]
ax.bar(np.arange(len(plot_s)) - 0.15, plot_s.smd, width=0.3, label="Unweighted")
ax.bar(np.arange(len(plot_s)) + 0.15, plot_s.smd_ipw, width=0.3, label="IPW")
ax.set_xticks(np.arange(len(plot_s)))
ax.set_xticklabels(plot_s.covariate)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Arm 1 vs control: standardized mean difference")
ax.legend()
fig.tight_layout()
plt.show()
'''
    calib = r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, arm, truth_arm in ((axes[0], 1, tau1), (axes[1], 2, tau2)):
    pred = arm_models[("X", arm)].predict(Xa)
    bins = pd.qcut(pred, 8, duplicates="drop")
    tab = pd.DataFrame({"pred": pred, "true": truth_arm, "bin": bins}).groupby("bin", observed=True).mean()
    ax.plot(tab.pred, tab.true, marker="o")
    lim = [min(tab.pred.min(), tab.true.min()), max(tab.pred.max(), tab.true.max())]
    ax.plot(lim, lim, color="grey", lw=0.8)
    ax.set_title(f"Arm {arm} calibration")
    ax.set_xlabel("Mean predicted CATE")
    ax.set_ylabel("Mean true CATE")
fig.tight_layout()
plt.show()
'''
    shap_fit = r'''
explanations = {}
for arm in (1, 2):
    Xc, wc, yc = contrast(Xa, wa, ya, arm)
    model = XLearner(learner="xgb", n_estimators=N_EST, random_state=arm)
    model.fit(Xc, wc, yc, feature_names=names_a)
    explanations[arm] = explain_cate(model, Xc, max_samples=20, n_permutations=5, random_state=arm)
    print("Arm", arm, np.round(explanations[arm].importance(), 3))
'''
    shap_plot = r'''
for arm, exp in explanations.items():
    exp.plot_importance(title=f"X-Learner SHAP importance, arm {arm}")
    plt.show()
'''
    lung = r'''
lung = pd.read_csv(tutorial_file("lung.csv"))
lung = lung.dropna(subset=["age", "sex", "ph.ecog", "time"]).copy()
lung = lung[lung["time"] > 0]
rng = np.random.default_rng(123)
lung["treatment"] = rng.choice([0, 1, 2], size=len(lung), p=[0.4, 0.3, 0.3])
lung["health_status"] = rng.binomial(
    1, 0.30 + 0.10 * (lung["treatment"] == 1) + 0.15 * (lung["treatment"] == 2)
)
lung["log_time"] = np.log(lung["time"])
X_lung = lung[["age", "sex", "ph.ecog"]].to_numpy(dtype=float)
w_lung = lung["treatment"].to_numpy(dtype=int)
lung_names = ["age", "sex", "ph.ecog"]
print(lung["treatment"].value_counts().sort_index().to_string())
print(lung.groupby("treatment")[["health_status", "log_time"]].mean().round(3))
'''
    lung_fit = r'''
def lung_table(y, outcome):
    rows = []
    for arm, contrast_name in ((1, "A vs placebo"), (2, "B vs placebo")):
        Xc, wc, yc = contrast(X_lung, w_lung, y, arm)
        for label, cls in (("S", SLearner), ("T", TLearner), ("X", XLearner), ("R", RLearner)):
            model = cls(learner="gbr", n_estimators=N_EST, random_state=3)
            model.fit(Xc, wc, yc, feature_names=lung_names)
            row = ate_row(f"{label}-Learner", model, Xc)
            row["outcome"] = outcome
            row["contrast"] = contrast_name
            rows.append(row)
    return pd.DataFrame(rows)

lung_ate = pd.concat([
    lung_table(lung["log_time"].to_numpy(dtype=float), "log_time"),
    lung_table(lung["health_status"].to_numpy(dtype=float), "health_status"),
], ignore_index=True)
print(lung_ate.round(3).to_string(index=False))
'''
    lung_shap = r'''
Xc, wc, yc = contrast(X_lung, w_lung, lung["log_time"].to_numpy(dtype=float), 1)
lung_x = XLearner(learner="xgb", n_estimators=N_EST, random_state=5)
lung_x.fit(Xc, wc, yc, feature_names=lung_names)
lung_exp = explain_cate(lung_x, Xc, max_samples=20, n_permutations=5, random_state=5)
lung_exp.plot_importance(title="Lung data: X-Learner SHAP (A vs placebo, log time)")
plt.show()
lung_exp.plot_waterfall(row=0)
plt.show()
'''
    inserts = [
        ("## Part 0:", part0),
        ("### Summary table (multiple treatment ATE)", summary),
        ("Generate synthetic multi-arm data", multi),
        ("### DR-Learner for multiple treatments", dr),
        ("### Visualization: heterogeneity across arms", viz),
        ("## Part B: Diagnostics", holdout),
        ("### Overlap diagnostics", overlap),
        ("### Balance assessment", balance),
        ("### CATE calibration", calib),
        ("## Part C: SHAP", shap_fit),
        ("### SHAP: beeswarm", shap_plot),
        ("## Part D: Use case with Lung", lung),
        ("#### T-, X-, and R-Learners", lung_fit),
        ("### SHAP for per-arm X-Learner", lung_shap),
    ]
    mod.finish(name, mod.assemble(_chunks(name), inserts, prefix=mod.banner(name)))


def build_continuous():
    name = "02_08_05_02_03_meta_learners_continuous_treatment.ipynb"

    def edit(c):
        return c.replace('treatment_type = "continuous"', "a continuous dose")

    sim = INSTALL + "\n" + IMPORTS + r'''
rng = np.random.default_rng(123)
n, p = 1000, 6
X = rng.normal(size=(n, p))
dose = np.clip(rng.normal(1.4 + 0.35 * X[:, 0], 0.55, size=n), 0.05, 4.0)
modifier = 1.0 + 0.45 * X[:, 0]
# Marginal slope (1.15 - 0.30 * d) * modifier. Outcome is the integral plus noise.
y = (
    0.5 * X[:, 1]
    + dose * modifier * (1.15 - 0.15 * dose)
    + rng.normal(scale=0.45, size=n)
)
true_slope = (1.15 - 0.30 * dose) * modifier
names = [f"X{i+1}" for i in range(p)]
grid = np.linspace(np.quantile(dose, 0.08), np.quantile(dose, 0.92), 11)
print(X.shape, "dose mean", round(float(dose.mean()), 3))
'''
    adr = r'''
from sklearn.ensemble import GradientBoostingRegressor

mu = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=0)
mu.fit(np.column_stack([X, dose]), y)
level = [float(mu.predict(np.column_stack([X, np.full(len(X), d)])).mean()) for d in grid]

m_y = np.zeros(len(y))
m_d = np.zeros(len(y))
rng_fold = np.random.default_rng(0)
fold = rng_fold.integers(0, 2, size=len(y))
for k in (0, 1):
    tr, te = fold != k, fold == k
    fy = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=0).fit(X[tr], y[tr])
    fd = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=1).fit(X[tr], dose[tr])
    m_y[te] = fy.predict(X[te])
    m_d[te] = fd.predict(X[te])
y_res, d_res = y - m_y, dose - m_d
bw = 1.06 * np.std(dose) * len(dose) ** (-0.2)

def local_slope(mask=None):
    dd = dose if mask is None else dose[mask]
    yr = y_res if mask is None else y_res[mask]
    dr = d_res if mask is None else d_res[mask]
    out = []
    for g in grid:
        wt = np.exp(-0.5 * ((dd - g) / bw) ** 2)
        out.append(float(np.sum(wt * dr * yr) / np.sum(wt * dr ** 2)))
    return np.asarray(out)

slope_hat = local_slope()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(grid, level, marker="o")
axes[0].set_title("S-learner average dose response")
axes[0].set_xlabel("Dose")
axes[0].set_ylabel("Mean predicted outcome")
axes[1].plot(grid, slope_hat, marker="o", label="R-learner")
axes[1].plot(grid, [((1.15 - 0.30 * g) * modifier).mean() for g in grid], color="black", label="True average slope")
axes[1].set_title("Average marginal slope")
axes[1].set_xlabel("Dose")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
'''
    hetero = r'''
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for mask, label in ((X[:, 0] > 0, "X1 > 0"), (X[:, 0] <= 0, "X1 ≤ 0")):
    ax.plot(grid, local_slope(mask), marker="o", label=label)
ax.set_xlabel("Dose")
ax.set_ylabel("Local slope")
ax.set_title("Dose-response slope by X1")
ax.legend()
fig.tight_layout()
plt.show()
'''
    native = r'''
d0, d1 = np.quantile(dose, [0.25, 0.75])
s_mu = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=2)
s_mu.fit(np.column_stack([X, dose]), y)
s_contrast = s_mu.predict(np.column_stack([X, np.full(len(X), d1)])) - s_mu.predict(np.column_stack([X, np.full(len(X), d0)]))
high = (dose > np.median(dose)).astype(int)
native_rows = [{"learner": "S contrast (q75-q25)", "mean": float(s_contrast.mean())}]
for label, cls in (("T median split", TLearner), ("X median split", XLearner), ("DR median split", DRLearner)):
    model = cls(learner="gbr", n_estimators=N_EST, random_state=2)
    model.fit(X, high, y, feature_names=names)
    native_rows.append({"learner": label, "mean": float(np.mean(model.predict(X)))})
plr = PartiallyLinearDML(discrete_treatment=False, model_y="gbr", n_estimators=N_EST, random_state=2)
plr.fit(X, dose, y, feature_names=names)
native_rows.append({"learner": "Partially linear DML slope", "mean": float(plr.effect())})
print(pd.DataFrame(native_rows).round(3).to_string(index=False))
print("True mean slope", round(float(true_slope.mean()), 3))
'''
    diag = r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(dose, bins=30, color="#2E86AB")
axes[0].set_title("Marginal dose")
axes[1].hist(m_d, bins=30, color="#1F4E79")
axes[1].set_title("Estimated E[D | X]")
fig.tight_layout()
plt.show()

quartiles = pd.qcut(dose, 4, labels=["Q1", "Q2", "Q3", "Q4"])
rows = []
for q in ["Q1", "Q2", "Q3", "Q4"]:
    mask = quartiles == q
    ref = quartiles == "Q1"
    for j, name in enumerate(names):
        rows.append({
            "quartile": q,
            "covariate": name,
            "smd_vs_q1": float((X[mask, j].mean() - X[ref, j].mean()) / (X[:, j].std() + 1e-8)),
        })
print(pd.DataFrame(rows).pivot(index="covariate", columns="quartile", values="smd_vs_q1").round(3))
'''
    calib = r'''
ref = float(np.median(dose))
pred_at_ref = np.array([
    local_slope()[np.argmin(np.abs(grid - ref))]
] * len(X))
# unit-level slope proxy: residual ratio, shrunk
unit_slope = y_res * d_res / (d_res ** 2 + 0.15)
bins = pd.qcut(unit_slope, 6, duplicates="drop")
tab = pd.DataFrame({"pred": unit_slope, "actual": true_slope, "bin": bins}).groupby("bin", observed=True).mean()
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(tab.pred, tab.actual, marker="o")
ax.set_xlabel("Mean residual slope proxy")
ax.set_ylabel("Mean true slope")
ax.set_title("Calibration of the local slope proxy")
fig.tight_layout()
plt.show()
'''
    nmes_fit = r'''
nmes = pd.read_csv(tutorial_file("nmes_data.csv"))
print(nmes.shape)
print(nmes[["packyears", "TOTALEXP"]].describe().round(2))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(nmes["packyears"], bins=30, color="#2E86AB")
axes[0].set_title("Pack-years")
axes[1].hist(np.log1p(nmes["TOTALEXP"]), bins=30, color="#1F4E79")
axes[1].set_title("log(1 + total expenditure)")
fig.tight_layout()
plt.show()

rng = np.random.default_rng(4)
take = rng.choice(len(nmes), size=1200, replace=False)
sub = nmes.iloc[take].reset_index(drop=True)
covars = [c for c in sub.columns if c not in ("TOTALEXP", "packyears")]
Xn = sub[covars].to_numpy(dtype=float)
dn = sub["packyears"].to_numpy(dtype=float)
yn = np.log1p(sub["TOTALEXP"].to_numpy(dtype=float))
nmes_names = covars
grid_n = np.linspace(np.quantile(dn, 0.05), np.quantile(dn, 0.95), 9)
mu_n = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=4)
mu_n.fit(np.column_stack([Xn, dn]), yn)
level_n = [float(mu_n.predict(np.column_stack([Xn, np.full(len(Xn), d)])).mean()) for d in grid_n]
print("NMES subsample", len(sub), "covariates", covars)
'''
    nmes_plot = r'''
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(grid_n, level_n, marker="o")
ax.set_xlabel("Pack-years")
ax.set_ylabel("Mean predicted log expenditure")
ax.set_title("NMES: S-learner average dose response")
fig.tight_layout()
plt.show()
'''
    nmes_overlap = r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(dn, bins=30, color="#2E86AB")
axes[0].set_title("Pack-years in the subsample")
male = sub["MALE"].to_numpy()
axes[1].hist(dn[male == 0], bins=25, density=True, histtype="step", label="Female")
axes[1].hist(dn[male == 1], bins=25, density=True, histtype="step", label="Male")
axes[1].set_title("Dose by sex")
axes[1].legend()
fig.tight_layout()
plt.show()
'''
    nmes_meta = r'''
high = (dn > np.median(dn)).astype(int)
rows = []
stored = {}
for label, cls in (("S", SLearner), ("T", TLearner), ("X", XLearner), ("R", RLearner), ("DR", DRLearner)):
    model = cls(learner="gbr", n_estimators=N_EST, random_state=4)
    model.fit(Xn, high, yn, feature_names=nmes_names)
    stored[label] = model
    rows.append({"learner": f"{label} high vs low pack-years", "mean_cate": float(np.mean(model.predict(Xn)))})
plr_n = PartiallyLinearDML(discrete_treatment=False, model_y="gbr", n_estimators=N_EST, random_state=4)
plr_n.fit(Xn, dn, yn, feature_names=nmes_names)
rows.append({"learner": "Partially linear DML (per pack-year)", "mean_cate": float(plr_n.effect())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
'''
    nmes_dml = r'''
lin = LinearDML(discrete_treatment=False, n_estimators=N_EST, model_y="gbr", random_state=4)
lin.fit(Xn, dn, yn, feature_names=nmes_names)
sparse = SparseLinearDML(discrete_treatment=False, n_estimators=N_EST, model_y="gbr", random_state=4, alpha=0.05)
sparse.fit(Xn, dn, yn, feature_names=nmes_names)
forest = CausalForestDML(num_trees=40, n_splits=2, random_state=4)
forest.fit(Xn, high, yn, feature_names=nmes_names)
rows = []
for label, model in (("LinearDML", lin), ("SparseLinearDML", sparse), ("CausalForestDML (median split)", forest)):
    info = model.average_treatment_effect()
    rows.append({"learner": label, "ate": info["estimate"], "se": info["std.err"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
spear = pd.DataFrame({
    "linear": lin.predict(Xn),
    "sparse": sparse.predict(Xn),
    "forest": forest.predict(Xn),
}).corr(method="spearman")
print("Spearman correlation of predicted effects")
print(spear.round(3))
'''
    nmes_shap = r'''
exp_t = explain_cate(stored["T"], Xn, max_samples=20, n_permutations=5, feature_names=nmes_names, random_state=4)
print(pd.Series(exp_t.importance(), index=exp_t.feature_names).round(3).sort_values(ascending=False).to_string())
'''
    nmes_shap_plot = r'''
exp_t.plot_importance(title="NMES T-Learner SHAP (high vs low pack-years)")
plt.show()
exp_t.plot_waterfall(row=int(np.argmax(np.abs(exp_t.predictions))))
plt.show()
'''
    inserts = [
        ("## Setup", INSTALL),
        ("## Part A:", sim),
        ("### Average Dose-Response Curve", adr),
        ("### Heterogeneous Dose-Response", hetero),
        ("### Native S/T/X/R/DR", native),
        ("### Diagnostics & Validation Protocol", diag),
        ("#### CATE Calibration Plot", calib),
        ("### Prepare analysis data", nmes_fit),
        ("### Average estimated local slope", nmes_plot),
        ("### Quick overlap check", nmes_overlap),
        ("### Meta-learners on NMES", nmes_meta),
        ("### Use case: Double machine learning", nmes_dml),
        ("### SHAP interpretability for predicted CATE", nmes_shap),
        ("### SHAP beeswarm and waterfall", nmes_shap_plot),
    ]
    # sim already includes INSTALL, and Setup also installs. Drop the duplicate by not putting INSTALL in sim.
    sim = IMPORTS + sim.split(IMPORTS, 1)[-1]
    inserts[1] = ("## Part A:", sim)
    mod.finish(name, mod.assemble(_chunks(name, edit), inserts, prefix=mod.banner(name)))


def build_tmle():
    name = "02_08_05_02_04_tmle_model.ipynb"
    setup = INSTALL + "\n" + IMPORTS + r'''
def tmle_ate(X, a, y, seed=0):
    """Continuous-outcome TMLE: initial Q and g, then a one-step fluctuation."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    a = np.asarray(a).astype(int)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    q_model = GradientBoostingRegressor(n_estimators=N_EST, max_depth=2, random_state=seed)
    q_model.fit(np.column_stack([X, a]), y)
    g_model = GradientBoostingClassifier(n_estimators=N_EST, max_depth=2, random_state=seed)
    g_model.fit(X, a)
    g = np.clip(g_model.predict_proba(X)[:, 1], 0.02, 0.98)
    q_hat = q_model.predict(np.column_stack([X, a]))
    h = a / g - (1 - a) / (1 - g)
    eps = float(np.dot(h, y - q_hat) / np.dot(h, h))
    q1 = q_model.predict(np.column_stack([X, np.ones(len(X))])) + eps / g
    q0 = q_model.predict(np.column_stack([X, np.zeros(len(X))])) + eps * (-1.0 / (1 - g))
    ate = float(np.mean(q1 - q0))
    influence = (q1 - q0) + h * (y - (q_hat + eps * h))
    se = float(np.std(influence, ddof=1) / np.sqrt(len(y)))
    return {"ate": ate, "se": se, "ci_lower": ate - 1.96 * se, "ci_upper": ate + 1.96 * se, "q1": q1, "q0": q0}

def cumgain(score, truth):
    order = np.argsort(-np.asarray(score, dtype=float))
    truth = np.asarray(truth, dtype=float)[order]
    cum = np.cumsum(truth)
    frac = np.arange(1, len(truth) + 1) / len(truth)
    norm = cum / cum[-1] if abs(cum[-1]) > 1e-8 else cum
    return frac, cum, norm

def qini_curve(y, w, score):
    order = np.argsort(-np.asarray(score, dtype=float))
    y = np.asarray(y, dtype=float)[order]
    w = np.asarray(w).astype(int)[order]
    nt = np.cumsum(w == 1)
    nc = np.cumsum(w == 0)
    cum_t = np.cumsum(y * (w == 1))
    cum_c = np.cumsum(y * (w == 0))
    q = np.where(nc > 0, cum_t - cum_c * nt / np.maximum(nc, 1), 0.0)
    return np.arange(1, len(y) + 1), q
'''
    data = r'''
dat = synthetic_data(mode=2, n=1600, p=10, sigma=1.5, random_state=42)
X, y, w, tau = dat["X"], dat["y"], dat["w"], dat["tau"]
rng = np.random.default_rng(42)
test = rng.choice(len(y), size=len(y) // 2, replace=False)
train = np.setdiff1d(np.arange(len(y)), test)
names = [f"X{i+1}" for i in range(X.shape[1])]
print("Train", len(train), "test", len(test), "true ATE", round(float(tau.mean()), 3))
'''
    ite = r'''
learner_x = XLearner(learner="rf", n_estimators=N_EST, random_state=42)
learner_x.fit(X[train], w[train], y[train], feature_names=names)
cate_test = learner_x.predict(X[test])
print("Test mean predicted CATE", round(float(cate_test.mean()), 3), "true", round(float(tau[test].mean()), 3))
'''
    dist = r'''
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(cate_test, bins=30, density=True, alpha=0.55, label="X-Learner")
ax.hist(tau[test], bins=30, density=True, alpha=0.45, label="Actual")
ax.set_title("Test-set CATE")
ax.legend()
fig.tight_layout()
plt.show()
'''
    with_truth = r'''
frac, cum, norm = cumgain(cate_test, tau[test])
frac_opt, cum_opt, norm_opt = cumgain(tau[test], tau[test])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(frac, norm, label="X-Learner")
ax.plot(frac_opt, norm_opt, color="black", label="Sorted by true effect")
ax.plot([0, 1], [0, 1], color="grey", lw=0.8, label="Random")
ax.set_xlabel("Fraction targeted")
ax.set_ylabel("Normalized cumulative gain")
ax.set_title("Uplift curve with ground truth")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
'''
    no_truth = r'''
frac_obs, gain_obs = None, None
order = np.argsort(-cate_test)
yy, ww = y[test][order], w[test][order]
n = len(yy)
gain_obs = np.empty(n)
for i in range(n):
    n1 = int((ww[: i + 1] == 1).sum())
    n0 = i + 1 - n1
    gain_obs[i] = 0.0 if n1 == 0 or n0 == 0 else (yy[: i + 1][ww[: i + 1] == 1].mean() - yy[: i + 1][ww[: i + 1] == 0].mean())
frac_obs = np.arange(1, n + 1) / n
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(frac_obs[::10], gain_obs[::10])
ax.set_title("Uplift from observed outcomes only")
ax.set_xlabel("Fraction targeted")
ax.set_ylabel("Cumulative outcome difference")
fig.tight_layout()
plt.show()
'''
    tmle_head = r'''
tmle_test = tmle_ate(X[test], w[test], y[test], seed=1)
print(pd.Series({k: tmle_test[k] for k in ("ate", "se", "ci_lower", "ci_upper")}).round(3))
print("X-Learner mean CATE on the test set", round(float(cate_test.mean()), 3))
'''
    tmle_curve = r'''
order = np.argsort(-cate_test)
n_seg = 5
segments = np.array_split(order, n_seg)
seg_ate = []
for seg in segments:
    fit = tmle_ate(X[test][seg], w[test][seg], y[test][seg], seed=2)
    seg_ate.append(fit["ate"])
seg_ate = np.asarray(seg_ate)
cum_seg = np.cumsum(seg_ate)
norm_seg = cum_seg / cum_seg[-1] if abs(cum_seg[-1]) > 1e-8 else cum_seg
frac_seg = np.arange(1, n_seg + 1) / n_seg
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(frac_seg, norm_seg, marker="o", color="purple")
ax.set_title("Segment TMLE uplift curve")
ax.set_xlabel("Fraction targeted (5 segments)")
ax.set_ylabel("Normalized cumulative TMLE")
fig.tight_layout()
plt.show()
print("Segment TMLE ATEs", np.round(seg_ate, 3))
'''
    auuc = r'''
auuc_truth = float(np.trapezoid(norm, frac))
auuc_tmle = float(np.trapezoid(norm_seg, frac_seg))
print(pd.DataFrame([
    {"curve": "X-Learner vs true tau", "auuc": auuc_truth},
    {"curve": "Segment TMLE", "auuc": auuc_tmle},
]).round(3).to_string(index=False))
'''
    qini = r'''
xs, q_model = qini_curve(y[test], w[test], cate_test)
xs_t, q_true = qini_curve(y[test], w[test], tau[test])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(xs, q_model, label="X-Learner")
ax.plot(xs_t, q_true, color="black", label="True effect")
ax.set_title("Qini curve")
ax.set_xlabel("Units targeted")
ax.set_ylabel("Cumulative incremental outcome")
ax.legend()
fig.tight_layout()
plt.show()
print("Qini score, model", round(float(np.trapezoid(q_model, xs) / len(xs)), 3))
print("Qini score, oracle", round(float(np.trapezoid(q_true, xs_t) / len(xs_t)), 3))
qini_tmle = np.cumsum(seg_ate)
print("Segment-TMLE Qini path", np.round(qini_tmle, 3))
'''
    qini_plot = r'''
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(np.arange(1, n_seg + 1) * (len(test) / n_seg), qini_tmle, marker="o", color="purple")
ax.set_title("Qini curve from segment TMLE")
ax.set_xlabel("Units targeted")
ax.set_ylabel("Cumulative segment TMLE")
fig.tight_layout()
plt.show()
'''
    lalonde = r'''
from causaldata import nsw_mixtape
nsw = nsw_mixtape.load_pandas().data.copy()
print("NSW / LaLonde columns:", list(nsw.columns))
print(nsw["data_id"].value_counts().to_string())
nsw = nsw[nsw["data_id"] == "Dehejia-Wahba Sample"].copy()
lal_features = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
X_l = nsw[lal_features].to_numpy(dtype=float)
y_l = nsw["re78"].to_numpy(dtype=float)
w_l = nsw["treat"].to_numpy(dtype=int)
print("n", len(y_l), "treated", int(w_l.sum()))
'''
    lal_split = r'''
rng = np.random.default_rng(42)
te = rng.choice(len(y_l), size=len(y_l) // 2, replace=False)
tr = np.setdiff1d(np.arange(len(y_l)), te)
learner_lal = XLearner(learner="rf", n_estimators=N_EST, random_state=42)
learner_lal.fit(X_l[tr], w_l[tr], y_l[tr], feature_names=lal_features)
cate_lal = learner_lal.predict(X_l[te])
print("Test mean CATE", round(float(cate_lal.mean()), 1))
'''
    lal_uplift = r'''
order = np.argsort(-cate_lal)
yy, ww = y_l[te][order], w_l[te][order]
gain = np.empty(len(yy))
for i in range(len(yy)):
    n1 = int((ww[: i + 1] == 1).sum())
    n0 = i + 1 - n1
    gain[i] = 0.0 if n1 == 0 or n0 == 0 else yy[: i + 1][ww[: i + 1] == 1].mean() - yy[: i + 1][ww[: i + 1] == 0].mean()
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(np.arange(1, len(gain) + 1) / len(gain), gain)
ax.set_title("NSW: uplift from observed earnings")
ax.set_xlabel("Fraction targeted")
fig.tight_layout()
plt.show()
'''
    lal_tmle = r'''
overall = tmle_ate(X_l[te], w_l[te], y_l[te], seed=3)
print("Test-set TMLE", {k: round(overall[k], 1) for k in ("ate", "se", "ci_lower", "ci_upper")})
order = np.argsort(-cate_lal)
n_seg_l = 5
seg_ate_l = []
for seg in np.array_split(order, n_seg_l):
    seg_ate_l.append(tmle_ate(X_l[te][seg], w_l[te][seg], y_l[te][seg], seed=4)["ate"])
seg_ate_l = np.asarray(seg_ate_l)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
frac = np.arange(1, n_seg_l + 1) / n_seg_l
cum = np.cumsum(seg_ate_l)
axes[0].plot(frac, cum / cum[-1] if abs(cum[-1]) > 1 else cum, marker="o", color="purple")
axes[0].set_title("NSW segment-TMLE uplift")
axes[1].plot(np.arange(1, n_seg_l + 1), np.cumsum(seg_ate_l), marker="o", color="purple")
axes[1].set_title("NSW segment-TMLE Qini path")
fig.tight_layout()
plt.show()
print("Segment ATEs", np.round(seg_ate_l, 1))
'''
    inserts = [
        ("## Implementation in Python", setup),
        ("### Generating Synthetic Data", data),
        ("### Calculating Individual Treatment Effect", ite),
        ("### Distribution of CATE Predictions", dist),
        ("### Uplift Curve With Ground Truth", with_truth),
        ("### Uplift Curve Without Ground Truth", no_truth),
        ("### Uplift Curve with TMLE as Ground Truth", tmle_head),
        ("#### Uplift curve without CI", tmle_curve),
        ("#### AUUC Score", auuc),
        ("#### Qini Score", qini),
        ("#### Qini curve with CI", qini_plot),
        ("## Part B: Use case — LaLonde", lalonde),
        ("### Train–test split and X-Learner", lal_split),
        ("### Uplift from observed outcomes only", lal_uplift),
        ("### TMLE on the test set and by segment", lal_tmle),
    ]
    mod.finish(name, mod.assemble(_chunks(name), inserts, prefix=mod.banner(name)))


def build_da():
    name = "02_08_05_02_05_meta_learners_domain_adaptation_learner.ipynb"
    setup = INSTALL + "\n" + IMPORTS
    data = r'''
rng = np.random.default_rng(123)
n = 1200
X = rng.normal(size=(n, 5))
prop = 1 / (1 + np.exp(-X[:, 0]))
w = rng.binomial(1, prop)
tau = 1.0 + 0.5 * X[:, 0]
y = tau * w + X[:, 0] + rng.normal(scale=0.8, size=n)
names = [f"X{i+1}" for i in range(X.shape[1])]
print("n", n, "p", X.shape[1], "treatment rate", round(float(w.mean()), 3))
print("Population mean of tau is 1 when X1 is standard normal; sample mean", round(float(tau.mean()), 3))
dal = DomainAdaptationLearner(learner="gbr", n_estimators=N_EST, random_state=123)
dal.fit(X, w, y, feature_names=names)
cate = dal.predict(X)
print("In-sample mean CATE", round(float(cate.mean()), 3))
'''
    summarize = r'''
print("Propensity model:", type(dal.propensity_).__name__)
print("Control outcome model:", type(dal.mu0_).__name__)
print("Treated outcome model:", type(dal.mu1_).__name__)
print("Final CATE model:", type(dal.final_).__name__)
'''
    ate = r'''
info = ate_row("DA-Learner", dal, X)
info["true_ate"] = float(tau.mean())
print(pd.Series(info).round(3).to_string())
'''
    # ate_row is in METRICS, which is not in setup. Include METRICS in data cell.
    data = data  # placeholder, patched below
    holdout = r'''
rng = np.random.default_rng(456)
te = rng.choice(n, size=n // 2, replace=False)
tr = np.setdiff1d(np.arange(n), te)
da_h = DomainAdaptationLearner(learner="gbr", n_estimators=N_EST, random_state=456)
t_h = TLearner(learner="gbr", n_estimators=N_EST, random_state=456)
da_h.fit(X[tr], w[tr], y[tr], feature_names=names)
t_h.fit(X[tr], w[tr], y[tr], feature_names=names)
rows = []
for label, model in (("DA-Learner", da_h), ("T-Learner", t_h)):
    pred = model.predict(X[te])
    rows.append({
        "learner": label,
        "rmse": float(np.sqrt(np.mean((pred - tau[te]) ** 2))),
        "mae": float(np.mean(np.abs(pred - tau[te]))),
        "corr": float(np.corrcoef(pred, tau[te])[0, 1]),
    })
hold_pred = {"DA-Learner": da_h.predict(X[te]), "T-Learner": t_h.predict(X[te])}
print(pd.DataFrame(rows).round(3).to_string(index=False))
'''
    scatter = r'''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, label in zip(axes, hold_pred):
    ax.scatter(tau[te], hold_pred[label], s=12, alpha=0.45)
    ax.set_title(label)
    ax.set_xlabel("True CATE")
    ax.set_ylabel("Predicted CATE")
fig.tight_layout()
plt.show()
'''
    shap = r'''
models = {
    "DA-Learner": da_h,
    "T-Learner": t_h,
    "S-Learner": SLearner(learner="gbr", n_estimators=N_EST, random_state=1),
    "X-Learner": XLearner(learner="xgb", n_estimators=N_EST, random_state=1),
}
models["S-Learner"].fit(X[tr], w[tr], y[tr], feature_names=names)
models["X-Learner"].fit(X[tr], w[tr], y[tr], feature_names=names)
explanations = {}
for label, model in models.items():
    explanations[label] = explain_cate(model, X[tr], max_samples=20, n_permutations=5, random_state=1)
    print(label, np.round(explanations[label].importance(), 3))
'''
    shap_plot = r'''
exp = explanations["T-Learner"]
exp.plot_importance(title="T-Learner SHAP importance")
plt.show()
exp.plot_waterfall(row=int(np.argmax(exp.predictions)), title="Waterfall at the highest predicted CATE")
plt.show()
'''
    data = METRICS + "\n" + data
    inserts = [
        ("## Implementation in Python", setup),
        ("### Data", data),
        ("#### Summarize", summarize),
        ("### Average treatment effect", ate),
        ("### Holdout validation", holdout),
        ("#### Scatter: predicted vs true CATE", scatter),
        ("### SHAP for predicted CATE", shap),
        ("### Visualize SHAP", shap_plot),
    ]
    mod.finish(name, mod.assemble(_chunks(name), inserts, prefix=mod.banner(name)))


def build_dr():
    name = "02_08_05_02_06_meta_learners_doubly_robust_learners.ipynb"

    def edit(c):
        if c.strip().startswith("## Implementation in Python"):
            return """## Implementation in Python

PyCausalML fits these doubly robust and double-machine-learning variants with `fit(X, treatment, y)` and `predict(X)`.

| Variant in this notebook | PyCausalML object | Final stage |
|---|---|---|
| Random-forest DR | `DRLearner(learner="rf")` | Random forest of the DR pseudo-outcome |
| Linear DR | `DRLearner(learner="lm")` | OLS of the DR pseudo-outcome |
| Sparse linear DR | Lasso on `dr_scores_` | L1-penalized linear CATE |
| XGBoost DR | `DRLearner(learner="xgb")` | Gradient boosting |
| Linear DML | `LinearDML` | Linear CATE on cross-fit residuals |

`learner` selects the base regressor (`"lm"`, `"rf"`, `"xgb"`, or `"gbr"`). Cross-fitting uses `n_splits`. `average_treatment_effect()` returns a point estimate and a standard error when an influence function is available. For the linear DR-Learner, slopes are the coefficients from regressing the doubly robust pseudo-outcome on `X`.
"""
        if c.strip().startswith("## Part B: Use case") and "abortion" in c.lower():
            return c.rstrip() + "\n\n" + ABORTION_COLUMNS
        if c.strip().startswith("# 5.2.6") or c.strip().startswith("# 2.6"):
            body = "\n".join(c.splitlines()[1:])
            return "# 2.6 Doubly Robust Learners\n" + body
        return c

    sim = INSTALL + "\n" + IMPORTS + "\n" + METRICS + r'''
rng = np.random.default_rng(42)
n = 1200
X_effect = rng.normal(size=(n, 5))
W_conf = rng.normal(size=(n, 3))
X = np.column_stack([X_effect, W_conf])
logit = -0.15 + 0.7 * W_conf[:, 0] - 0.45 * W_conf[:, 1]
prop = 1 / (1 + np.exp(-logit))
treat = rng.binomial(1, prop)
tau = 1.5 * X_effect[:, 0]
y = 0.4 * X_effect[:, 1] + W_conf[:, 0] + treat * tau + rng.normal(size=n)
feature_names = [f"X{i+1}" for i in range(5)] + [f"W{i+1}" for i in range(3)]
print("n", n, "treatment rate", round(float(treat.mean()), 3), "true ATE", round(float(tau.mean()), 3))
print("Mapping: random-forest DR -> DRLearner(learner='rf'); linear DR -> learner='lm';")
print("sparse DR -> Lasso on the doubly robust pseudo-outcome; XGBoost DR -> learner='xgb'.")
'''
    forest = r'''
dr_forest = DRLearner(learner="rf", n_estimators=N_EST, n_splits=2, random_state=42)
dr_forest.fit(X, treat, y, feature_names=feature_names)
print(pd.Series(ate_row("Random-forest DR", dr_forest, X)).round(3).to_string())
'''
    overlap = r'''
from sklearn.ensemble import GradientBoostingClassifier
propensity = GradientBoostingClassifier(n_estimators=N_EST, max_depth=2, random_state=42)
propensity.fit(X, treat)
phat = propensity.predict_proba(X)[:, 1]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.hist(phat[treat == 0], bins=25, alpha=0.65, label="Control")
ax.hist(phat[treat == 1], bins=25, alpha=0.65, label="Treated")
ax.set_xlabel("Estimated propensity")
ax.set_title("Overlap")
ax.legend()
fig.tight_layout()
plt.show()
print("Propensity range", round(float(phat.min()), 3), round(float(phat.max()), 3))
'''
    linear = r'''
from sklearn.linear_model import LinearRegression
dr_lin = DRLearner(learner="lm", n_estimators=N_EST, random_state=42)
dr_lin.fit(X, treat, y, feature_names=feature_names)
coef = LinearRegression().fit(X, dr_lin.dr_scores_)
print(pd.Series(ate_row("Linear DR", dr_lin, X)).round(3).to_string())
print("Final-stage slopes (true effect loads on X1 only)")
print(pd.Series(coef.coef_, index=feature_names).round(3).to_string())
'''
    xgb = r'''
dr_xgb = DRLearner(learner="xgb", n_estimators=N_EST, random_state=42)
dr_xgb.fit(X, treat, y, feature_names=feature_names)
print(pd.Series(ate_row("XGBoost DR", dr_xgb, X)).round(3).to_string())
'''
    generic = r'''
dr_mixed = DRLearner(learner="gbr", n_estimators=N_EST, random_state=42)
dr_mixed.fit(X, treat, y, feature_names=feature_names)
print(pd.Series(ate_row("Gradient-boosting DR", dr_mixed, X)).round(3).to_string())
print("Nuisance outcome models use the same learner; the propensity score is cross-fit with gradient boosting.")
'''
    dml = r'''
from sklearn.linear_model import Lasso
dml_lin = LinearDML(discrete_treatment=True, n_estimators=N_EST, model_y="gbr", random_state=42)
dml_lin.fit(X, treat, y, feature_names=feature_names)
sparse = Lasso(alpha=0.05, max_iter=5000).fit(X, dr_lin.dr_scores_)
print("Linear DML", {k: round(float(v), 3) for k, v in dml_lin.average_treatment_effect().items() if np.ndim(v) == 0})
print("Sparse DR slopes")
print(pd.Series(sparse.coef_, index=feature_names).round(3).to_string())
'''
    compare = r'''
rows = [ate_row(label, model, X) for label, model in (
    ("Random-forest DR", dr_forest),
    ("Linear DR", dr_lin),
    ("XGBoost DR", dr_xgb),
    ("Gradient-boosting DR", dr_mixed),
)]
rows.append({
    "learner": "Linear DML",
    "ate": dml_lin.average_treatment_effect()["estimate"],
    "se": dml_lin.average_treatment_effect()["std.err"],
    "ci_lower": dml_lin.average_treatment_effect()["ci_lower"],
    "ci_upper": dml_lin.average_treatment_effect()["ci_upper"],
})
out = pd.DataFrame(rows)
out["true_ate"] = float(tau.mean())
print(out.round(3).to_string(index=False))
'''
    ab_fit = r'''
dr_models = {}
for label, kwargs in (
    ("Random-forest DR", dict(learner="rf")),
    ("Linear DR", dict(learner="lm")),
    ("XGBoost DR", dict(learner="xgb")),
):
    model = DRLearner(n_estimators=N_EST, n_splits=2, random_state=333, **kwargs)
    model.fit(X_ab_tr, W_ab_tr, Y_ab_tr, feature_names=feature_names)
    dr_models[label] = model
dml_ab = LinearDML(discrete_treatment=True, n_estimators=N_EST, model_y="gbr", random_state=333)
dml_ab.fit(X_ab_tr, W_ab_tr, Y_ab_tr, feature_names=feature_names)
sparse_ab = Lasso(alpha=0.02, max_iter=5000).fit(X_ab_tr, dr_models["Linear DR"].dr_scores_)
print("Sparse DR nonzero covariates", int(np.sum(np.abs(sparse_ab.coef_) > 1e-6)))
'''
    ab_compare = r'''
rows = []
for label, model in dr_models.items():
    row = ate_row(label, model, X_ab_tr)
    row["mean_cate_test"] = float(np.mean(model.predict(X_ab_te)))
    rows.append(row)
info = dml_ab.average_treatment_effect()
rows.append({
    "learner": "Linear DML",
    "ate": info["estimate"],
    "se": info["std.err"],
    "ci_lower": info["ci_lower"],
    "ci_upper": info["ci_upper"],
    "mean_cate_test": float(np.mean(dml_ab.predict(X_ab_te))),
})
print(pd.DataFrame(rows).round(3).to_string(index=False))
'''
    ab_hist = r'''
fig, ax = plt.subplots(figsize=(7.5, 4))
for label, model in dr_models.items():
    ax.hist(model.predict(X_ab_tr), bins=30, histtype="step", density=True, label=label)
ax.set_xlabel("Predicted CATE on lnr")
ax.set_title("Abortion data: training-set CATE")
ax.legend()
fig.tight_layout()
plt.show()
'''
    ab_shap = r'''
ab_exp = explain_cate(
    dr_models["Linear DR"], X_ab_tr, max_samples=24, n_permutations=6,
    feature_names=feature_names, random_state=333,
)
print(pd.Series(ab_exp.importance(), index=ab_exp.feature_names).round(3).sort_values(ascending=False).to_string())
'''
    ab_plot = r'''
ab_exp.plot_importance(title="Abortion data: linear DR-Learner SHAP")
plt.show()
ab_exp.plot_waterfall(row=0, title="Abortion data: linear DR-Learner waterfall")
plt.show()
'''
    inserts = [
        ("## Implementation in Python", sim),
        ("### `ForestDRLearner`: random forest final stage", forest),
        ("Overlap check using fitted propensity", overlap),
        ("### `LinearDRLearner`: OLS final stage", linear),
        ("### `XGBDRLearner` (optional)", xgb),
        ("### Generic `DRLearner`", generic),
        ("### `LinearDML`", dml),
        ("### Quick comparison on this DGP", compare),
        ("### Data Preparation", ABORTION),
        ("We **fit** each learner", ab_fit),
        ("### Comparison (abortion, train / test split)", ab_compare),
        ("### Visualize predicted CATE distribution", ab_hist),
        ("### SHAP for predicted CATE (abortion", ab_shap),
        ("### Visualize SHAP with beeswarm", ab_plot),
    ]
    mod.finish(name, mod.assemble(_chunks(name, edit), inserts, prefix=mod.banner(name)))


if __name__ == "__main__":
    mod.build_intro()
    mod.build_binary()
    build_multi()
    build_continuous()
    build_tmle()
    build_da()
    build_dr()

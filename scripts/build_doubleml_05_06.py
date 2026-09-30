#!/usr/bin/env python3
"""Notebooks 4.5 and 4.6."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_doubleml_02_04 import HELPERS
from build_doubleml_notebooks import BANNER, C, IMPORTS, M, SETUP, write


def nb05():
    write(
        "02_08_05_04_05_double_ml_dynamic_ml.ipynb",
        [
            M(BANNER),
            M("""
# 4.5 Dynamic Double Machine Learning

Standard DML is for a treatment that is assigned once. Many decisions repeat: a treatment at time $t$ depends on the past, the outcome is realized at the end, and the confounders move. Dynamic DML keeps machine-learning nuisances for those confounders and still estimates how treatment in each period shifts the final outcome.
"""),
            M("""
## Overview

The panel for one unit is a short history of controls and treatments and a final outcome $Y$. With a linear effect of each period's treatment,

$$Y = \\sum_{t=1}^{m} \\theta_t T_t + \\gamma^\\top W + \\eta.$$

Cross-fitting residualizes $Y$ and every $T_t$ on the controls. The final stage regresses the outcome residual on the treatment residuals, one coefficient per period. When the effect also depends on baseline covariates $X$,

$$\\theta_t(X) = a_t + X^\\top b_t,$$

the same residuals are interacted with $(1, X)$.

A policy contrast compares a baseline path (all zeros) with a target path (all ones). For a constant effect the contrast is the sum of the period coefficients. For a heterogeneous effect it is $\\sum_t \\hat\\theta_t(X_i)$ for each unit.
"""),
            M("""
## Implementation in Python

Part A builds a balanced panel in wide form (one row per unit, one column per period). Part B uses the hospital deterioration panel in `Data/hospital-deterioration-dataset`: a supervised risk model for `deterioration_next_12h`, then a three-hour dynamic fit with `nurse_alert` as the treatment. The extract is large, so the notebook keeps a patient subsample.
"""),
            M("## Set Up\n"),
            C(SETUP),
            C(IMPORTS),
            C(HELPERS),
            C(r'''
def fit_dynamic(T, controls, y, X=None, kind="ridge", n_splits=2, seed=0):
    """Period-specific linear effects of the columns of T on the final outcome."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    y_res = y - cross_fit_predict(controls, y, n_splits, seed, kind)
    t_res = np.column_stack([
        T[:, j] - cross_fit_predict(controls, T[:, j], n_splits, seed + 1 + j, kind)
        for j in range(T.shape[1])
    ])
    if X is None:
        coef, *_ = np.linalg.lstsq(t_res, y_res, rcond=None)
        return np.asarray(coef, dtype=float)
    X = np.asarray(X, dtype=float)
    phi = np.column_stack([np.ones(len(X)), X])
    design = np.column_stack([t_res[:, j:j + 1] * phi for j in range(T.shape[1])])
    beta, *_ = np.linalg.lstsq(design, y_res, rcond=None)
    return np.asarray(beta, dtype=float).reshape(T.shape[1], phi.shape[1])


def simulate_dynamic(n=900, m=4, seed=0, heterogeneous=False):
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(n, 3))
    baseline = rng.normal(size=(n, 2))
    treatment = rng.binomial(1, 0.45, size=(n, m)).astype(float)
    gamma = np.array([0.5, -0.3, 0.2])
    if not heterogeneous:
        theta = np.linspace(0.4, 1.3, m)
        outcome = treatment @ theta + controls @ gamma + rng.normal(scale=0.5, size=n)
        return treatment, controls, baseline, outcome, theta
    intercepts = np.linspace(0.3, 1.0, m)
    slopes = np.linspace(0.2, 0.7, m)
    effect = intercepts + baseline[:, 0:1] * slopes
    outcome = np.sum(treatment * effect, axis=1) + controls @ gamma + rng.normal(scale=0.5, size=n)
    return treatment, controls, baseline, outcome, intercepts, slopes


print("Dynamic DML helpers ready")
'''),
            M("""
## Part A: Dynamic DML with Synthetic Data

### Average effects

Each unit has four periods. The true period effects increase from 0.4 to 1.3. Controls $W$ confound treatment and the final outcome.
"""),
            C('''
T, W, X, Y, theta_true = simulate_dynamic(seed=1)
coef = fit_dynamic(T, W, Y, kind="ridge", seed=1)
print("True period effects:", np.round(theta_true, 3))
print("Estimated period effects:", np.round(coef, 3))
print("True policy effect (all zeros to all ones):", round(float(theta_true.sum()), 3))
print("Estimated policy effect:", round(float(coef.sum()), 3))
'''),
            C('''
fig, ax = plt.subplots(figsize=(6.2, 4.0))
periods = np.arange(1, len(theta_true) + 1)
ax.plot(periods, theta_true, "o-", color="black", label="True")
ax.plot(periods, coef, "s--", color="#1F4E79", label="Estimated")
ax.set_xlabel("Period")
ax.set_ylabel("Effect on the final outcome")
ax.set_title("Dynamic DML: constant period effects")
ax.legend(frameon=False)
fig.tight_layout()
'''),
            M("""
### Heterogeneous effects

Now $\\theta_t(X) = a_t + b_t X_1$. The final stage is one linear model per period, with an intercept and slopes on the baseline covariates.
"""),
            C('''
Th, Wh, Xh, Yh, a_true, b_true = simulate_dynamic(seed=2, heterogeneous=True)
betas = fit_dynamic(Th, Wh, Yh, X=Xh, kind="ridge", seed=2)
print("Intercepts, true vs estimated:")
print(np.round(np.column_stack([a_true, betas[:, 0]]), 3))
print("Slope on the first baseline covariate, true vs estimated:")
print(np.round(np.column_stack([b_true, betas[:, 1]]), 3))
'''),
            C('''
rng = np.random.default_rng(3)
X_test = rng.normal(size=(400, 2))
phi = np.column_stack([np.ones(len(X_test)), X_test])
cate = phi @ betas.T
policy = cate.sum(axis=1)
print("Mean period effects on the test units:", np.round(cate.mean(axis=0), 3))
print("Mean policy effect:", round(float(policy.mean()), 3))
print("Policy effect, first five units:", np.round(policy[:5], 3))
'''),
            C('''
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.scatter(a_true, betas[:, 0], label="Intercept", s=40)
ax.scatter(b_true, betas[:, 1], label="Slope on X1", s=40)
low = min(a_true.min(), b_true.min(), betas.min()) - 0.1
high = max(a_true.max(), b_true.max(), betas.max()) + 0.1
ax.plot([low, high], [low, high], color="grey", ls="--")
ax.set_xlabel("True parameter")
ax.set_ylabel("Estimated parameter")
ax.set_title("Heterogeneous dynamic DML")
ax.legend(frameon=False)
fig.tight_layout()
'''),
            M("""
## Part B: Hospital clinical deterioration

The files in `Data/hospital-deterioration-dataset` are an hourly cohort: vitals, labs, static attributes, and `deterioration_next_12h`. This notebook reads the bundled hourly panel, keeps a random set of patients, and builds 3-hour rolling means and hour-to-hour changes on that subsample.

### Data processing
"""),
            C('''
hospital = data_dir / "hospital-deterioration-dataset" / "hospital_deterioration_hourly_panel.csv"
usecols = [
    "patient_id", "hour_from_admission", "heart_rate", "respiratory_rate", "spo2_pct",
    "temperature_c", "systolic_bp", "diastolic_bp", "oxygen_device", "oxygen_flow",
    "mobility_score", "nurse_alert", "wbc_count", "lactate", "creatinine", "crp_level",
    "hemoglobin", "sepsis_risk_score", "age", "gender", "comorbidity_index",
    "admission_type", "deterioration_next_12h",
]
raw = pd.read_csv(hospital, usecols=usecols)
rng = np.random.default_rng(11)
patients = raw["patient_id"].drop_duplicates().to_numpy()
keep = set(rng.choice(patients, size=min(600, len(patients)), replace=False))
raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour_from_admission"]).copy()
roll_cols = ["heart_rate", "respiratory_rate", "spo2_pct", "systolic_bp", "lactate", "creatinine"]
for col in roll_cols:
    grouped = raw.groupby("patient_id")[col]
    raw[f"{col}_roll3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    raw[f"{col}_dh"] = grouped.diff().fillna(0.0)
print(raw.shape, "patients", raw["patient_id"].nunique())
raw.head(3)
'''),
            C('''
features = [
    "hour_from_admission", "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
    "systolic_bp", "diastolic_bp", "oxygen_flow", "mobility_score", "wbc_count", "lactate",
    "creatinine", "crp_level", "hemoglobin", "sepsis_risk_score", "age", "comorbidity_index",
] + [f"{c}_roll3" for c in roll_cols] + [f"{c}_dh" for c in roll_cols]
encoded = pd.get_dummies(raw[["oxygen_device", "gender", "admission_type"]], drop_first=True)
work = pd.concat([raw.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
feature_names = features + list(encoded.columns)
work[feature_names] = work[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
print("Feature count", len(feature_names))
'''),
            M("### Train / test split and the supervised risk model\n"),
            C('''
ids = work["patient_id"].drop_duplicates().to_numpy()
rng = np.random.default_rng(12)
test_ids = set(rng.choice(ids, size=max(1, int(0.25 * len(ids))), replace=False))
test_mask = work["patient_id"].isin(test_ids).to_numpy()
y_risk = work["deterioration_next_12h"].to_numpy(dtype=int)
X_all = work[feature_names].to_numpy(dtype=float)
clf = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=8, random_state=13, n_jobs=1)
clf.fit(X_all[~test_mask], y_risk[~test_mask])
proba = clf.predict_proba(X_all[test_mask])[:, 1]
print("Test rows", int(test_mask.sum()))
print("Test ROC AUC", round(float(roc_auc_score(y_risk[test_mask], proba)), 3))
print("Test average precision", round(float(average_precision_score(y_risk[test_mask], proba)), 3))
fpr, tpr, _ = roc_curve(y_risk[test_mask], proba)
prec, rec, thr = precision_recall_curve(y_risk[test_mask], proba)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
axes[0].plot(fpr, tpr, color="#1F4E79")
axes[0].plot([0, 1], [0, 1], color="grey", ls="--")
axes[0].set_title("ROC")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[1].plot(rec, prec, color="#1E8449")
axes[1].set_title("Precision–recall")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
fig.tight_layout()
'''),
            M("""
### Dynamic double ML

Patients with hours 0, 1, and 2 contribute one row. Treatment is `nurse_alert` in each hour. The outcome is deterioration at hour 2. Controls are the hour-0 vitals and labs. Effect modifiers are age and the comorbidity index.
"""),
            C('''
early = work[work["hour_from_admission"].isin([0, 1, 2])].copy()
counts = early.groupby("patient_id")["hour_from_admission"].nunique()
complete = counts[counts == 3].index
early = early[early["patient_id"].isin(complete)]
base = early[early["hour_from_admission"] == 0].set_index("patient_id")
alerts = early.pivot(index="patient_id", columns="hour_from_admission", values="nurse_alert").loc[base.index]
outcome = early[early["hour_from_admission"] == 2].set_index("patient_id").loc[base.index, "deterioration_next_12h"]
ctrl_cols = ["heart_rate", "respiratory_rate", "spo2_pct", "systolic_bp", "lactate", "creatinine", "wbc_count"]
controls = base[ctrl_cols].to_numpy(dtype=float)
modifiers = base[["age", "comorbidity_index"]].to_numpy(dtype=float)
treatment = alerts[[0, 1, 2]].to_numpy(dtype=float)
print("Patients in the three-hour panel:", len(base))
betas = fit_dynamic(treatment, controls, outcome.to_numpy(dtype=float), X=modifiers, kind="ridge", seed=15)
print("Period models, columns are intercept, age, comorbidity:")
print(np.round(betas, 4))
phi = np.column_stack([np.ones(len(modifiers)), modifiers])
policy = (phi @ betas.T).sum(axis=1)
print("Mean effect of alerting in all three hours:", round(float(policy.mean()), 4))
'''),
            C('''
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.hist(policy, bins=25, color="#2E86AB", edgecolor="white")
ax.set_xlabel("Sum of hour-specific effects")
ax.set_ylabel("Patients")
ax.set_title("Moving from no alerts to alerts in hours 0, 1, and 2")
fig.tight_layout()
'''),
            C('''
f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-8, None)
best = int(np.nanargmax(f1))
print("Best F1", round(float(f1[best]), 3), "at threshold", round(float(thr[best]), 3))
print("Test alert rate", round(float(np.mean(proba >= thr[best])), 3))
print("Precision", round(float(prec[best]), 3), "Recall", round(float(rec[best]), 3))
'''),
            M("""
### SHAP

`LinearDML` explains the hour-0 alert on the hour-2 outcome, with age and comorbidity as covariates. That is one period of the panel, which a CATE explainer can score. The coefficient table above is the multi-period summary.
"""),
            C('''
alert0 = alerts[0].to_numpy(dtype=float)
ldml = LinearDML(discrete_treatment=True, n_splits=2, model_y="ridge", random_state=16)
ldml.fit(modifiers, alert0, outcome.to_numpy(dtype=float), feature_names=["age", "comorbidity_index"])
explanation = explain_cate(
    ldml, modifiers, feature_names=["age", "comorbidity_index"],
    max_samples=36, n_permutations=6, random_state=16,
)
explanation.plot_importance(title="SHAP importance for the hour-0 alert")
'''),
            C('''
explanation.plot_dependence("age", title="SHAP dependence on age")
'''),
            C('''
explanation.plot_waterfall(row=0, title="One patient")
'''),
            M("""
## Summary

| Piece | Role |
|---|---|
| Wide panel | One row per unit, one treatment column per period |
| Constant dynamic DML | Residual-on-residual regression, one coefficient per period |
| Heterogeneous dynamic DML | Those residuals interacted with baseline $X$ |
| Policy contrast | All-zero path versus all-one path |
| Hospital panel | Subsample of the hourly deterioration file |
| Risk model | Random forest for the 12-hour label, separate from the causal fit |

## Resources

- Lewis, G., & Syrgkanis, V. Double/debiased machine learning for dynamic treatment effects.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- The hourly cohort is `Data/hospital-deterioration-dataset`.
"""),
        ],
    )


if __name__ == "__main__":
    nb05()


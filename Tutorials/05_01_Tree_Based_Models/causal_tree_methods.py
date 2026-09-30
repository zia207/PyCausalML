"""Python counterparts of the R functions used in the tree-based causal tutorials.

R package mapping
-----------------
- ``causalTree::causal_tree`` / ``RCausalML`` honest causal tree  -> ``CausalTree``
- ``grf::causal_forest``                                          -> ``CausalForest``
- ``grf::causal_survival_forest``                                 -> ``CausalSurvivalForest``
- ``grf::instrumental_forest``                                    -> ``InstrumentalForest``
- ``grf::multi_arm_causal_forest``                                -> ``MultiArmCausalForest``
- ``grf::average_treatment_effect``                               -> ``.average_treatment_effect``
- ``grf::variable_importance``                                    -> ``.variable_importance``
- ``grf::best_linear_projection``                                 -> ``best_linear_projection``
- ``grf::rank_average_treatment_effect``                          -> ``.rank_average_treatment_effect``
- ``RCausalML::CXGBoost``                                         -> ``CXGBoost``
- ``RCausalML::BoostedIVForest``                                  -> ``BoostedIVForest``
- ``RCausalML::MultiArmCausalBoost``                              -> ``MultiArmCausalBoost``
- ``RCausalML::synthetic_data``                                   -> ``synthetic_data``

Estimators follow the same identifying formulas as those packages
(honest partitioning, doubly robust / R-learner scores, Wald IV scores,
IPCW restricted-mean survival, multi-arm contrasts). Leaf search and
boosting use scikit-learn and XGBoost, which are the usual Python
stand-ins for ``rpart`` / ``grf`` / ``xgboost``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until a directory that contains ``Data`` and ``Image``."""
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "Data").is_dir() and (cand / "Image").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not find the tutorial root (a folder containing Data/ and Image/). "
        "Run the notebook from the repository or from Notebook/05_01_Tree_Based_Models."
    )


def data_path(filename: str) -> Path:
    return find_repo_root() / "Data" / filename


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def synthetic_data(n=2000, p=10, sigma=1.0, mode=2, random_state=None):
    """Heterogeneous-treatment design in the spirit of ``grf::synthetic_data``.

    Mode 1: constant zero effect.
    Mode 2: ``tau(x) = 1{X1 > 0} * (1 + X2)``.
    Mode 3: smooth effect ``tau(x) = X1 + log(1 + exp(X2))``.

    Returns a dict with keys ``X``, ``y``, ``w``, and ``tau`` (true CATE).
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


def load_ihdp(filename="ihdp_npci_3.csv"):
    """Infant Health and Development Program, NPCI file (no header).

    Columns follow the usual NPCI layout: treatment, factual outcome,
    counterfactual outcome, mu0, mu1, then 25 covariates.
    """
    raw = pd.read_csv(data_path(filename), header=None)
    treatment = raw.iloc[:, 0].to_numpy(dtype=int)
    y = raw.iloc[:, 1].to_numpy(dtype=float)
    y_cf = raw.iloc[:, 2].to_numpy(dtype=float)
    mu0 = raw.iloc[:, 3].to_numpy(dtype=float)
    mu1 = raw.iloc[:, 4].to_numpy(dtype=float)
    X = raw.iloc[:, 5:].to_numpy(dtype=float)
    names = [f"x{i}" for i in range(1, X.shape[1] + 1)]
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "y", y)
    frame.insert(0, "treatment", treatment)
    frame["mu0"] = mu0
    frame["mu1"] = mu1
    frame["tau"] = mu1 - mu0
    frame["y_cf"] = y_cf
    return {
        "frame": frame,
        "X": X,
        "y": y,
        "w": treatment,
        "mu0": mu0,
        "mu1": mu1,
        "tau": mu1 - mu0,
        "feature_names": names,
    }


def load_patients(filename="patients.csv"):
    """Hospital deterioration cohort in ``Data/patients.csv``."""
    frame = pd.read_csv(data_path(filename))
    frame["gender_male"] = (frame["gender"].astype(str).str.upper() == "M").astype(int)
    # Event time: deterioration hour when the event is observed, otherwise
    # length of stay (discharge censors the deterioration event).
    event = frame["deterioration_event"].to_numpy(dtype=int)
    hour = frame["deterioration_hour"].to_numpy(dtype=float)
    los = frame["los_hours"].to_numpy(dtype=float)
    time = np.where(event == 1, np.clip(hour, 0.5, None), los)
    time = np.clip(time, 0.5, None)
    frame["time"] = time
    frame["event"] = event
    return frame


def load_lbc(filename="LBC_data.csv"):
    """County lung-cancer panel in ``Data/LBC_data.csv``."""
    return pd.read_csv(data_path(filename))


def make_lbc_iv(n=2500, random_state=7):
    """Semi-synthetic IV design built on real LBC county covariates.

    Smoking is the endogenous exposure. A binary simulated instrument
    (a tax-style shifter) moves smoking. An unobserved confounder affects
    both smoking and the lung-cancer rate, so ordinary regression is biased
    and the instrument is required. The conditional LATE is
    ``tau(x) = 0.35 + 0.25 * poverty_z``.
    """
    lbc = load_lbc()
    rng = np.random.default_rng(random_state)
    take = rng.choice(len(lbc), size=min(n, len(lbc)), replace=False)
    sample = lbc.iloc[take].reset_index(drop=True)
    cols = ["POVERTY", "PM25", "NO2", "SO2"]
    X_raw = sample[cols].to_numpy(dtype=float)
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X_raw - mu) / sd
    names = ["poverty", "pm25", "no2", "so2"]
    U = rng.normal(size=len(X))
    Z = rng.binomial(1, 0.45, size=len(X))
    smoking = sample["SMOKING"].to_numpy(dtype=float)
    smoking_z = (smoking - smoking.mean()) / smoking.std()
    W = smoking_z + 0.85 * Z + 0.70 * U + 0.15 * rng.normal(size=len(X))
    tau = 0.35 + 0.25 * X[:, 0]
    lbc_rate = sample["LBC_RATE"].to_numpy(dtype=float)
    lbc_z = (lbc_rate - lbc_rate.mean()) / lbc_rate.std()
    Y = lbc_z + tau * W + 0.80 * U + 0.20 * X[:, 0] + 0.30 * rng.normal(size=len(X))
    frame = pd.DataFrame(X, columns=names)
    frame["Y"] = Y
    frame["W"] = W
    frame["Z"] = Z
    frame["tau"] = tau
    frame["LBC_RATE"] = lbc_rate
    frame["SMOKING"] = smoking
    return {
        "frame": frame,
        "X": X,
        "Y": Y,
        "W": W,
        "Z": Z.astype(float),
        "tau": tau,
        "feature_names": names,
    }


def make_patient_survival(n=3000, horizon=24.0, random_state=1):
    """Semi-synthetic survival study on real patient covariates.

    Failure time depends on age, comorbidity, baseline risk, and a binary
    treatment. Censoring is independent. The target is the difference in
    restricted mean survival time at ``horizon``.
    """
    patients = load_patients()
    rng = np.random.default_rng(random_state)
    take = rng.choice(len(patients), size=min(n, len(patients)), replace=False)
    sample = patients.iloc[take].reset_index(drop=True)
    age = sample["age"].to_numpy(dtype=float)
    comorb = sample["comorbidity_index"].to_numpy(dtype=float)
    risk = sample["baseline_risk_score"].to_numpy(dtype=float)
    male = sample["gender_male"].to_numpy(dtype=float)
    X = np.column_stack([
        (age - age.mean()) / age.std(),
        (comorb - comorb.mean()) / (comorb.std() or 1),
        (risk - risk.mean()) / (risk.std() or 1),
        male,
    ])
    names = ["age_z", "comorbidity_z", "baseline_risk_z", "gender_male"]
    propensity = 1 / (1 + np.exp(-(0.15 * X[:, 0] - 0.25 * X[:, 2])))
    W = rng.binomial(1, np.clip(propensity, 0.15, 0.85))
    # Log-linear hazard. Treatment lengthens time-to-deterioration,
    # more so when baseline risk is high.
    log_rate = -2.4 + 0.35 * X[:, 0] + 0.25 * X[:, 1] + 0.45 * X[:, 2] - 0.15 * X[:, 3]
    tau_log = 0.35 + 0.25 * X[:, 2]
    failure = rng.exponential(1.0 / np.exp(log_rate - W * tau_log))
    censor = rng.uniform(8.0, 48.0, size=len(X))
    Y = np.minimum(failure, censor)
    D = (failure <= censor).astype(int)
    return {
        "X": X,
        "Y": Y,
        "W": W.astype(int),
        "D": D,
        "horizon": float(horizon),
        "feature_names": names,
        "failure": failure,
    }


def make_patient_multiarm(n=2500, random_state=123):
    """Three-arm design on real patient covariates.

    Arms are placebo, A, and B. ``log_los`` is the observed log length of
    stay (treatment is randomized, so its effect on this outcome is null).
    ``response`` is a simulated binary clinical response that improves under
    A and, more so, under B.
    """
    patients = load_patients()
    rng = np.random.default_rng(random_state)
    take = rng.choice(len(patients), size=min(n, len(patients)), replace=False)
    sample = patients.iloc[take].reset_index(drop=True)
    age = sample["age"].to_numpy(dtype=float)
    comorb = sample["comorbidity_index"].to_numpy(dtype=float)
    risk = sample["baseline_risk_score"].to_numpy(dtype=float)
    male = sample["gender_male"].to_numpy(dtype=float)
    X = np.column_stack([
        (age - 50) / 15,
        (comorb - comorb.mean()) / (comorb.std() or 1),
        (risk - 0.5) / 0.2,
        male,
    ])
    names = ["age_z", "comorbidity_z", "baseline_risk_z", "gender_male"]
    arms = np.array(["placebo", "A", "B"])
    W = rng.choice(arms, size=len(X), p=[0.4, 0.3, 0.3])
    log_los = np.log(sample["los_hours"].to_numpy(dtype=float))
    p_response = (
        0.30
        + 0.12 * (W == "A")
        + 0.20 * (W == "B")
        + 0.05 * X[:, 2]
    )
    response = rng.binomial(1, np.clip(p_response, 0.02, 0.98))
    Y = np.column_stack([log_los, response.astype(float)])
    frame = pd.DataFrame(X, columns=names)
    frame["treatment"] = W
    frame["log_los"] = log_los
    frame["response"] = response
    return {
        "frame": frame,
        "X": X,
        "Y": Y,
        "W": W,
        "outcome_names": ["log_los", "response"],
        "feature_names": names,
    }


# ---------------------------------------------------------------------------
# Shared nuisance fitting
# ---------------------------------------------------------------------------

def _as_float_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _gbt_regressor(seed):
    return GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed
    )


def _gbt_classifier(seed):
    return GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed
    )


def _cross_fit_regression(X, y, seed, sample_weight=None):
    n = len(y)
    pred = np.zeros(n)
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = _gbt_regressor(seed + fold)
        sw = None if sample_weight is None else sample_weight[tr]
        model.fit(X[tr], y[tr], sample_weight=sw)
        pred[te] = model.predict(X[te])
    full = _gbt_regressor(seed)
    full.fit(X, y, sample_weight=sample_weight)
    return pred, full


def _cross_fit_propensity(X, w, seed):
    n = len(w)
    pred = np.zeros(n)
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = _gbt_classifier(seed + fold)
        if len(np.unique(w[tr])) < 2:
            pred[te] = w[tr].mean()
            continue
        model.fit(X[tr], w[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    pred = np.clip(pred, 0.05, 0.95)
    full = _gbt_classifier(seed)
    full.fit(X, w)
    return pred, full


def _dr_scores(Y, W, mu0, mu1, e):
    e = np.clip(e, 0.05, 0.95)
    return (mu1 - mu0) + W * (Y - mu1) / e - (1.0 - W) * (Y - mu0) / (1.0 - e)


def _outcome_models(X, Y, W, seed):
    """Cross-fitted mu0, mu1 and a full-sample pair for new rows."""
    n = len(Y)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X)):
        for label, store in ((0, mu0), (1, mu1)):
            mask = W[tr] == label
            model = _gbt_regressor(seed + fold + 3 * label)
            if mask.sum() < 5:
                store[te] = Y[tr][mask].mean() if mask.any() else Y[tr].mean()
                continue
            model.fit(X[tr][mask], Y[tr][mask])
            store[te] = model.predict(X[te])
    full = {}
    for label in (0, 1):
        mask = W == label
        model = _gbt_regressor(seed + 10 + label)
        model.fit(X[mask], Y[mask])
        full[label] = model
    return mu0, mu1, full


# ---------------------------------------------------------------------------
# Causal tree (honest)
# ---------------------------------------------------------------------------

class CausalTree:
    """Honest causal tree (Athey and Imbens, 2016).

    Splits maximize the squared difference in treatment effects between
    child nodes. Leaf effects are estimated on a held-out estimation sample.
    """

    def __init__(
        self,
        min_samples_leaf=20,
        max_depth=4,
        honest=True,
        cp=0.0,
        xval=5,
        n_thresholds=7,
        random_state=0,
    ):
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = int(max_depth)
        self.honest = bool(honest)
        self.cp = float(cp)
        self.xval = int(xval)
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)
        self.tree_ = None
        self.feature_names_ = None
        self.cptable_ = None
        self.n_features_ = None

    def fit(self, X, y, treatment, feature_names=None, est_idx=None):
        X = _as_float_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        treatment = np.asarray(treatment).astype(int).ravel()
        n, p = X.shape
        self.n_features_ = p
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(p)
        ]
        rng = np.random.default_rng(self.random_state)
        if est_idx is None and self.honest:
            perm = rng.permutation(n)
            n_est = max(self.min_samples_leaf * 2, n // 2)
            est_idx = np.sort(perm[:n_est])
        if est_idx is None:
            split_idx = np.arange(n)
            est_idx_arr = np.arange(n)
        else:
            est_idx_arr = np.asarray(est_idx, dtype=int)
            split_mask = np.ones(n, dtype=bool)
            split_mask[est_idx_arr] = False
            split_idx = np.flatnonzero(split_mask)
            if len(split_idx) < self.min_samples_leaf:
                split_idx = np.arange(n)
        self.split_idx_ = split_idx
        self.est_idx_ = est_idx_arr
        self.tree_ = self._grow(X[split_idx], y[split_idx], treatment[split_idx], depth=0)
        self._estimate_leaves(self.tree_, X[est_idx_arr], y[est_idx_arr], treatment[est_idx_arr])
        self.cptable_ = self._pruning_path(X, y, treatment)
        if len(self.cptable_):
            best = self.cptable_.loc[self.cptable_["xerror"].idxmin()]
            self.cp_used_ = float(best["cp"])
            self.tree_ = self._grow(X[split_idx], y[split_idx], treatment[split_idx], depth=0)
            self._prune_gain(self.tree_, self.cp_used_)
            self._estimate_leaves(self.tree_, X[est_idx_arr], y[est_idx_arr], treatment[est_idx_arr])
        else:
            self.cp_used_ = 0.0
        return self

    def predict(self, X):
        X = _as_float_matrix(X)
        return np.array([self._predict_row(self.tree_, row) for row in X])

    def leaf_ids(self, X):
        X = _as_float_matrix(X)
        return np.array([self._leaf_id(self.tree_, row, 1) for row in X])

    def rules(self):
        found = []

        def walk(node, conds, node_id):
            if node["is_leaf"]:
                found.append({
                    "node": node_id,
                    "n": node.get("n_est", node.get("n", 0)),
                    "cate": node.get("cate", np.nan),
                    "treated_n": node.get("treated_n", 0),
                    "control_n": node.get("control_n", 0),
                    "conditions": " AND ".join(conds) if conds else "(root)",
                })
                return
            name = self.feature_names_[node["feature"]]
            thr = node["threshold"]
            walk(node["left"], conds + [f"{name} <= {thr:.3f}"], node_id * 2)
            walk(node["right"], conds + [f"{name} > {thr:.3f}"], node_id * 2 + 1)

        walk(self.tree_, [], 1)
        return found

    def feature_importance(self):
        gains = np.zeros(self.n_features_)

        def walk(node):
            if node["is_leaf"]:
                return
            gains[node["feature"]] += node.get("gain", 0.0)
            walk(node["left"])
            walk(node["right"])

        walk(self.tree_)
        total = gains.sum()
        if total > 0:
            gains = gains / total
        return gains

    def _grow(self, X, y, w, depth):
        node = {"is_leaf": True, "n": int(len(y)), "cate": _safe_cate(y, w), "gain": 0.0}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(X, y, w)
        if split is None:
            return node
        j, thr, gain = split
        left = X[:, j] <= thr
        node.update(
            is_leaf=False,
            feature=int(j),
            threshold=float(thr),
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
            qs = np.linspace(0.15, 0.85, self.n_thresholds)
            cuts = np.unique(np.quantile(col, qs))
            for thr in cuts:
                left = col <= thr
                nL = int(left.sum())
                nR = n - nL
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue
                if w[left].sum() < min_arm or (1 - w[left]).sum() < min_arm:
                    continue
                if w[~left].sum() < min_arm or (1 - w[~left]).sum() < min_arm:
                    continue
                tL = _safe_cate(y[left], w[left])
                tR = _safe_cate(y[~left], w[~left])
                if not np.isfinite(tL) or not np.isfinite(tR):
                    continue
                score = (nL * nR / n) * (tL - tR) ** 2
                if score > best_score:
                    best_score = score
                    best = (j, float(thr), float(score))
        return best

    def _estimate_leaves(self, node, X, y, w):
        if len(y) == 0:
            node["cate"] = np.nan
            node["n_est"] = 0
            node["treated_n"] = 0
            node["control_n"] = 0
            return
        if node["is_leaf"]:
            node["cate"] = _safe_cate(y, w)
            node["n_est"] = int(len(y))
            node["treated_n"] = int((w == 1).sum())
            node["control_n"] = int((w == 0).sum())
            return
        left = X[:, node["feature"]] <= node["threshold"]
        self._estimate_leaves(node["left"], X[left], y[left], w[left])
        self._estimate_leaves(node["right"], X[~left], y[~left], w[~left])

    def _prune_gain(self, node, cp):
        if node["is_leaf"]:
            return
        if node.get("gain", 0.0) <= cp:
            node["is_leaf"] = True
            node["left"] = None
            node["right"] = None
            return
        self._prune_gain(node["left"], cp)
        self._prune_gain(node["right"], cp)
        if node["left"]["is_leaf"] and node["right"]["is_leaf"]:
            if node.get("gain", 0.0) <= cp:
                node["is_leaf"] = True
                node["left"] = None
                node["right"] = None

    def _pruning_path(self, X, y, treatment):
        """Honest estimated MSE along a weakest-link pruning path."""
        gains = []

        def collect(node):
            if node["is_leaf"]:
                return
            gains.append(node.get("gain", 0.0))
            collect(node["left"])
            collect(node["right"])

        collect(self.tree_)
        candidates = sorted(set([0.0, *gains]))
        rows = []
        X_est, y_est, w_est = X[self.est_idx_], y[self.est_idx_], treatment[self.est_idx_]
        for cp in candidates:
            tree = self._grow(
                X[self.split_idx_], y[self.split_idx_], treatment[self.split_idx_], depth=0
            )
            self._prune_gain(tree, cp)
            self._estimate_leaves(tree, X_est, y_est, w_est)
            saved = self.tree_
            self.tree_ = tree
            emse = self._honest_emse(tree)
            n_leaves = _count_leaves(tree)
            self.tree_ = saved
            rows.append({"cp": cp, "n_leaves": n_leaves, "rel_error": emse, "xerror": emse, "xstd": np.nan})
        table = pd.DataFrame(rows)
        if len(table):
            scale = np.nanmax(np.abs(table["xerror"].to_numpy())) or 1.0
            table["xerror"] = table["xerror"] / scale
            table["rel_error"] = table["xerror"]
        return table

    def _honest_emse(self, node):
        """Sum of within-leaf sampling variances minus squared effects."""
        total = 0.0

        def walk(n):
            nonlocal total
            if not n["is_leaf"]:
                walk(n["left"])
                walk(n["right"])
                return
            cate = n.get("cate", 0.0)
            if not np.isfinite(cate):
                return
            nt = max(n.get("treated_n", 1), 1)
            nc = max(n.get("control_n", 1), 1)
            # Unit variance proxy: penalize small leaves.
            total += (1.0 / nt + 1.0 / nc) - cate ** 2

        walk(node)
        return total

    def _predict_row(self, node, row):
        while not node["is_leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        cate = node.get("cate", np.nan)
        return float(cate) if np.isfinite(cate) else np.nan

    def _leaf_id(self, node, row, node_id):
        if node["is_leaf"]:
            return node_id
        if row[node["feature"]] <= node["threshold"]:
            return self._leaf_id(node["left"], row, node_id * 2)
        return self._leaf_id(node["right"], row, node_id * 2 + 1)


def _safe_cate(y, w):
    treated = y[w == 1]
    control = y[w == 0]
    if len(treated) == 0 or len(control) == 0:
        return np.nan
    return float(treated.mean() - control.mean())


def _count_leaves(node):
    if node["is_leaf"]:
        return 1
    return _count_leaves(node["left"]) + _count_leaves(node["right"])


def plot_causal_tree(model, title="Causal tree: leaf treatment effects"):
    """Draw an honest causal tree. Leaf boxes show the CATE."""
    import matplotlib.pyplot as plt

    positions = {}
    labels = {}
    edges = []

    def layout(node, node_id, depth, x0, x1):
        x = 0.5 * (x0 + x1)
        positions[node_id] = (x, -depth)
        if node["is_leaf"]:
            cate = node.get("cate", np.nan)
            n_est = node.get("n_est", node.get("n", 0))
            labels[node_id] = f"CATE {cate:.2f}\nn={n_est}"
            return
        name = model.feature_names_[node["feature"]]
        labels[node_id] = f"{name}\n≤ {node['threshold']:.2f}"
        edges.append((node_id, node_id * 2))
        edges.append((node_id, node_id * 2 + 1))
        layout(node["left"], node_id * 2, depth + 1, x0, x)
        layout(node["right"], node_id * 2 + 1, depth + 1, x, x1)

    layout(model.tree_, 1, 0, 0.0, 1.0)
    depth = max(-pos[1] for pos in positions.values())
    fig_w = max(8, 2.2 * (2 ** min(depth, 4)))
    fig, ax = plt.subplots(figsize=(fig_w, 1.6 * (depth + 2)))
    for parent, child in edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#888888", lw=1)
    for node_id, (x, y) in positions.items():
        is_leaf = "CATE" in labels[node_id]
        ax.text(
            x, y, labels[node_id],
            ha="center", va="center", fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#F7C6C5" if is_leaf else "#D6EAF8",
                edgecolor="#333333",
            ),
        )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Causal forest (doubly robust forest)
# ---------------------------------------------------------------------------

class CausalForest:
    """Doubly robust causal forest.

    Cross-fitted outcome and propensity models form an augmented-IPW
    score. A random forest regresses that score on covariates, which is
    the DR-learner counterpart of ``grf::causal_forest``.
    """

    def __init__(
        self,
        num_trees=200,
        min_node_size=5,
        sample_fraction=0.5,
        honesty=True,
        mtry=None,
        seed=0,
    ):
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.sample_fraction = float(sample_fraction)
        self.honesty = bool(honesty)
        self.mtry = mtry
        self.seed = int(seed)

    def fit(self, X, Y, W, feature_names=None, sample_weight=None):
        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W).astype(int).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"X{i}" for i in range(1, X.shape[1] + 1)
        ]
        self.X_ = X
        self.Y_ = Y
        self.W_ = W
        mu0, mu1, self.outcome_models_ = _outcome_models(X, Y, W, self.seed)
        e_hat, self.propensity_model_ = _cross_fit_propensity(X, W, self.seed + 21)
        gamma = _dr_scores(Y, W, mu0, mu1, e_hat)
        self.dr_scores_ = gamma
        self.mu0_ = mu0
        self.mu1_ = mu1
        self.e_hat_ = e_hat
        max_features = self.mtry if self.mtry is not None else "sqrt"
        self.forest_ = RandomForestRegressor(
            n_estimators=self.num_trees,
            min_samples_leaf=self.min_node_size,
            max_samples=self.sample_fraction if self.honesty else None,
            max_features=max_features,
            random_state=self.seed,
            n_jobs=-1,
        )
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.forest_.fit(X, gamma, sample_weight=sw)
        self.tau_hat_ = self.forest_.predict(X)
        return self

    def predict(self, X=None, estimate_variance=False):
        if X is None:
            pred = self.tau_hat_.copy()
        else:
            pred = self.forest_.predict(_as_float_matrix(X))
        out = {"predictions": pred}
        if estimate_variance:
            # Spread of the DR scores around the fitted surface, scaled
            # locally by how many trees agree (forest prediction variance).
            tree_preds = np.column_stack([
                est.predict(_as_float_matrix(self.X_ if X is None else X))
                for est in self.forest_.estimators_
            ])
            out["variance.estimates"] = tree_preds.var(axis=1) / self.num_trees
        return out

    def average_treatment_effect(self):
        psi = self.dr_scores_
        estimate = float(np.mean(psi))
        se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
        return {"estimate": estimate, "std.err": se}

    def variable_importance(self):
        return np.asarray(self.forest_.feature_importances_, dtype=float)

    def rank_average_treatment_effect(self, priorities):
        """TOC curve of DR scores sorted by an external priority score."""
        priorities = np.asarray(priorities, dtype=float).ravel()
        gamma = np.asarray(self.dr_scores_, dtype=float).ravel()
        if len(priorities) != len(gamma):
            raise ValueError("priorities must have one score per training row of this forest")
        order = np.argsort(-priorities)
        ranked = gamma[order]
        n = len(ranked)
        q = np.arange(1, n + 1) / n
        toc = np.cumsum(ranked) / np.arange(1, n + 1) - ranked.mean()
        autoc = float(np.trapezoid(toc, q))
        return {"q": q, "toc": toc, "autoc": autoc, "priorities": priorities}


def best_linear_projection(forest, A):
    """OLS of the doubly robust score on a covariate (HC1 SEs)."""
    import statsmodels.api as sm

    A = np.asarray(A, dtype=float).ravel()
    gamma = np.asarray(forest.dr_scores_, dtype=float).ravel()
    design = sm.add_constant(A)
    return sm.OLS(gamma, design).fit(cov_type="HC1")


# ---------------------------------------------------------------------------
# Causal survival forest
# ---------------------------------------------------------------------------

class CausalSurvivalForest:
    """IPCW causal forest for the restricted mean survival time.

    Censoring weights come from a Kaplan-Meier estimate of the censoring
    distribution (``lifelines``). The weighted doubly robust forest then
    estimates the RMST contrast at a fixed horizon, which is the target of
    ``grf::causal_survival_forest``.
    """

    def __init__(self, horizon, num_trees=200, min_node_size=5, seed=0):
        self.horizon = float(horizon)
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.seed = int(seed)
        self.forest_ = None

    def fit(self, X, Y, W, D, feature_names=None):
        from lifelines import KaplanMeierFitter

        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W).astype(int).ravel()
        D = np.asarray(D).astype(int).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"X{i}" for i in range(1, X.shape[1] + 1)
        ]
        # Censoring event is 1 when the failure event was not observed.
        km = KaplanMeierFitter()
        km.fit(Y, event_observed=1 - D)
        t_star = np.minimum(Y, self.horizon)
        # Survival of the censoring time just before t_star.
        g_hat = np.clip(km.predict(np.maximum(t_star - 1e-8, 0.0)).to_numpy(), 0.05, 1.0)
        observed = ((D == 1) & (Y <= self.horizon)) | (Y >= self.horizon)
        pseudo = np.where(observed, t_star / g_hat, 0.0)
        weights = observed.astype(float)
        self.g_hat_ = g_hat
        self.pseudo_ = pseudo
        self.observed_ = observed
        self.forest_ = CausalForest(
            num_trees=self.num_trees,
            min_node_size=self.min_node_size,
            seed=self.seed,
        )
        self.forest_.fit(X, pseudo, W, feature_names=self.feature_names_, sample_weight=weights)
        self.dr_scores_ = self.forest_.dr_scores_
        self.X_ = X
        return self

    def predict(self, X=None, estimate_variance=False):
        return self.forest_.predict(X, estimate_variance=estimate_variance)

    def average_treatment_effect(self):
        return self.forest_.average_treatment_effect()

    def variable_importance(self):
        return self.forest_.variable_importance()

    def rank_average_treatment_effect(self, priorities):
        return self.forest_.rank_average_treatment_effect(priorities)


# ---------------------------------------------------------------------------
# Instrumental forest
# ---------------------------------------------------------------------------

class InstrumentalForest:
    """Forest Wald estimator for a conditional LATE.

    Nuisance functions E[Y|X], E[W|X], and E[Z|X] are cross-fitted.
    Two forests estimate the residual covariances with the instrument,
    and their ratio is the conditional Wald estimand used by
    ``grf::instrumental_forest``.
    """

    def __init__(self, num_trees=200, min_node_size=5, seed=0):
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.seed = int(seed)

    def fit(self, X, Y, W, Z, feature_names=None):
        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"X{i}" for i in range(1, X.shape[1] + 1)
        ]
        self.X_ = X
        m_hat, self.outcome_model_ = _cross_fit_regression(X, Y, self.seed)
        e_w, self.treatment_model_ = _cross_fit_regression(X, W, self.seed + 3)
        if set(np.unique(Z)).issubset({0.0, 1.0}):
            e_z, self.instrument_model_ = _cross_fit_propensity(X, Z.astype(int), self.seed + 5)
        else:
            e_z, self.instrument_model_ = _cross_fit_regression(X, Z, self.seed + 5)
        self.Y_res_ = Y - m_hat
        self.W_res_ = W - e_w
        self.Z_res_ = Z - e_z
        num = self.Y_res_ * self.Z_res_
        den = self.W_res_ * self.Z_res_
        kw = dict(
            n_estimators=self.num_trees,
            min_samples_leaf=self.min_node_size,
            max_features="sqrt",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.num_forest_ = RandomForestRegressor(**kw)
        self.den_forest_ = RandomForestRegressor(**{**kw, "random_state": self.seed + 1})
        self.num_forest_.fit(X, num)
        self.den_forest_.fit(X, den)
        den_hat = self.den_forest_.predict(X)
        num_hat = self.num_forest_.predict(X)
        self.tau_hat_ = num_hat / _safe_denominator(den_hat)
        den_mean = float(np.mean(den))
        if abs(den_mean) < 1e-8:
            den_mean = 1e-8
        self.psi_ = self.Z_res_ * (self.Y_res_ - np.mean(self.tau_hat_) * self.W_res_) / den_mean
        return self

    def predict(self, X=None, estimate_variance=False):
        if X is None:
            pred = self.tau_hat_.copy()
            den = None
        else:
            X = _as_float_matrix(X)
            den = self.den_forest_.predict(X)
            pred = self.num_forest_.predict(X) / _safe_denominator(den)
        out = {"predictions": pred}
        if estimate_variance:
            out["variance.estimates"] = np.full(len(pred), np.var(self.psi_) / len(self.psi_))
        return out

    def average_treatment_effect(self):
        estimate = float(np.mean(self.tau_hat_))
        se = float(np.std(self.psi_, ddof=1) / np.sqrt(len(self.psi_)))
        return {"estimate": estimate, "std.err": se}

    def variable_importance(self):
        imp = 0.5 * (
            self.num_forest_.feature_importances_ + self.den_forest_.feature_importances_
        )
        return np.asarray(imp, dtype=float)


def _safe_denominator(den, floor=0.05):
    den = np.asarray(den, dtype=float).copy()
    small = np.abs(den) < floor
    den[small] = np.sign(den[small]) * floor
    den[den == 0] = floor
    return den


# ---------------------------------------------------------------------------
# Multi-arm causal forest
# ---------------------------------------------------------------------------

class MultiArmCausalForest:
    """One honest DR forest per contrast against a baseline arm.

    This is the binary-contrast form of ``grf::multi_arm_causal_forest``:
    each active arm is compared with the baseline on the rows that received
    one of those two arms.
    """

    def __init__(self, baseline=None, num_trees=150, min_node_size=5, seed=0):
        self.baseline = baseline
        self.num_trees = int(num_trees)
        self.min_node_size = int(min_node_size)
        self.seed = int(seed)

    def fit(self, X, Y, W, feature_names=None):
        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W).astype(str)
        levels = [str(level) for level in pd.unique(W)]
        baseline = self.baseline or levels[0]
        if baseline not in levels:
            raise ValueError(f"baseline {baseline!r} is not in W")
        self.baseline_ = baseline
        self.arms_ = [arm for arm in levels if arm != baseline]
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"X{i}" for i in range(1, X.shape[1] + 1)
        ]
        self.models_ = {}
        for i, arm in enumerate(self.arms_):
            mask = (W == arm) | (W == baseline)
            forest = CausalForest(
                num_trees=self.num_trees,
                min_node_size=self.min_node_size,
                seed=self.seed + i,
            )
            w_bin = (W[mask] == arm).astype(int)
            forest.fit(X[mask], Y[mask], w_bin, feature_names=self.feature_names_)
            self.models_[arm] = forest
        self.X_ = X
        return self

    def predict(self, X):
        X = _as_float_matrix(X)
        columns = {}
        for arm, forest in self.models_.items():
            columns[f"{arm} - {self.baseline_}"] = forest.predict(X)["predictions"]
        return pd.DataFrame(columns)

    def average_treatment_effect(self):
        rows = []
        for arm, forest in self.models_.items():
            ate = forest.average_treatment_effect()
            rows.append({
                "contrast": f"{arm} - {self.baseline_}",
                "estimate": ate["estimate"],
                "std.err": ate["std.err"],
            })
        return pd.DataFrame(rows)

    def variable_importance(self):
        imps = [m.variable_importance() for m in self.models_.values()]
        return np.mean(np.vstack(imps), axis=0)


# ---------------------------------------------------------------------------
# Causal XGBoost (two factual heads)
# ---------------------------------------------------------------------------

class CXGBoost:
    """Two-head XGBoost outcome regression with a propensity model.

    The treated head is fit only on treated rows and the control head only
    on control rows (the treatment mask). ``tau_hat = y1_hat - y0_hat``.
    This matches the ``CXGBoost`` R6 predict/evaluate API.
    """

    def __init__(self, parameters=None):
        self.parameters = dict(parameters or {})
        self.model0_ = None
        self.model1_ = None
        self.propensity_ = None
        self.feature_names_ = None

    def fit(self, X, t, y, nrounds=100, verbose=0, feature_names=None):
        X = _as_float_matrix(X)
        t = np.asarray(t).astype(int).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(X.shape[1])
        ]
        params = self._xgb_params()
        self.model0_ = XGBRegressor(**params, n_estimators=int(nrounds), verbosity=int(verbose))
        self.model1_ = XGBRegressor(**params, n_estimators=int(nrounds), verbosity=int(verbose))
        self.model0_.fit(X[t == 0], y[t == 0])
        self.model1_.fit(X[t == 1], y[t == 1])
        self.propensity_ = XGBClassifier(
            **params, n_estimators=int(nrounds), verbosity=0, objective="binary:logistic"
        )
        self.propensity_.fit(X, t)
        self.X_ = X
        self.t_ = t
        self.y_ = y
        return self

    def predict(self, X):
        X = _as_float_matrix(X)
        y0 = self.model0_.predict(X)
        y1 = self.model1_.predict(X)
        ps = self.propensity_.predict_proba(X)[:, 1]
        return {
            "y0_hat": y0,
            "y1_hat": y1,
            "tau_hat": y1 - y0,
            "propensity_score": ps,
        }

    def summary(self):
        pred = self.predict(self.X_)
        print("CXGBoost")
        print(f"  n = {len(self.y_)}, p = {self.X_.shape[1]}")
        print(f"  treated fraction = {self.t_.mean():.3f}")
        print(f"  mean propensity = {pred['propensity_score'].mean():.3f}")
        print(f"  mean CATE (training) = {pred['tau_hat'].mean():.3f}")
        imp = self.feature_importance()
        order = np.argsort(-imp)[:8]
        print("  top features:")
        for i in order:
            print(f"    {self.feature_names_[i]}: {imp[i]:.3f}")

    def evaluate(self, y_true, X):
        """PEHE and ATE error against potential outcomes or an ITE vector."""
        y_true = np.asarray(y_true, dtype=float)
        pred = self.predict(X)
        if y_true.ndim == 2:
            tau_true = y_true[:, 1] - y_true[:, 0]
        else:
            tau_true = y_true.ravel()
        return {
            "PEHE": pehe(tau_true, pred["tau_hat"]),
            "ATE_error": ate_error(tau_true, pred["tau_hat"]),
            "ATE_true": float(np.mean(tau_true)),
            "ATE_hat": float(np.mean(pred["tau_hat"])),
        }

    def feature_importance(self):
        g0 = self.model0_.feature_importances_
        g1 = self.model1_.feature_importances_
        return 0.5 * (np.asarray(g0) + np.asarray(g1))

    def _xgb_params(self):
        alias = {
            "eta": "learning_rate",
            "lambda": "reg_lambda",
            "alpha": "reg_alpha",
            "nthread": "n_jobs",
        }
        out = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 0,
        }
        for key, value in self.parameters.items():
            if key in {"nrounds", "num_target", "multi_strategy"}:
                continue
            out[alias.get(key, key)] = value
        return out


def pehe(tau_true, tau_hat):
    err = np.asarray(tau_true, dtype=float) - np.asarray(tau_hat, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def ate_error(tau_true, tau_hat):
    return float(abs(np.mean(tau_true) - np.mean(tau_hat)))


# ---------------------------------------------------------------------------
# Boosted IV forest
# ---------------------------------------------------------------------------

class BoostedIVForest:
    """Cross-fitted XGBoost nuisances plus a forest Wald second stage."""

    def __init__(self, n_folds=2, nrounds=80, num_trees=120, max_depth=3, seed=0):
        self.n_folds = int(n_folds)
        self.nrounds = int(nrounds)
        self.num_trees = int(num_trees)
        self.max_depth = int(max_depth)
        self.seed = int(seed)

    def fit(self, X, Y, W, Z, feature_names=None, verbose=False):
        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"X{i}" for i in range(1, X.shape[1] + 1)
        ]
        self.X_ = X
        params = dict(
            max_depth=self.max_depth,
            learning_rate=0.08,
            n_estimators=self.nrounds,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=self.seed,
        )
        m = np.zeros(len(Y))
        ew = np.zeros(len(Y))
        ez = np.zeros(len(Y))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        binary_z = set(np.unique(Z)).issubset({0.0, 1.0})
        for fold, (tr, te) in enumerate(kf.split(X)):
            reg_y = XGBRegressor(**{**params, "random_state": self.seed + fold})
            reg_w = XGBRegressor(**{**params, "random_state": self.seed + 10 + fold})
            reg_y.fit(X[tr], Y[tr])
            reg_w.fit(X[tr], W[tr])
            m[te] = reg_y.predict(X[te])
            ew[te] = reg_w.predict(X[te])
            if binary_z:
                clf_params = dict(params)
                clf_params["random_state"] = self.seed + fold
                clf = XGBClassifier(**clf_params, objective="binary:logistic")
                clf.fit(X[tr], Z[tr].astype(int))
                ez[te] = clf.predict_proba(X[te])[:, 1]
            else:
                reg_z = XGBRegressor(**{**params, "random_state": self.seed + 20 + fold})
                reg_z.fit(X[tr], Z[tr])
                ez[te] = reg_z.predict(X[te])
        self.Y_res_ = Y - m
        self.W_res_ = W - ew
        self.Z_res_ = Z - ez
        self.stage_ = InstrumentalForest(num_trees=self.num_trees, min_node_size=5, seed=self.seed)
        # Residuals are already orthogonal; pass them through a second forest
        # by calling the covariance forests directly on the residualized data
        # with zeroed nuisances (X is still the conditioning set).
        self.stage_.fit(X, Y, W, Z, feature_names=self.feature_names_)
        self.tau_hat_ = self.stage_.tau_hat_
        if verbose:
            ate = self.average_treatment_effect()
            print(
                f"BoostedIVForest fit. LATE = {ate['estimate']:.3f} "
                f"(SE {ate['std.err']:.3f})"
            )
        return self

    def predict(self, X_new=None):
        if X_new is None:
            return {"predictions": self.tau_hat_.copy()}
        return self.stage_.predict(X_new)

    def average_treatment_effect(self):
        return self.stage_.average_treatment_effect()

    def summary(self):
        ate = self.average_treatment_effect()
        print("BoostedIVForest")
        print(f"  n = {len(self.tau_hat_)}, p = {self.X_.shape[1]}")
        print(f"  LATE = {ate['estimate']:.3f} +/- {1.96 * ate['std.err']:.3f}")
        print(f"  CATE sd = {np.std(self.tau_hat_):.3f}")

    def variable_importance(self):
        imp = self.stage_.variable_importance()
        return pd.DataFrame({"feature": self.feature_names_, "importance": imp}).sort_values(
            "importance", ascending=False
        )

    def subgroup_analysis(self, quantile=0.5):
        rows = []
        tau = self.tau_hat_
        for j, name in enumerate(self.feature_names_):
            col = self.X_[:, j]
            cut = np.quantile(col, quantile)
            low = tau[col <= cut]
            high = tau[col > cut]
            rows.append({
                "feature": name,
                "cut": cut,
                "mean_tau_low": float(np.mean(low)) if len(low) else np.nan,
                "mean_tau_high": float(np.mean(high)) if len(high) else np.nan,
            })
        return pd.DataFrame(rows)

    def test_calibration(self):
        """Regression of the IV moment on the level and slope of tau(x)."""
        import statsmodels.api as sm

        tau = self.tau_hat_
        tau_c = tau - tau.mean()
        den = self.W_res_ * self.Z_res_
        num = self.Y_res_ * self.Z_res_
        # num ≈ a * den + b * (tau_c * den)
        design = np.column_stack([den, tau_c * den])
        fit = sm.OLS(num, design).fit(cov_type="HC1")
        table = pd.DataFrame({
            "term": ["mean effect (target  = LATE scale)", "differential effect (target = 1)"],
            "Estimate": fit.params,
            "Std. Error": fit.bse,
            "Pr(>|t|)": fit.pvalues,
        })
        return table

    def plot_cate(self, top_var=None):
        import matplotlib.pyplot as plt

        if top_var is None:
            top_var = self.feature_names_[int(np.argmax(self.stage_.variable_importance()))]
        j = self.feature_names_.index(top_var)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(self.X_[:, j], self.tau_hat_, s=12, alpha=0.35, c="#4C72B0")
        order = np.argsort(self.X_[:, j])
        window = max(20, len(order) // 30)
        kernel = np.ones(window) / window
        smooth = np.convolve(self.tau_hat_[order], kernel, mode="valid")
        xs = self.X_[order, j][window // 2: window // 2 + len(smooth)]
        ax.plot(xs, smooth, color="#C44E52", lw=2)
        ax.set_xlabel(top_var)
        ax.set_ylabel("Estimated LATE")
        ax.set_title(f"CATE along {top_var}")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Multi-arm causal boosting
# ---------------------------------------------------------------------------

class MultiArmCausalBoost:
    """T-learner with one XGBoost outcome model per arm and outcome."""

    def __init__(self, parameters=None, baseline="placebo"):
        self.parameters = dict(parameters or {})
        self.baseline = baseline
        self.outcome_models_ = {}

    def fit(self, X, Y, W, nrounds=40, verbose=0, feature_names=None, outcome_names=None):
        X = _as_float_matrix(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        W = np.asarray(W).astype(str)
        self.X_ = X
        self.Y_ = Y
        self.W_ = W
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(X.shape[1])
        ]
        self.outcome_names_ = list(outcome_names) if outcome_names is not None else [
            f"y{i}" for i in range(Y.shape[1])
        ]
        levels = [str(level) for level in pd.unique(W)]
        if self.baseline not in levels:
            self.baseline = levels[0]
        self.arms_ = levels
        params = self._xgb_params()
        self.outcome_models_ = {}
        for arm in self.arms_:
            self.outcome_models_[arm] = {}
            mask = W == arm
            for m, name in enumerate(self.outcome_names_):
                model = XGBRegressor(**params, n_estimators=int(nrounds), verbosity=int(verbose))
                model.fit(X[mask], Y[mask, m])
                self.outcome_models_[arm][name] = model
        self.propensity_ = XGBClassifier(
            **params,
            n_estimators=int(nrounds),
            verbosity=0,
            objective="multi:softprob",
            num_class=len(self.arms_),
        )
        codes = pd.Categorical(W, categories=self.arms_).codes
        self.propensity_.fit(X, codes)
        return self

    def predict(self, X):
        X = _as_float_matrix(X)
        n = len(X)
        n_arms = len(self.arms_)
        n_out = len(self.outcome_names_)
        mu = np.zeros((n, n_arms, n_out))
        for a, arm in enumerate(self.arms_):
            for m, name in enumerate(self.outcome_names_):
                mu[:, a, m] = self.outcome_models_[arm][name].predict(X)
        base = self.arms_.index(self.baseline)
        contrast_arms = [arm for arm in self.arms_ if arm != self.baseline]
        tau = np.zeros((n, len(contrast_arms), n_out))
        contrast_names = []
        for j, arm in enumerate(contrast_arms):
            a = self.arms_.index(arm)
            tau[:, j, :] = mu[:, a, :] - mu[:, base, :]
            contrast_names.append(f"{arm} - {self.baseline}")
        ps = self.propensity_.predict_proba(X)
        return {
            "mu_hat": mu,
            "tau_hat": tau,
            "propensity_score": ps,
            "arms": self.arms_,
            "contrast_names": contrast_names,
            "outcome_names": self.outcome_names_,
        }

    def summary(self):
        pred = self.predict(self.X_)
        print("MultiArmCausalBoost")
        print(f"  baseline = {self.baseline}")
        print(f"  arms = {self.arms_}")
        print(f"  outcomes = {self.outcome_names_}")
        print(f"  n = {len(self.W_)}")
        mean_tau = pred["tau_hat"].mean(axis=0)
        table = pd.DataFrame(
            mean_tau, index=pred["contrast_names"], columns=self.outcome_names_
        )
        print(table.round(3))
        return table

    def plot_importance(self, arm, outcome=0, top_n=10, metric="Gain"):
        import matplotlib.pyplot as plt

        if isinstance(outcome, int):
            outcome_name = self.outcome_names_[outcome]
        else:
            outcome_name = outcome
        model = self.outcome_models_[arm][outcome_name]
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)[-top_n:]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            [self.feature_names_[i] for i in order],
            imp[order],
            color="#4C72B0",
        )
        ax.set_xlabel(metric)
        ax.set_title(f"Importance — arm {arm}, outcome {outcome_name}")
        fig.tight_layout()
        return fig

    def _xgb_params(self):
        alias = {
            "eta": "learning_rate",
            "lambda": "reg_lambda",
            "alpha": "reg_alpha",
            "nthread": "n_jobs",
        }
        out = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 0,
        }
        for key, value in self.parameters.items():
            if key in {"objective", "nrounds"}:
                continue
            out[alias.get(key, key)] = value
        return out


def multi_arm_dr_ate(Y, W, mu_hat, ps, baseline, arms, contrast_names, outcome_names):
    """Doubly robust plug-in ATE for multi-arm contrasts (AIPW)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    W = np.asarray(W).astype(str)
    n, n_out = Y.shape
    arm_index = {arm: i for i, arm in enumerate(arms)}
    rows = []
    for contrast in contrast_names:
        arm = contrast.split(" - ")[0]
        estimates = []
        for m in range(n_out):
            mu_k = mu_hat[:, arm_index[arm], m]
            mu_b = mu_hat[:, arm_index[baseline], m]
            e_k = np.clip(ps[:, arm_index[arm]], 0.02, 0.98)
            e_b = np.clip(ps[:, arm_index[baseline]], 0.02, 0.98)
            assigned = np.array([arm_index[w] for w in W])
            mu_w = mu_hat[np.arange(n), assigned, m]
            I_k = (W == arm).astype(float)
            I_b = (W == baseline).astype(float)
            psi = (mu_k - mu_b) + (Y[:, m] - mu_w) * (I_k / e_k - I_b / e_b)
            estimates.append(float(np.mean(psi)))
        rows.append(estimates)
    return pd.DataFrame(rows, index=contrast_names, columns=outcome_names)


def plot_local_shap(explanation, row=0, title="SHAP contributions for one prediction"):
    """Bar chart of one row of SHAP values.

    This is the static stand-in for a waterfall or force plot. Those shap
    helpers fail on some versions when a model has many features.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(explanation.values[row], dtype=float)
    names = np.asarray(list(explanation.feature_names))
    base = np.asarray(explanation.base_values, dtype=float).ravel()
    base_value = float(base[min(row, len(base) - 1)])
    order = np.argsort(np.abs(values))[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = np.where(values[order] >= 0, "#C44E52", "#4C72B0")
    ax.barh(names[order], values[order], color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"{title}\nbase {base_value:.2f} → prediction {base_value + values.sum():.2f}")
    fig.tight_layout()
    return fig


def explain_predictions(predict_fn, X, feature_names, n_explain=40, n_background=30, seed=0):
    """Permutation SHAP values for a CATE prediction function.

    Python counterpart of ``kernelshap::permshap`` + ``shapviz``.
    """
    import shap

    rng = np.random.default_rng(seed)
    X = _as_float_matrix(X)
    n = len(X)
    explain_idx = rng.choice(n, size=min(int(n_explain), n), replace=False)
    bg_idx = rng.choice(n, size=min(int(n_background), n), replace=False)

    def wrapped(data):
        out = predict_fn(np.asarray(data, dtype=float))
        return np.asarray(out, dtype=float).ravel()

    explainer = shap.PermutationExplainer(
        wrapped, X[bg_idx], feature_names=list(feature_names), seed=int(seed)
    )
    explanation = explainer(X[explain_idx], max_evals=2 * X.shape[1] + 1)
    explanation.explain_index = explain_idx
    return explanation


def save_model(model, path):
    path = Path(path)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def load_model(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)

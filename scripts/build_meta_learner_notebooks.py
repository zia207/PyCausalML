"""Rebuild the meta-learner tutorials as Python / PyCausalML notebooks.

Reads the current notebooks, keeps the explanatory markdown, drops R-package
setup, and inserts Python code that uses pycausalml. Run once on the R
notebooks; do not re-run after the notebooks have been converted.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "Tutorials" / "05_02_Meta_Learners"

DROP_IF = (
    "rpy2",
    "Mount Google Drive",
    "Install Missing Packages",
    "Verify Installation",
    "Load R Packages",
    "Check Loaded Packages",
    "Following R packages are required",
    "Check and Install Required R Packages",
    "Optional packages for the appendix",
    "Simulation sizes, learners, and fast-render flags",
)

SETUP_MD = """## Setup

This notebook uses **[PyCausalML](https://github.com/zia207/PyCausalML)** (`pycausalml`).

`fit(X, treatment, y)` returns the fitted estimator. `predict` returns a NumPy array of conditional average treatment effects. A fitted model also exposes `explain` and `plot_shap`.

Install the package from this repository, with the plotting and boosting extras.
"""

INSTALL_CODE = r'''import subprocess
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
try:
    import pycausalml  # noqa: F401
    import causaldata  # noqa: F401
    import matplotlib  # noqa: F401
    import xgboost  # noqa: F401
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-e", f"{root}[plot,boost,shap]",
        "statsmodels", "causaldata",
    ])
print("PyCausalML ready.")
'''

IMPORTS = r'''%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from pycausalml import (
    DRLearner,
    DomainAdaptationLearner,
    LinearDML,
    RLearner,
    SLearner,
    SparseLinearDML,
    TLearner,
    XLearner,
    explain_cate,
    synthetic_data,
)
from pycausalml.data import make_multiarm_data
from pycausalml.double_machine_learning import CausalForestDML, PartiallyLinearDML

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
N_EST = 25
print("Imported PyCausalML")
'''

METRICS = r'''def ate_row(name, model, X):
    cate = np.asarray(model.predict(X), dtype=float)
    info = model.average_treatment_effect(X)
    est = float(info["estimate"])
    se = float(info["std.err"])
    if not np.isfinite(se):
        se = float(np.std(cate, ddof=1) / np.sqrt(len(cate)))
        est = float(np.mean(cate))
    return {
        "learner": name,
        "ate": est,
        "se": se,
        "ci_lower": est - 1.96 * se,
        "ci_upper": est + 1.96 * se,
    }

def abs_pct_ate(pred, truth):
    denom = abs(float(np.mean(truth)))
    if denom < 1e-8:
        return np.nan
    return 100.0 * abs(float(np.mean(pred) - np.mean(truth))) / denom

def kl_divergence(pred, truth, bins=15):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    lo = float(min(pred.min(), truth.min()))
    hi = float(max(pred.max(), truth.max()))
    if not np.isfinite(lo) or hi <= lo:
        return np.nan
    hp, _ = np.histogram(pred, bins=bins, range=(lo, hi))
    hq, _ = np.histogram(truth, bins=bins, range=(lo, hi))
    hp = hp.astype(float) + 1e-3
    hq = hq.astype(float) + 1e-3
    hp /= hp.sum()
    hq /= hq.sum()
    return float(np.sum(hp * np.log(hp / hq)))

def cumulative_gain(y, w, score):
    order = np.argsort(-np.asarray(score, dtype=float))
    y = np.asarray(y, dtype=float)[order]
    w = np.asarray(w)[order]
    n = len(y)
    frac = np.arange(1, n + 1) / n
    gain = np.empty(n)
    for i in range(n):
        ww = w[: i + 1]
        yy = y[: i + 1]
        n1 = int((ww == 1).sum())
        n0 = int((ww == 0).sum())
        gain[i] = 0.0 if n1 == 0 or n0 == 0 else (yy[ww == 1].mean() - yy[ww == 0].mean()) * (i + 1) / n
    return frac, gain

def tutorial_file(name: str) -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        for rel in (
            Path("Tutorials") / "05_02_Meta_Learners" / "data" / name,
            Path("data") / name,
        ):
            path = cand / rel
            if path.is_file():
                return path
    raise FileNotFoundError(name)
'''

ABORTION_LOAD = r'''from causaldata import abortion

abortion_df = abortion.load_pandas().data.copy()
print("Abortion dataset loaded. Columns:", list(abortion_df.columns))
print(abortion.NOTE)
print(abortion.DESCRLONG)
print(abortion.SOURCE)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
plot_df = abortion_df.dropna(subset=["lnr"]).copy()
counts = plot_df["repeal"].value_counts().sort_index()
axes[0, 0].bar(["No Early Repeal", "Early Repeal"], counts.to_numpy(), color=["skyblue", "tomato"])
axes[0, 0].set_title("Treatment group sizes")
axes[0, 0].set_ylabel("Number of state-year observations")
plot_df.boxplot(column="lnr", by="repeal", ax=axes[0, 1])
axes[0, 1].set_title("Logged gonorrhea rate by repeal status")
axes[0, 1].set_xlabel("Early repeal (1=Yes, 0=No)")
plot_df.boxplot(column="income", by="repeal", ax=axes[1, 0])
axes[1, 0].set_title("Real income per capita by repeal status")
axes[1, 0].set_xlabel("Early repeal (1=Yes, 0=No)")
plot_df.boxplot(column="alcohol", by="repeal", ax=axes[1, 1])
axes[1, 1].set_title("Alcohol consumption by repeal status")
axes[1, 1].set_xlabel("Early repeal (1=Yes, 0=No)")
plt.suptitle("Abortion dataset: treatment, outcome, and key covariates")
fig.tight_layout()
plt.show()

feature_names = [
    c for c in plot_df.columns
    if c not in ("lnr", "repeal") and plot_df[c].notna().all()
]
X_ab = plot_df[feature_names].to_numpy(dtype=float)
Y_ab = plot_df["lnr"].to_numpy(dtype=float)
W_ab = plot_df["repeal"].to_numpy(dtype=int)
print("Covariates used:", feature_names)
print("n =", len(Y_ab), " treated =", int(W_ab.sum()))

rng_ab = np.random.default_rng(333)
n_ab = len(Y_ab)
train_idx = rng_ab.choice(n_ab, size=round(0.8 * n_ab), replace=False)
test_idx = np.setdiff1d(np.arange(n_ab), train_idx)
X_ab_tr, X_ab_te = X_ab[train_idx], X_ab[test_idx]
Y_ab_tr, Y_ab_te = Y_ab[train_idx], Y_ab[test_idx]
W_ab_tr, W_ab_te = W_ab[train_idx], W_ab[test_idx]
print("Training set size:", len(Y_ab_tr))
print("Test set size:", len(Y_ab_te))
'''

ABORTION_COLUMNS = """### Data

The **abortion** dataset contains state-year observations on gonorrhea rates among 15–19 year olds, with a binary indicator for early repeal of abortion prohibition. Treatment is `repeal` (1 = early repeal, 0 = otherwise) and the outcome is `lnr`, the logged gonorrhea rate. Rows with a missing outcome are dropped, and covariates with missing values are left out of `X`.

- **repeal** — treatment indicator
- **lnr** — logged gonorrhea rate among 15–19 year olds (outcome)
- **age, race, year, sex, totpop** — demographics
- **ir, crack, alcohol, income, ur, poverty** — state context
- **pi, bf15, younger, t** — additional policy and composition measures
"""


def _id() -> str:
    return uuid.uuid4().hex[:12]


def md(text: str):
    cell = new_markdown_cell(text.strip() + "\n")
    cell["id"] = _id()
    return cell


def code(text: str):
    cell = new_code_cell(text.strip() + "\n")
    cell["id"] = _id()
    return cell


def clean_md(text: str) -> str | None:
    lines = text.strip().splitlines()
    first = lines[0] if lines else ""
    if any(key in first for key in DROP_IF):
        rest = []
        keep = False
        for line in lines[1:]:
            if line.startswith("#"):
                keep = True
            if keep:
                rest.append(line)
        text = "\n".join(rest)
        if not text.strip():
            return None
    for marker in (
        "Following R packages are required",
        "### Check and Install Required",
        "### Install Missing Packages",
        "Optional packages for the appendix",
        "## Set Up",
    ):
        if marker in text and not text.strip().startswith(marker):
            text = text.split(marker)[0]
    if text.strip() in {"## Setup", "## Set Up", "## Setup  "}:
        return SETUP_MD
    text = text.replace("RCausalML", "PyCausalML")
    text = text.replace("## Implementation in R", "## Implementation in Python")
    text = text.replace("R/meta_learners.R", "`pycausalml.meta_learners`")
    text = text.replace("R/DMLearner.R", "`pycausalml.double_machine_learning`")
    text = text.replace("{kernelshap}", "permutation SHAP")
    text = text.replace("kernelshap", "permutation SHAP")
    text = text.replace("shapviz", "the PyCausalML SHAP plots")
    text = text.replace("{causaldata}", "`causaldata`")
    text = text.replace("{causaldrf}", "the NMES table shipped with this tutorial")
    text = text.replace("{survival}", "the lung-cancer table shipped with this tutorial")
    text = text.replace("R package **ranger**", "random forests (`learner=\"rf\"`)")
    text = text.replace("R package **xgboost**", "XGBoost (`learner=\"xgb\"`)")
    text = text.replace("**ranger**", "random forests")
    text = text.replace("`ranger`", '`learner="rf"`')
    text = text.replace("grf::causal_forest", "a causal forest")
    text = text.replace("`grf`", "generalized random forests")
    text = re.sub(r"\{[A-Za-z0-9_.]+\}", "", text)
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if "cran.r-project.org" in low or "r package" in low:
            continue
        if "install.packages" in low or "library(" in low:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return None
    return text + "\n"


def load_markdown(filename: str) -> list[str]:
    nb = json.loads((NB_DIR / filename).read_text())
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        raw = "".join(cell["source"]).strip()
        if raw.startswith("<a ") or raw.startswith("![R Banner]") or raw.startswith("![Python Banner]"):
            continue
        cleaned = clean_md(raw)
        if cleaned:
            out.append(cleaned)
    return out


def assemble(chunks: list[str], inserts: list[tuple[str, str]], prefix: list | None = None):
    """Place each code block after the first markdown chunk that contains its anchor."""
    used = set()
    cells = list(prefix or [])
    remaining = list(inserts)
    for chunk in chunks:
        cells.append(md(chunk))
        still = []
        for anchor, src in remaining:
            if anchor not in used and anchor in chunk:
                cells.append(code(src))
                used.add(anchor)
            else:
                still.append((anchor, src))
        remaining = still
    for anchor, src in remaining:
        raise SystemExit(f"Unmatched anchor: {anchor}")
    return cells


def banner(filename: str):
    colab = (
        f'<a href="https://colab.research.google.com/github/zia207/Causal_Inference_Python/blob/main/'
        f'Notebook/05_02_Meta_Learners/{filename}" target="_parent">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    return [md(colab), md("![Python Banner](../../Image/05_02_meta_learners.png)")]


def finish(filename: str, cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
    }
    path = NB_DIR / filename
    nbformat.write(nb, path)
    print(f"wrote {filename} ({len(cells)} cells)")


def build_intro():
    name = "02_08_05_02_00_meta_learners_introduction.ipynb"
    chunks = load_markdown(name)
    chunks = [c for c in chunks if not c.strip().startswith("<a ")]
    chunks = [c.replace("![R Banner]", "![Python Banner]") for c in chunks]
    resources = """## Resources

1. **Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).** *Metalearners for estimating heterogeneous treatment effects using machine learning.* PNAS.

2. **Nie, X., & Wager, S. (2021).** *Quasi-oracle estimation of heterogeneous treatment effects.* Biometrika.

3. **Chernozhukov, V., et al. (2018).** *Double/Debiased Machine Learning for Treatment and Structural Parameters.* The Econometrics Journal.

4. **Kennedy, E. H. (2023).** *Semiparametric doubly robust targeted double machine learning: a review.* Statistical Science.

5. **Athey, S., & Imbens, G. W. (2016).** *Recursive partitioning for heterogeneous causal effects.* PNAS.

6. **PyCausalML** — S-, T-, X-, R-, DR-, and domain-adaptation learners in `pycausalml.meta_learners`, used throughout this section.
"""
    chunks = [resources if c.strip().startswith("## Resources") else c for c in chunks]
    intro_code = '''from pycausalml import (
    DRLearner,
    DomainAdaptationLearner,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)

print("Meta-learners in PyCausalML:")
for cls in (SLearner, TLearner, XLearner, RLearner, DRLearner, DomainAdaptationLearner):
    print("-", cls.__name__)
'''
    cells = assemble(chunks, [("## Setup", INSTALL_CODE)], prefix=banner(name))
    finish(name, _inject_after_install(cells, intro_code))


def _inject_after_install(cells, extra_code: str):
    out = []
    for cell in cells:
        out.append(cell)
        if cell["cell_type"] == "code" and "PyCausalML ready" in "".join(cell["source"]):
            out.append(code(extra_code))
    return out


BINARY_GEN = r'''rng = np.random.default_rng(42)
dat = synthetic_data(mode=2, n=1200, p=10, sigma=1.0, random_state=42)
X = dat["X"]
y = dat["y"]
w = dat["w"]
tau = dat["tau"]
feature_names = [f"X{i+1}" for i in range(X.shape[1])]
print(X.shape, "treatment rate", round(float(w.mean()), 3), "true ATE", round(float(tau.mean()), 3))
'''

BINARY_ATE = r'''bases = {"Linear regression": "lm", "Random forest": "rf", "XGBoost": "xgb"}
families = {
    "S-Learner": SLearner,
    "T-Learner": TLearner,
    "X-Learner": XLearner,
    "R-Learner": RLearner,
}
fitted = {}
rows = []
for family, cls in families.items():
    for base_name, base in bases.items():
        model = cls(learner=base, n_estimators=N_EST, random_state=42)
        model.fit(X, w, y, feature_names=feature_names)
        key = f"{family} ({base_name})"
        fitted[key] = model
        row = ate_row(key, model, X)
        row["true_ate"] = float(tau.mean())
        rows.append(row)
ate_table = pd.DataFrame(rows)
print(ate_table.round(3).to_string(index=False))
print("PyCausalML estimates the propensity score inside the X-Learner and the R-Learner.")
'''

BINARY_CATE = r'''cate = {name: np.asarray(model.predict(X), dtype=float) for name, model in fitted.items()}
show = [k for k in cate if k.startswith(("T-Learner", "X-Learner", "R-Learner"))]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for name in show:
    ax.hist(cate[name], bins=28, histtype="step", density=True, lw=1.4, label=name)
ax.hist(tau, bins=28, histtype="step", density=True, color="black", lw=2, label="Actual")
ax.set_xlabel("CATE")
ax.set_ylabel("Density")
ax.set_title("Predicted CATE versus the true conditional effect")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
plt.show()
'''

BINARY_VALID = METRICS + r'''
k_sims = 3
records = []
for sim in range(k_sims):
    draw = synthetic_data(mode=2, n=900, p=8, sigma=1.0, random_state=100 + sim)
    Xs, ys, ws, taus = draw["X"], draw["y"], draw["w"], draw["tau"]
    names = [f"X{i+1}" for i in range(Xs.shape[1])]
    rng = np.random.default_rng(sim)
    hold = rng.choice(len(ys), size=round(0.2 * len(ys)), replace=False)
    train = np.setdiff1d(np.arange(len(ys)), hold)
    for family, cls in families.items():
        for base_name, base in bases.items():
            model = cls(learner=base, n_estimators=N_EST, random_state=sim)
            model.fit(Xs[train], ws[train], ys[train], feature_names=names)
            for split, idx in (("train", train), ("validation", hold)):
                pred = model.predict(Xs[idx])
                records.append({
                    "learner": f"{family} ({base_name})",
                    "split": split,
                    "abs_pct_ate": abs_pct_ate(pred, taus[idx]),
                    "mse": float(np.mean((pred - taus[idx]) ** 2)),
                    "kl": kl_divergence(pred, taus[idx]),
                })
valid_df = pd.DataFrame(records)
summary = (
    valid_df.groupby(["learner", "split"], as_index=False)[["abs_pct_ate", "mse", "kl"]]
    .mean()
)
train_summary = summary[summary["split"] == "train"].drop(columns="split")
validation_summary = summary[summary["split"] == "validation"].drop(columns="split")
print("Train summary")
print(train_summary.round(3).to_string(index=False))
print("\nValidation summary")
print(validation_summary.round(3).to_string(index=False))
'''

BINARY_SCATTER = r'''merged = train_summary.merge(validation_summary, on="learner", suffixes=("_train", "_valid"))
metrics = ["abs_pct_ate", "mse", "kl"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, metric in zip(axes, metrics):
    ax.scatter(merged[f"{metric}_train"], merged[f"{metric}_valid"], s=40)
    for _, row in merged.iterrows():
        ax.annotate(row["learner"].split()[0], (row[f"{metric}_train"], row[f"{metric}_valid"]), fontsize=7)
    lims = [
        min(merged[f"{metric}_train"].min(), merged[f"{metric}_valid"].min()),
        max(merged[f"{metric}_train"].max(), merged[f"{metric}_valid"].max()),
    ]
    ax.plot(lims, lims, color="grey", lw=0.8)
    ax.set_xlabel("Train")
    ax.set_ylabel("Validation")
    ax.set_title(metric)
fig.tight_layout()
plt.show()
'''

BINARY_BAR = r'''plot_df = validation_summary[~validation_summary["learner"].str.startswith("S-Learner")].copy()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric in zip(axes, ["abs_pct_ate", "mse", "kl"]):
    ax.barh(plot_df["learner"], plot_df[metric], color="#1F4E79")
    ax.set_title(metric + " (validation)")
fig.tight_layout()
plt.show()
'''

BINARY_SINGLE = r'''draw = synthetic_data(mode=2, n=1000, p=8, sigma=1.0, random_state=7)
rng = np.random.default_rng(7)
hold = rng.choice(len(draw["y"]), size=200, replace=False)
train = np.setdiff1d(np.arange(len(draw["y"])), hold)
names = [f"X{i+1}" for i in range(draw["X"].shape[1])]
single = {}
for family, cls in (("T-Learner", TLearner), ("X-Learner", XLearner), ("R-Learner", RLearner)):
    model = cls(learner="xgb", n_estimators=N_EST, random_state=7)
    model.fit(draw["X"][train], draw["w"][train], draw["y"][train], feature_names=names)
    single[family] = {
        "train": model.predict(draw["X"][train]),
        "valid": model.predict(draw["X"][hold]),
        "tau_train": draw["tau"][train],
        "tau_valid": draw["tau"][hold],
        "model": model,
        "X_train": draw["X"][train],
        "y_train": draw["y"][train],
        "w_train": draw["w"][train],
        "names": names,
    }
print("Single simulation fitted:", list(single))
'''

BINARY_DIST = r'''fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, split, truth_key in ((axes[0], "train", "tau_train"), (axes[1], "valid", "tau_valid")):
    for family, payload in single.items():
        ax.hist(payload[split], bins=25, histtype="step", density=True, label=family)
    ax.hist(single["T-Learner"][truth_key], bins=25, histtype="step", density=True, color="black", label="Actual")
    ax.set_title(f"CATE density — {split}")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
'''

BINARY_PRED_SCATTER = r'''fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
for col, family in enumerate(single):
    for row, (split, truth_key) in enumerate((("train", "tau_train"), ("valid", "tau_valid"))):
        ax = axes[row, col]
        ax.scatter(single[family][truth_key], single[family][split], s=10, alpha=0.5)
        ax.set_title(f"{family} — {split}")
        ax.set_xlabel("Actual CATE")
        ax.set_ylabel("Predicted CATE")
fig.tight_layout()
plt.show()
'''

BINARY_AUUC = r'''fig, ax = plt.subplots(figsize=(6.5, 4))
auuc_rows = []
for family, payload in single.items():
    frac, gain = cumulative_gain(payload["y_train"], payload["w_train"], payload["train"])
    ax.plot(frac, gain, label=family)
    auuc_rows.append({"learner": family, "auuc": float(np.trapezoid(gain, frac))})
frac, gain = cumulative_gain(payload["y_train"], payload["w_train"], payload["tau_train"])
ax.plot(frac, gain, color="black", label="Actual")
auuc_rows.append({"learner": "Actual", "auuc": float(np.trapezoid(gain, frac))})
ax.set_xlabel("Fraction of units targeted")
ax.set_ylabel("Cumulative gain")
ax.set_title("Cumulative gain on the training split")
ax.legend()
fig.tight_layout()
plt.show()
print(pd.DataFrame(auuc_rows).round(3).to_string(index=False))
'''

BINARY_SHAP = r'''shap_models = {
    "S-Learner": fitted["S-Learner (XGBoost)"],
    "T-Learner": fitted["T-Learner (XGBoost)"],
    "X-Learner": fitted["X-Learner (XGBoost)"],
    "R-Learner": fitted["R-Learner (XGBoost)"],
}
explanations = {}
for name, model in shap_models.items():
    explanations[name] = explain_cate(model, X, max_samples=24, n_permutations=6, random_state=1)
    print(name, "mean |SHAP|", np.round(explanations[name].importance(), 3))
'''

BINARY_SHAP_PLOT = r'''exp = explanations["T-Learner"]
exp.plot_importance(title="T-Learner (XGBoost): SHAP importance")
plt.show()
exp.plot_waterfall(row=int(np.argmax(exp.predictions)), title="T-Learner waterfall at the highest predicted CATE")
plt.show()
'''

BINARY_AB_FIT = r'''ab_models = {}
ab_rows = []
for family, cls in families.items():
    model = cls(learner="rf", n_estimators=N_EST, random_state=333)
    model.fit(X_ab_tr, W_ab_tr, Y_ab_tr, feature_names=feature_names)
    ab_models[family] = model
    row = ate_row(family, model, X_ab_tr)
    row["mean_cate_test"] = float(np.mean(model.predict(X_ab_te)))
    ab_rows.append(row)
print(pd.DataFrame(ab_rows).round(3).to_string(index=False))
'''

BINARY_AB_HIST = r'''fig, ax = plt.subplots(figsize=(7.5, 4))
for family, model in ab_models.items():
    ax.hist(model.predict(X_ab_tr), bins=30, histtype="step", density=True, label=family)
ax.set_xlabel("Predicted CATE on lnr")
ax.set_title("Abortion data: training-set CATE")
ax.legend()
fig.tight_layout()
plt.show()
'''

BINARY_AB_SHAP = r'''ab_explain = explain_cate(
    ab_models["T-Learner"], X_ab_tr, max_samples=24, n_permutations=6,
    feature_names=feature_names, random_state=333,
)
print("T-Learner mean |SHAP|")
print(pd.Series(ab_explain.importance(), index=ab_explain.feature_names).round(3).sort_values(ascending=False).to_string())
'''

BINARY_AB_SHAP_PLOT = r'''ab_explain.plot_importance(title="Abortion data: T-Learner SHAP importance")
plt.show()
ab_explain.plot_waterfall(row=0, title="Abortion data: T-Learner waterfall")
plt.show()
'''


def build_binary():
    name = "02_08_05_02_01_meta_learners_binary_treatments.ipynb"
    chunks = load_markdown(name)
    chunks = [
        c.replace(
            "with `PyCausalML`",
            "with **PyCausalML**",
        ).replace("In this notebook, we will generate", "In this notebook, we generate")
        for c in chunks
    ]
    inserts = [
        ("## Implementation in Python", INSTALL_CODE),
        ("### Generate synthetic data", IMPORTS + "\n" + METRICS + "\n" + BINARY_GEN),
        ("#### R-Learner with and without propensity score input", BINARY_ATE),
        ("### Calculate Individual Treatment Effect (ITE/CATE)", BINARY_CATE),
        ("## Part B: Validating Meta-Learner Accuracy", BINARY_VALID),
        ("### Scatter plot: Train vs Validation", BINARY_SCATTER),
        ("### Bar plots", BINARY_BAR),
        ("### Single simulation: train and validation predictions", BINARY_SINGLE),
        ("### Distribution plot for single simulation", BINARY_DIST),
        ("### Scatter plots: predicted CATE vs actual", BINARY_PRED_SCATTER),
        ("### Cumulative Gain AUUC", BINARY_AUUC),
        ("## Part C: SHAP and interpretability", BINARY_SHAP),
        ("### Beeswarm and waterfall plots for one meta-learner (T-Learner with XGBoost base)", BINARY_SHAP_PLOT),
        ("### Data Preparation", ABORTION_LOAD),
        ("### Fit ATE meta-learners on training split and estimate ATE", BINARY_AB_FIT),
        ("### Visualize predicted CATE distribution on training split", BINARY_AB_HIST),
        ("### SHAP for predicted CATE (abortion, training subsample)", BINARY_AB_SHAP),
        ("### Visualize SHAP with beeswarm and waterfall plots for one meta-learner (T-Learner with ranger base)", BINARY_AB_SHAP_PLOT),
    ]
    # Setup cell from clean_md is SETUP_MD, which contains "## Setup"
    cells = assemble(chunks, inserts, prefix=banner(name))
    finish(name, cells)


if __name__ == "__main__":
    build_intro()
    build_binary()

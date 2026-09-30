"""Convert the uplift-learner tutorials from R to PyCausalML notebooks.

Reads each notebook, keeps the teaching markdown, replaces R code and R-package
wording, and writes the Python notebooks in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "Tutorials" / "05_03_Uplift_Learners"
BANNER = "![Python Banner](../../Image/05_03_uplift_learners.png)\n"

INSTALL = """
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
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-e", f"{root}[plot,boost,shap]",
    "causaldata", "statsmodels",
])
print("PyCausalML ready.")
""".strip() + "\n"

IMPORTS_SYNTH = """
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycausalml import CausalTree, InteractionTree, TLearner, UpliftRandomForest, UpliftTree
from pycausalml.tree_based_models import plot_causal_tree
from pycausalml.uplift_learners import (
    format_uplift_tree,
    plot_uplift_curve,
    plot_uplift_tree,
    qini_coefficient,
    qini_curve,
)

plt.rcParams["figure.dpi"] = 110


def make_uplift_classification(
    treatment_name=("control", "treatment1", "treatment2", "treatment3"),
    y_name="conversion",
    n_samples=2000,
    n_classification_features=10,
    n_classification_informative=5,
    n_uplift_increase_dict=None,
    n_uplift_decrease_dict=None,
    delta_uplift_increase_dict=None,
    delta_uplift_decrease_dict=None,
    random_seed=111,
):
    \"\"\"Synthetic multi-arm classification data for uplift trees.

    Informative features shift the baseline conversion probability. A separate
    block of features raises or lowers the uplift of each treatment arm.
    \"\"\"
    rng = np.random.default_rng(random_seed)
    names = list(treatment_name)
    arms = names[1:]
    X = rng.normal(size=(n_samples, n_classification_features))
    beta = np.zeros(n_classification_features)
    n_info = min(n_classification_informative, n_classification_features)
    beta[:n_info] = 0.55
    baseline = 1.0 / (1.0 + np.exp(-(X @ beta)))
    arm_id = rng.integers(0, len(names), size=n_samples)
    uplift = np.zeros(n_samples)
    for j, arm in enumerate(arms):
        n_inc = int(n_uplift_increase_dict[arm])
        n_dec = int(n_uplift_decrease_dict[arm])
        d_inc = float(delta_uplift_increase_dict[arm])
        d_dec = float(delta_uplift_decrease_dict[arm])
        inc = (X[:, :n_inc] > 0).mean(axis=1)
        dec = (X[:, n_inc : n_inc + n_dec] > 0).mean(axis=1)
        uplift[arm_id == (j + 1)] = d_inc * inc[arm_id == (j + 1)] + d_dec * dec[arm_id == (j + 1)]
    probability = np.clip(baseline + uplift, 0.02, 0.98)
    y = rng.binomial(1, probability)
    columns = [f"x{i}" for i in range(n_classification_features)]
    frame = pd.DataFrame(X, columns=columns)
    frame["treatment_group_key"] = np.asarray(names)[arm_id]
    frame[y_name] = y
    return frame, columns


def make_uplift_regression(
    treatment_name=("control", "treatment1", "treatment2", "treatment3"),
    y_name="outcome",
    n_samples=2000,
    n_regression_features=10,
    n_regression_informative=5,
    n_uplift_increase_dict=None,
    n_uplift_decrease_dict=None,
    delta_uplift_increase_dict=None,
    delta_uplift_decrease_dict=None,
    sigma=1.0,
    random_seed=111,
):
    \"\"\"Synthetic multi-arm regression data. Same feature blocks as classification.\"\"\"
    rng = np.random.default_rng(random_seed)
    names = list(treatment_name)
    arms = names[1:]
    X = rng.normal(size=(n_samples, n_regression_features))
    beta = np.zeros(n_regression_features)
    n_info = min(n_regression_informative, n_regression_features)
    beta[:n_info] = 0.8
    baseline = X @ beta
    arm_id = rng.integers(0, len(names), size=n_samples)
    uplift = np.zeros(n_samples)
    for j, arm in enumerate(arms):
        n_inc = int(n_uplift_increase_dict[arm])
        n_dec = int(n_uplift_decrease_dict[arm])
        d_inc = float(delta_uplift_increase_dict[arm])
        d_dec = float(delta_uplift_decrease_dict[arm])
        inc = (X[:, :n_inc] > 0).mean(axis=1)
        dec = (X[:, n_inc : n_inc + n_dec] > 0).mean(axis=1)
        uplift[arm_id == (j + 1)] = d_inc * inc[arm_id == (j + 1)] + d_dec * dec[arm_id == (j + 1)]
    y = baseline + uplift + rng.normal(0.0, sigma, size=n_samples)
    columns = [f"x{i}" for i in range(n_regression_features)]
    frame = pd.DataFrame(X, columns=columns)
    frame["treatment_group_key"] = np.asarray(names)[arm_id]
    frame[y_name] = y
    return frame, columns


def fit_multi(df, x_names, y_name, factory, control="control"):
    models = {}
    for arm in [a for a in df["treatment_group_key"].unique() if a != control]:
        sub = df[df["treatment_group_key"].isin([control, arm])]
        model = factory()
        model.fit(
            sub[x_names].to_numpy(),
            (sub["treatment_group_key"].to_numpy() != control).astype(int),
            sub[y_name].to_numpy(),
            feature_names=x_names,
        )
        models[arm] = model
    return models


def predict_multi(models, X, kind="uplift"):
    frame = {}
    for i, (arm, model) in enumerate(models.items()):
        if kind == "rates" and hasattr(model, "predict_rates"):
            p_control, p_treat = model.predict_rates(X)
            if i == 0:
                frame["control"] = p_control
            frame[arm] = p_treat
        elif kind == "means" and hasattr(model, "mu0_"):
            if i == 0:
                frame["control"] = np.asarray(model.mu0_.predict(X), dtype=float)
            frame[arm] = np.asarray(model.mu1_.predict(X), dtype=float)
        else:
            frame[arm] = np.asarray(model.predict(X), dtype=float)
    return pd.DataFrame(frame)


def plot_forest(forest, title="Uplift forest", n_show=4):
    n_show = min(n_show, len(forest.trees_))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, tree in zip(np.ravel(axes), forest.trees_[:n_show]):
        plot_uplift_tree(tree, title=tree.criterion.upper() if hasattr(tree, "criterion") else "Tree", ax=ax)
    for ax in np.ravel(axes)[n_show:]:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


print("Imported PyCausalML")
""".strip() + "\n"

IMPORTS_ABORTION = """
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pycausalml import (
    CausalTree,
    InteractionTree,
    RLearner,
    SLearner,
    TLearner,
    UpliftRandomForest,
    UpliftXGBoost,
)
from pycausalml.tree_based_models import plot_causal_tree
from pycausalml.uplift_learners import (
    format_uplift_tree,
    plot_uplift_curve,
    plot_uplift_tree,
    qini_coefficient,
    qini_curve,
)

plt.rcParams["figure.dpi"] = 110


def f_filter(X, y, w, names, order=1):
    \"\"\"Rank features by the partial F statistic of treatment interactions.

    Order 1 tests ``Y ~ X + W + X:W``. Higher orders add powers of ``X`` and
    the matching treatment interactions, following the F-filter idea in
    Zhao et al. (2020).
    \"\"\"
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    rows = []
    for j, name in enumerate(names):
        xj = np.asarray(X[:, j], dtype=float)
        scale = float(np.std(xj))
        if scale < 1e-8:
            rows.append({"feature": name, "score": np.nan})
            continue
        xj = (xj - float(np.mean(xj))) / scale
        main = [np.ones(len(y)), w, xj]
        for power in range(2, order + 1):
            main.append(xj ** power)
        n_main = len(main)
        interactions = [w * (xj ** power) for power in range(1, order + 1)]
        design = np.column_stack(main + interactions)
        try:
            full = sm.OLS(y, design).fit()
            restricted = sm.OLS(y, design[:, :n_main]).fit()
            q = design.shape[1] - n_main
            score = ((restricted.ssr - full.ssr) / q) / (full.ssr / full.df_resid)
        except Exception:
            score = np.nan
        rows.append({"feature": name, "score": float(score)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def plot_qini_models(y, treatment, scores, title):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    table = []
    for name, score in scores.items():
        curve = qini_curve(y, treatment, score)
        ax.plot(curve["fraction"], curve["qini"], label=name)
        table.append({"model": name, "qini": qini_coefficient(y, treatment, score)})
    ax.set_xlabel("Fraction targeted")
    ax.set_ylabel("Cumulative incremental gain")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame(table)


print("Imported PyCausalML")
""".strip() + "\n"


def md(text: str):
    return new_markdown_cell(text.strip("\n") + "\n")


def code(text: str):
    return new_code_cell(text.strip("\n") + "\n")


def src(cell) -> str:
    return "".join(cell.get("source", []))


def set_src(cell, text: str) -> None:
    cell["source"] = text if text.endswith("\n") else text + "\n"
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def load(name: str):
    path = TUTORIALS / name
    return json.loads(path.read_text()), path


def save(nb_dict, path: Path) -> None:
    nb = nbformat.from_dict(nb_dict)
    nbformat.validator.normalize(nb)
    path.write_text(nbformat.writes(nb))
    print("wrote", path.name, "cells", len(nb.cells))


def banner_first(nb) -> None:
    cells = nb["cells"]
    if cells and "05_03_uplift_learners.png" in src(cells[0]):
        return
    if cells and ("R Banner" in src(cells[0]) or "drive.google.com" in src(cells[0])):
        set_src(cells[0], BANNER)
        cells[0]["cell_type"] = "markdown"
        return
    cells.insert(0, new_markdown_cell(BANNER))


def drop_cells(nb, predicates) -> None:
    kept = []
    for cell in nb["cells"]:
        text = src(cell)
        if any(pred(text) for pred in predicates):
            continue
        kept.append(cell)
    nb["cells"] = kept


def insert_after(nb, heading_snippet: str, new_cells) -> None:
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        if heading_snippet in src(cell):
            for offset, new in enumerate(new_cells, start=1):
                cells.insert(i + offset, new)
            return
    raise RuntimeError(f"heading not found: {heading_snippet}")


def replace_code_containing(nb, needle: str, new_source: str, nth: int = 1) -> None:
    seen = 0
    for cell in nb["cells"]:
        text = src(cell)
        if cell["cell_type"] == "code" and "%%R" in text and needle in text:
            seen += 1
            if seen == nth:
                set_src(cell, new_source)
                return
    raise RuntimeError(f"code cell not found ({nth}): {needle[:80]}")


def replace_markdown_containing(nb, needle: str, new_source: str, nth: int = 1) -> None:
    seen = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and needle in src(cell):
            seen += 1
            if seen == nth:
                set_src(cell, new_source)
                return
    raise RuntimeError(f"markdown cell not found ({nth}): {needle[:80]}")


SYNTH_ARGS = """
treatment_name=("control", "treatment1", "treatment2", "treatment3"),
y_name="{y_name}",
n_samples=2000,
n_{kind}_features=10,
n_{kind}_informative=5,
n_uplift_increase_dict={{"treatment1": 5, "treatment2": 5, "treatment3": 5}},
n_uplift_decrease_dict={{"treatment1": 3, "treatment2": 3, "treatment3": 3}},
delta_uplift_increase_dict={{"treatment1": {d1}, "treatment2": {d2}, "treatment3": {d3}}},
delta_uplift_decrease_dict={{"treatment1": {dec}, "treatment2": {dec}, "treatment3": {dec}}},
random_seed=111,
""".strip()


def classification_call(y_name="conversion"):
    args = SYNTH_ARGS.format(y_name=y_name, kind="classification", d1="0.1", d2="0.15", d3="0.2", dec="-0.1")
    return f"df, x_names = make_uplift_classification(\n    {args}\n)\n"


def regression_call():
    args = SYNTH_ARGS.format(
        y_name="outcome", kind="regression", d1="0.4", d2="0.6", d3="0.8", dec="-0.35"
    )
    return (
        "df, x_names = make_uplift_regression(\n    "
        + args
        + ",\n    sigma=1.0,\n)\n"
    )


def build_intro() -> None:
    nb, path = load("02_08_05_03_00_uplift_trees_introduction.ipynb")
    banner_first(nb)
    text = src(nb["cells"][1]) if len(nb["cells"]) > 1 else ""
    if not text.lstrip().startswith("#"):
        # Banner replacement keeps the title in cell 1.
        pass
    save(nb, path)


def _already_python(nb) -> bool:
    blob = "\n".join(src(cell) for cell in nb["cells"])
    return "%%R" not in blob and "RCausalML" not in blob and "R Banner" not in blob


def build_classification() -> None:
    nb, path = load("02_08_05_03_01_uplift_trees_classification.ipynb")
    if _already_python(nb):
        print("skip classification (already Python)")
        return
    banner_first(nb)
    drop_cells(nb, [lambda t: "rpart.plot" in t or "use_rpart_plot" in t])

    replace_markdown_containing(
        nb,
        "RCausalML",
        """
## Uplift Trees in Python

In this section we use PyCausalML to fit an uplift tree and an uplift random forest. A multi-arm experiment is handled by fitting one binary model per treatment arm (control versus that arm). Predictions on the test set are the arm-specific uplift scores. Kernel-style explanations use `explain`, which attributes the predicted uplift (CATE) to covariates.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "R/synthetic_data.R",
        """
## Getting started with simulated data

We simulate a multi-arm conversion experiment. Informative features move the baseline conversion probability. Separate feature blocks raise or lower the uplift of each treatment. The outcome is binary, so the trees split with Kullback-Leibler divergence.

### Generate synthetic dataset
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "UpliftTreeClassifier",
        """
#### Single uplift tree

`UpliftTree` fits one divergence tree on a binary treatment. For several treatments we fit one tree per arm, always against the same control group. `predict` returns uplift. `predict_rates` returns the leaf-level control and treated response probabilities.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "UpliftRandomForestClassifier",
        """
#### Uplift random forest

`UpliftRandomForest` averages uplift trees grown on bootstrap samples. With `n_trees` greater than one, the test-set prediction is the mean leaf uplift across trees. The same one-model-per-arm pattern covers multiple treatments.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "Uplift p_value",
        """
### Read the tree

- **Split line**: feature and threshold, with the gain of that split.
- **tau**: treatment-minus-control outcome at that node. On a leaf this is the uplift score.
- **n**: number of training rows in the node.
- Leaves with a positive uplift are the Persuadable region; a negative uplift flags Do-Not-Disturbs.

There is no separate validation-fill step. The next cell compares the mean predicted uplift on the training rows and the test rows.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "validation uplift score",
        """
### Train versus test uplift

The fitted tree is scored on the training matrix and the test matrix. A large gap in mean predicted uplift is a warning that the tree is fitting noise.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "trees[[1]]",
        """
### Print and plot one tree in the forest

`UpliftRandomForest.trees_` holds the fitted trees. The first tree is `trees_[0]`.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_forest[0]",
        """
### Forest plot

The panels below are four trees from the same forest. Leaf color follows the sign of the uplift: green when the leaf uplift is positive, red when it is negative.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_rf_multi()",
        """
Split the sample and fit one KL uplift tree on control versus any treatment. The leaf value is the uplift of receiving one of the active treatments rather than control.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "### Kernel SHAP values",
        """
### Permutation SHAP for predicted uplift

`explain` attributes each unit's predicted uplift to covariates. We explain a subset of the training rows. Positive SHAP means the feature pushes predicted uplift up.

This explains heterogeneity in $\\hat{\\tau}(X)$, not the raw conversion probability.
""".strip(),
        nth=1,
    )
    replace_markdown_containing(
        nb,
        "Interpretation notes",
        """
#### Interpretation notes

- **Positive SHAP** pushes predicted uplift up. **Negative SHAP** pushes it down.
- The explained quantity is $\\hat{\\tau}(X)$ from the uplift forest, not the conversion outcome.
- Dependence plots show how a covariate's contribution changes as its value changes.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "demonstrated the use of uplift trees",
        """
## Summary and Conclusion

This notebook fitted uplift trees and uplift forests with PyCausalML on a synthetic multi-arm conversion experiment. One tree per treatment arm estimates the incremental effect of that arm relative to control. The Qini curve checks whether ranking units by predicted uplift captures more incremental conversions than random targeting. `explain` shows which features drive the predicted uplift.

Uplift models separate Persuadables from Sure Things. That ranking is what a targeting policy needs: treat people whose outcome changes because of the intervention.
""".strip(),
    )

    insert_after(nb, "## Set Up", [code(INSTALL), code(IMPORTS_SYNTH)])

    replace_code_containing(nb, "make_uplift_classification", classification_call())
    replace_code_containing(nb, "head(df", "df.head()\n")
    replace_code_containing(
        nb,
        "aggregate(conversion",
        """
summary = (
    df.groupby("treatment_group_key")["conversion"]
    .agg(mean="mean", size="size")
)
summary.loc["All"] = [df["conversion"].mean(), len(df)]
print(summary.round(4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "idx_test",
        """
rng = np.random.default_rng(111)
idx_test = rng.choice(len(df), size=round(0.2 * len(df)), replace=False)
df_train = df.drop(index=idx_test).reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)
print("Train:", len(df_train), "| Test:", len(df_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_rf_multi",
        """
def one_tree():
    return UpliftTree(
        criterion="kl", max_depth=4, min_samples_leaf=40, min_samples_treatment=10, random_state=111
    )

clf = fit_multi(df_train, x_names, "conversion", one_tree)
X_test = df_test[x_names].to_numpy()
df_res_tree = predict_multi(clf, X_test, kind="rates")
df_res_tree.head()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "n_trees = 10",
        """
def one_forest():
    return UpliftRandomForest(
        criterion="kl",
        n_trees=8,
        max_depth=4,
        min_samples_leaf=30,
        random_state=111,
    )

uplift_model = fit_multi(df_train, x_names, "conversion", one_forest)
df_res = predict_multi(uplift_model, X_test, kind="rates")
uplift_hat = predict_multi(uplift_model, X_test, kind="uplift")
df_res["max_delta"] = uplift_hat.max(axis=1).to_numpy()
df_res["recommended_treatment"] = np.where(
    df_res["max_delta"] < 0, "control", uplift_hat.idxmax(axis=1)
)
print("Shape:", df_res.shape)
df_res.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "dim(y_pred)",
        """
y_pred = uplift_hat.copy()
print(y_pred.shape)
y_pred.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "names(result)",
        """
result = y_pred.copy()
result.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "best_treatment",
        """
best_treatment = np.where(result.max(axis=1).to_numpy() < 0, "control", result.idxmax(axis=1).to_numpy())
actual_is_best = (df_test["treatment_group_key"].to_numpy() == best_treatment).astype(int)
actual_is_control = (df_test["treatment_group_key"].to_numpy() == "control").astype(int)
""".strip(),
    )
    replace_code_containing(
        nb,
        "synthetic <-",
        """
synthetic = (actual_is_best == 1) | (actual_is_control == 1)
synth = result.loc[synthetic].copy()
print("Synthetic population:", int(synthetic.sum()), "of", len(df_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "auuc_metrics",
        """
auuc_metrics = pd.DataFrame({
    "is_treated": 1 - actual_is_control[synthetic],
    "conversion": df_test.loc[synthetic, "conversion"].to_numpy(),
    "uplift_tree": synth.max(axis=1).to_numpy(),
})
auuc_metrics.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "plot_gain",
        """
plot_uplift_curve(
    auuc_metrics["conversion"],
    auuc_metrics["is_treated"],
    auuc_metrics["uplift_tree"],
    title="Uplift curve — multi-arm forest",
)
print("Qini coefficient:", round(qini_coefficient(
    auuc_metrics["conversion"], auuc_metrics["is_treated"], auuc_metrics["uplift_tree"]
), 4))
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "df_bin",
        """
rng_bin = np.random.default_rng(111)
df_bin = df[df["treatment_group_key"].isin(["control", "treatment1"])].reset_index(drop=True)
X_all = df_bin[x_names].to_numpy()
y_all = df_bin["conversion"].to_numpy()
w_all = (df_bin["treatment_group_key"].to_numpy() != "control").astype(int)
idx_bin = rng_bin.choice(len(df_bin), size=int(0.8 * len(df_bin)), replace=False)
test_bin = np.setdiff1d(np.arange(len(df_bin)), idx_bin)
X_train, y_train, w_train = X_all[idx_bin], y_all[idx_bin], w_all[idx_bin]
X_test_bin, y_test, w_test = X_all[test_bin], y_all[test_bin], w_all[test_bin]
print(X_train.shape, X_test_bin.shape)
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_rf_kl",
        """
uplift_model_bin = UpliftTree(
    criterion="kl", max_depth=4, min_samples_leaf=40, min_samples_treatment=10, random_state=111
)
uplift_model_bin.fit(X_train, w_train, y_train, feature_names=x_names)
print(format_uplift_tree(uplift_model_bin))
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "uplift_tree_plot(uplift_tree_fitted",
        """
plot_uplift_tree(uplift_model_bin, title="Uplift tree (KL, control vs treatment1)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "Mean predicted CATE",
        """
pred_train = uplift_model_bin.predict(X_train)
pred_test = uplift_model_bin.predict(X_test_bin)
print(
    "Mean predicted CATE — train:", round(float(pred_train.mean()), 4),
    " test:", round(float(pred_test.mean()), 4),
)
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_rf <-",
        """
uplift_rf = UpliftRandomForest(
    criterion="kl", n_trees=5, max_depth=4, min_samples_leaf=30, random_state=111
)
uplift_rf.fit(X_train, w_train, y_train, feature_names=x_names)
print("Trees:", len(uplift_rf.trees_))
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_tree_string",
        """
print(format_uplift_tree(uplift_rf.trees_[0]))
""".strip(),
    )
    replace_code_containing(
        nb,
        "First tree in uplift random forest",
        """
plot_uplift_tree(uplift_rf.trees_[0], title="First tree in the uplift random forest (KL)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_forest_plot",
        """
plot_forest(uplift_rf, title="Four trees from the KL uplift forest")
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "color_leaves_by_tau",
        """
# Leaf boxes are green when uplift is positive and red when it is negative.
plot_uplift_tree(uplift_rf.trees_[0], title="Leaf color follows the sign of uplift")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "out_multi",
        """
df_multi, X_names_multi = make_uplift_classification(
    treatment_name=("control", "treatment1", "treatment2", "treatment3"),
    y_name="conversion",
    n_samples=2000,
    n_classification_features=10,
    n_classification_informative=5,
    n_uplift_increase_dict={"treatment1": 5, "treatment2": 5, "treatment3": 5},
    n_uplift_decrease_dict={"treatment1": 3, "treatment2": 3, "treatment3": 3},
    delta_uplift_increase_dict={"treatment1": 0.1, "treatment2": 0.15, "treatment3": 0.2},
    delta_uplift_decrease_dict={"treatment1": -0.1, "treatment2": -0.1, "treatment3": -0.1},
    random_seed=111,
)
""".strip(),
    )
    replace_code_containing(
        nb,
        "agg_multi",
        """
summary_multi = (
    df_multi.groupby("treatment_group_key")["conversion"].agg(mean="mean", size="size")
)
summary_multi.loc["All"] = [df_multi["conversion"].mean(), len(df_multi)]
print(summary_multi.round(4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "w_multi",
        """
rng_m = np.random.default_rng(111)
idx_m = rng_m.choice(len(df_multi), size=int(0.8 * len(df_multi)), replace=False)
w_multi = (df_multi["treatment_group_key"].to_numpy() != "control").astype(int)
X_multi = df_multi[X_names_multi].to_numpy()
y_multi = df_multi["conversion"].to_numpy()
print("Any-treatment rate:", round(float(w_multi.mean()), 3))
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_model_multi",
        """
uplift_model_multi = UpliftTree(
    criterion="kl", max_depth=3, min_samples_leaf=40, min_samples_treatment=10, random_state=111
)
uplift_model_multi.fit(X_multi[idx_m], w_multi[idx_m], y_multi[idx_m], feature_names=X_names_multi)
print(format_uplift_tree(uplift_model_multi))
""".strip(),
    )
    replace_code_containing(
        nb,
        "one control + multiple treatments",
        """
plot_uplift_tree(uplift_model_multi, title="Uplift tree (any treatment vs control)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "pred_cate <-",
        """
uf = UpliftRandomForest(criterion="kl", n_trees=5, max_depth=4, min_samples_leaf=25, random_state=111)
uf.fit(X_all, w_all, y_all, feature_names=x_names)
pred_cate = uf.predict(X_all)
print("Mean predicted CATE:", round(float(pred_cate.mean()), 4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "hist(pred_cate",
        """
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(pred_cate, bins=30, color="steelblue", edgecolor="white")
ax.axvline(pred_cate.mean(), color="red", ls="--", label=f"Mean = {pred_cate.mean():.3f}")
ax.set_xlabel("CATE")
ax.set_title("Uplift random forest (KL): predicted CATE")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "kernelshap(",
        """
sv = uplift_rf.explain(X_train, max_samples=30, n_permutations=6, random_state=2026)
print("SHAP array shape:", sv.values.shape)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_importance",
        """
sv.plot_importance(title="SHAP importance for predicted uplift")
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_dependence(shp_uplift, v = important_vars",
        """
order = np.argsort(sv.importance())[::-1][:4]
for j in order:
    sv.plot_dependence(sv.feature_names[j])
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "color_var",
        """
if len(order) >= 2:
    j, k = int(order[1]), int(order[0])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    points = ax.scatter(sv.data[:, j], sv.values[:, j], c=sv.data[:, k], cmap="viridis", s=28)
    fig.colorbar(points, ax=ax, label=sv.feature_names[k])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel(sv.feature_names[j])
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{sv.feature_names[j]} colored by {sv.feature_names[k]}")
    fig.tight_layout()
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "sv_waterfall",
        """
row_id = int(np.argmax(sv.predictions))
print("Highest predicted uplift in the explain set:", round(float(sv.predictions[row_id]), 4))
sv.plot_waterfall(row_id, title="SHAP waterfall — highest estimated uplift")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "shap_test",
        """
sv_test = uplift_rf.explain(X_test_bin[:10], max_samples=10, n_permutations=6, random_state=2026)
sv_test.plot_importance(title="SHAP importance on 10 test rows")
plt.show()
""".strip(),
    )
    save(nb, path)


def build_regression() -> None:
    nb, path = load("02_08_05_03_02_uplift_trees_regression.ipynb")
    if _already_python(nb):
        print("skip regression (already Python)")
        return
    banner_first(nb)
    replace_markdown_containing(
        nb,
        "R/dataset.R",
        """
## Simulated regression uplift data

`make_uplift_regression` uses the same feature blocks as the classification generator. The outcome is continuous: a baseline mean, plus a treatment-specific uplift, plus Gaussian noise.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_rf_multi()",
        """
### Multi-arm uplift: one T-learner per treatment

Each non-control arm is paired with control and fitted with a `TLearner`. The base learner is a random forest. Predicted uplift for arm $k$ is $\\hat{\\mu}_k(x) - \\hat{\\mu}_0(x)$.
""".strip(),
        nth=1,
    )
    replace_markdown_containing(
        nb,
        "n_trees = 1",
        """
### Shallow T-learner per arm

`n_estimators=1` grows a single tree in each arm. That is a fast check that both arms fit, not a production model.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "single transparent tree",
        """
For a single transparent tree on a continuous outcome, `InteractionTree` splits to maximize the squared difference in treatment effects between child nodes. The leaf value is the difference in sample means.
""".strip(),
        nth=1,
    )
    replace_markdown_containing(
        nb,
        "causal inference tree",
        """
### Honest causal tree

`CausalTree` is the honest single-tree estimator: one sample chooses the splits and a held-out sample estimates the leaf treatment effects. It is the interpretable counterpart of a causal inference tree.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_rf_kl()",
        """
### Train versus test mean CATE

A `TLearner` with random-forest base learners estimates the conditional means on the same binary sample. Compare the mean predicted CATE on train and test rows.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "forest$trees",
        """
A T-learner stores two outcome models, not a list of uplift trees, so a forest diagram does not apply. Variable importance below is the average impurity importance of the two random forests. `InteractionTree` is the model to draw when the split rules themselves matter.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "Kernel SHAP",
        """
### Permutation SHAP for predicted uplift

`explain` attributes $\\hat{\\tau}(X)$ from the binary T-learner. Positive SHAP pushes the predicted CATE up. The explanation is about treatment-effect heterogeneity, not the raw outcome.
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_rf_multi()` and `uplift_rf_kl()",
        """
## Summary

With a continuous outcome, a `TLearner` (or `UpliftXGBoost`) estimates $\\hat{\\mu}_1(x) - \\hat{\\mu}_0(x)$ separately in each arm. `InteractionTree` and `CausalTree` publish the partition. The Qini curve and `explain` are the same evaluation tools as in the classification notebook, with the outcome column `outcome` instead of `conversion`.
""".strip(),
    )

    insert_after(nb, "## Set Up", [code(INSTALL), code(IMPORTS_SYNTH)])

    replace_code_containing(nb, "make_uplift_regression", regression_call(), nth=1)
    replace_code_containing(nb, "head(df", "df.head()\n")
    replace_code_containing(
        nb,
        "aggregate(outcome",
        """
summary = df.groupby("treatment_group_key")["outcome"].agg(mean="mean", sd="std", size="size")
summary.loc["All"] = [df["outcome"].mean(), df["outcome"].std(), len(df)]
print(summary.round(4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "idx_test",
        """
rng = np.random.default_rng(111)
idx_test = rng.choice(len(df), size=round(0.2 * len(df)), replace=False)
df_train = df.drop(index=idx_test).reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)
X_test = df_test[x_names].to_numpy()
print("Train:", len(df_train), "| Test:", len(df_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "clf_shallow",
        """
def shallow():
    return TLearner(learner="rf", n_estimators=1, random_state=111)

clf_shallow = fit_multi(df_train, x_names, "outcome", shallow)
df_res_shallow = predict_multi(clf_shallow, X_test, kind="means")
df_res_shallow.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "n_trees = 50",
        """
def full_learner():
    return TLearner(learner="rf", n_estimators=40, random_state=111)

uplift_model = fit_multi(df_train, x_names, "outcome", full_learner)
df_res = predict_multi(uplift_model, X_test, kind="means")
print("Shape:", df_res.shape)
df_res.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "names(result)",
        """
result = predict_multi(uplift_model, X_test, kind="uplift")
result.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "best_treatment",
        """
best_treatment = np.where(result.max(axis=1).to_numpy() < 0, "control", result.idxmax(axis=1).to_numpy())
actual_is_best = (df_test["treatment_group_key"].to_numpy() == best_treatment).astype(int)
actual_is_control = (df_test["treatment_group_key"].to_numpy() == "control").astype(int)
""".strip(),
    )
    replace_code_containing(
        nb,
        "synthetic <-",
        """
synthetic = (actual_is_best == 1) | (actual_is_control == 1)
synth = result.loc[synthetic].copy()
print("Synthetic population:", int(synthetic.sum()))
""".strip(),
    )
    replace_code_containing(
        nb,
        "auuc_metrics",
        """
auuc_metrics = pd.DataFrame({
    "is_treated": 1 - actual_is_control[synthetic],
    "outcome": df_test.loc[synthetic, "outcome"].to_numpy(),
    "uplift_model": synth.max(axis=1).to_numpy(),
})
auuc_metrics.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "plot_gain",
        """
plot_uplift_curve(
    auuc_metrics["outcome"],
    auuc_metrics["is_treated"],
    auuc_metrics["uplift_model"],
    title="Uplift curve — continuous outcome",
)
print("Qini coefficient:", round(qini_coefficient(
    auuc_metrics["outcome"], auuc_metrics["is_treated"], auuc_metrics["uplift_model"]
), 4))
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "df_bin",
        """
rng_bin = np.random.default_rng(111)
df_bin = df[df["treatment_group_key"].isin(["control", "treatment1"])].reset_index(drop=True)
X_all = df_bin[x_names].to_numpy()
y_all = df_bin["outcome"].to_numpy()
w_all = (df_bin["treatment_group_key"].to_numpy() != "control").astype(int)
idx_bin = rng_bin.choice(len(df_bin), size=int(0.8 * len(df_bin)), replace=False)
test_bin = np.setdiff1d(np.arange(len(df_bin)), idx_bin)
X_train, y_train, w_train = X_all[idx_bin], y_all[idx_bin], w_all[idx_bin]
X_test_bin, y_test, w_test = X_all[test_bin], y_all[test_bin], w_all[test_bin]
print(X_train.shape, X_test_bin.shape)
""".strip(),
    )
    replace_code_containing(
        nb,
        "interaction_tree(",
        """
it_fit = InteractionTree(max_depth=3, min_samples_leaf=40, random_state=111)
it_fit.fit(X_train, w_train, y_train, feature_names=x_names)
print(format_uplift_tree(it_fit))
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "Interaction tree — continuous",
        """
plot_uplift_tree(it_fit, title="Interaction tree — continuous outcome (control vs treatment1)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "causal_inference_tree(",
        """
cit_fit = CausalTree(max_depth=3, min_samples_leaf=40, honest=True, random_state=111)
cit_fit.fit(X_train, w_train, y_train, feature_names=x_names)
print(pd.DataFrame(cit_fit.rules()).round(3))
plot_causal_tree(cit_fit, title="Honest causal tree — continuous outcome")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_bin_rf",
        """
uplift_bin = TLearner(learner="rf", n_estimators=40, random_state=111)
uplift_bin.fit(X_train, w_train, y_train, feature_names=x_names)
pred_tr = uplift_bin.predict(X_train)
pred_te = uplift_bin.predict(X_test_bin)
print(
    "Mean predicted CATE — train:", round(float(pred_tr.mean()), 4),
    " test:", round(float(pred_te.mean()), 4),
)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "variable.importance",
        """
imp = 0.5 * (
    np.asarray(uplift_bin.mu0_.feature_importances_)
    + np.asarray(uplift_bin.mu1_.feature_importances_)
)
order = np.argsort(imp)[::-1]
print(pd.Series(imp[order], index=np.asarray(x_names)[order]).round(4).head(8))
""".strip(),
    )
    replace_code_containing(nb, "out_multi", regression_call().replace("df, x_names", "df_multi, X_names_multi"), nth=1)
    replace_code_containing(
        nb,
        "w_multi",
        """
rng_m = np.random.default_rng(111)
idx_m = rng_m.choice(len(df_multi), size=int(0.8 * len(df_multi)), replace=False)
w_multi = (df_multi["treatment_group_key"].to_numpy() != "control").astype(int)
X_multi = df_multi[X_names_multi].to_numpy()
y_multi = df_multi["outcome"].to_numpy()
""".strip(),
    )
    replace_code_containing(
        nb,
        "it_multi",
        """
it_multi = InteractionTree(max_depth=3, min_samples_leaf=40, random_state=111)
it_multi.fit(X_multi[idx_m], w_multi[idx_m], y_multi[idx_m], feature_names=X_names_multi)
print(format_uplift_tree(it_multi))
""".strip(),
    )
    replace_code_containing(
        nb,
        "any treatment vs control",
        """
plot_uplift_tree(it_multi, title="Interaction tree: any treatment vs control (continuous outcome)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "pred_cate <-",
        """
uf = TLearner(learner="rf", n_estimators=40, random_state=111)
uf.fit(X_all, w_all, y_all, feature_names=x_names)
pred_cate = uf.predict(X_all)
""".strip(),
    )
    replace_code_containing(
        nb,
        "hist(",
        """
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(pred_cate, bins=30, color="steelblue", edgecolor="white")
ax.axvline(pred_cate.mean(), color="red", ls="--", label=f"Mean = {pred_cate.mean():.3f}")
ax.set_xlabel("Predicted treatment effect")
ax.set_title("T-learner forest: predicted CATE (continuous outcome)")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "kernelshap(",
        """
sv = uplift_bin.explain(X_train, max_samples=30, n_permutations=6, random_state=2026)
print("SHAP array shape:", sv.values.shape)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_importance",
        """
sv.plot_importance(title="SHAP importance for predicted CATE")
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_dependence(shp_uplift, v = important_vars",
        """
order = np.argsort(sv.importance())[::-1][:4]
for feature in np.asarray(x_names)[order]:
    sv.plot_dependence(feature)
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "color_var",
        """
if len(order) >= 2:
    j, k = int(order[1]), int(order[0])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    points = ax.scatter(sv.data[:, j], sv.values[:, j], c=sv.data[:, k], cmap="viridis", s=28)
    fig.colorbar(points, ax=ax, label=x_names[k])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel(x_names[j])
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{x_names[j]} colored by {x_names[k]}")
    fig.tight_layout()
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "sv_waterfall",
        """
row_id = int(np.argmax(sv.predictions))
print("Highest predicted CATE in the explain set:", round(float(sv.predictions[row_id]), 4))
sv.plot_waterfall(row_id, title="SHAP waterfall — highest estimated CATE")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "shap_test",
        """
sv_test = uplift_bin.explain(X_test_bin[:10], max_samples=10, n_permutations=6, random_state=2026)
sv_test.plot_importance(title="SHAP importance on 10 test rows")
plt.show()
""".strip(),
    )
    save(nb, path)


ABORTION_LOAD = """
from causaldata import abortion

abortion_df = abortion.load_pandas().data.copy()
print("Abortion dataset loaded. Columns:", list(abortion_df.columns))
print(abortion.NOTE)
print(abortion.DESCRLONG)
print(abortion.SOURCE)
""".strip() + "\n"

ABORTION_INSPECT = """
print(abortion_df.head())
print(abortion_df["repeal"].value_counts().sort_index())
print(pd.crosstab(abortion_df["repeal"], abortion_df["race"]))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
counts = abortion_df["repeal"].value_counts().sort_index()
axes[0, 0].bar(["No early repeal", "Early repeal"], counts.to_numpy(), color=["skyblue", "tomato"])
axes[0, 0].set_title("Treatment group sizes")
axes[0, 0].set_ylabel("State-year observations")
abortion_df.boxplot(column="lnr", by="repeal", ax=axes[0, 1])
axes[0, 1].set_title("Logged gonorrhea rate by repeal")
axes[0, 1].set_xlabel("Early repeal")
abortion_df.boxplot(column="income", by="repeal", ax=axes[1, 0])
axes[1, 0].set_title("Real income per capita by repeal")
axes[1, 0].set_xlabel("Early repeal")
abortion_df.boxplot(column="alcohol", by="repeal", ax=axes[1, 1])
axes[1, 1].set_title("Alcohol consumption by repeal")
axes[1, 1].set_xlabel("Early repeal")
fig.suptitle("Abortion dataset: treatment, outcome, and covariates")
fig.tight_layout()
plt.show()
""".strip() + "\n"

ABORTION_PREP = """
missing_rate = abortion_df.isna().mean()
sparse = missing_rate[missing_rate >= 0.2].index.tolist()
print("Dropped sparse columns:", sparse)
model_df = abortion_df.drop(columns=sparse).dropna().reset_index(drop=True)
feature_names = [c for c in model_df.columns if c not in {"lnr", "repeal"}]
X = model_df[feature_names].to_numpy(dtype=float)
Y = model_df["lnr"].to_numpy(dtype=float)
W = model_df["repeal"].to_numpy(dtype=int)
keep = np.var(X, axis=0) > 1e-10
X = X[:, keep]
feature_names = [name for name, flag in zip(feature_names, keep) if flag]
x_names = feature_names
print("Covariates used:", x_names)
print("n =", len(Y), " treated =", int(W.sum()))
""".strip() + "\n"

ABORTION_SPLIT = """
rng_abortion = np.random.default_rng(42)
n = len(Y)
train_idx = rng_abortion.choice(n, size=round(0.8 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]
W_train, W_test = W[train_idx], W[test_idx]
print("Training rows:", len(Y_train))
print("Test rows:    ", len(Y_test))
print("Columns:      ", x_names)
""".strip() + "\n"

ABORTION_AGG = """
arm = pd.DataFrame({"lnr": Y, "repeal": W})
summary = arm.groupby("repeal")["lnr"].agg(mean="mean", sd="std", size="size")
summary.loc["All"] = [float(Y.mean()), float(Y.std()), len(Y)]
print(summary.round(4))
""".strip() + "\n"


def _shared_abortion(nb, t_learner: str, t_kwargs: str) -> None:
    banner_first(nb)
    insert_after(nb, "## Set Up", [code(INSTALL), code(IMPORTS_ABORTION)])
    replace_markdown_containing(
        nb,
        "{causaldata}",
        """
### Data

The abortion dataset is loaded from `causaldata`, the same source as the causal-tree tutorial. It records state-year observations on gonorrhea rates, with a binary indicator for early repeal of abortion prohibition. Risky sexual behavior is proxied by gonorrhea incidence.

**Key variables:**

| Variable | Description |
|----------|-------------|
| `fip` | State FIPS code |
| `age` | Age in years |
| `race` | Race (1 = white, 2 = black) |
| `sex` | Sex (1 = male, 2 = female) |
| `year` | Calendar year |
| `t` | Rescaled time variable |
| `totpop` | Total population (state-level) |
| `ir` | Incarcerated males per 100,000 |
| `crack` | Crack cocaine index |
| `alcohol` | Alcohol consumption per capita |
| `income` | Real income per capita |
| `ur` | State unemployment rate |
| `poverty` | Poverty rate |
| **`repeal`** | **Treatment**: 1 = early repeal of abortion prohibition, 0 = otherwise |
| **`lnr`** | **Outcome**: log gonorrhea rate per 100,000 |
| `pi` | Parental involvement law in effect |
| `bf15` | Indicator: black female aged 15–19 |

Rows with a missing outcome are dropped. A column that is missing for at least 20% of rows (in this file, `fa`) is left out of the covariate matrix. Near-zero-variance columns are dropped, matching the causal-tree tutorial.

> **Causal note:** This is observational panel data. Uplift models estimate conditional contrasts $\\hat{\\tau}(x)$. A causal reading still needs an identification argument (for example difference-in-differences or fixed effects).
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "get_importance",
        """
### Feature selection (F-filter)

Following Zhao et al. (2020), each covariate is ranked by the partial F statistic of its interaction with treatment in a linear model of `lnr`. Order 1 tests `lnr ~ x + repeal + x:repeal`. Orders 2 and 3 add higher powers of `x`. Ranking uses the training rows only.

The likelihood-ratio and KL filters in the uplift literature are for binary outcomes. `lnr` is continuous, so the F-filter is the matching choice.
""".strip(),
    )
    replace_code_containing(nb, "list_causaldata_datasets", ABORTION_LOAD)
    replace_code_containing(nb, "load_causaldata", ABORTION_INSPECT)
    replace_code_containing(nb, "abortion_clean", ABORTION_PREP)
    replace_code_containing(nb, "train_idx", ABORTION_SPLIT)
    replace_code_containing(nb, "aggregate(", ABORTION_AGG)
    replace_code_containing(
        nb,
        "df_fs_train",
        """
print("Training matrix:", X_train.shape)
""".strip(),
    )
    replace_code_containing(
        nb,
        "get_importance(",
        """
f_imp_1 = f_filter(X_train, Y_train, W_train, x_names, order=1)
f_imp_2 = f_filter(X_train, Y_train, W_train, x_names, order=2)
f_imp_3 = f_filter(X_train, Y_train, W_train, x_names, order=3)
print("*** Order 1 ***")
print(f_imp_1.head(8).round(3).to_string(index=False))
print("*** Order 2 ***")
print(f_imp_2.head(8).round(3).to_string(index=False))
print("*** Order 3 ***")
print(f_imp_3.head(8).round(3).to_string(index=False))
""".strip(),
    )
    replace_code_containing(
        nb,
        "barplot(",
        """
top_bar = f_imp_1.head(min(20, len(f_imp_1))).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(top_bar["feature"], top_bar["score"], color="steelblue")
ax.set_xlabel("Partial F statistic")
ax.set_title("F1 filter: feature importance (abortion, train)")
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "ranked_f",
        """
ranked_f = f_imp_1["feature"].tolist()
top_3_features = ranked_f[: min(3, len(ranked_f))]
top_6_features = ranked_f[: min(6, len(ranked_f))]
print("Top 3 (F1):", ", ".join(top_3_features))
print("Top 6 (F1):", ", ".join(top_6_features))
""".strip(),
    )
    replace_code_containing(
        nb,
        "model_cols",
        """
def fit_learner(Learner, feats, **kwargs):
    cols = [x_names.index(name) for name in feats]
    model = Learner(**kwargs)
    model.fit(X_train[:, cols], W_train, Y_train, feature_names=list(feats))
    return model.predict(X_test[:, cols])

feature_sets = {
    "All": x_names,
    "Top_3": top_3_features,
    "Top_6": top_6_features,
}
""".strip(),
    )
    replace_code_containing(
        nb,
        "fit_tl",
        f"""
pred_tl = {{
    name: fit_learner(TLearner, feats, {t_kwargs})
    for name, feats in feature_sets.items()
}}
pred_tl["Random"] = np.random.default_rng(42).random(len(Y_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "T-Learner",
        f"""
fig, auuc_tl = plot_qini_models(
    Y_test, W_test, pred_tl, title="{t_learner}: cumulative gain — abortion test set"
)
plt.show()
print(auuc_tl.round(4))
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "fit_rl",
        """
pred_rl = {
    name: fit_learner(RLearner, feats, learner="rf", n_estimators=15, n_splits=2, random_state=111)
    for name, feats in feature_sets.items()
}
pred_rl["Random"] = np.random.default_rng(42).random(len(Y_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "R-Learner",
        """
fig, auuc_rl = plot_qini_models(
    Y_test, W_test, pred_rl, title="R-learner: cumulative gain — abortion test set"
)
plt.show()
print(auuc_rl.round(4))
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "fit_sl",
        """
pred_sl = {
    name: fit_learner(SLearner, feats, learner="rf", n_estimators=25, random_state=111)
    for name, feats in feature_sets.items()
}
pred_sl["Random"] = np.random.default_rng(42).random(len(Y_test))
""".strip(),
    )
    replace_code_containing(
        nb,
        "S-Learner",
        """
fig, auuc_sl = plot_qini_models(
    Y_test, W_test, pred_sl, title="S-learner: cumulative gain — abortion test set"
)
plt.show()
print(auuc_sl.round(4))
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "auuc_tab",
        """
auuc_tab = pd.concat([
    auuc_tl.assign(learner="T-learner"),
    auuc_rl.assign(learner="R-learner"),
    auuc_sl.assign(learner="S-learner"),
], ignore_index=True)
print(auuc_tab.round(4).to_string(index=False))
""".strip(),
    )


def build_random_forest() -> None:
    nb, path = load("02_08_05_03_03_uplift_randomForest.ipynb")
    if _already_python(nb):
        print("skip random forest (already Python)")
        return
    replace_markdown_containing(
        nb,
        "RCausalML` argument",
        """
### Continuous outcomes

PyCausalML has two forest styles for a continuous outcome:

| Approach | Idea | PyCausalML class |
|----------|------|------------------|
| **T-learner forest** | Separate outcome forests for $T=0$ and $T=1$; CATE is the difference of predicted means | `TLearner(learner="rf")` |
| **Uplift forest** | Bagged trees whose splits maximize the squared gap in $\\bar{Y}^{(1)} - \\bar{Y}^{(0)}$ | `UpliftRandomForest` |

The forest output is $\\hat{\\tau}(x)$ on the scale of $Y$. Here $Y$ is `lnr` (log gonorrhea rate per 100,000), so a CATE of $-0.12$ means early repeal is associated with a **0.12 log-point lower** gonorrhea rate for that covariate profile.

#### From ATE to targeted policy

The **average treatment effect (ATE)** summarises the treatment in one number. An uplift forest estimates the **full profile** $\\tau(x)$, enabling:

| CATE region | Interpretation | Targeting decision |
|-------------|----------------|-------------------|
| **High positive** $\\tau(x)$ | Large expected benefit from treatment | Priority targets |
| **Near zero** | Treatment adds little | Skip / low priority |
| **Negative** $\\tau(x)$ | Treatment may worsen outcomes | Actively avoid treating |

Standard outcome prediction ranks high-baseline units highly even when treatment adds nothing. Uplift forests rank by **incremental effect**.

The abortion dataset has a single policy indicator (`repeal`), so this notebook stays on the binary path. A multi-arm experiment would fit one forest per non-control arm.
""".strip(),
    )
    _shared_abortion(
        nb,
        "T-learner (random forest)",
        'learner="rf", n_estimators=25, random_state=111',
    )
    replace_markdown_containing(
        nb,
        "uplift_randomForestRegressor()",
        """
## Uplift random forest

`UpliftRandomForest` bags trees that split on treatment-effect heterogeneity. On a continuous outcome the split is the squared difference of child CATEs, the same rule as `InteractionTree`. `TLearner(learner="rf")` is the two-forest T-learner: one random forest per arm, and CATE equals the difference of predicted means.

### T-learner forest
""".strip(),
    )
    replace_code_containing(
        nb,
        'evaluation_function = "TLearner"',
        """
uplift_model = TLearner(learner="rf", n_estimators=40, random_state=111)
uplift_model.fit(X_train, W_train, Y_train, feature_names=x_names)
pred_tr = uplift_model.predict(X_train)
pred_te = uplift_model.predict(X_test)
print(
    "Mean predicted CATE — train:", round(float(pred_tr.mean()), 4),
    "| test:", round(float(pred_te.mean()), 4),
)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "full_output",
        """
pred_full = pd.DataFrame({
    "control": uplift_model.mu0_.predict(X_test),
    "treated": uplift_model.mu1_.predict(X_test),
    "cate": pred_te,
})
pred_full.head()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        'evaluation_function = "IT"',
        """
### Uplift forest (treatment-effect splits)

`UpliftRandomForest` averages leaf contrasts from trees that split to separate treatment effects, rather than fitting two outcome models.
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_it_rf",
        """
uplift_it_rf = UpliftRandomForest(
    criterion="ed", n_trees=12, max_depth=4, min_samples_leaf=60, random_state=111
)
uplift_it_rf.fit(X_train, W_train, Y_train, feature_names=x_names)
pred_it_te = uplift_it_rf.predict(X_test)
print("Uplift forest — mean predicted CATE (test):", round(float(pred_it_te.mean()), 4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "print(uplift_model)",
        """
print("T-learner importances:")
print(pd.Series(
    0.5 * (uplift_model.mu0_.feature_importances_ + uplift_model.mu1_.feature_importances_),
    index=x_names,
).sort_values(ascending=False).round(4).head(8))
print("Uplift-forest importances:")
print(pd.Series(uplift_it_rf.feature_importance(), index=x_names).sort_values(ascending=False).round(4).head(8))
""".strip(),
    )
    replace_code_containing(
        nb,
        "df_gain",
        """
df_gain = pd.DataFrame({
    "is_treated": W_test,
    "outcome": Y_test,
    "uplift_model": pred_te,
})
df_gain.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "plot_gain(df_gain",
        """
plot_uplift_curve(Y_test, W_test, pred_te, title="T-learner forest — abortion test set")
print("Qini coefficient:", round(qini_coefficient(Y_test, W_test, pred_te), 4))
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "uf_full",
        """
uf_full = TLearner(learner="rf", n_estimators=30, random_state=111)
uf_full.fit(X, W, Y, feature_names=x_names)
pred_cate_full = uf_full.predict(X)
""".strip(),
    )
    replace_code_containing(
        nb,
        "hist(",
        """
fig, ax = plt.subplots(figsize=(6.6, 4))
ax.hist(pred_cate_full, bins=30, color="steelblue", edgecolor="white")
ax.axvline(pred_cate_full.mean(), color="red", ls="--", lw=2)
ax.set_xlabel("Predicted treatment effect (lnr scale)")
ax.set_title("T-learner forest: distribution of predicted CATE")
ax.legend([f"Mean CATE = {pred_cate_full.mean():.3f}"], frameon=False)
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_forest_plot()",
        """
### Variable importance

For the T-learner, average the impurity importance of the control forest and the treated forest. The uplift forest reports split-gain importance from `feature_importance`. Individual uplift trees are drawn from `uplift_it_rf.trees_`.
""".strip(),
    )
    replace_code_containing(
        nb,
        "variable.importance",
        """
imp_avg = 0.5 * (
    np.asarray(uplift_model.mu0_.feature_importances_)
    + np.asarray(uplift_model.mu1_.feature_importances_)
)
print(pd.Series(imp_avg, index=x_names).sort_values(ascending=False).round(4).head(8))
plot_uplift_tree(uplift_it_rf.trees_[0], title="First tree in the uplift forest")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "interaction_tree(",
        """
it_fit = InteractionTree(max_depth=3, min_samples_leaf=80, random_state=111)
it_fit.fit(X_train, W_train, Y_train, feature_names=x_names)
print(format_uplift_tree(it_fit))
""".strip(),
    )
    replace_code_containing(
        nb,
        "lnr vs repeal",
        """
plot_uplift_tree(it_fit, title="Interaction tree — lnr vs repeal (abortion data, train)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "causal_inference_tree(",
        """
cit_fit = CausalTree(max_depth=3, min_samples_leaf=80, honest=True, random_state=111)
cit_fit.fit(X_train, W_train, Y_train, feature_names=x_names)
print(pd.DataFrame(cit_fit.rules()).round(3))
plot_causal_tree(cit_fit, title="Honest causal tree — abortion data (train)")
plt.show()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "Explaining Predicted CATE with Kernel SHAP",
        """
## Explaining predicted CATE

`explain` decomposes predicted CATE into covariate contributions. A positive value means that feature pushes the predicted treatment effect up. The explanation is about $\\hat{\\tau}(X)$, not the raw outcome. The call below uses a subset of the training rows.
""".strip(),
    )
    replace_code_containing(
        nb,
        "kernelshap(",
        """
sv = uplift_model.explain(X_train, max_samples=24, n_permutations=4, random_state=2026)
print("SHAP array shape:", sv.values.shape)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_importance",
        """
sv.plot_importance(title="SHAP importance for predicted CATE")
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_dependence(shp_uplift, v = important_vars",
        """
order = np.argsort(sv.importance())[::-1][:4]
for feature in np.asarray(x_names)[order]:
    sv.plot_dependence(feature)
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "color_var",
        """
if len(order) >= 2:
    j, k = int(order[1]), int(order[0])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    points = ax.scatter(sv.data[:, j], sv.values[:, j], c=sv.data[:, k], cmap="viridis", s=28)
    fig.colorbar(points, ax=ax, label=x_names[k])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel(x_names[j])
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{x_names[j]} colored by {x_names[k]}")
    fig.tight_layout()
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "sv_waterfall",
        """
row_id = int(np.argmax(sv.predictions))
print("Highest predicted CATE in the explain set:", round(float(sv.predictions[row_id]), 4))
sv.plot_waterfall(row_id, title="SHAP waterfall — highest estimated CATE")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "shap_test",
        """
sv_test = uplift_model.explain(X_test[:10], max_samples=10, n_permutations=4, random_state=2026)
sv_test.plot_importance(title="SHAP importance on 10 test rows")
plt.show()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "full workflow for uplift random forest",
        """
## Summary

This tutorial walks through uplift regression on the abortion dataset with PyCausalML.

| Step | What we did |
|------|-------------|
| **Data** | Loaded `abortion` from `causaldata`, as in the causal-tree tutorial. Outcome `lnr`, treatment `repeal` |
| **Feature selection** | F-filter on the training rows; compared top-3, top-6, and all features |
| **Learner comparison** | T-, R-, and S-learners scored by the Qini coefficient on the test set |
| **Uplift forest** | `TLearner` random forests and `UpliftRandomForest` |
| **Evaluation** | Qini curves, CATE distribution, feature importance |
| **Single trees** | `InteractionTree` and honest `CausalTree` |
| **SHAP** | `explain` — importance, dependence, and a waterfall plot |

### Key reminders

- **Scale of CATE:** predictions are on the `lnr` scale. A CATE of $-0.10$ is 0.10 fewer log-points of gonorrhea rate.
- **Observational data:** the forest estimates conditional contrasts. Identification is a separate argument.
- **T-learner vs uplift forest:** `TLearner(learner="rf")` fits two outcome forests. `UpliftRandomForest` splits on treatment-effect heterogeneity and keeps the individual trees.
""".strip(),
    )
    save(nb, path)


def build_xgboost() -> None:
    nb, path = load("02_08_05_03_04_uplift_xgboost.ipynb")
    if _already_python(nb):
        print("skip xgboost (already Python)")
        return
    replace_markdown_containing(
        nb,
        "uplift_xgboost()",
        """
## Why gradient boosting for uplift?

`UpliftXGBoost` is a **T-learner** with gradient-boosted trees as the base learner:

1. Fit one ensemble on control units ($T = 0$) to estimate $\\hat{\\mu}_0(x)$.
2. Fit a separate ensemble on treated units ($T = 1$) to estimate $\\hat{\\mu}_1(x)$.
3. Predict CATE as the difference:

$$\\hat{\\tau}(x) = \\hat{\\mu}_1(x) - \\hat{\\mu}_0(x)$$

| Aspect | Uplift random forest | Uplift XGBoost |
|--------|----------------------|----------------|
| Class | `UpliftRandomForest` | `UpliftXGBoost` |
| Base learner | Bagged uplift trees | Gradient-boosted trees |
| Splitting rule | Squared CATE gap | T-learner only |
| Strengths | Treatment-aware splits; individual trees can be drawn | Flexible nonlinear surfaces |

The model output is $\\hat{\\tau}(x)$ on the scale of $Y$. Here $Y$ is `lnr`, so a CATE of $-0.12$ is a **0.12 log-point lower** gonorrhea rate for that profile.

#### From ATE to targeted policy

| CATE region | Interpretation | Targeting decision |
|-------------|----------------|-------------------|
| **High positive** $\\tau(x)$ | Large expected benefit from treatment | Priority targets |
| **Near zero** | Treatment adds little | Skip / low priority |
| **Negative** $\\tau(x)$ | Treatment may worsen outcomes | Actively avoid treating |

The abortion dataset has a single policy indicator (`repeal`), so this notebook stays on the binary path.
""".strip(),
        nth=1,
    )
    _shared_abortion(
        nb,
        "T-learner (XGBoost)",
        'learner="xgb", n_estimators=40, random_state=111',
    )
    replace_markdown_containing(
        nb,
        "uplift_xgboostRegressor()",
        """
## Uplift XGBoost

`UpliftXGBoost` fits one boosted tree ensemble per treatment arm and returns CATE as the difference of the two predicted means. Treatment-effect split criteria (KL, interaction trees) live on `UpliftTree` and `UpliftRandomForest`.

### Fit the T-learner
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "uplift_xgboostRegressor(",
        """
uplift_model = UpliftXGBoost(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=111,
)
uplift_model.fit(X_train, W_train, Y_train, feature_names=x_names)
pred_tr = uplift_model.predict(X_train)
pred_te = uplift_model.predict(X_test)
print(
    "Mean predicted CATE — train:", round(float(pred_tr.mean()), 4),
    "| test:", round(float(pred_te.mean()), 4),
)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "full_output",
        """
mu0, mu1 = uplift_model.predict_potential_outcomes(X_test)
pred_full = pd.DataFrame({"control": mu0, "treated": mu1, "cate": pred_te})
pred_full.head()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "print(uplift_model)",
        """
print(pd.Series(uplift_model.feature_importance(), index=x_names).sort_values(ascending=False).round(4))
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "uplift_xgboostRegressor()` vs `uplift_randomForestRegressor()",
        """
## Comparing uplift XGBoost and an uplift random forest

Both models return CATE on the outcome scale. XGBoost is a T-learner of boosted trees. The random-forest comparison uses `TLearner(learner="rf")` and `UpliftRandomForest` on the same train/test split.

### T-learner random forest
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_rf_tl",
        """
uplift_rf_tl = TLearner(learner="rf", n_estimators=30, random_state=111)
uplift_rf_tl.fit(X_train, W_train, Y_train, feature_names=x_names)
pred_rf_tl_tr = uplift_rf_tl.predict(X_train)
pred_rf_tl_te = uplift_rf_tl.predict(X_test)
print(
    "Uplift RF (T-learner) — mean CATE train:", round(float(pred_rf_tl_tr.mean()), 4),
    "| test:", round(float(pred_rf_tl_te.mean()), 4),
)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "pred_rf_tl_full",
        """
pred_rf_tl_full = pd.DataFrame({
    "control": uplift_rf_tl.mu0_.predict(X_test),
    "treated": uplift_rf_tl.mu1_.predict(X_test),
    "cate": pred_rf_tl_te,
})
pred_rf_tl_full.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_rf_it",
        """
uplift_rf_it = UpliftRandomForest(
    criterion="ed", n_trees=12, max_depth=4, min_samples_leaf=60, random_state=111
)
uplift_rf_it.fit(X_train, W_train, Y_train, feature_names=x_names)
pred_rf_it_te = uplift_rf_it.predict(X_test)
print("Uplift RF — mean CATE (test):", round(float(pred_rf_it_te.mean()), 4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "print(uplift_rf_tl)",
        """
print("RF T-learner importance:")
print(pd.Series(
    0.5 * (uplift_rf_tl.mu0_.feature_importances_ + uplift_rf_tl.mu1_.feature_importances_),
    index=x_names,
).sort_values(ascending=False).round(4).head(8))
print("Uplift-forest importance:")
print(pd.Series(uplift_rf_it.feature_importance(), index=x_names).sort_values(ascending=False).round(4).head(8))
""".strip(),
    )
    replace_code_containing(
        nb,
        "cmp_mean",
        """
cmp_mean = pd.DataFrame({
    "Model": ["Uplift XGBoost", "T-learner random forest", "Uplift random forest"],
    "Mean_CATE_train": [pred_tr.mean(), pred_rf_tl_tr.mean(), np.nan],
    "Mean_CATE_test": [pred_te.mean(), pred_rf_tl_te.mean(), pred_rf_it_te.mean()],
})
print(cmp_mean.round(4).to_string(index=False))
corr_tab = pd.DataFrame({
    "Pair": ["XGBoost vs RF T-learner", "XGBoost vs uplift forest", "RF T-learner vs uplift forest"],
    "Correlation": [
        np.corrcoef(pred_te, pred_rf_tl_te)[0, 1],
        np.corrcoef(pred_te, pred_rf_it_te)[0, 1],
        np.corrcoef(pred_rf_tl_te, pred_rf_it_te)[0, 1],
    ],
})
print(corr_tab.round(4).to_string(index=False))
""".strip(),
    )
    replace_code_containing(
        nb,
        "par(mfrow",
        """
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(pred_rf_tl_te, pred_te, s=12, alpha=0.45, c="#3366AA")
axes[0].axline((0, 0), slope=1, color="gray", ls="--")
axes[0].set_xlabel("T-learner random forest")
axes[0].set_ylabel("Uplift XGBoost")
axes[0].set_title("Predicted CATE — test set")
axes[1].scatter(pred_rf_it_te, pred_te, s=12, alpha=0.45, c="#CC6633")
axes[1].axline((0, 0), slope=1, color="gray", ls="--")
axes[1].set_xlabel("Uplift random forest")
axes[1].set_ylabel("Uplift XGBoost")
axes[1].set_title("Predicted CATE — test set")
fig.tight_layout()
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "df_gain",
        """
df_gain = pd.DataFrame({
    "is_treated": W_test,
    "outcome": Y_test,
    "uplift_xgb": pred_te,
    "uplift_rf_tl": pred_rf_tl_te,
    "uplift_rf_it": pred_rf_it_te,
    "Random": np.random.default_rng(42).random(len(X_test)),
})
df_gain.head()
""".strip(),
    )
    replace_code_containing(
        nb,
        "uplift_cols",
        """
scores = {
    "Uplift XGBoost": df_gain["uplift_xgb"],
    "T-learner RF": df_gain["uplift_rf_tl"],
    "Uplift forest": df_gain["uplift_rf_it"],
    "Random": df_gain["Random"],
}
fig, auuc_uplift = plot_qini_models(
    Y_test, W_test, scores, title="Uplift XGBoost vs uplift forest — abortion test set"
)
plt.show()
print(auuc_uplift.round(4).to_string(index=False))
""".strip(),
    )
    replace_code_containing(
        nb,
        "ux_full",
        """
ux_full = UpliftXGBoost(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=111)
ux_full.fit(X, W, Y, feature_names=x_names)
uf_tl_full = TLearner(learner="rf", n_estimators=20, random_state=111)
uf_tl_full.fit(X, W, Y, feature_names=x_names)
uf_it_full = UpliftRandomForest(
    criterion="ed", n_trees=8, max_depth=3, min_samples_leaf=80, random_state=111
)
uf_it_full.fit(X, W, Y, feature_names=x_names)
pred_cate_xgb = ux_full.predict(X)
pred_cate_rf_tl = uf_tl_full.predict(X)
pred_cate_rf_it = uf_it_full.predict(X)
""".strip(),
    )
    replace_code_containing(
        nb,
        "hist(",
        """
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(pred_cate_xgb, bins=30, color="#3366AA", alpha=0.55, label=f"XGBoost (mean = {pred_cate_xgb.mean():.3f})")
ax.hist(pred_cate_rf_tl, bins=30, color="#2E8B57", alpha=0.45, label=f"RF T-learner (mean = {pred_cate_rf_tl.mean():.3f})")
ax.hist(pred_cate_rf_it, bins=30, color="#E07A3D", alpha=0.45, label=f"Uplift forest (mean = {pred_cate_rf_it.mean():.3f})")
ax.set_xlabel("Predicted treatment effect (lnr scale)")
ax.set_title("Predicted CATE distributions (full sample)")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "xgb.importance",
        """
imp_xgb = pd.Series(uplift_model.feature_importance(), index=x_names).sort_values(ascending=False)
print(imp_xgb.head(8).round(4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "imp_rf0",
        """
imp_rf = pd.Series(
    0.5 * (uplift_rf_tl.mu0_.feature_importances_ + uplift_rf_tl.mu1_.feature_importances_),
    index=x_names,
).sort_values(ascending=False)
print(imp_rf.head(8).round(4))
""".strip(),
    )
    replace_code_containing(
        nb,
        "top_n",
        """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
left = imp_xgb.head(6).iloc[::-1]
right = imp_rf.head(6).iloc[::-1]
axes[0].barh(left.index, left.to_numpy(), color="darkorange")
axes[0].set_title("Uplift XGBoost (average gain)")
axes[1].barh(right.index, right.to_numpy(), color="forestgreen")
axes[1].set_title("T-learner random forest")
fig.tight_layout()
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "interaction_tree(",
        """
it_fit = InteractionTree(max_depth=3, min_samples_leaf=80, random_state=111)
it_fit.fit(X_train, W_train, Y_train, feature_names=x_names)
print(format_uplift_tree(it_fit))
""".strip(),
    )
    replace_code_containing(
        nb,
        "lnr vs repeal",
        """
plot_uplift_tree(it_fit, title="Interaction tree — lnr vs repeal (abortion data, train)")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "causal_inference_tree(",
        """
cit_fit = CausalTree(max_depth=3, min_samples_leaf=80, honest=True, random_state=111)
cit_fit.fit(X_train, W_train, Y_train, feature_names=x_names)
print(pd.DataFrame(cit_fit.rules()).round(3))
plot_causal_tree(cit_fit, title="Honest causal tree — abortion data (train)")
plt.show()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "Kernel SHAP is model-agnostic",
        """
## Explaining predicted CATE

`explain` decomposes predicted CATE into covariate contributions. A positive value means that feature pushes the predicted treatment effect up. The explanation is about $\\hat{\\tau}(X)$, not the raw outcome.
""".strip(),
    )
    replace_code_containing(
        nb,
        "kernelshap(",
        """
sv = uplift_model.explain(X_train, max_samples=24, n_permutations=4, random_state=2026)
print("SHAP array shape:", sv.values.shape)
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_importance",
        """
sv.plot_importance(title="SHAP importance for predicted CATE")
plt.show()
""".strip(),
        nth=1,
    )
    replace_code_containing(
        nb,
        "sv_dependence(shp_uplift, v = important_vars",
        """
order = np.argsort(sv.importance())[::-1][:4]
for feature in np.asarray(x_names)[order]:
    sv.plot_dependence(feature)
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "color_var",
        """
if len(order) >= 2:
    j, k = int(order[1]), int(order[0])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    points = ax.scatter(sv.data[:, j], sv.values[:, j], c=sv.data[:, k], cmap="viridis", s=28)
    fig.colorbar(points, ax=ax, label=x_names[k])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel(x_names[j])
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{x_names[j]} colored by {x_names[k]}")
    fig.tight_layout()
    plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "sv_waterfall",
        """
row_id = int(np.argmax(sv.predictions))
print("Highest predicted CATE in the explain set:", round(float(sv.predictions[row_id]), 4))
sv.plot_waterfall(row_id, title="SHAP waterfall — highest estimated CATE")
plt.show()
""".strip(),
    )
    replace_code_containing(
        nb,
        "shap_test",
        """
sv_test = uplift_model.explain(X_test[:10], max_samples=10, n_permutations=4, random_state=2026)
sv_test.plot_importance(title="SHAP importance on 10 test rows")
plt.show()
""".strip(),
    )
    replace_markdown_containing(
        nb,
        "full workflow for uplift XGBoost",
        """
## Summary

This tutorial fits uplift XGBoost on the abortion dataset with PyCausalML and compares it with a T-learner random forest and an uplift forest.

| Step | What we did |
|------|-------------|
| **Data** | Loaded `abortion` from `causaldata`, as in the causal-tree tutorial |
| **Feature selection** | F-filter on the training rows |
| **Learner comparison** | T-learner XGBoost, plus R- and S-learners, scored by the Qini coefficient |
| **Uplift XGBoost** | `UpliftXGBoost` T-learner |
| **Uplift forest** | `TLearner(learner="rf")` and `UpliftRandomForest` on the same split |
| **Single trees** | `InteractionTree` and honest `CausalTree` |
| **SHAP** | `explain` on the XGBoost CATE |

### Key reminders

- Predictions are on the `lnr` scale.
- The sample is observational. Conditional contrasts are not, by themselves, causal effects.
- `UpliftXGBoost` is a T-learner. Use `UpliftRandomForest` when the splits themselves should target treatment-effect heterogeneity.
""".strip(),
    )
    save(nb, path)


if __name__ == "__main__":
    build_intro()
    build_classification()
    build_regression()
    build_random_forest()
    build_xgboost()



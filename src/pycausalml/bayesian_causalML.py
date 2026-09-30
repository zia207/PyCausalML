"""Bayesian causal inference for quasi-experiments, networks, and time series.

This module is self-contained. It uses NumPy, SciPy, and NetworkX. It does
not import PyMC, pgmpy, or CausalNex. Those libraries are the references for
the three families below; the estimators here are the dependency-free
counterparts.

Quasi-experiments
    Conjugate Gaussian posteriors for difference-in-differences, regression
    discontinuity, and interrupted time series, plus a Bayesian bootstrap for
    two-stage least squares and synthetic control. The designs follow
    `CausalPy <https://causalpy.readthedocs.io/en/latest/>`_.

Bayesian causal networks
    A DAG, backdoor and frontdoor identification, a linear-Gaussian Bayesian
    network, and a discrete network with Dirichlet–multinomial conditionals.
    ``do`` queries mutilate the graph and propagate. This covers the same
    operations as `pgmpy.inference.CausalInference
    <https://pgmpy.org/api/causal_inference.html>`_ (adjustment, frontdoor,
    naive adjustment, naive IV) and the fit / intervene workflow in
    `CausalNex <https://github.com/mckinsey/causalnex>`_. CausalNex itself
    was discontinued on 24 June 2026; do not take a dependency on it.
    Partially linear double machine learning already lives in
    :mod:`pycausalml.double_machine_learning`.

Structural models and time series
    A joint treatment–outcome structural model with exclusion restrictions,
    in the sense of the CausalPy note on `Bayesian structural causal models
    <https://causalpy.readthedocs.io/en/latest/knowledgebase/structural_causal_models.html>`_,
    and a lagged structural causal model whose coefficients have Gaussian
    posteriors. Hard ``do`` interventions replace a structural equation and
    are rolled forward.

Coefficient vectors are posterior means under a weak ridge prior. Credible
intervals are quantiles of posterior draws. ``summary()`` omits those draws;
they stay on ``draws_``.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "BayesianDifferenceInDifferences",
    "BayesianRegressionDiscontinuity",
    "BayesianInterruptedTimeSeries",
    "BayesianInstrumentalVariable",
    "BayesianSyntheticControl",
    "CausalGraph",
    "GaussianBayesianNetwork",
    "DiscreteBayesianNetwork",
    "NaiveAdjustmentRegressor",
    "NaiveIVRegressor",
    "BayesianStructuralSCM",
    "BayesianTimeSeriesSCM",
    "simulate_structural_scm",
]


def _as_1d(values, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    out = np.asarray(values, dtype=float).ravel()
    if out.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return out


def _as_2d(values, name: str) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        values = values.to_numpy(dtype=float).reshape(-1, 1)
    out = np.asarray(values, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 1- or 2-dimensional.")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return out


def _check_lengths(n: int, **arrays) -> None:
    for name, array in arrays.items():
        if len(array) != n:
            raise ValueError(f"{name} has length {len(array)}; expected {n}.")


def _summarize(estimate: float, draws: np.ndarray, rope=None) -> dict:
    draws = np.asarray(draws, dtype=float).ravel()
    low, high = np.quantile(draws, [0.025, 0.975])
    out = {
        "estimate": float(estimate),
        "posterior_mean": float(np.mean(draws)),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
    }
    if rope is not None:
        lo, hi = float(rope[0]), float(rope[1])
        out["rope_probability"] = float(np.mean((draws >= lo) & (draws <= hi)))
    return out


def _draw_normal(mean: np.ndarray, cov: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov.flat[:: cov.shape[0] + 1] += 1e-12
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 0.0, None)
    noise = rng.normal(size=(n_draws, len(mean)))
    return mean + noise @ (vectors * np.sqrt(values)).T


def _normal_posterior(design: np.ndarray, y: np.ndarray, prior_precision: float, intercept_index: int = 0):
    """Gaussian posterior for a linear mean with a ridge prior.

    ``intercept_index`` is almost unpenalized. The noise variance is the
    residual variance, so the coefficient posterior is the usual conditional
    normal posterior.
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = design.shape
    precision = design.T @ design
    penalty = np.full(p, float(prior_precision))
    if intercept_index is not None:
        penalty[int(intercept_index)] = min(float(prior_precision), 1e-8)
    precision.flat[:: p + 1] += penalty
    coef = np.linalg.solve(precision, design.T @ y)
    resid = y - design @ coef
    sigma2 = float(np.sum(resid ** 2) / max(n - p, 1))
    cov = sigma2 * np.linalg.inv(precision)
    cov = 0.5 * (cov + cov.T)
    return coef, cov, sigma2


def _wls(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.asarray(weights, dtype=float))
    beta, _, _, _ = np.linalg.lstsq(design * scale[:, None], y * scale, rcond=None)
    return np.asarray(beta, dtype=float)


def _names_of(nodes) -> list:
    if nodes is None:
        return []
    if isinstance(nodes, (str, bytes)):
        return [str(nodes)]
    return [str(node) for node in nodes]


def _node_set(nodes) -> set:
    return set(_names_of(nodes))


# ---------------------------------------------------------------------------
# Quasi-experimental posteriors
# ---------------------------------------------------------------------------


class _BayesianLinearDesign:
    """Shared posterior for a linear quasi-experimental design."""

    def __init__(self, prior_precision: float = 1e-4, n_draws: int = 1000, rope=None, random_state: int = 0):
        self.prior_precision = float(prior_precision)
        self.n_draws = int(n_draws)
        self.rope = rope
        self.random_state = int(random_state)

    def _fit_design(self, design, y, names, effect_index: int):
        y = _as_1d(y, "y")
        design = np.asarray(design, dtype=float)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        _check_lengths(len(design), y=y)
        if not (0 <= effect_index < design.shape[1]):
            raise ValueError("effect_index is outside the design.")
        coef, cov, sigma2 = _normal_posterior(design, y, self.prior_precision)
        rng = np.random.default_rng(self.random_state)
        coef_draws = _draw_normal(coef, cov, self.n_draws, rng)
        self.design_ = design
        self.coef_ = coef
        self.coef_draws_ = coef_draws
        self.posterior_cov_ = cov
        self.sigma2_ = sigma2
        self.feature_names_ = list(names)
        self.effect_index_ = int(effect_index)
        self.draws_ = coef_draws[:, effect_index]
        self.effect_ = float(coef[effect_index])
        self.summary_ = _summarize(self.effect_, self.draws_, self.rope)
        return self

    def predict(self, design=None) -> np.ndarray:
        if design is None:
            design = self.design_
        design = np.asarray(design, dtype=float)
        return design @ self.coef_

    def summary(self) -> dict:
        return dict(self.summary_)


class BayesianDifferenceInDifferences(_BayesianLinearDesign):
    """Bayesian two-way difference-in-differences.

    ``Y = a + b * treated + c * post + tau * treated * post``. ``tau`` is
    the average treatment effect on the treated under parallel trends.
    ``treated`` and ``post`` are 0/1 indicators.
    """

    def fit(self, y, treated, post):
        y = _as_1d(y, "y")
        treated = _as_1d(treated, "treated")
        post = _as_1d(post, "post")
        _check_lengths(len(y), treated=treated, post=post)
        design = np.column_stack([np.ones(len(y)), treated, post, treated * post])
        return self._fit_design(
            design,
            y,
            ["intercept", "treated", "post", "treated_x_post"],
            effect_index=3,
        )


class BayesianRegressionDiscontinuity(_BayesianLinearDesign):
    """Bayesian local linear regression discontinuity.

    The design is ``1, (R - c), D, (R - c) * D``. The effect is the jump
    in ``D`` at ``cutoff``. Rows outside ``bandwidth`` are dropped when a
    bandwidth is set. If ``treatment`` is omitted, treatment is ``R >= cutoff``.
    """

    def __init__(self, cutoff: float = 0.0, bandwidth: float = None, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth

    def fit(self, running, y, treatment=None):
        running = _as_1d(running, "running")
        y = _as_1d(y, "y")
        _check_lengths(len(running), y=y)
        if treatment is None:
            treatment = (running >= self.cutoff).astype(float)
        else:
            treatment = _as_1d(treatment, "treatment")
            _check_lengths(len(running), treatment=treatment)
        centered = running - self.cutoff
        mask = np.ones(len(y), dtype=bool)
        if self.bandwidth is not None:
            mask = np.abs(centered) <= float(self.bandwidth)
        if int(mask.sum()) < 4:
            raise ValueError("The regression-discontinuity window has fewer than 4 rows.")
        centered, y, treated = centered[mask], y[mask], treatment[mask]
        design = np.column_stack([np.ones(len(y)), centered, treated, centered * treated])
        self.n_window_ = int(len(y))
        return self._fit_design(
            design,
            y,
            ["intercept", "slope", "treatment", "slope_x_treatment"],
            effect_index=2,
        )


class BayesianInterruptedTimeSeries(_BayesianLinearDesign):
    """Bayesian level and slope change after an intervention time.

    ``Y = a + b * time + tau * post + d * time_since``. The reported effect
    is the immediate level shift ``tau``. The slope change is ``slope_change_``.
    """

    def fit(self, time, y, intervention_time: float):
        time = _as_1d(time, "time")
        y = _as_1d(y, "y")
        _check_lengths(len(time), y=y)
        post = (time >= float(intervention_time)).astype(float)
        since = np.maximum(time - float(intervention_time), 0.0)
        design = np.column_stack([np.ones(len(y)), time, post, since])
        fitted = self._fit_design(
            design,
            y,
            ["intercept", "time", "level_shift", "slope_change"],
            effect_index=2,
        )
        self.slope_change_ = float(self.coef_[3])
        self.intervention_time_ = float(intervention_time)
        return fitted


class BayesianInstrumentalVariable:
    """Bayesian two-stage least squares.

    The first stage is ``treatment ~ instrument, X``. The second stage is
    ``y ~ treatment_hat, X``. Draws are a Bayesian bootstrap (Dirichlet
    weights on the rows) of both stages, so first-stage uncertainty is
    included. The structural effect is the coefficient on the fitted treatment.
    """

    def __init__(self, n_draws: int = 400, rope=None, random_state: int = 0):
        self.n_draws = int(n_draws)
        self.rope = rope
        self.random_state = int(random_state)

    def fit(self, y, treatment, instrument, X=None):
        y = _as_1d(y, "y")
        treatment = _as_1d(treatment, "treatment")
        instrument = _as_2d(instrument, "instrument")
        _check_lengths(len(y), treatment=treatment, instrument=instrument)
        if X is None:
            covariates = np.zeros((len(y), 0))
            covariate_names = []
        else:
            covariates = _as_2d(X, "X")
            _check_lengths(len(y), X=covariates)
            covariate_names = [f"X{i}" for i in range(1, covariates.shape[1] + 1)]
            if isinstance(X, pd.DataFrame):
                covariate_names = [str(c) for c in X.columns]
        self.feature_names_ = ["intercept", "treatment", *covariate_names]
        self._y = y
        self._treatment = treatment
        self._instrument = instrument
        self._X = covariates

        point = self._coef(y, treatment, instrument, covariates, np.ones(len(y)))
        rng = np.random.default_rng(self.random_state)
        draws = np.empty(self.n_draws, dtype=float)
        for b in range(self.n_draws):
            weights = rng.dirichlet(np.ones(len(y)))
            draws[b] = self._coef(y, treatment, instrument, covariates, weights)[1]
        self.coef_ = point
        self.effect_ = float(point[1])
        self.draws_ = draws
        self.summary_ = _summarize(self.effect_, draws, self.rope)
        return self

    @staticmethod
    def _coef(y, treatment, instrument, covariates, weights):
        n = len(y)
        ones = np.ones((n, 1))
        if covariates.size:
            first = np.hstack([ones, instrument, covariates])
        else:
            first = np.hstack([ones, instrument])
        fitted = first @ _wls(first, treatment, weights)
        if covariates.size:
            second = np.hstack([ones, fitted.reshape(-1, 1), covariates])
        else:
            second = np.hstack([ones, fitted.reshape(-1, 1)])
        return _wls(second, y, weights)

    def predict(self, treatment=None, X=None) -> np.ndarray:
        """Structural prediction at an intervened treatment level."""
        if treatment is None:
            treatment = self._treatment
        treatment = _as_1d(treatment, "treatment")
        if X is None:
            covariates = self._X
            if len(covariates) != len(treatment):
                covariates = np.zeros((len(treatment), self._X.shape[1]))
        else:
            covariates = _as_2d(X, "X")
            _check_lengths(len(treatment), X=covariates)
        prediction = self.coef_[0] + self.coef_[1] * treatment
        if covariates.size:
            prediction = prediction + covariates @ self.coef_[2:]
        return prediction

    def summary(self) -> dict:
        return dict(self.summary_)


class BayesianSyntheticControl:
    """Bayesian synthetic control for one treated unit.

    Donor weights are non-negative and sum to one. They minimize the weighted
    pre-period squared error. The point estimate uses equal weights. Posterior
    draws reweight the pre-period with a Dirichlet Bayesian bootstrap.
    ``control_pre`` has shape ``(n_pre, n_donors)``.
    """

    def __init__(self, n_draws: int = 100, random_state: int = 0):
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, control_pre, treated_pre, control_post, treated_post):
        control_pre = _as_2d(control_pre, "control_pre")
        treated_pre = _as_1d(treated_pre, "treated_pre")
        control_post = _as_2d(control_post, "control_post")
        treated_post = _as_1d(treated_post, "treated_post")
        if control_pre.shape[0] != len(treated_pre):
            raise ValueError("control_pre and treated_pre must share the pre-period length.")
        if control_post.shape[0] != len(treated_post):
            raise ValueError("control_post and treated_post must share the post-period length.")
        if control_pre.shape[1] != control_post.shape[1]:
            raise ValueError("Pre and post donor matrices must have the same donors.")
        self.weights_ = self._weights(control_pre, treated_pre, np.ones(len(treated_pre)))
        self.synth_post_ = control_post @ self.weights_
        self.effect_path_ = treated_post - self.synth_post_
        self.effect_ = float(np.mean(self.effect_path_))
        rng = np.random.default_rng(self.random_state)
        draws = np.empty(self.n_draws, dtype=float)
        for b in range(self.n_draws):
            weights = rng.dirichlet(np.ones(len(treated_pre)))
            donor_w = self._weights(control_pre, treated_pre, weights)
            draws[b] = float(np.mean(treated_post - control_post @ donor_w))
        self.draws_ = draws
        self.summary_ = _summarize(self.effect_, draws)
        self.feature_names_ = [f"donor_{i}" for i in range(control_pre.shape[1])]
        return self

    @staticmethod
    def _weights(control_pre, treated_pre, row_weights):
        n_donors = control_pre.shape[1]
        scale_rows = np.sqrt(np.asarray(row_weights, dtype=float))
        # Scale the pre-period so the quadratic loss is order 1. A numerical
        # gradient on the raw pack counts treats the equal-weight start as optimal.
        level = float(np.std(treated_pre))
        if not np.isfinite(level) or level < 1e-8:
            level = 1.0
        design = np.asarray(control_pre, dtype=float) / level
        target = np.asarray(treated_pre, dtype=float) / level

        def loss(weights):
            resid = (design @ weights - target) * scale_rows
            return float(np.dot(resid, resid))

        def gradient(weights):
            resid = (design @ weights - target) * scale_rows
            return 2.0 * (design.T @ (resid * scale_rows))

        constraints = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
            "jac": lambda weights: np.ones(n_donors),
        }
        bounds = [(0.0, 1.0)] * n_donors
        start = np.full(n_donors, 1.0 / n_donors)
        result = minimize(
            loss,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        weights = np.clip(result.x, 0.0, None)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return start
        return weights / total

    def predict(self, control_post=None) -> np.ndarray:
        if control_post is None:
            return self.synth_post_
        return _as_2d(control_post, "control_post") @ self.weights_

    def summary(self) -> dict:
        return dict(self.summary_)


# ---------------------------------------------------------------------------
# Causal graphs: backdoor, frontdoor, instruments
# ---------------------------------------------------------------------------


def _d_separated(digraph: nx.DiGraph, xs, ys, zs) -> bool:
    """d-separation by moralizing the ancestral graph."""
    x_set, y_set, z_set = _node_set(xs), _node_set(ys), _node_set(zs)
    if not x_set or not y_set:
        raise ValueError("d-separation needs at least one node on each side.")
    if x_set & y_set:
        return False
    relevant = set()
    stack = list(x_set | y_set | z_set)
    seen = set(stack)
    while stack:
        node = stack.pop()
        if node not in digraph:
            continue
        relevant.add(node)
        for parent in digraph.predecessors(node):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    moral = nx.Graph()
    moral.add_nodes_from(relevant)
    for parent, child in digraph.edges:
        if parent in relevant and child in relevant:
            moral.add_edge(parent, child)
    for node in relevant:
        parents = [parent for parent in digraph.predecessors(node) if parent in relevant]
        for left, right in combinations(parents, 2):
            moral.add_edge(left, right)
    moral.remove_nodes_from([node for node in z_set if node in moral])
    for source in x_set:
        if source not in moral:
            continue
        reachable = nx.node_connected_component(moral, source)
        if any(target in reachable for target in y_set):
            return False
    return True


class CausalGraph:
    """Directed acyclic graph for identification.

    Edges are ``(parent, child)`` pairs: ``parent`` causes ``child``. This is
    the structure object behind pgmpy's ``Adjustment`` and ``Frontdoor``
    helpers and CausalNex's ``StructureModel``. The graph is assumed to be
    causally sufficient: latent confounding has to be drawn as a node.
    """

    def __init__(self, edges, nodes=None):
        graph = nx.DiGraph()
        if nodes is not None:
            graph.add_nodes_from(_names_of(nodes))
        graph.add_edges_from((str(parent), str(child)) for parent, child in edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Causal graph must be a directed acyclic graph.")
        self.graph = graph
        self.edges = list(graph.edges)
        self.nodes = list(nx.topological_sort(graph))

    def parents(self, node: str) -> list:
        return list(self.graph.predecessors(str(node)))

    def children(self, node: str) -> list:
        return list(self.graph.successors(str(node)))

    def descendants(self, node: str) -> set:
        return set(nx.descendants(self.graph, str(node)))

    def ancestors(self, node: str) -> set:
        return set(nx.ancestors(self.graph, str(node)))

    def d_separated(self, x, y, z=None) -> bool:
        """Whether ``x`` and ``y`` are d-separated by ``z``."""
        return _d_separated(self.graph, x, y, [] if z is None else z)

    def satisfies_backdoor(self, treatment: str, outcome: str, adjustment) -> bool:
        """Backdoor criterion: ``adjustment`` blocks every non-causal path."""
        treatment, outcome = str(treatment), str(outcome)
        adjustment = _node_set(adjustment)
        if treatment in adjustment or outcome in adjustment:
            return False
        if adjustment & self.descendants(treatment):
            return False
        mutilated = self.graph.copy()
        mutilated.remove_edges_from(list(mutilated.out_edges(treatment)))
        return _d_separated(mutilated, [treatment], [outcome], adjustment)

    def adjustment_sets(self, treatment: str, outcome: str, max_size: int = 3) -> list:
        """Inclusion-minimal observed backdoor adjustment sets.

        The empty set and the parents of ``treatment`` are always checked.
        Other candidates are subsets of non-descendants up to ``max_size``.
        """
        treatment, outcome = str(treatment), str(outcome)
        self._require(treatment, outcome)
        forbidden = {treatment, outcome} | self.descendants(treatment)
        pool = [node for node in self.nodes if node not in forbidden]
        limit = int(max_size)
        if len(pool) > 10:
            limit = min(limit, 2)
        candidates = [set(), set(self.parents(treatment)) - forbidden]
        for size in range(1, min(limit, len(pool)) + 1):
            for combo in combinations(pool, size):
                candidates.append(set(combo))
        valid = []
        seen = set()
        for candidate in candidates:
            key = frozenset(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self.satisfies_backdoor(treatment, outcome, candidate):
                valid.append(key)
        return _inclusion_minimal(valid)

    def minimal_adjustment_set(self, treatment: str, outcome: str) -> list:
        """One smallest backdoor adjustment set, as a sorted list of names."""
        found = self.adjustment_sets(treatment, outcome)
        if not found:
            raise ValueError(
                f"No observed adjustment set identifies the effect of {treatment} on {outcome}."
            )
        return sorted(found[0])

    def satisfies_frontdoor(self, treatment: str, outcome: str, mediators) -> bool:
        """Whether ``mediators`` meet Pearl's frontdoor criterion."""
        treatment, outcome = str(treatment), str(outcome)
        mediators = _node_set(mediators)
        if not mediators or treatment in mediators or outcome in mediators:
            return False
        without_mediators = self.graph.copy()
        without_mediators.remove_nodes_from(mediators)
        if treatment in without_mediators and outcome in without_mediators:
            if nx.has_path(without_mediators, treatment, outcome):
                return False
        opened = self.graph.copy()
        opened.remove_edges_from(list(opened.out_edges(treatment)))
        if not _d_separated(opened, [treatment], mediators, []):
            return False
        blocked = self.graph.copy()
        for mediator in mediators:
            blocked.remove_edges_from(list(blocked.out_edges(mediator)))
        return _d_separated(blocked, mediators, [outcome], [treatment])

    def frontdoor_sets(self, treatment: str, outcome: str, max_size: int = 2) -> list:
        """Mediator sets that satisfy the frontdoor criterion."""
        treatment, outcome = str(treatment), str(outcome)
        self._require(treatment, outcome)
        mediators = self.descendants(treatment) & self.ancestors(outcome)
        valid = []
        for size in range(1, min(int(max_size), len(mediators)) + 1):
            for combo in combinations(sorted(mediators), size):
                if self.satisfies_frontdoor(treatment, outcome, combo):
                    valid.append(frozenset(combo))
        return _inclusion_minimal(valid)

    def instruments(self, treatment: str, outcome: str) -> list:
        """Observed nodes that affect ``outcome`` only through ``treatment``.

        A node is an instrument when it has a directed path into the treatment
        and it is d-separated from the outcome once the treatment is removed.
        """
        treatment, outcome = str(treatment), str(outcome)
        self._require(treatment, outcome)
        found = []
        for node in self.nodes:
            if node in (treatment, outcome):
                continue
            if not nx.has_path(self.graph, node, treatment):
                continue
            reduced = self.graph.copy()
            reduced.remove_node(treatment)
            if node not in reduced or outcome not in reduced or _d_separated(reduced, [node], [outcome], []):
                found.append(node)
        return found

    def _require(self, *nodes) -> None:
        missing = [node for node in nodes if node not in self.graph]
        if missing:
            raise ValueError(f"Nodes not in the graph: {missing}.")

    def adjacency(self) -> pd.DataFrame:
        """Adjacency matrix with ``A.loc[child, parent] = 1``."""
        matrix = nx.to_pandas_adjacency(self.graph, nodelist=self.nodes, dtype=float).T
        matrix = matrix.reindex(index=self.nodes, columns=self.nodes).fillna(0.0)
        return matrix


def _inclusion_minimal(sets: list) -> list:
    minimal = []
    for candidate in sets:
        if any(other < candidate for other in sets):
            continue
        if candidate not in minimal:
            minimal.append(candidate)
    minimal.sort(key=lambda item: (len(item), sorted(item)))
    return minimal


# ---------------------------------------------------------------------------
# Bayesian networks
# ---------------------------------------------------------------------------


class GaussianBayesianNetwork:
    """Linear-Gaussian Bayesian network with a normal coefficient posterior.

    Each node is ``intercept + parents @ beta + noise``. ``ate`` is the total
    derivative of the outcome with respect to a hard intervention on the
    treatment, averaged over the coefficient posterior. ``sample`` draws from
    the posterior-mean structural equations, optionally under ``do``.
    """

    def __init__(self, edges, prior_precision: float = 1e-4, n_draws: int = 400, random_state: int = 0):
        self.causal_graph = CausalGraph(edges)
        self.prior_precision = float(prior_precision)
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, data):
        frame = _as_frame(data)
        missing = [node for node in self.causal_graph.nodes if node not in frame.columns]
        if missing:
            raise ValueError(f"Data is missing graph nodes: {missing}.")
        self.names_ = list(self.causal_graph.nodes)
        self.parents_ = {node: self.causal_graph.parents(node) for node in self.names_}
        self.equations_ = {}
        for node in self.names_:
            parents = self.parents_[node]
            design = np.column_stack([np.ones(len(frame)), frame[parents].to_numpy(dtype=float)])
            coef, cov, sigma2 = _normal_posterior(design, frame[node].to_numpy(dtype=float), self.prior_precision)
            self.equations_[node] = {
                "coef": coef,
                "cov": cov,
                "sigma2": sigma2,
                "feature_names": ["intercept", *parents],
            }
        self.n_obs_ = int(len(frame))
        return self

    def ate(self, treatment: str, outcome: str, treated: float = 1.0, control: float = 0.0) -> dict:
        """Posterior of ``E[Y | do(T=treated)] - E[Y | do(T=control)]``.

        For a linear structural assignment this difference is the total effect
        times ``treated - control``.
        """
        self._require_fitted()
        treatment, outcome = str(treatment), str(outcome)
        self._require_nodes(treatment, outcome)
        delta = float(treated) - float(control)
        mean_effect = self._total_effect({node: eq["coef"] for node, eq in self.equations_.items()}, treatment, outcome)
        rng = np.random.default_rng(self.random_state)
        draws = np.empty(self.n_draws, dtype=float)
        for b in range(self.n_draws):
            drawn = {}
            for node, eq in self.equations_.items():
                drawn[node] = _draw_normal(eq["coef"], eq["cov"], 1, rng)[0]
            draws[b] = self._total_effect(drawn, treatment, outcome) * delta
        self.effect_ = float(mean_effect * delta)
        self.draws_ = draws
        self.summary_ = _summarize(self.effect_, draws)
        self.treatment_ = treatment
        self.outcome_ = outcome
        return self.summary_

    def sample(self, n: int = 500, do=None, random_state=None) -> pd.DataFrame:
        """Ancestral draws. ``do`` maps a node name to a fixed numeric value."""
        self._require_fitted()
        do = {} if do is None else {str(k): float(v) for k, v in do.items()}
        self._require_nodes(*do)
        seed = self.random_state if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)
        out = np.zeros((int(n), len(self.names_)))
        index = {node: i for i, node in enumerate(self.names_)}
        for node in self.names_:
            column = index[node]
            if node in do:
                out[:, column] = do[node]
                continue
            parents = self.parents_[node]
            coef = self.equations_[node]["coef"]
            mean = np.full(n, coef[0])
            for k, parent in enumerate(parents):
                mean = mean + coef[k + 1] * out[:, index[parent]]
            scale = np.sqrt(max(self.equations_[node]["sigma2"], 0.0))
            out[:, column] = mean + rng.normal(0.0, scale, size=n)
        return pd.DataFrame(out, columns=self.names_)

    def summary(self) -> dict:
        if not hasattr(self, "summary_"):
            raise RuntimeError("Call ate() before summary().")
        return dict(self.summary_)

    def _total_effect(self, coefs: dict, treatment: str, outcome: str) -> float:
        effect = {node: 0.0 for node in self.names_}
        effect[treatment] = 1.0
        for node in self.names_:
            if node == treatment:
                continue
            acc = 0.0
            for k, parent in enumerate(self.parents_[node]):
                acc += float(coefs[node][k + 1]) * effect[parent]
            effect[node] = acc
        return float(effect[outcome])

    def _require_fitted(self) -> None:
        if not hasattr(self, "equations_"):
            raise RuntimeError("Call fit() first.")

    def _require_nodes(self, *nodes) -> None:
        missing = [node for node in nodes if node not in self.parents_]
        if missing:
            raise ValueError(f"Nodes not in the network: {missing}.")


class DiscreteBayesianNetwork:
    """Categorical Bayesian network with Dirichlet–multinomial conditionals.

    ``query`` uses exact enumeration when the joint table has at most
    ``exact_limit`` cells, and likelihood weighting otherwise. A ``do`` map
    replaces the conditional of that node with a point mass. ``evidence``
    conditions on observed values and still uses the node's conditional.
    """

    def __init__(self, edges, alpha: float = 1.0, exact_limit: int = 8192, random_state: int = 0):
        self.causal_graph = CausalGraph(edges)
        self.alpha = float(alpha)
        self.exact_limit = int(exact_limit)
        self.random_state = int(random_state)

    def fit(self, data):
        frame = _as_frame(data)
        missing = [node for node in self.causal_graph.nodes if node not in frame.columns]
        if missing:
            raise ValueError(f"Data is missing graph nodes: {missing}.")
        self.names_ = list(self.causal_graph.nodes)
        self.parents_ = {node: self.causal_graph.parents(node) for node in self.names_}
        self.states_ = {}
        self.codes_ = {}
        for node in frame.columns:
            labels = [_python_scalar(value) for value in pd.unique(frame[node].to_numpy())]
            labels = sorted(labels, key=lambda value: (str(type(value)), str(value)))
            index = {label: i for i, label in enumerate(labels)}
            self.states_[node] = labels
            self.codes_[node] = np.array([index[label] for label in frame[node].map(_python_scalar)], dtype=int)
        self.card_ = {node: len(self.states_[node]) for node in self.names_}
        self.cpd_ = {}
        for node in self.names_:
            parents = self.parents_[node]
            counts: dict = {}
            for row in range(len(frame)):
                key = tuple(int(self.codes_[parent][row]) for parent in parents)
                bucket = counts.setdefault(key, np.zeros(self.card_[node], dtype=float))
                bucket[int(self.codes_[node][row])] += 1.0
            alpha = self.alpha
            card = self.card_[node]
            self.cpd_[node] = {
                key: (count + alpha) / (count.sum() + alpha * card) for key, count in counts.items()
            }
        self.n_obs_ = int(len(frame))
        return self

    def query(self, variables, evidence=None, do=None, n_samples: int = 4000) -> dict:
        """Posterior marginals under optional evidence and interventions.

        ``variables`` is a name or a list of names. Returned probabilities are
        keyed by the original state labels.
        """
        self._require_fitted()
        variables = _names_of(variables)
        evidence = {} if evidence is None else {str(k): v for k, v in evidence.items()}
        do = {} if do is None else {str(k): v for k, v in do.items()}
        overlap = set(evidence) & set(do)
        if overlap:
            raise ValueError(f"A node cannot be both evidence and an intervention: {sorted(overlap)}.")
        encoded_evidence = {node: self._state_index(node, value) for node, value in evidence.items()}
        encoded_do = {node: self._state_index(node, value) for node, value in do.items()}
        joint = int(np.prod([self.card_[node] for node in self.names_]))
        if joint <= self.exact_limit:
            method = "exact"
            distributions = {
                node: self._exact(node, encoded_do, encoded_evidence) for node in variables
            }
        else:
            method = "likelihood_weighting"
            distributions = self._weighted(variables, encoded_do, encoded_evidence, int(n_samples))
        labeled = {}
        for node, probs in distributions.items():
            labeled[node] = {self.states_[node][i]: float(probs[i]) for i in range(self.card_[node])}
        return {"method": method, "distributions": labeled}

    def ate(self, treatment: str, outcome: str, treated=1, control=0) -> dict:
        """``E[Y | do(T=treated)] - E[Y | do(T=control)]`` on numeric states."""
        self._require_fitted()
        labels = self.states_[str(outcome)]
        if not all(isinstance(label, (int, float)) for label in labels):
            raise TypeError("ate() needs a numeric outcome. Use query() for other labels.")
        high = self.query(outcome, do={treatment: treated})["distributions"][str(outcome)]
        low = self.query(outcome, do={treatment: control})["distributions"][str(outcome)]
        expectation_high = sum(float(label) * high[label] for label in labels)
        expectation_low = sum(float(label) * low[label] for label in labels)
        estimate = float(expectation_high - expectation_low)
        self.effect_ = estimate
        self.summary_ = {
            "estimate": estimate,
            "treated_mean": float(expectation_high),
            "control_mean": float(expectation_low),
        }
        return dict(self.summary_)

    def cpd(self, node: str) -> pd.DataFrame:
        """Posterior-mean conditional probability table for one node."""
        self._require_fitted()
        node = str(node)
        rows = []
        parents = self.parents_[node]
        for key, probs in self.cpd_[node].items():
            row = {}
            for i, parent in enumerate(parents):
                row[parent] = self.states_[parent][key[i]]
            for i, label in enumerate(self.states_[node]):
                row[f"P({label})"] = float(probs[i])
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        if not hasattr(self, "summary_"):
            raise RuntimeError("Call ate() before summary().")
        return dict(self.summary_)

    def _lookup(self, node: str, parent_codes: tuple) -> np.ndarray:
        table = self.cpd_[node]
        if parent_codes in table:
            probs = np.asarray(table[parent_codes], dtype=float)
        else:
            probs = np.full(self.card_[node], 1.0 / self.card_[node])
        total = float(probs.sum())
        return probs / total if total > 0 else np.full(self.card_[node], 1.0 / self.card_[node])

    def _exact(self, query: str, do: dict, evidence: dict) -> np.ndarray:
        probs = np.zeros(self.card_[query], dtype=float)
        assignment = {}

        def recurse(index: int, weight: float) -> None:
            if weight == 0.0:
                return
            if index == len(self.names_):
                probs[assignment[query]] += weight
                return
            node = self.names_[index]
            parents = self.parents_[node]
            key = tuple(assignment[parent] for parent in parents)
            if node in do:
                assignment[node] = do[node]
                recurse(index + 1, weight)
                return
            if node in evidence:
                state = evidence[node]
                assignment[node] = state
                recurse(index + 1, weight * float(self._lookup(node, key)[state]))
                return
            conditional = self._lookup(node, key)
            for state, probability in enumerate(conditional):
                assignment[node] = state
                recurse(index + 1, weight * float(probability))

        recurse(0, 1.0)
        total = float(probs.sum())
        if total <= 0:
            raise RuntimeError("The query has probability zero under this model.")
        return probs / total

    def _weighted(self, variables, do, evidence, n_samples: int) -> dict:
        rng = np.random.default_rng(self.random_state)
        totals = {node: np.zeros(self.card_[node]) for node in variables}
        weight_sum = 0.0
        for _ in range(n_samples):
            assignment = {}
            weight = 1.0
            for node in self.names_:
                key = tuple(assignment[parent] for parent in self.parents_[node])
                conditional = self._lookup(node, key)
                if node in do:
                    assignment[node] = do[node]
                elif node in evidence:
                    state = evidence[node]
                    assignment[node] = state
                    weight *= float(conditional[state])
                else:
                    assignment[node] = int(rng.choice(self.card_[node], p=conditional))
            if weight <= 0:
                continue
            weight_sum += weight
            for node in variables:
                totals[node][assignment[node]] += weight
        if weight_sum <= 0:
            raise RuntimeError("Likelihood weighting received zero total weight.")
        return {node: totals[node] / weight_sum for node in variables}

    def _state_index(self, node: str, value) -> int:
        node = str(node)
        if node not in self.states_:
            raise ValueError(f"Node not in the network: {node}.")
        target = _python_scalar(value)
        for i, label in enumerate(self.states_[node]):
            if _same_state(label, target):
                return i
        raise ValueError(f"{value!r} is not a state of {node}: {self.states_[node]}.")

    def _require_fitted(self) -> None:
        if not hasattr(self, "cpd_"):
            raise RuntimeError("Call fit() first.")


class NaiveAdjustmentRegressor:
    """Bayesian outcome regression on a backdoor adjustment set.

    The adjustment set is the smallest set returned by
    :meth:`CausalGraph.adjustment_sets`. The treatment coefficient is the
    average treatment effect under that graph and a linear outcome model.
    This is the native counterpart of pgmpy's ``NaiveAdjustmentRegressor``.
    """

    def __init__(self, causal_graph: CausalGraph, treatment: str, outcome: str, prior_precision: float = 1e-6, n_draws: int = 800, random_state: int = 0):
        if not isinstance(causal_graph, CausalGraph):
            causal_graph = CausalGraph(causal_graph)
        self.causal_graph = causal_graph
        self.treatment = str(treatment)
        self.outcome = str(outcome)
        self.prior_precision = float(prior_precision)
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, data):
        frame = _as_frame(data)
        adjustment = self.causal_graph.minimal_adjustment_set(self.treatment, self.outcome)
        for column in [self.treatment, self.outcome, *adjustment]:
            if column not in frame.columns:
                raise ValueError(f"Data is missing column {column!r}.")
        columns = [self.treatment, *adjustment]
        design = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
        coef, cov, sigma2 = _normal_posterior(
            design, frame[self.outcome].to_numpy(dtype=float), self.prior_precision
        )
        rng = np.random.default_rng(self.random_state)
        draws = _draw_normal(coef, cov, self.n_draws, rng)
        self.adjustment_set_ = adjustment
        self.feature_names_ = ["intercept", *columns]
        self.coef_ = coef
        self.posterior_cov_ = cov
        self.sigma2_ = sigma2
        self.effect_ = float(coef[1])
        self.draws_ = draws[:, 1]
        self.summary_ = _summarize(self.effect_, self.draws_)
        return self

    def predict(self, data) -> np.ndarray:
        frame = _as_frame(data)
        columns = [self.treatment, *self.adjustment_set_]
        design = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
        return design @ self.coef_

    def summary(self) -> dict:
        public = dict(self.summary_)
        public["adjustment_set"] = list(self.adjustment_set_)
        return public


class NaiveIVRegressor:
    """Two-stage least squares with instruments read off the causal graph.

    Instruments are the nodes returned by :meth:`CausalGraph.instruments`.
    This is the native counterpart of pgmpy's ``NaiveIVRegressor``.
    """

    def __init__(self, causal_graph: CausalGraph, treatment: str, outcome: str, n_draws: int = 400, random_state: int = 0):
        if not isinstance(causal_graph, CausalGraph):
            causal_graph = CausalGraph(causal_graph)
        self.causal_graph = causal_graph
        self.treatment = str(treatment)
        self.outcome = str(outcome)
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, data):
        frame = _as_frame(data)
        instruments = self.causal_graph.instruments(self.treatment, self.outcome)
        if not instruments:
            raise ValueError(
                f"The graph has no instrument for the effect of {self.treatment} on {self.outcome}."
            )
        for column in [self.treatment, self.outcome, *instruments]:
            if column not in frame.columns:
                raise ValueError(f"Data is missing column {column!r}.")
        self.instruments_ = instruments
        self.model_ = BayesianInstrumentalVariable(n_draws=self.n_draws, random_state=self.random_state)
        self.model_.fit(
            frame[self.outcome].to_numpy(),
            frame[self.treatment].to_numpy(),
            frame[instruments].to_numpy(),
        )
        self.effect_ = self.model_.effect_
        self.draws_ = self.model_.draws_
        self.coef_ = self.model_.coef_
        self.summary_ = dict(self.model_.summary_)
        return self

    def predict(self, treatment) -> np.ndarray:
        return self.model_.predict(treatment)

    def summary(self) -> dict:
        public = dict(self.summary_)
        public["instruments"] = list(self.instruments_)
        return public


# ---------------------------------------------------------------------------
# Structural causal model and structural time series
# ---------------------------------------------------------------------------


def simulate_structural_scm(n: int = 1500, alpha: float = 3.0, rho: float = 0.6, sigma: float = 1.0, random_state: int = 0, binary_treatment: bool = False) -> dict:
    """Confounded structural system with instruments and one outcome covariate.

    Columns 0 and 1 are confounders, 2 and 3 are instruments (they do not
    enter the outcome equation), and column 4 predicts the outcome only.
    Treatment and outcome errors are bivariate normal with correlation ``rho``.
    The structural treatment effect is ``alpha``. This is the simulation design
    used in the CausalPy structural-model notes.
    """
    rng = np.random.default_rng(random_state)
    covariates = rng.normal(size=(int(n), 5))
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    beta_y = np.array([0.5, -0.3, 0.0, 0.0, 0.4])
    beta_t = np.array([0.7, 0.1, -0.4, 0.3, 0.0])
    covariance = float(sigma) ** 2 * np.array([[1.0, rho], [rho, 1.0]])
    errors = rng.multivariate_normal(np.zeros(2), covariance, size=int(n))
    outcome_error, treatment_error = errors[:, 0], errors[:, 1]
    latent = covariates @ beta_t + treatment_error
    if binary_treatment:
        treatment = (latent > 0.0).astype(float)
    else:
        treatment = latent
    outcome = float(alpha) * treatment + covariates @ beta_y + outcome_error
    return {
        "X": covariates,
        "treatment": treatment,
        "y": outcome,
        "alpha": float(alpha),
        "rho": float(rho),
        "confounders": [0, 1],
        "instruments": [2, 3],
        "outcome_only": [4],
        "feature_names": [f"feature_{i}" for i in range(5)],
    }


class BayesianStructuralSCM:
    """Joint structural model for an endogenous treatment.

    The assignments are ::

        T = a_t + X_t beta_t + V
        Y = alpha * T + a_y + X_y beta_y + U

    with ``Corr(U, V) = rho``. Ordinary least squares is biased when ``rho``
    is not zero, even after the observed covariates are included, because ``V``
    remains inside ``T``. Identification uses exclusion restrictions:

    * ``instruments`` enter the treatment equation only.
    * ``outcome_only`` enters the outcome equation only.
    * every other column is a confounder and enters both.

    The estimator is two-stage least squares. Posterior draws are a Bayesian
    bootstrap of both stages. Predictions under ``do(T=t)`` use the outcome
    equation only: the control residual is an estimation device, not part of
    the intervened structural assignment.

    If ``instruments`` is omitted, the treatment is taken to be conditionally
    ignorable given ``X`` and the fit is a Bayesian outcome regression.
    """

    def __init__(self, n_draws: int = 400, random_state: int = 0):
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, X, treatment, y, instruments=None, outcome_only=None, feature_names=None):
        if isinstance(X, pd.DataFrame):
            names = [str(column) for column in X.columns]
            covariates = X.to_numpy(dtype=float)
        else:
            covariates = _as_2d(X, "X")
            names = [f"X{i}" for i in range(covariates.shape[1])]
        if feature_names is not None:
            names = [str(name) for name in feature_names]
            if len(names) != covariates.shape[1]:
                raise ValueError("feature_names length does not match X.shape[1].")
        treatment = _as_1d(treatment, "treatment")
        y = _as_1d(y, "y")
        _check_lengths(len(covariates), treatment=treatment, y=y)
        instrument_idx = _resolve_columns(names, instruments)
        outcome_idx = _resolve_columns(names, outcome_only)
        if set(instrument_idx) & set(outcome_idx):
            raise ValueError("A covariate cannot be both an instrument and outcome-only.")
        confounder_idx = [i for i in range(covariates.shape[1]) if i not in instrument_idx and i not in outcome_idx]
        self.feature_names_ = names
        self.instrument_index_ = instrument_idx
        self.outcome_only_index_ = outcome_idx
        self.confounder_index_ = confounder_idx
        self._X = covariates
        self._treatment = treatment
        self._y = y

        if instrument_idx:
            self.identified_by_ = "instrumental_variable"
            point = self._iv_coef(np.ones(len(y)))
            rng = np.random.default_rng(self.random_state)
            draws = np.empty((self.n_draws, len(point)), dtype=float)
            for b in range(self.n_draws):
                draws[b] = self._iv_coef(rng.dirichlet(np.ones(len(y))))
        else:
            self.identified_by_ = "conditional_ignorability"
            design = self._outcome_design(covariates, treatment)
            coef, cov, sigma2 = _normal_posterior(design, y, 1e-4, intercept_index=1)
            self.sigma2_ = sigma2
            self.posterior_cov_ = cov
            point = coef
            draws = _draw_normal(coef, cov, self.n_draws, np.random.default_rng(self.random_state))
        self.coef_ = point
        self.coef_draws_ = draws
        self.effect_ = float(point[0])
        self.draws_ = draws[:, 0]
        self.summary_ = _summarize(self.effect_, self.draws_)
        self.summary_["identified_by"] = self.identified_by_
        return self

    def predict(self, X, treatment) -> np.ndarray:
        """Mean outcome under ``do(treatment)`` at covariates ``X``."""
        covariates = self._align(X)
        treatment = _as_1d(treatment, "treatment")
        _check_lengths(len(covariates), treatment=treatment)
        design = self._outcome_design(covariates, treatment)
        return design @ self.coef_

    def predict_interval(self, X, treatment, level: float = 0.95):
        """Pointwise credible interval for ``predict`` from coefficient draws."""
        covariates = self._align(X)
        treatment = _as_1d(treatment, "treatment")
        design = self._outcome_design(covariates, treatment)
        paths = self.coef_draws_ @ design.T
        alpha = (1.0 - float(level)) / 2.0
        low, high = np.quantile(paths, [alpha, 1.0 - alpha], axis=0)
        return design @ self.coef_, low, high

    def potential_outcomes(self, X):
        """``Y(0)`` and ``Y(1)`` from the homogeneous structural effect."""
        covariates = self._align(X)
        zeros = np.zeros(len(covariates))
        ones = np.ones(len(covariates))
        return self.predict(covariates, zeros), self.predict(covariates, ones)

    def average_treatment_effect(self) -> dict:
        return self.summary()

    def summary(self) -> dict:
        return dict(self.summary_)

    def _iv_coef(self, weights: np.ndarray) -> np.ndarray:
        first = self._with_intercept(self._X[:, self.confounder_index_ + self.instrument_index_])
        fitted = first @ _wls(first, self._treatment, weights)
        outcome = self._outcome_design(self._X, fitted)
        return _wls(outcome, self._y, weights)

    def _outcome_design(self, covariates: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        exogenous = self._with_intercept(covariates[:, self.confounder_index_ + self.outcome_only_index_])
        return np.column_stack([np.asarray(treatment, dtype=float).ravel(), exogenous])

    @staticmethod
    def _with_intercept(block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=float)
        if block.size == 0:
            return np.ones((block.shape[0] if block.ndim == 2 else 0, 1))
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        return np.column_stack([np.ones(block.shape[0]), block])

    def _align(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [name for name in self.feature_names_ if name not in X.columns]
            if missing:
                raise ValueError(f"X is missing columns {missing}.")
            return X[self.feature_names_].to_numpy(dtype=float)
        covariates = _as_2d(X, "X")
        if covariates.shape[1] != len(self.feature_names_):
            raise ValueError("X has a different number of columns than the fitted model.")
        return covariates


class BayesianTimeSeriesSCM:
    """Bayesian structural causal model for a multivariate series.

    Contemporaneous parents form a DAG. Each equation also includes its own
    lags and the lags of its contemporaneous parents, up to ``lag``. Coefficients
    have an independent Gaussian posterior per equation. A hard intervention
    replaces that equation for the rollout horizon. ``impulse_response`` adds a
    one-period shock to the structural residual instead of clamping the variable.

    ``edges`` are contemporaneous ``(parent, child)`` pairs. An empty edge list
    is a system of independent autoregressions.
    """

    def __init__(self, edges=None, lag: int = 1, prior_precision: float = 1e-4, n_draws: int = 200, random_state: int = 0):
        self.edges = [] if edges is None else [(str(parent), str(child)) for parent, child in edges]
        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError("lag must be at least 1.")
        self.prior_precision = float(prior_precision)
        self.n_draws = int(n_draws)
        self.random_state = int(random_state)

    def fit(self, data, names=None):
        values, names = _series_matrix(data, names)
        if len(values) <= self.lag + 2:
            raise ValueError("The series is shorter than lag + 3 rows.")
        known = set(names)
        for parent, child in self.edges:
            if parent not in known or child not in known:
                raise ValueError(f"Edge {(parent, child)} uses a name outside {names}.")
        graph = CausalGraph(self.edges, nodes=names)
        self.names_ = list(graph.nodes)
        order = {name: i for i, name in enumerate(names)}
        values = values[:, [order[name] for name in self.names_]]
        self.index_ = {name: i for i, name in enumerate(self.names_)}
        self.parents_ = {name: graph.parents(name) for name in self.names_}
        self.data_ = values
        self.equations_ = {}
        for name in self.names_:
            rows = []
            target = []
            for t in range(self.lag, len(values)):
                assigned = {parent: values[t, self.index_[parent]] for parent in self.parents_[name]}
                window = [values[t - self.lag + j] for j in range(self.lag)]
                rows.append(self._feature_row(name, assigned, window))
                target.append(values[t, self.index_[name]])
            design = np.vstack(rows)
            coef, cov, sigma2 = _normal_posterior(design, np.asarray(target), self.prior_precision)
            self.equations_[name] = {
                "coef": coef,
                "cov": cov,
                "sigma2": sigma2,
                "feature_names": self._feature_names(name),
            }
        return self

    def forecast(self, horizon: int, history=None) -> pd.DataFrame:
        """Observational mean path over ``horizon`` steps."""
        path = self._rollout(self._mean_coefs(), self._history(history), int(horizon), None, None)
        return pd.DataFrame(path, columns=self.names_)

    def counterfactual(self, horizon: int, interventions: dict, baseline=None, history=None) -> dict:
        """Effect of hard interventions against a baseline rollout.

        ``interventions`` maps a variable to a scalar or a horizon-length
        sequence. ``baseline=None`` is the observational forecast. Pass a dict
        to contrast two interventions, for example ``do(X=1)`` against ``do(X=0)``.
        """
        self._require_fitted()
        horizon = int(horizon)
        history = self._history(history)
        mean = self._mean_coefs()
        treated = self._rollout(mean, history, horizon, interventions, None)
        base = self._rollout(mean, history, horizon, baseline, None)
        effect = treated - base
        rng = np.random.default_rng(self.random_state)
        cloud = np.empty((self.n_draws, horizon, len(self.names_)))
        for b in range(self.n_draws):
            drawn = self._draw_coefs(rng)
            cloud[b] = self._rollout(drawn, history, horizon, interventions, None) - self._rollout(
                drawn, history, horizon, baseline, None
            )
        low, high = np.quantile(cloud, [0.025, 0.975], axis=0)
        return {
            "names": list(self.names_),
            "counterfactual": treated,
            "baseline": base,
            "effect": effect,
            "effect_lower": low,
            "effect_upper": high,
        }

    def impulse_response(self, variable: str, shock: float = 1.0, horizon: int = 10, history=None) -> dict:
        """One-period structural shock, then the equations return to normal."""
        self._require_fitted()
        variable = str(variable)
        if variable not in self.index_:
            raise ValueError(f"Unknown variable {variable!r}.")
        horizon = int(horizon)
        history = self._history(history)
        mean = self._mean_coefs()
        base = self._rollout(mean, history, horizon, None, None)
        shocked = self._rollout(mean, history, horizon, None, {variable: float(shock)})
        return {
            "names": list(self.names_),
            "response": shocked,
            "baseline": base,
            "effect": shocked - base,
        }

    def summary(self, node: str) -> dict:
        equation = self.equations_[str(node)]
        return {
            "feature_names": list(equation["feature_names"]),
            "coef": np.asarray(equation["coef"], dtype=float).copy(),
            "sigma2": float(equation["sigma2"]),
        }

    def _rollout(self, coefs, history, horizon, interventions, shock):
        interventions = {} if interventions is None else interventions
        hist = [row.astype(float).copy() for row in history]
        out = np.zeros((horizon, len(self.names_)))
        for step in range(horizon):
            current = np.zeros(len(self.names_))
            assigned = {}
            for name in self.names_:
                column = self.index_[name]
                if name in interventions:
                    spec = interventions[name]
                    value = float(spec[step] if np.ndim(spec) else spec)
                else:
                    value = float(self._feature_row(name, assigned, hist) @ coefs[name])
                    if shock and step == 0 and name in shock:
                        value += float(shock[name])
                current[column] = value
                assigned[name] = value
            out[step] = current
            hist.append(current)
            hist = hist[-self.lag :]
        return out

    def _feature_row(self, node: str, assigned: dict, hist: list) -> np.ndarray:
        features = [1.0]
        for parent in self.parents_[node]:
            features.append(float(assigned[parent]))
        for lag in range(1, self.lag + 1):
            past = hist[-lag]
            features.append(float(past[self.index_[node]]))
            for parent in self.parents_[node]:
                features.append(float(past[self.index_[parent]]))
        return np.asarray(features, dtype=float)

    def _feature_names(self, node: str) -> list:
        names = ["intercept"]
        names.extend(f"{parent}[t]" for parent in self.parents_[node])
        for lag in range(1, self.lag + 1):
            names.append(f"{node}[t-{lag}]")
            names.extend(f"{parent}[t-{lag}]" for parent in self.parents_[node])
        return names

    def _mean_coefs(self) -> dict:
        self._require_fitted()
        return {name: eq["coef"] for name, eq in self.equations_.items()}

    def _draw_coefs(self, rng: np.random.Generator) -> dict:
        drawn = {}
        for name, eq in self.equations_.items():
            drawn[name] = _draw_normal(eq["coef"], eq["cov"], 1, rng)[0]
        return drawn

    def _history(self, history) -> list:
        self._require_fitted()
        if history is None:
            history = self.data_[-self.lag :]
        history = np.asarray(history, dtype=float)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.shape[1] != len(self.names_):
            raise ValueError("history must have one column per series, in the fitted order.")
        if history.shape[0] < self.lag:
            raise ValueError(f"history must contain at least {self.lag} rows.")
        return [history[-self.lag + j] for j in range(self.lag)]

    def _require_fitted(self) -> None:
        if not hasattr(self, "equations_"):
            raise RuntimeError("Call fit() first.")


def _as_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame()
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError("Tabular data must be 2-dimensional.")
        frame = pd.DataFrame(array, columns=[f"X{i}" for i in range(array.shape[1])])
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Tabular data is empty.")
    return frame


def _series_matrix(data, names):
    if isinstance(data, pd.DataFrame):
        inferred = [str(column) for column in data.columns]
        values = data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        inferred = [f"X{i}" for i in range(values.shape[1])]
    if names is not None:
        inferred = [str(name) for name in names]
        if len(inferred) != values.shape[1]:
            raise ValueError("names length does not match the number of series.")
    if not np.isfinite(values).all():
        raise ValueError("The series contains NaN or infinite values.")
    return values, inferred


def _resolve_columns(names: list, spec) -> list:
    if spec is None:
        return []
    if isinstance(spec, (str, int)):
        spec = [spec]
    resolved = []
    for item in spec:
        if isinstance(item, str):
            if item not in names:
                raise ValueError(f"Unknown covariate {item!r}.")
            resolved.append(names.index(item))
        else:
            index = int(item)
            if index < 0 or index >= len(names):
                raise ValueError(f"Covariate index {index} is outside 0..{len(names) - 1}.")
            resolved.append(index)
    return resolved


def _python_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _same_state(left, right) -> bool:
    if left == right:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return False

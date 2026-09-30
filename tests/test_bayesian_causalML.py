"""Recovery checks for the standalone Bayesian causal module."""

import numpy as np
import pandas as pd
import pytest

from pycausalml.bayesian_causalML import (
    BayesianDifferenceInDifferences,
    BayesianInstrumentalVariable,
    BayesianInterruptedTimeSeries,
    BayesianRegressionDiscontinuity,
    BayesianStructuralSCM,
    BayesianSyntheticControl,
    BayesianTimeSeriesSCM,
    CausalGraph,
    DiscreteBayesianNetwork,
    GaussianBayesianNetwork,
    NaiveAdjustmentRegressor,
    NaiveIVRegressor,
    simulate_structural_scm,
)
from pycausalml.data import make_iv_data


class TestIdentification:
    def test_d_separation_canonical_graphs(self):
        chain = CausalGraph([("A", "B"), ("B", "C")])
        assert chain.d_separated("A", "C", "B")
        assert not chain.d_separated("A", "C")

        fork = CausalGraph([("B", "A"), ("B", "C")])
        assert fork.d_separated("A", "C", "B")
        assert not fork.d_separated("A", "C")

        collider = CausalGraph([("A", "B"), ("C", "B")])
        assert collider.d_separated("A", "C")
        assert not collider.d_separated("A", "C", "B")

    def test_backdoor_and_frontdoor(self):
        confounded = CausalGraph([("C", "T"), ("C", "Y"), ("T", "Y")])
        assert confounded.minimal_adjustment_set("T", "Y") == ["C"]
        assert not confounded.satisfies_backdoor("T", "Y", [])
        assert confounded.adjacency().loc["T", "C"] == 1.0

        frontdoor = CausalGraph([("U", "T"), ("U", "Y"), ("T", "M"), ("M", "Y")])
        assert frontdoor.frontdoor_sets("T", "Y") == [frozenset({"M"})]
        assert frontdoor.instruments("T", "Y") == []

        instrument = CausalGraph([("Z", "T"), ("T", "Y")])
        assert instrument.instruments("T", "Y") == ["Z"]
        leaked = CausalGraph([("Z", "T"), ("T", "Y"), ("Z", "Y")])
        assert leaked.instruments("T", "Y") == []

    def test_cycle_is_rejected(self):
        with pytest.raises(ValueError):
            CausalGraph([("A", "B"), ("B", "A")])


class TestQuasiExperiments:
    def test_did_rd_its_iv_synthetic(self):
        rng = np.random.default_rng(0)
        n = 240
        treated = np.repeat([0.0, 1.0], n // 2)
        post = np.tile([0.0, 1.0], n // 2)
        y = 1.0 + 0.2 * treated + 0.3 * post + 1.5 * treated * post + rng.normal(scale=0.15, size=n)
        did = BayesianDifferenceInDifferences(n_draws=400, random_state=0)
        did.fit(y, treated, post)
        assert abs(did.effect_ - 1.5) < 0.2
        assert did.summary()["ci_lower"] < 1.5 < did.summary()["ci_upper"]

        running = rng.uniform(-1, 1, size=400)
        jump = (running >= 0).astype(float)
        outcome = 0.4 * running + 1.2 * jump + rng.normal(scale=0.1, size=400)
        rd = BayesianRegressionDiscontinuity(cutoff=0.0, n_draws=400, random_state=0)
        rd.fit(running, outcome)
        assert abs(rd.effect_ - 1.2) < 0.2

        time = np.arange(80, dtype=float)
        series = 0.05 * time + np.where(time >= 40, 2.0, 0.0) + rng.normal(scale=0.05, size=80)
        its = BayesianInterruptedTimeSeries(n_draws=400, random_state=0)
        its.fit(time, series, intervention_time=40)
        assert abs(its.effect_ - 2.0) < 0.25
        assert abs(its.slope_change_) < 0.05

        data = make_iv_data(n=800, random_state=1)
        iv = BayesianInstrumentalVariable(n_draws=80, random_state=1)
        iv.fit(data["Y"], data["W"], data["Z"], data["X"])
        assert abs(iv.effect_ - 1.5) < 0.35
        assert iv.summary()["ci_lower"] < 1.5 < iv.summary()["ci_upper"]

        pre = rng.normal(size=(12, 3))
        weights = np.array([0.2, 0.5, 0.3])
        treated_pre = pre @ weights
        post_donors = rng.normal(size=(6, 3))
        treated_post = post_donors @ weights + 1.0
        sc = BayesianSyntheticControl(n_draws=8, random_state=0)
        sc.fit(pre, treated_pre, post_donors, treated_post)
        assert abs(sc.effect_ - 1.0) < 0.05
        assert abs(sc.weights_.sum() - 1.0) < 1e-6


class TestBayesianNetworks:
    def test_gaussian_total_effect_adjusts_for_confounding(self):
        rng = np.random.default_rng(0)
        n = 3000
        confounder = rng.normal(size=n)
        treatment = 0.5 * confounder + rng.normal(scale=0.4, size=n)
        outcome = 1.5 * treatment - 0.7 * confounder + rng.normal(scale=0.4, size=n)
        frame = pd.DataFrame({"C": confounder, "T": treatment, "Y": outcome})
        naive = np.linalg.lstsq(np.column_stack([np.ones(n), treatment]), outcome, rcond=None)[0][1]
        model = GaussianBayesianNetwork(
            [("C", "T"), ("C", "Y"), ("T", "Y")], n_draws=200, random_state=0
        )
        model.fit(frame)
        ate = model.ate("T", "Y")
        assert abs(ate["estimate"] - 1.5) < 0.15
        assert ate["ci_lower"] < 1.5 < ate["ci_upper"]
        assert abs(naive - 1.5) > 0.4

        draws = model.sample(4000, do={"T": 1.0}, random_state=1)
        draws0 = model.sample(4000, do={"T": 0.0}, random_state=2)
        assert abs((draws["Y"].mean() - draws0["Y"].mean()) - 1.5) < 0.2

    def test_discrete_intervention_differs_from_conditioning(self):
        rng = np.random.default_rng(0)
        n = 8000
        treatment = rng.binomial(1, 0.5, size=n)
        mediator = rng.binomial(1, np.where(treatment == 1, 0.8, 0.2))
        outcome = rng.binomial(1, np.where(mediator == 1, 0.9, 0.1))
        chain = pd.DataFrame({"T": treatment, "M": mediator, "Y": outcome})
        network = DiscreteBayesianNetwork([("T", "M"), ("M", "Y")], alpha=0.5, random_state=0)
        network.fit(chain)
        ate = network.ate("T", "Y", treated=1, control=0)
        assert abs(ate["estimate"] - 0.48) < 0.06
        assert network.query("Y", do={"T": 1})["method"] == "exact"

        confounder = rng.binomial(1, 0.5, size=n)
        treatment = rng.binomial(1, np.where(confounder == 1, 0.9, 0.1))
        probability = 0.1 + 0.2 * treatment + 0.6 * confounder
        outcome = rng.binomial(1, probability)
        frame = pd.DataFrame({"C": confounder, "T": treatment, "Y": outcome})
        confounded = DiscreteBayesianNetwork(
            [("C", "T"), ("C", "Y"), ("T", "Y")], alpha=0.5, random_state=0
        )
        confounded.fit(frame)
        interventional = confounded.query("Y", do={"T": 1})["distributions"]["Y"][1]
        conditional = confounded.query("Y", evidence={"T": 1})["distributions"]["Y"][1]
        assert abs(interventional - 0.60) < 0.07
        assert conditional > interventional + 0.1

    def test_naive_adjustment_and_iv(self):
        rng = np.random.default_rng(1)
        n = 2500
        confounder = rng.normal(size=n)
        treatment = 0.8 * confounder + rng.normal(scale=0.5, size=n)
        outcome = 1.5 * treatment + 1.2 * confounder + rng.normal(scale=0.4, size=n)
        frame = pd.DataFrame({"C": confounder, "T": treatment, "Y": outcome})
        adjusted = NaiveAdjustmentRegressor(CausalGraph([("C", "T"), ("C", "Y"), ("T", "Y")]), "T", "Y")
        adjusted.fit(frame)
        assert adjusted.adjustment_set_ == ["C"]
        assert abs(adjusted.effect_ - 1.5) < 0.15

        instrument = rng.normal(size=n)
        latent = rng.normal(size=n)
        endogenous = 0.9 * instrument + 0.7 * latent + rng.normal(scale=0.1, size=n)
        response = 1.5 * endogenous + 0.8 * latent + rng.normal(scale=0.2, size=n)
        iv_frame = pd.DataFrame({"Z": instrument, "T": endogenous, "Y": response})
        iv = NaiveIVRegressor(CausalGraph([("Z", "T"), ("T", "Y")]), "T", "Y", n_draws=40, random_state=0)
        iv.fit(iv_frame)
        assert iv.instruments_ == ["Z"]
        assert abs(iv.effect_ - 1.5) < 0.35


class TestStructuralModels:
    def test_structural_scm_uses_the_exclusion_restriction(self):
        data = simulate_structural_scm(n=2500, alpha=3.0, rho=0.6, random_state=0)
        design = np.column_stack([np.ones(len(data["y"])), data["treatment"], data["X"]])
        ols = np.linalg.lstsq(design, data["y"], rcond=None)[0][1]
        model = BayesianStructuralSCM(n_draws=40, random_state=0)
        model.fit(
            data["X"],
            data["treatment"],
            data["y"],
            instruments=data["instruments"],
            outcome_only=data["outcome_only"],
        )
        assert model.identified_by_ == "instrumental_variable"
        assert abs(model.effect_ - 3.0) < 0.3
        assert abs(ols - 3.0) > abs(model.effect_ - 3.0)
        y0, y1 = model.potential_outcomes(data["X"])
        assert abs(np.mean(y1 - y0) - model.effect_) < 1e-8

    def test_time_series_intervention(self):
        rng = np.random.default_rng(0)
        n = 1500
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.4 * x[t - 1] + rng.normal(0.0, 0.5)
            y[t] = 0.8 * x[t] + 0.2 * y[t - 1] + rng.normal(0.0, 0.3)
        model = BayesianTimeSeriesSCM(edges=[("X", "Y")], lag=1, n_draws=25, random_state=0)
        model.fit(pd.DataFrame({"X": x, "Y": y}))
        equation = model.summary("Y")
        names = equation["feature_names"]
        contemporaneous = float(equation["coef"][names.index("X[t]")])
        own_lag = float(equation["coef"][names.index("Y[t-1]")])
        parent_lag = float(equation["coef"][names.index("X[t-1]")])
        assert abs(contemporaneous - 0.8) < 0.12

        irf = model.impulse_response("X", shock=1.0, horizon=6, history=np.zeros((1, 2)))
        y_index = irf["names"].index("Y")
        assert abs(irf["effect"][0, y_index] - contemporaneous) < 1e-8

        steady = (contemporaneous + parent_lag) / (1.0 - own_lag)
        contrast = model.counterfactual(
            40,
            {"X": 1.0},
            baseline={"X": 0.0},
            history=np.zeros((1, 2)),
        )
        assert abs(contrast["effect"][-1, y_index] - steady) < 1e-4
        assert abs(steady - 1.0) < 0.2
        assert contrast["effect_lower"].shape == contrast["effect"].shape

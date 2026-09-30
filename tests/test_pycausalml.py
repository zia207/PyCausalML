"""Tests for the classical PyCausalML modules."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pycausalml
from pycausalml import (
    BayesianLinearCATE,
    BoostedIVForest,
    CausalForest,
    CausalForestDML,
    CausalSurvivalForest,
    CausalTree,
    CXGBoost,
    DifferenceInDifferences,
    DomainAdaptationLearner,
    DRLearner,
    DRPolicyForest,
    DRPolicyTree,
    InstrumentalForest,
    InstrumentalVariable,
    InteractionTree,
    InteractiveRegressionDML,
    InterruptedTimeSeries,
    KernelDML,
    LinearDML,
    MultiArmCausalBoost,
    MultiArmCausalForest,
    NonParamDML,
    PartiallyLinearDML,
    PolicyLearner,
    RegressionDiscontinuity,
    RLearner,
    SLearner,
    SparseLinearDML,
    SyntheticControl,
    TARNet,
    TLearner,
    UpliftRandomForest,
    UpliftTree,
    UpliftXGBoost,
    XLearner,
    __version__,
    available_backends,
    explain_cate,
    synthetic_data,
)
from pycausalml.bayesian_causal_inference import linear_shap, rope_probability
from pycausalml.data import (
    make_binary_uplift,
    make_iv_data,
    make_linear_treatment,
    make_multiarm_data,
    make_survival_data,
)
from pycausalml.integrations import (
    causallearn_direct_lingam,
    causallearn_ges,
    causallearn_pc,
    causalml_slearner,
    causalml_tlearner,
    causalpy_difference_in_differences,
    doubleml_irm,
    doubleml_plr,
    dowhy_ate,
    econml_causal_forest,
    econml_dml,
    econml_drlearner,
    require_module,
)
from pycausalml.plotting import plot_cate_hist
from pycausalml.policy_learners import policy_value
from pycausalml.tree_based_models import (
    average_treatment_effect,
    best_linear_projection,
    fit_skgrf_causal_forest,
    plot_causal_tree,
    variable_importance,
)
from pycausalml.uplift_learners import plot_uplift_curve, qini_coefficient, qini_curve


def _pehe(truth, estimate) -> float:
    truth = np.asarray(truth, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    return float(np.sqrt(np.mean((truth - estimate) ** 2)))


class TestPackageSurface:
    def test_version_and_deep_reexport(self):
        assert __version__ == "0.2.0"
        from pydeepcausalml import TARNet as DeepTARNet

        assert TARNet is DeepTARNet

    def test_backend_registry(self):
        found = available_backends()
        assert set(found) >= {"dowhy", "econml", "causallearn", "causalml", "doubleml", "causalpy", "skgrf"}
        assert all(isinstance(value, bool) for value in found.values())

    def test_every_public_name_is_exported(self):
        missing = [name for name in pycausalml.__all__ if not hasattr(pycausalml, name)]
        assert missing == []


class TestTreeModels:
    def test_causal_tree_and_plot(self):
        data = synthetic_data(n=360, p=4, sigma=0.5, mode=2, random_state=0)
        tree = CausalTree(min_samples_leaf=20, max_depth=3, random_state=0)
        tree.fit(data["X"], data["w"], data["y"])
        pred = tree.predict(data["X"])
        assert pred.shape == (360,)
        assert np.isfinite(pred).all()
        assert tree.rules()
        assert tree.feature_importance().shape == (4,)
        fig = plot_causal_tree(tree)
        assert fig is not None
        plt.close(fig)

    def test_causal_forest_ate(self):
        data = synthetic_data(n=500, p=4, sigma=0.5, mode=2, random_state=1)
        forest = CausalForest(num_trees=25, nuisance_estimators=15, random_state=1)
        forest.fit(data["X"], data["w"], data["y"])
        ate = forest.average_treatment_effect()
        assert abs(ate["estimate"] - data["tau"].mean()) < 0.75
        assert ate["std.err"] > 0
        assert forest.feature_importance().shape == (4,)
        ranked = forest.rank_average_treatment_effect(forest.predict(data["X"]))
        assert "autoc" in ranked
        projection = best_linear_projection(forest.dr_scores_, data["X"][:, 0])
        assert projection["params"].shape == (2,)
        summary = average_treatment_effect(forest)
        assert summary["estimate"] == ate["estimate"]
        np.testing.assert_allclose(variable_importance(forest), forest.feature_importance())

    def test_instrumental_and_boosted_iv(self):
        data = make_iv_data(n=700, random_state=0)
        forest = InstrumentalForest(num_trees=20, random_state=0)
        forest.fit(data["X"], data["W"], data["Y"], data["Z"])
        ate = forest.average_treatment_effect()["estimate"]
        assert abs(ate - 1.5) < 0.8
        boosted = BoostedIVForest(n_estimators=20, learner="gbr", random_state=0)
        boosted.fit(data["X"], data["W"], data["Y"], data["Z"])
        pred = boosted.predict(data["X"])
        assert np.isfinite(pred).all()
        assert abs(pred.mean() - 1.5) < 1.0

    def test_multiarm_and_survival(self):
        data = make_multiarm_data(n=420, random_state=0)
        forest = MultiArmCausalForest(num_trees=15, random_state=0)
        forest.fit(data["X"], data["w"], data["y"])
        pred = forest.predict(data["X"])
        assert pred.shape == (420, 2)
        assert np.isfinite(pred).all()
        boost = MultiArmCausalBoost(n_estimators=15, learner="gbr", random_state=0)
        boost.fit(data["X"], data["w"], data["y"])
        assert boost.predict(data["X"]).shape == (420, 2)

        survival = make_survival_data(n=280, random_state=0)
        surv = CausalSurvivalForest(horizon=survival["horizon"], num_trees=15, random_state=0)
        surv.fit(survival["X"], survival["w"], survival["time"], survival["event"])
        effect = surv.average_treatment_effect()["estimate"]
        assert np.isfinite(effect)
        assert effect > 0

    def test_cxgboost(self):
        data = synthetic_data(n=400, p=4, sigma=0.6, mode=2, random_state=2)
        model = CXGBoost(n_estimators=25, learner="gbr", random_state=2)
        model.fit(data["X"], data["w"], data["y"])
        pred = model.predict(data["X"])
        assert _pehe(data["tau"], pred) < 1.4

    def test_skgrf_wrapper(self):
        data = synthetic_data(n=80, p=3, sigma=0.5, mode=2, random_state=0)
        if not available_backends()["skgrf"]:
            with pytest.raises(ImportError, match="skgrf"):
                fit_skgrf_causal_forest(data["X"], data["y"], data["w"])
            return
        est = fit_skgrf_causal_forest(data["X"], data["y"], data["w"], n_estimators=10)
        pred = np.asarray(est.predict(data["X"]), dtype=float).ravel()
        assert pred.shape == (80,)
        assert np.isfinite(pred).all()


class TestDataGenerators:
    def test_modes_and_invalid_inputs(self):
        zero = synthetic_data(n=40, p=3, mode=1, random_state=0)
        assert np.all(zero["tau"] == 0)
        smooth = synthetic_data(n=40, p=3, mode=3, random_state=0)
        assert np.isfinite(smooth["tau"]).all()
        assert smooth["tau"].std() > 0
        with pytest.raises(ValueError, match="mode"):
            synthetic_data(mode=9)
        with pytest.raises(ValueError, match="p >= 3"):
            synthetic_data(p=2)


class TestUplift:
    def test_criteria_and_curve(self):
        data = make_binary_uplift(n=400, random_state=0)
        for criterion in ("kl", "ed", "chi", "cts", "ddp"):
            tree = UpliftTree(criterion=criterion, max_depth=2, min_samples_leaf=25, random_state=0)
            tree.fit(data["X"], data["w"], data["y"])
            pred = tree.predict(data["X"])
            assert pred.shape == (400,)
            assert np.isfinite(pred).all()
        forest = UpliftRandomForest(criterion="kl", n_trees=6, max_depth=2, random_state=0)
        forest.fit(data["X"], data["w"], data["y"])
        score = forest.predict(data["X"])
        curve = qini_curve(data["y"], data["w"], score)
        assert len(curve["qini"]) == 400
        assert np.isfinite(qini_coefficient(data["y"], data["w"], score))
        fig = plot_uplift_curve(data["y"], data["w"], score)
        plt.close(fig)
        interaction = InteractionTree(max_depth=2, min_samples_leaf=25, random_state=0)
        interaction.fit(data["X"], data["w"], data["y"])
        assert np.isfinite(interaction.predict(data["X"])).all()

    def test_uplift_xgboost(self):
        data = synthetic_data(n=350, p=4, sigma=0.7, mode=2, random_state=3)
        model = UpliftXGBoost(n_estimators=20, learner="gbr", random_state=3)
        model.fit(data["X"], data["w"], data["y"])
        mu0, mu1 = model.predict_potential_outcomes(data["X"])
        np.testing.assert_allclose(mu1 - mu0, model.predict(data["X"]))


class TestMetaLearners:
    def test_learners_recover_heterogeneous_effect(self):
        data = synthetic_data(n=500, p=4, sigma=0.55, mode=2, random_state=4)
        learners = [
            SLearner(learner="gbr", n_estimators=25, random_state=4),
            TLearner(learner="gbr", n_estimators=25, random_state=4),
            XLearner(learner="gbr", n_estimators=20, random_state=4),
            RLearner(learner="gbr", n_estimators=20, random_state=4),
            DRLearner(learner="gbr", n_estimators=20, random_state=4),
            DomainAdaptationLearner(learner="gbr", n_estimators=20, random_state=4),
        ]
        for learner in learners:
            learner.fit(data["X"], data["w"], data["y"])
            pred = learner.predict(data["X"])
            assert pred.shape == (500,)
            assert np.isfinite(pred).all()
            assert _pehe(data["tau"], pred) < 1.6
        boosted = SLearner(learner="xgb", n_estimators=15, random_state=4)
        boosted.fit(data["X"], data["w"], data["y"])
        assert np.isfinite(boosted.predict(data["X"])).all()

    def test_dr_ate_interval(self):
        data = make_linear_treatment(n=600, tau=2.0, sigma=0.8, random_state=0)
        learner = DRLearner(learner="ridge", n_estimators=10, random_state=0)
        learner.fit(data["X"], data["w"], data["y"])
        summary = learner.average_treatment_effect()
        assert abs(summary["estimate"] - 2.0) < 0.45
        assert summary["std.err"] > 0


class TestDoubleML:
    def test_partially_linear_constant_effect(self):
        data = make_linear_treatment(n=700, tau=2.0, sigma=0.7, random_state=1)
        model = PartiallyLinearDML(model_y="ridge", random_state=1)
        model.fit(data["X"], data["w"], data["y"])
        assert abs(model.effect() - 2.0) < 0.35
        low, high = model.confidence_interval()
        assert low < 2.0 < high

    def test_linear_and_sparse_cate(self):
        rng = np.random.default_rng(0)
        n, p = 600, 4
        X = rng.normal(size=(n, p))
        propensity = 1 / (1 + np.exp(-0.3 * X[:, 0]))
        w = rng.binomial(1, propensity)
        tau = 1.0 + 0.8 * X[:, 1]
        y = 0.4 * X[:, 0] + w * tau + rng.normal(scale=0.6, size=n)
        for cls in (LinearDML, SparseLinearDML):
            model = cls(model_y="ridge", n_estimators=10, random_state=0, alpha=0.02)
            model.fit(X, w, y)
            pred = model.predict(X)
            assert np.corrcoef(pred, tau)[0, 1] > 0.45
            assert np.isfinite(model.average_treatment_effect()["std.err"])

    def test_nonparam_kernel_irm(self):
        data = make_linear_treatment(n=450, tau=1.5, sigma=0.8, random_state=2)
        nonparam = NonParamDML(model_y="ridge", model_final="ridge", n_estimators=10, random_state=2)
        nonparam.fit(data["X"], data["w"], data["y"])
        assert abs(nonparam.average_treatment_effect()["estimate"] - 1.5) < 0.5
        kernel = KernelDML(n_components=12, random_state=2)
        kernel.fit(data["X"], data["w"], data["y"])
        assert np.isfinite(kernel.predict(data["X"])).all()
        irm = InteractiveRegressionDML(model_y="ridge", n_estimators=10, random_state=2)
        irm.fit(data["X"], data["w"], data["y"])
        assert abs(irm.effect() - 1.5) < 0.5
        forest = CausalForestDML(num_trees=20, random_state=2)
        forest.fit(data["X"], data["w"], data["y"])
        assert np.isfinite(forest.predict(data["X"])).all()


class TestPolicy:
    def test_policy_assigns_treatment(self):
        data = synthetic_data(n=360, p=4, sigma=0.6, mode=2, random_state=5)
        policy = PolicyLearner(max_depth=3, n_estimators=15, random_state=5)
        policy.fit(data["X"], data["w"], data["y"])
        assignment = policy.predict(data["X"])
        assert set(np.unique(assignment)).issubset({0, 1})
        proba = policy.predict_proba(data["X"])
        assert proba.shape[0] == 360
        assert np.isfinite(policy.training_policy_value())
        tree = DRPolicyTree(max_depth=2, n_estimators=12, random_state=5)
        tree.fit(data["X"], data["w"], data["y"])
        assert tree.predict_value(data["X"]).shape == (360,)
        forest = DRPolicyForest(max_depth=2, n_estimators=12, random_state=5)
        forest.fit(data["X"], data["w"], data["y"])
        recommended = forest.predict(data["X"])
        assert set(np.unique(recommended)).issubset({0, 1})
        value = policy_value(data["y"], data["w"], recommended, np.full(360, 0.5))
        assert np.isfinite(value)
        assert np.isfinite(forest.training_policy_value())


class TestBayesianAndQuasi:
    def test_bayesian_linear_cate_and_shap(self):
        data = make_linear_treatment(n=400, tau=2.0, sigma=0.5, random_state=3)
        model = BayesianLinearCATE(random_state=3)
        model.fit(data["X"], data["w"], data["y"])
        low, high = model.credible_interval(data["X"])
        mean = model.predict(data["X"])
        assert np.mean(low <= mean) > 0.95
        assert np.mean(mean <= high) == 1.0
        explanation = model.explain(data["X"][:15])
        np.testing.assert_allclose(
            explanation.base_values + explanation.values.sum(axis=1),
            explanation.predictions,
            atol=1e-6,
        )
        fig = explanation.plot_waterfall(0)
        plt.close(fig)
        assert abs(model.average_treatment_effect()["estimate"] - 2.0) < 0.6

    def test_did_rd_its_iv_synthetic(self):
        rng = np.random.default_rng(0)
        n = 240
        treated = np.repeat([0, 1], n // 2)
        post = np.tile([0, 1], n // 2)
        y = 1.0 + 0.2 * treated + 0.3 * post + 1.5 * treated * post + rng.normal(scale=0.15, size=n)
        did = DifferenceInDifferences(n_boot=40, random_state=0)
        did.fit(y, treated, post)
        assert abs(did.effect_ - 1.5) < 0.25
        assert did.summary()["ci_lower"] < 1.5 < did.summary()["ci_upper"]
        assert 0 <= rope_probability(did.draws_, -0.1, 0.1) <= 1

        running = rng.uniform(-1, 1, size=300)
        jump = (running >= 0).astype(float)
        outcome = 0.4 * running + 1.2 * jump + rng.normal(scale=0.1, size=300)
        rd = RegressionDiscontinuity(cutoff=0.0, n_boot=30, random_state=0)
        rd.fit(running, outcome)
        assert abs(rd.effect_ - 1.2) < 0.3

        time = np.arange(80, dtype=float)
        series = 0.05 * time + np.where(time >= 40, 2.0, 0.0) + rng.normal(scale=0.05, size=80)
        its = InterruptedTimeSeries(n_boot=30, random_state=0)
        its.fit(time, series, intervention_time=40)
        assert abs(its.effect_ - 2.0) < 0.4

        data = make_iv_data(n=400, random_state=1)
        iv = InstrumentalVariable(n_boot=30, random_state=1)
        iv.fit(data["Y"], data["W"], data["Z"], data["X"])
        assert abs(iv.effect_ - 1.5) < 0.45
        assert iv.explain().values.shape[1] == 3

        pre = rng.normal(size=(12, 3))
        weights = np.array([0.2, 0.5, 0.3])
        treated_pre = pre @ weights
        post = rng.normal(size=(6, 3))
        treated_post = post @ weights + 1.0
        sc = SyntheticControl(n_boot=8, random_state=0)
        sc.fit(pre, treated_pre, post, treated_post)
        assert abs(sc.effect_ - 1.0) < 0.35
        donor_shap = sc.explain(pre)
        assert donor_shap.values.shape == pre.shape

    def test_linear_shap_adds_up(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 3))
        coef = np.array([0.5, -1.0, 0.25])
        explanation = linear_shap(coef, X, ["a", "b", "c"], intercept=0.3)
        np.testing.assert_allclose(
            explanation.base_values + explanation.values.sum(axis=1),
            explanation.predictions,
            atol=1e-8,
        )
        np.testing.assert_allclose(explanation.predictions, 0.3 + X @ coef)


class TestShap:
    def test_explain_cate_adds_up(self):
        data = synthetic_data(n=200, p=4, sigma=0.5, mode=2, random_state=6)
        model = TLearner(learner="gbr", n_estimators=15, random_state=6)
        model.fit(data["X"], data["w"], data["y"])
        explanation = explain_cate(model, data["X"], n_permutations=6, max_samples=16, random_state=0)
        np.testing.assert_allclose(
            explanation.base_values + explanation.values.sum(axis=1),
            explanation.predictions,
            atol=1e-5,
        )
        fig = model.plot_shap(data["X"], kind="dependence", feature=0, n_permutations=4, max_samples=12)
        plt.close(fig)
        fig = explanation.plot_importance()
        plt.close(fig)
        fig = plot_cate_hist(explanation.predictions)
        plt.close(fig)
        library = explain_cate(model, data["X"], method="shap", n_permutations=4, max_samples=12, random_state=0)
        assert library.values.shape[1] == 4
        with pytest.raises(ValueError, match="method"):
            explain_cate(model, data["X"][:8], method="nope")


class TestIntegrations:
    def test_dowhy_linear_ate(self):
        if not available_backends()["dowhy"]:
            with pytest.raises(ImportError):
                dowhy_ate(np.zeros((10, 2)), np.zeros(10), np.zeros(10))
            return
        data = make_linear_treatment(n=400, tau=2.0, sigma=0.4, random_state=0)
        result = dowhy_ate(data["X"], data["w"], data["y"])
        assert abs(float(result.value) - 2.0) < 0.4

    def test_causallearn_pc_shape(self):
        if not available_backends()["causallearn"]:
            with pytest.raises(ImportError):
                causallearn_pc(np.zeros((20, 3)))
            return
        rng = np.random.default_rng(0)
        latent = rng.normal(size=180)
        data = np.column_stack([latent, latent + rng.normal(scale=0.1, size=180), rng.normal(size=180)])
        adjacency = causallearn_pc(data, alpha=0.05)
        assert adjacency.shape == (3, 3)
        assert np.trace(adjacency) == 0

    def test_causallearn_lingam_shape(self):
        if not available_backends()["causallearn"]:
            pytest.skip("causal-learn is not installed")
        rng = np.random.default_rng(0)
        data = rng.normal(size=(150, 3))
        adjacency = causallearn_direct_lingam(data)
        assert adjacency.shape == (3, 3)

    def test_causallearn_ges_shape(self):
        if not available_backends()["causallearn"]:
            with pytest.raises(ImportError, match="causallearn"):
                causallearn_ges(np.zeros((20, 3)))
            return
        rng = np.random.default_rng(1)
        data = rng.normal(size=(80, 3))
        adjacency = causallearn_ges(data)
        assert adjacency.shape == (3, 3)
        assert np.trace(adjacency) == 0

    def test_missing_upstream_wrappers_raise(self):
        X = np.zeros((12, 2))
        t = np.array([0, 1] * 6)
        y = np.zeros(12)
        if not available_backends()["econml"]:
            with pytest.raises(ImportError, match="econml"):
                econml_dml(X, t, y)
        if not available_backends()["causalml"]:
            with pytest.raises(ImportError, match="causalml"):
                causalml_slearner(X, t, y)
        if not available_backends()["doubleml"]:
            with pytest.raises(ImportError, match="doubleml"):
                doubleml_irm(X, t, y)
        if not available_backends()["causalpy"]:
            with pytest.raises(ImportError, match="causalpy"):
                causalpy_difference_in_differences(y, t, t)
        if not available_backends()["econml"]:
            with pytest.raises(ImportError, match="econml"):
                econml_causal_forest(X, t, y)
            with pytest.raises(ImportError, match="econml"):
                econml_drlearner(X, t, y)
        if not available_backends()["causallearn"]:
            with pytest.raises(ImportError, match="causallearn"):
                causallearn_ges(X)
        if not available_backends()["causalml"]:
            with pytest.raises(ImportError, match="causalml"):
                causalml_tlearner(X, t, y)
        if not available_backends()["doubleml"]:
            with pytest.raises(ImportError, match="doubleml"):
                doubleml_plr(X, t, y)
        with pytest.raises(ImportError, match="not-a-real-module"):
            require_module("not-a-real-module")

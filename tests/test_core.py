"""Tests for utilities, datasets, and metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from pydeepcausalml import datasets, metrics, set_seed
from pydeepcausalml.base import EarlyStopping, MLP, mmd_rbf
from pydeepcausalml.utils import (
    check_array,
    check_binary_treatment,
    module_to_device,
    notears_acyclicity_h,
    to_numpy,
    to_tensor,
)


class TestUtils:
    def test_set_seed_reproducible(self):
        set_seed(0)
        a = torch.randn(5)
        set_seed(0)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_to_tensor_roundtrip(self):
        x = np.arange(6, dtype=float).reshape(3, 2)
        t = to_tensor(x)
        assert t.dtype == torch.float32
        np.testing.assert_allclose(to_numpy(t), x)

    def test_to_tensor_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert to_tensor(df).shape == (2, 2)

    def test_check_array_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            check_array(np.array([[1.0, np.nan]]))

    def test_check_array_promotes_1d(self):
        assert check_array(np.arange(3.0)).shape == (3, 1)

    def test_check_binary_treatment_rejects_nonbinary(self):
        with pytest.raises(ValueError, match="binary"):
            check_binary_treatment([0, 1, 2])

    def test_module_to_device_and_acyclicity(self):
        layer = torch.nn.Linear(2, 1)
        moved = module_to_device(layer, torch.device("cpu"))
        assert next(moved.parameters()).device.type == "cpu"
        h = notears_acyclicity_h(torch.zeros(3, 3))
        assert float(h) == pytest.approx(0.0, abs=1e-6)


class TestBaseBuildingBlocks:
    def test_early_stopping(self):
        stopper = EarlyStopping(patience=2, min_delta=0.1)
        assert stopper.step(1.0) is False
        assert stopper.step(0.5) is False
        assert stopper.step(0.5) is False
        assert stopper.step(0.5) is True
        assert stopper.should_stop is True

    def test_mlp_forward_shape(self):
        net = MLP(3, 2, hidden=(4,), final_activation=torch.nn.Sigmoid())
        out = net(torch.zeros(5, 3))
        assert out.shape == (5, 2)
        assert torch.isfinite(out).all()

    def test_mmd_zero_for_identical_samples(self):
        a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        value = mmd_rbf(a, a.clone())
        assert float(value) == pytest.approx(0.0, abs=1e-6)


class TestDatasets:
    def test_confounded_data_schema(self):
        df = datasets.make_confounded_data(n=200, random_state=1)
        assert len(df) == 200
        assert {"treatment", "outcome", "true_cate", "propensity"} <= set(df.columns)
        assert set(df["treatment"].unique()) <= {0, 1}

    def test_confounded_data_reproducible(self):
        a = datasets.make_confounded_data(n=100, random_state=7)
        b = datasets.make_confounded_data(n=100, random_state=7)
        pd.testing.assert_frame_equal(a, b)

    def test_var_data_graph(self):
        x, a_true = datasets.make_var_data(n_steps=300, random_state=0)
        assert x.shape == (300, 5)
        assert a_true.sum() == 4
        assert np.isfinite(x).all()

    def test_intervention_series(self):
        df = datasets.make_intervention_series(n_units=10, horizon=8, random_state=0)
        assert len(df) == 80
        assert df.loc[df["treated"] == 0, "intervention"].sum() == 0


class TestMetrics:
    def test_pehe_zero_for_perfect(self):
        cate = np.array([1.0, 2.0, 3.0])
        assert metrics.pehe(cate, cate) == 0.0

    def test_pehe_value(self):
        assert metrics.pehe([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))

    def test_ate_error(self):
        assert metrics.ate_error(5.0, 3.0) == 2.0

    def test_graph_metrics_perfect(self):
        a = np.array([[0, 1], [0, 0]])
        result = metrics.graph_recovery_metrics(a, a)
        assert result["f1"] == 1.0 and result["shd"] == 0

    def test_graph_metrics_counts(self):
        a_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        a_pred = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        result = metrics.graph_recovery_metrics(a_true, a_pred)
        assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
        assert result["shd"] == metrics.shd(a_true, a_pred) == 2

    def test_graph_metrics_shape_mismatch(self):
        with pytest.raises(ValueError):
            metrics.graph_recovery_metrics(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_mase_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert metrics.mase(y, y) == 0.0

    def test_mase_matches_manual(self):
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([1.0, 3.0, 4.0])
        expected = 1.0 / ((3 / 2) * 3.0)
        assert metrics.mase(y_true, y_pred) == pytest.approx(expected)

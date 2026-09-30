"""Every public submodule exports the names listed in its ``__all__``."""

import importlib

import pytest

MODULES = [
    "pycausalml",
    "pycausalml.data",
    "pycausalml.interpretation",
    "pycausalml.plotting",
    "pycausalml.tree_based_models",
    "pycausalml.uplift_learners",
    "pycausalml.meta_learners",
    "pycausalml.double_machine_learning",
    "pycausalml.policy_learners",
    "pycausalml.bayesian_causal_inference",
    "pycausalml.bayesian_causalML",
    "pycausalml.integrations",
    "pydeepcausalml",
    "pydeepcausalml.base",
    "pydeepcausalml.datasets",
    "pydeepcausalml.metrics",
    "pydeepcausalml.plotting",
    "pydeepcausalml.utils",
    "pydeepcausalml.effect",
    "pydeepcausalml.generative",
    "pydeepcausalml.discovery",
    "pydeepcausalml.timeseries",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_public_names_import(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
    assert module.__all__

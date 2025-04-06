# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Test utils.hydra_utils module."""

import pytest
from omegaconf import OmegaConf

from utils.hydra_utils import instantiate_recursively


class Dummy:
    """Dummy class"""

    def __init__(self, value: int) -> None:
        self.value = value


@pytest.fixture
def basic_cfg() -> OmegaConf:
    """Basic config structure."""
    return OmegaConf.create({
        "a": {
            "_target_": "tests.utils.test_hydra_utils.Dummy",
            "value": 42,
        },
        "b": "not_instantiable",
    })


@pytest.fixture
def nested_cfg() -> OmegaConf:
    """Config with nested structure."""
    return OmegaConf.create({
        "a": {
            "x": {
                "_target_": "tests.utils.test_hydra_utils.Dummy",
                "value": 7,
            },
        },
        "list": [
            {"_target_": "tests.utils.test_hydra_utils.Dummy", "value": 3},
            "static",
        ],
    })


def test_basic_instantiation(basic_cfg: OmegaConf) -> None:
    """Test with no nested structure."""
    out = instantiate_recursively(basic_cfg)
    assert type(out["a"]).__name__ == "Dummy"
    assert out["a"].value == 42
    assert out["b"] == "not_instantiable"


def test_nested_instantiation(nested_cfg: OmegaConf) -> None:
    """Test nested structure."""
    out = instantiate_recursively(nested_cfg)
    assert type(out["a"]["x"]).__name__ == "Dummy"
    assert out["a"]["x"].value == 7
    assert type(out["list"][0]).__name__ == "Dummy"
    assert out["list"][0].value == 3
    assert out["list"][1] == "static"

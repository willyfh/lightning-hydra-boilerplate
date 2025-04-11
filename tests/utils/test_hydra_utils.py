# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Test utils.hydra_utils module."""

from typing import cast

import pytest
from omegaconf import OmegaConf

from utils.hydra_utils import InstantiatedConfig, instantiate_recursively


class Dummy:
    """Dummy class for testing instantiation."""

    def __init__(self, value: int) -> None:
        self.value = value


@pytest.fixture
def basic_cfg() -> OmegaConf:
    """Basic config structure."""
    return OmegaConf.create({
        "a": {
            "_target_": f"{__name__}.Dummy",  # Ensures importable path
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
                "_target_": f"{__name__}.Dummy",
                "value": 7,
            },
        },
        "list": [
            {"_target_": f"{__name__}.Dummy", "value": 3},
            "static",
        ],
    })


def test_basic_instantiation(basic_cfg: OmegaConf) -> None:
    """Test with no nested structure."""
    out = cast(dict[str, InstantiatedConfig], instantiate_recursively(basic_cfg))
    a = cast(Dummy, out["a"])

    assert isinstance(a, Dummy)
    assert a.value == 42
    assert out["b"] == "not_instantiable"


def test_nested_instantiation(nested_cfg: OmegaConf) -> None:
    """Test nested structure."""
    out = cast(dict[str, InstantiatedConfig], instantiate_recursively(nested_cfg))
    a_x = cast(Dummy, cast(dict[str, InstantiatedConfig], out["a"])["x"])
    list0 = cast(Dummy, cast(list[InstantiatedConfig], out["list"])[0])

    assert isinstance(a_x, Dummy)
    assert a_x.value == 7
    assert isinstance(list0, Dummy)
    assert list0.value == 3
    assert cast(list[InstantiatedConfig], out["list"])[1] == "static"

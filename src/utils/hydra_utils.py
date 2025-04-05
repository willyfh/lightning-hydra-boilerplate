# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Hydra utilities for recursive instantiation of configuration objects."""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

ConfigType: TypeAlias = DictConfig | ListConfig | Mapping | Sequence | str | int | float | bool | None


def instantiate_recursively(cfg: DictConfig) -> dict:
    """Recursively instantiate all objects in the config that have a `_target_` key.

    Args:
        cfg (DictConfig): A Hydra config dictionary.

    Returns:
        dict: A dictionary where all instantiable elements have been instantiated.
    """

    def helper(value: ConfigType) -> ConfigType:
        if isinstance(value, DictConfig):
            if "_target_" in value:
                return instantiate(value)
            return {key: helper(val) for key, val in value.items()}
        if isinstance(value, ListConfig):
            return [helper(item) for item in value]
        return value

    return {key: helper(value) for key, value in cfg.items()}

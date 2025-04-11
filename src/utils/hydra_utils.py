# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Hydra utilities."""

from typing import TypeAlias

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

InstantiatedConfig: TypeAlias = (
    dict[str, "InstantiatedConfig"] | list["InstantiatedConfig"] | str | int | float | bool | None
)


def instantiate_recursively(cfg: DictConfig | ListConfig) -> InstantiatedConfig:
    """Recursively instantiate all objects in the config that have a `_target_` key.

    Args:
        cfg (DictConfig | ListConfig): A Hydra configuration node.

    Returns:
        InstantiatedConfig: The instantiated config structure with all `_target_` entries resolved.
    """
    if isinstance(cfg, DictConfig):
        if "_target_" in cfg:
            return instantiate(cfg)
        return {k: instantiate_recursively(v) for k, v in cfg.items()}
    if isinstance(cfg, ListConfig):
        return [instantiate_recursively(i) for i in cfg]
    return cfg

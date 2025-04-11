# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Hydra utilities."""

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig


def instantiate_recursively(cfg: DictConfig | ListConfig) -> dict | list | object:
    """Recursively instantiate all objects in the config that have a `_target_` key.

    Args:
        cfg (DictConfig | ListConfig): A Hydra configuration node.

    Returns:
        dict | list | object: The instantiated config structure with all `_target_` entries resolved.
    """
    if isinstance(cfg, DictConfig):
        if "_target_" in cfg:
            return instantiate(cfg)
        return {k: instantiate_recursively(v) for k, v in cfg.items()}
    if isinstance(cfg, ListConfig):
        return [instantiate_recursively(i) for i in cfg]
    return cfg

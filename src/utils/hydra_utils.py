
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

def instantiate_recursively(cfg: DictConfig):
    """A utility function to recursively instantiate all objects with '_target_'."""
    def helper(value):
        if isinstance(value, DictConfig):
            if "_target_" in value:
                return instantiate(value)
            return {key: helper(val) for key, val in value.items()}
        elif isinstance(value, ListConfig):
            return [helper(item) for item in value]
        else:
            return value

    return {key: helper(value) for key, value in cfg.items()}
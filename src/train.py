# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Training entry point using Hydra and PyTorch Lightning."""

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.hydra_utils import instantiate_recursively


def train(cfg: DictConfig) -> None:
    """Instantiate and train the model using PyTorch Lightning.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    trainer = pl.Trainer(**trainer_params)

    # Start training
    trainer.fit(model, datamodule=data_module)

    # Optionally run test
    if not cfg.skip_test:
        trainer.test(model, datamodule=data_module)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point, triggered by Hydra with config injection."""
    train(cfg)


if __name__ == "__main__":
    main()

# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Training entry point using Hydra and PyTorch Lightning."""

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from utils.hydra_utils import instantiate_recursively


def train(cfg: DictConfig) -> None:
    """Instantiate and train the model using PyTorch Lightning.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    logger.info("Using configuration:\n" + OmegaConf.to_yaml(cfg))

    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    # Setup the datasets
    data_module.setup()
    train_size = len(data_module.train_dataloader().dataset)
    val_size = len(data_module.val_dataloader().dataset)
    test_size = len(data_module.test_dataloader().dataset)

    # Log dataset sizes
    logger.info(f"Training dataset size: {train_size}")
    logger.info(f"Validation dataset size: {val_size}")
    logger.info(f"Test dataset size: {test_size}")

    trainer = pl.Trainer(**trainer_params)

    # Start training
    trainer.fit(model, datamodule=data_module)

    # Optionally run test
    if not cfg.skip_test:
        trainer.test(model, datamodule=data_module)
    else:
        logger.info("Test stage skipped.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point, triggered by Hydra with config injection."""
    train(cfg)


if __name__ == "__main__":
    main()

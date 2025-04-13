# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Training entry point using Hydra and PyTorch Lightning."""

import logging

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.hydra_utils import instantiate_recursively
from utils.logger_utils import setup_logger


def train(cfg: DictConfig) -> None:
    """Instantiate and train the model using PyTorch Lightning.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    logging.info("Using configuration:\n" + OmegaConf.to_yaml(cfg))
    model = instantiate_recursively(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    if isinstance(trainer_params, DictConfig):
        callbacks_cfg = trainer_params.get("callbacks")
        if isinstance(callbacks_cfg, DictConfig):
            trainer_params.callbacks = list(callbacks_cfg.values())

    # Setup the datasets
    datamodule.setup()
    train_size = len(datamodule.train_dataloader().dataset)
    val_size = len(datamodule.val_dataloader().dataset)
    test_size = len(datamodule.test_dataloader().dataset)

    # Log dataset sizes
    logging.info(f"Training dataset size: {train_size}")
    logging.info(f"Validation dataset size: {val_size}")
    logging.info(f"Test dataset size: {test_size}")

    trainer = pl.Trainer(**trainer_params)

    # Start training
    trainer.fit(model, datamodule=datamodule)

    # Optionally run test
    if not cfg.skip_test:
        results = trainer.test(model, datamodule=datamodule)
        logging.info(f"Test results:\n{results}")
    else:
        logging.info("Test stage skipped.")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main entry point, triggered by Hydra with config injection."""
    pl.seed_everything(cfg.seed)
    setup_logger()
    train(cfg)


if __name__ == "__main__":
    main()

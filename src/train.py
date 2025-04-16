# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Training entry point using Hydra and PyTorch Lightning."""

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.logger_utils import log_message, setup_logger


def train(cfg: DictConfig) -> dict:
    """Instantiates and trains a Lightning model using the provided config.

    Also evaluates the model and returns combined training and test metrics.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        dict: A dictionary containing training and test metrics.
    """
    log_message("info", "Using configuration:\n" + OmegaConf.to_yaml(cfg))
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer_params = instantiate(cfg.trainer)
    trainer_params["callbacks"] = list(trainer_params["callbacks"].values())

    # Setup the datasets
    datamodule.setup()
    train_size = len(datamodule.train_dataloader().dataset)
    val_size = len(datamodule.val_dataloader().dataset)
    test_size = len(datamodule.test_dataloader().dataset)

    # Log dataset sizes
    log_message("info", f"Training dataset size: {train_size}")
    log_message("info", f"Validation dataset size: {val_size}")
    log_message("info", f"Test dataset size: {test_size}")

    trainer = pl.Trainer(**trainer_params)

    # Start training
    trainer.fit(model, datamodule=datamodule)

    # Gather metrics after training
    train_metrics = trainer.callback_metrics
    log_message("info", f"Metrics after training:\n{train_metrics}")

    # Optionally run test
    if not cfg.skip_test:
        trainer.test(datamodule=datamodule, ckpt_path="best")
        # Metrics after testing
        test_metrics = trainer.callback_metrics
        log_message("info", f"Test results:\n{test_metrics}")
    else:
        test_metrics = {}
        log_message("info", "Test stage skipped.")

    return {**train_metrics, **test_metrics}


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> float | None:
    """Main function triggered by Hydra.

    Seeds everything, runs training, and returns the optimized metric
    if specified (useful for Optuna sweeps).

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        float | None: The value of the optimized metric, if specified.
    """
    pl.seed_everything(cfg.seed)
    setup_logger()
    results = train(cfg)

    optimized_value = results.get(cfg.get("optimized_metric"))
    log_message("info", f"Returning optimized metric ({cfg.get('optimized_metric')}): {optimized_value}")
    return optimized_value


if __name__ == "__main__":
    main()

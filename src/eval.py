# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Evaluation entry point using Hydra and PyTorch Lightning."""

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.logger_utils import log_message, setup_logger


def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model from a checkpoint on a specified data split.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    log_message("info", f"Running evaluation with configuration:\n{OmegaConf.to_yaml(cfg)}")

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer_params = instantiate(cfg.trainer)
    trainer_params["callbacks"] = list(trainer_params["callbacks"].values())

    trainer = pl.Trainer(**trainer_params)
    datamodule.setup()

    data_split = cfg.data_split
    ckpt_path = cfg.ckpt_path

    if data_split == "test":
        dataset = datamodule.test_dataloader().dataset
        split_name = "Test"
        results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif data_split == "val":
        dataset = datamodule.val_dataloader().dataset
        split_name = "Validation"
        results = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        error_msg = f"Unknown data_split: {data_split}"
        raise ValueError(error_msg)

    log_message("info", f"{split_name} dataset size: {len(dataset)}")
    log_message("info", f"Evaluation results on {split_name} set:\n{results}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point triggered by Hydra config."""
    setup_logger()
    evaluate(cfg)


if __name__ == "__main__":
    main()

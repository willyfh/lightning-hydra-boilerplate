# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Evaluation entry point using Hydra and PyTorch Lightning."""

import logging

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.hydra_utils import instantiate_recursively
from utils.logger_utils import setup_logger


def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model from a checkpoint on a specified data split.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    logging.info("Running evaluation with configuration:\n%s", OmegaConf.to_yaml(cfg))

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    trainer = pl.Trainer(**trainer_params)
    datamodule.setup()

    eval_split = cfg.eval_split
    ckpt_path = cfg.ckpt_path

    if eval_split == "test":
        dataloader = datamodule.test_dataloader()
        split_name = "Test"
    elif eval_split == "val":
        dataloader = datamodule.val_dataloader()
        split_name = "Validation"
    elif eval_split == "train":
        dataloader = datamodule.train_dataloader()
        split_name = "Training"
    else:
        error_msg = f"Unknown eval_split: {eval_split}"
        raise ValueError(error_msg)

    logging.info(f"{split_name} dataset size: {len(dataloader.dataset)}")
    results = trainer.validate(model=model, dataloaders=dataloader, ckpt_path=ckpt_path)

    logging.info(f"Evaluation results on {split_name} set:\n{results}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point triggered by Hydra config."""
    setup_logger()
    evaluate(cfg)


if __name__ == "__main__":
    main()

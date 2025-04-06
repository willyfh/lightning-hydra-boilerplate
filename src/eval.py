# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Evaluation entry point using Hydra and PyTorch Lightning."""

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from utils.hydra_utils import instantiate_recursively


def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model from checkpoint using PyTorch Lightning.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    logger.info("Running evaluation with configuration:\n" + OmegaConf.to_yaml(cfg))

    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    trainer = pl.Trainer(**trainer_params)

    data_module.setup()

    if cfg.eval_split == "test":
        test_loader = data_module.test_dataloader()
        trainer.validate(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
    elif cfg.eval_split == "val":
        val_loader = data_module.val_dataloader()
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
    elif cfg.eval_split == "train":
        train_loader = data_module.train_dataloader()
        trainer.validate(model=model, dataloaders=train_loader, ckpt_path=cfg.ckpt_path)
    else:
        error_msg = f"Unknown eval_split : {cfg.eval_split}"
        raise ValueError(error_msg)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation entrypoint triggered by Hydra config."""
    evaluate(cfg)


if __name__ == "__main__":
    main()

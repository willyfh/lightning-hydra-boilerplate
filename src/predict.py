# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Inference script to generate and save model predictions."""

from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.logger_utils import log_message, setup_logger
from utils.pred_utils import save_predictions


def predict(cfg: DictConfig) -> None:
    """Run inference on a specified data split using a trained model from a checkpoint.

    Args:
        cfg (DictConfig): Hydra configuration object containing model, data, and trainer settings.
    """
    log_message("info", f"Running prediction with configuration:\n{OmegaConf.to_yaml(cfg)}")

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer_params = instantiate(cfg.trainer)
    trainer_params["callbacks"] = list(trainer_params["callbacks"].values())

    trainer = pl.Trainer(**trainer_params)

    datamodule.setup()

    # Select appropriate dataloader
    if cfg.data_split == "train":
        dataloader = datamodule.train_dataloader()
    elif cfg.data_split == "val":
        dataloader = datamodule.val_dataloader()
    elif cfg.data_split == "test":
        dataloader = datamodule.test_dataloader()
    elif cfg.data_split == "predict":
        dataloader = datamodule.predict_dataloader()
    else:
        error_msg = f"Unsupported data_split: {cfg.data_split}"
        raise ValueError(error_msg)

    predictions_batches = trainer.predict(
        model=model,
        dataloaders=dataloader,
        ckpt_path=cfg.get("ckpt_path", None),
    )
    # Flatten and sort predictions by idx
    flat_preds = [
        {"idx": batch["idx"][i].item(), "pred": batch["pred"][i].item()}
        for batch in predictions_batches
        for i in range(len(batch["idx"]))
    ]

    flat_preds.sort(key=lambda x: x["idx"])

    # Save the predictions
    run_dir = HydraConfig.get().run.dir
    output_path = Path(run_dir) / f"predictions.{cfg.save_format}"
    save_predictions(flat_preds, output_path, cfg.save_format)
    log_message("info", f"Predictions of {cfg.data_split} split are saved to {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    """Main prediction entry point triggered by Hydra config."""
    setup_logger()
    predict(cfg)


if __name__ == "__main__":
    main()

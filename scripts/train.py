import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L
from hydra.utils import instantiate


def train(cfg: DictConfig):

    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    logger = instantiate(cfg.logger)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.gpus,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
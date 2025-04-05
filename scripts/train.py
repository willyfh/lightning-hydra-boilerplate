import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L
from hydra.utils import instantiate


def train(cfg: DictConfig):

    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    logger = instantiate(cfg.logger)

    callbacks = []
    if "callback" in cfg:
        for cb_cfg in cfg.callback.values():
            callbacks.append(instantiate(cb_cfg))

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.gpus,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)

    if not cfg.trainer.skip_test:
        trainer.test(model, datamodule=data_module)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
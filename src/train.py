import hydra
from omegaconf import DictConfig
import lightning.pytorch as L
from hydra.utils import instantiate
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils.hydra_utils import instantiate_recursively

def train(cfg: DictConfig):
    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    trainer_params = instantiate_recursively(cfg.trainer)

    trainer = L.Trainer(
        **trainer_params,
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

    # Check if we should skip the test phase after training
    if not cfg.skip_test:
        print("Running evaluation (test) after training...")
        trainer.test(model, datamodule=data_module)
    else:
        print("Skipping test phase after training.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()

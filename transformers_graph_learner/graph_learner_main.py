import hydra
from omegaconf import DictConfig, OmegaConf

from train_model import train_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    train_model(cfg)


if __name__ == "__main__":
    main()

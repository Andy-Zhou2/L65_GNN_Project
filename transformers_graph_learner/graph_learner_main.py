import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import glob
import os
import shutil

from train_model import train_model


# os.environ["WANDB_MODE"] = "disabled"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if not HydraConfig.initialized():
        HydraConfig().set_config(cfg)

        # Retrieve Hydra's output directory
    output_dir = HydraConfig.get().runtime.output_dir

    # Save all .py files from the original working directory to Hydra's output directory
    original_cwd = hydra.utils.get_original_cwd()
    src_py_files = glob.glob(os.path.join(original_cwd, "*.py"))

    for file_path in src_py_files:
        shutil.copy(file_path, output_dir)
    shutil.copy("./configs/config.yaml", output_dir)

    train_model(cfg)


if __name__ == "__main__":
    main()

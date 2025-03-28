import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import glob
import os
import shutil

from .train_model import train_model


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
    shutil.copy(f"transformers_graph_learner/configs/{HydraConfig.get().job.config_name}.yaml", output_dir)

    for lr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
        cfg.training.lr = lr
        print(f"<<< Training with LR {lr} (seed {cfg.seed}) >>>")
        train_model(cfg)
        print(f"<<< Done training with LR {lr} graphs (seed {cfg.seed}) >>>")


if __name__ == "__main__":
    main()

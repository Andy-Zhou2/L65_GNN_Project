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

    for seed in [1, 2, 3, 4]:
        for ecc in range(2, 7):
            for num_layers in range(1, 6):
                for num_heads in [1]: #, 2, 4, 8, 16]:
                    for supervision in [False]:  # [False, True]:
                        if supervision and ecc > num_layers:  # Supervision doesn't make sense in this case
                            continue
                        cfg.seed = seed
                        cfg.model.num_layers = num_layers
                        cfg.model.nhead = num_heads
                        cfg.dataset.eccentricity = ecc
                        cfg.model.intermediate_supervision = supervision
                        print(f"<<< Training with {num_layers} layers, {num_heads} heads, ecc {ecc}, supervision {supervision} (seed {cfg.seed}) >>>")
                        train_model(cfg)
                        print(f"<<< Done training with {num_layers} layers, {num_heads} heads, ecc {ecc}, supervision {supervision} (seed {cfg.seed}) >>>")


if __name__ == "__main__":
    main()

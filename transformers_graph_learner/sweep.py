import wandb
import os
from hydra import initialize, compose

from train_model import train_model


def sweep_train():
    with wandb.init():
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name="config")

        # Extract hyperparameters from wandb config with defaults.
        print("wandb config:", wandb.config)
        seed = wandb.config["seed"]
        nhead = wandb.config["model.nhead"]
        num_layers = wandb.config["model.num_layers"]
        n_nodes_range = wandb.config["dataset.n_nodes_range"]

        custom_name = f"{num_layers} x {nhead} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) seed {seed}"
        wandb.run.name = custom_name

        # override the config with the new hyperparameters
        cfg.seed = seed
        cfg.model.nhead = nhead
        cfg.model.num_layers = num_layers
        cfg.dataset.n_nodes_range = n_nodes_range

        train_model(cfg)


if __name__ == "__main__":
    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {
    #         'name': 'test_loss',  # the metric to optimize
    #         'goal': 'minimize'
    #     },
    #     'parameters': {
    #         'seed': {
    #             'values': [42, 43, 44, 45, 46]
    #         },
    #         'model.nhead': {
    #             'values': [1]
    #         },
    #         'model.num_layers': {
    #             'values': [1, 2, 3, 4, 5]
    #         },
    #         'dataset.n_nodes_range': {
    #             'values': [(i, i) for i in [2, 3, 4, 5, 8, 16, 32]]
    #         }
    #     }
    # }
    #
    # # Create the sweep and get the sweep id.
    # sweep_id = wandb.sweep(sweep_config, project="transformer-graph-learner")
    # print(f"Sweep ID: {sweep_id}")

    # Start the sweep agent; count sets how many runs to execute.
    wandb.agent(
        "vf4vneqk",
        function=sweep_train,
        project="transformer-graph-learner",
        entity="wz337",
        count=5000,
    )

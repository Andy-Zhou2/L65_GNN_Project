import wandb
import os
from hydra import initialize, compose

from .train_model import train_model


def sweep_train():
    with wandb.init(group="lr_search"):
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name="config")

        # Extract hyperparameters from wandb config with defaults.
        print("wandb config:", wandb.config)
        lr = wandb.config.get("lr")

        cfg.training.lr = lr

        num_layers = cfg.model.num_layers
        nhead = cfg.model.nhead
        n_nodes_range = cfg.dataset.n_nodes_range
        seed = cfg.seed
        lr = cfg.training.lr
        custom_name = f"{num_layers} x {nhead} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) seed {seed} lr {lr}"
        wandb.run.name = custom_name

        train_model(cfg)


if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'test_loss',  # the metric to optimize
            'goal': 'minimize'
        },
        'parameters': {
            "lr": {
                'values': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            }
            # 'seed': {
            #     'values': [42, 43, 44, 45, 46]
            # },
            # 'model.nhead': {
            #     'values': [1]
            # },
            # 'model.num_layers': {
            #     'values': [1, 2, 3, 4, 5]
            # },
            # 'dataset.n_nodes_range': {
            #     'values': [(i, i) for i in [2, 3, 4, 5, 8, 16, 32]]
            # }
        }
    }

    # Create the sweep and get the sweep id.
    sweep_id = wandb.sweep(sweep_config, project="transformer-graph-learner", entity="L65_Project")
    print(f"Sweep ID: {sweep_id}")

    # Start the sweep agent; count sets how many runs to execute.
    wandb.agent(
        sweep_id,
        function=sweep_train,
        project="transformer-graph-learner",
        entity="L65_Project",
        count=5000,
    )

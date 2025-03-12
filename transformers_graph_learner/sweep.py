import wandb
import os
from hydra import initialize, compose

from train_model import train_model


def sweep_train():
    with wandb.init():
        with initialize(config_path="configs", version_base=None):
            cfg = compose(config_name="config")

        # Extract hyperparameters from wandb config with defaults.
        nhead = wandb.config.get("model.nhead", 8)
        num_layers = wandb.config.get("model.num_layers", 4)
        lr = wandb.config.get("training.lr", 1e-5)

        custom_name = f"{num_layers} x {nhead} at {lr}"
        wandb.run.name = custom_name

        # override the config with the new hyperparameters
        cfg.model.nhead = nhead
        cfg.model.num_layers = num_layers
        cfg.training.lr = lr

        train_model(cfg)



if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test_loss',  # the metric to optimize
            'goal': 'minimize'
        },
        'parameters': {
            'model.nhead': {
                'values': [1, 2, 4, 8, 16, 32]
            },
            'model.num_layers': {
                'values': [2, 3, 4, 5, 6, 8, 10]
            },
            'training.lr': {
                'values': [1e-5] #[1e-6, 1e-5, 1e-4, 1e-3]
            }
        }
    }

    # Create the sweep and get the sweep id.
    sweep_id = wandb.sweep(sweep_config, project="transformer-graph-learner")
    print(f"Sweep ID: {sweep_id}")

    # Start the sweep agent; count sets how many runs to execute.
    wandb.agent(sweep_id, function=sweep_train, count=500)

import wandb
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
import hydra

from .train_model import train_model


def flatten_omegaconf(conf):
    def _flatten(c, prefix=""):
        items = {}
        for k, v in c.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, full_key))
            else:
                items[full_key] = v
        return items

    return _flatten(OmegaConf.to_container(conf, resolve=True))


def unflatten_dot_dict(dot_dict):
    result = {}
    for key, value in dot_dict.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result

@hydra.main(config_path="configs", config_name="config")
def sweep_train(base_cfg):
    with wandb.init(group="layer_search_ecc_2_node_6"):
        # Create sweep config from wandb config (flat dot-dict)
        sweep_cfg = OmegaConf.create(unflatten_dot_dict(wandb.config))
        # Merge sweep overrides with base config
        cfg = OmegaConf.merge(base_cfg, sweep_cfg)

        num_layers = cfg.model.num_layers
        nhead = cfg.model.nhead
        n_nodes_range = cfg.dataset.n_nodes_range
        seed = cfg.seed
        lr = cfg.training.lr
        custom_name = f"{num_layers} x {nhead} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) seed {seed} lr {lr} C"
        wandb.run.name = custom_name

        train_model(cfg)


if __name__ == "__main__":
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")

    cfg = flatten_omegaconf(cfg)
    for key in cfg:
        cfg[key] = {'value': cfg[key]}

    cfg['seed'] = {'values': [1, 2, 3, 4, 5]}
    # cfg['training.lr'] = {'values': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]}
    cfg['model.num_layers'] = {'values': [1, 2, 3, 4, 5]}

    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'test_loss',  # the metric to optimize
            'goal': 'minimize'
        },
        'parameters': cfg
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

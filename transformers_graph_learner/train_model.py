import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pytorch_warmup as warmup
import time

from .graph_gen import SSSPDataset, collate_fn
from .token_graph_transformer import TokenGT
from .utils import to_device
from .evaluate_model import evaluate, evaluate_on_graph
from .early_stopper import EarlyStopping


def train_model(cfg: DictConfig):
    # Print the loaded configuration.
    print(OmegaConf.to_yaml(cfg))

    num_layers = cfg.model.num_layers
    nhead = cfg.model.nhead
    n_nodes_range = cfg.dataset.n_nodes_range
    seed = cfg.seed
    custom_name = f"{num_layers} x {nhead} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) seed {seed} lr {cfg.training.lr}"

    # Initialize Weights & Biases.
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=custom_name,
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Set random seeds for reproducibility.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Token generation parameters.
    d_p = cfg.dataset.d_p
    d_e = cfg.dataset.d_e
    node_id_encode = cfg.dataset.node_id_encode
    in_feat_dim = cfg.dataset.in_feat_dim
    token_in_dim = in_feat_dim + 2 * d_p + d_e

    # Create the dataset.
    dataset = SSSPDataset(
        num_graphs=cfg.dataset.num_graphs,
        d_p=d_p,
        n_nodes_range=cfg.dataset.n_nodes_range,
        node_identifier_encoding=node_id_encode,
    )
    print(f"Total graphs in dataset: {len(dataset)}")

    # Split dataset into train and test (e.g., 80/20 split).
    num_train = int(cfg.dataset.split * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Device configuration.
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize the model.
    model = TokenGT(
        token_in_dim=token_in_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        d_e=d_e,
        dropout=cfg.model.dropout,
        input_dropout=cfg.model.input_dropout,
    ).to(device)

    # Define optimizer, scheduler, and loss function.
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(patience=cfg.training.early_stopping.patience,
                                       verbose=cfg.training.early_stopping.verbose,
                                       delta=cfg.training.early_stopping.min_delta)

    def lr_lambda(step, warmup_steps, t_total):
        if warmup_steps is None:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(t_total - step) / float(max(1, t_total - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, cfg.scheduler.warmup_steps, cfg.training.num_epochs))
    criterion = nn.MSELoss()

    training_start_time = time.time()
    # Training loop.
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = to_device(data, device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate(test_loader, model, criterion, device)
        scheduler.step()

        if cfg.training.early_stopping.enabled:
            early_stopping(test_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # Save the model based on the configuration.
        if (epoch + 1) % cfg.training.save_every == 0:
            model_path = os.path.join(
                cfg.paths.models,
                "/".join(HydraConfig.get().runtime.output_dir.split("/")[-2:]),
                # f"{cfg.model.num_layers}_layers_{cfg.model.nhead}_heads_{cfg.training.lr}_lr",
            )

            os.makedirs(model_path, exist_ok=True)
            save_path = os.path.join(model_path, f"model_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        current_lr = scheduler.get_last_lr()[0]
        current_lr_log = np.log10(current_lr)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Learning Rate: 1e{current_lr_log}"
            )
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "learning_rate": current_lr,
                "learning_rate_log": current_lr_log,
                "time": time.time() - training_start_time,
            },
            step=epoch,
        )

    # evaluate_on_graph(model, test_dataset, device)

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
import pickle

from .graph_gen import SSSPDataset, collate_fn
from .token_graph_transformer import TokenGT
from .utils import to_device
from .evaluate_model import evaluate, evaluate_ood, evaluate_on_graph
from .early_stopper import EarlyStopping


def train_model(cfg: DictConfig):
    # Print the loaded configuration.
    print(OmegaConf.to_yaml(cfg))

    num_layers = cfg.model.num_layers
    nhead = cfg.model.nhead
    n_nodes_range = cfg.dataset.n_nodes_range
    seed = cfg.seed
    eccentricity = cfg.dataset.eccentricity
    intermediate_supervision = cfg.model.intermediate_supervision
    custom_name = f"{num_layers} x {nhead} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) ecc {eccentricity} seed {seed} lr {cfg.training.lr} int_sup {intermediate_supervision}"

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

    # Create the training dataset.
    dataset_name = f'{cfg.dataset.num_graphs} graphs ({cfg.dataset.n_nodes_range[0]}-{cfg.dataset.n_nodes_range[1]}) ecc {cfg.dataset.eccentricity} layer {cfg.model.num_layers}'
    if cfg.dataset.use_existing and os.path.exists(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl')):
        with open(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        assert len(dataset) >= cfg.dataset.num_graphs, f'Existing dataset has {len(dataset)} graphs, but requested {cfg.dataset.num_graphs}'
        dataset = dataset[:cfg.dataset.num_graphs]
        print(f'Using {len(dataset)} graphs from existing dataset')
    else:
        dataset = SSSPDataset(
            num_graphs=cfg.dataset.num_graphs,
            d_p=d_p,
            n_nodes_range=cfg.dataset.n_nodes_range,
            node_identifier_encoding=node_id_encode,
            max_hops=cfg.dataset.eccentricity,
            m=cfg.dataset.m,
            p=cfg.dataset.p,
            q=cfg.dataset.q,
            intermediate_supervision_layers=cfg.model.num_layers,
        )
        os.makedirs(cfg.dataset.dataset_path, exist_ok=True)
        with open(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
    print(f"Total graphs in dataset for train: {len(dataset)}")
    # Split dataset into train and test (e.g., 80/20 split).
    num_train = int(cfg.dataset.split * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    num_test = len(test_dataset)
    print(f"Train graphs: {len(train_dataset)}")
    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Create the OOD testing dataset.
    ood_test_loaders = []
    ood_configs = cfg.dataset.test_set_configs
    for i, ood_config in enumerate(ood_configs):
        print(f"Loading test dataset {i+1}/{len(ood_configs)}, with config {ood_config}")
        n_low, n_high, ecc = ood_config.split(",")
        n_low, n_high, ecc = int(n_low), int(n_high), int(ecc)
        dataset_name = f'{num_test} graphs ({n_low}-{n_high}) ecc {ecc} layer {cfg.model.num_layers}'
        if cfg.dataset.use_existing and os.path.exists(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl')):
            with open(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            assert len(dataset) >= num_test, f'Existing dataset has {len(dataset)} graphs, but requested {num_test}'
            dataset = dataset[:num_test]
            print(f'Using {len(dataset)} graphs from existing dataset')
        else:
            dataset = SSSPDataset(
                num_graphs=num_test,
                d_p=d_p,
                n_nodes_range=(n_low, n_high),
                node_identifier_encoding=node_id_encode,
                max_hops=ecc,
                m=cfg.dataset.m,
                p=cfg.dataset.p,
                q=cfg.dataset.q,
                intermediate_supervision_layers=cfg.model.num_layers,
            )
            os.makedirs(cfg.dataset.dataset_path, exist_ok=True)
            with open(os.path.join(cfg.dataset.dataset_path, f'{dataset_name}.pkl'), 'wb') as f:
                pickle.dump(dataset, f)
        print(f"Total graphs in test dataset with ({n_low}, {n_high} nodes, {ecc} ecc): {len(dataset)}")
        print(f"Test graphs: {len(dataset)}")
        # Create Dataloader
        loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        ood_test_loaders.append(loader)

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

    best_test_loss = float("inf")
    best_ood_losses = [float("inf") for _ in ood_configs]
    training_start_time = time.time()
    # Training loop.
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = to_device(data, device)
            optimizer.zero_grad()
            pred = model(data, cfg.model.intermediate_supervision)
            if cfg.model.intermediate_supervision:
                loss = criterion(pred, data["intermediate_ys"])
            else:
                loss = criterion(pred, data["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate(test_loader, model, criterion, device)
        ood_losses = evaluate_ood(ood_test_loaders, model, criterion, device)
        scheduler.step()

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
        # if (epoch + 1) % 10 == 0:
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
        if test_loss < best_test_loss:
            wandb.summary["best_test_loss"] = test_loss
            best_test_loss = test_loss

        for i, ood_loss in enumerate(ood_losses):
            ood_config = ood_configs[i]
            best_ood_loss = best_ood_losses[i]
            # n_low, n_high, ecc = ood_config.split(",")
            # n_low, n_high, ecc = int(n_low), int(n_high), int(ecc)
            wandb.log(
                {
                    f"losses/{ood_config}_loss": ood_loss,
                },
                step=epoch,
            )
            if ood_loss < best_ood_loss:
                wandb.summary[f"best_{ood_config}_loss"] = ood_loss
                best_ood_losses[i] = ood_loss
        

        # Early stopping
        if cfg.training.early_stopping.enabled:
            early_stopping(test_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    wandb.finish()

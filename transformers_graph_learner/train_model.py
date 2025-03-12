import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf

from graph_gen import SSSPDataset
from token_graph_transformer import TokenGT
from utils import to_device
from evaluate_model import evaluate, evaluate_on_graph


def train_model(cfg: DictConfig):
    # Print the loaded configuration.
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases.
    wandb.init(project=cfg.wandb.project)

    # Set random seeds for reproducibility.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Token generation parameters.
    d_p = cfg.dataset.d_p
    d_e = cfg.dataset.d_e
    in_feat_dim = cfg.dataset.in_feat_dim
    token_in_dim = in_feat_dim + 2 * d_p + d_e

    # Create the dataset.
    dataset = SSSPDataset(num_graphs=cfg.dataset.num_graphs, d_p=d_p, d_e=d_e)
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
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
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
    ).to(device)

    # Define optimizer, scheduler, and loss function.
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler.factor, patience=cfg.scheduler.patience
    )
    criterion = nn.MSELoss()

    # Training loop.
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            data = {k: v[0] for k, v in batch.items()}
            data = to_device(data, device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate(test_loader, model, criterion, device)
        scheduler.step(test_loss)

        # Save the model based on the configuration.
        if (epoch + 1) % cfg.training.save_every == 0:
            model_path =os.path.join(cfg.paths.models,
                                     f"{cfg.model.num_layers}_layers_{cfg.model.nhead}_heads_{cfg.training.lr}_lr")

            os.makedirs(model_path, exist_ok=True)
            save_path = os.path.join(model_path, f"model_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "test_loss": test_loss}, step=epoch)

    evaluate_on_graph(model, test_dataset, device)
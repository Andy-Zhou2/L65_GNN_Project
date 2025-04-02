import os
import time
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, ModuleList, Sequential
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max
import networkx as nx
import random
import matplotlib.pyplot as plt
import wandb


from .graph_gen import SSSPDataset
from .model import ShortestPathModel
from transformers_graph_learner.early_stopper import EarlyStopping

# os.environ["WANDB_MODE"] = "disabled"


def train(train_loader, model, optimizer, total_nodes_train, device):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # Predicted distances, shape (total_num_nodes_in_batch,)
        # Compute Mean Squared Error loss over all nodes in the batch.
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # * data.x.size(0)
    return total_loss / len(train_loader)  # / total_nodes_train


@torch.no_grad()
def test(loader, model, total_nodes_test, device):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y)
        total_loss += loss.item()  # * data.x.size(0)
    return total_loss / len(loader)  # / total_nodes_test


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if not HydraConfig.initialized():
        HydraConfig().set_config(cfg)

    # Print the loaded configuration.
    print(OmegaConf.to_yaml(cfg))

    num_layers = cfg.model.num_layers
    n_nodes_range = cfg.dataset.n_nodes_range
    seed = cfg.seed
    custom_name = (
        f"GCN {num_layers} nodes ({n_nodes_range[0]}-{n_nodes_range[1]}) {cfg.dataset.num_graphs} graphs"
        f" seed {seed} lr {cfg.training.lr}"
    )

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

    # Check compatibility: The dataset provides in_dim=1 and edge_dim=1,
    # so we initialize the model accordingly.
    dataset = SSSPDataset(
        root="sssp_dataset",
        num_graphs=cfg.dataset.num_graphs,
        n_nodes_range=cfg.dataset.n_nodes_range,
        m=cfg.dataset.m,
        p=cfg.dataset.p,
        q=cfg.dataset.q,
        max_hops=cfg.dataset.get("eccentricity", None),
    )
    dataset = dataset[
        : cfg.dataset.num_graphs
    ]  # The dataset may generate more graphs than requested!
    print(f"Dataset size: {len(dataset)}")
    # print(dataset[0])

    # Split dataset into train and test (e.g., 80/20 split).
    num_train = int(cfg.dataset.split * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    # Create loaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Calculate total number of nodes in training and testing sets for averaging.
    total_nodes_train = sum(data.x.size(0) for data in train_dataset)
    total_nodes_test = sum(data.x.size(0) for data in test_dataset)

    # Set up device, model, and optimizer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortestPathModel(
        num_layers=cfg.model.num_layers,
        emb_dim=cfg.model.d_model,
        in_dim=1,
        edge_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            verbose=cfg.training.early_stopping.verbose,
            delta=cfg.training.early_stopping.min_delta,
        )

    def lr_lambda(step, warmup_steps, t_total):
        if warmup_steps is None:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(t_total - step) / float(max(1, t_total - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(
            step, cfg.scheduler.warmup_steps, cfg.training.num_epochs
        ),
    )

    best_test_loss = float("inf")
    training_start_time = time.time()
    # Training loop.
    num_epochs = cfg.training.num_epochs
    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train(
            train_loader, model, optimizer, total_nodes_train, device
        )
        test_loss = test(test_loader, model, total_nodes_test, device)
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
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, LR {current_lr}"
            )
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "learning_rate": current_lr,
                # "learning_rate_log": current_lr_log,
                "time": time.time() - training_start_time,
            },
            step=epoch,
        )
        if test_loss < best_test_loss:
            wandb.summary["best_test_loss"] = test_loss
            best_test_loss = test_loss
        # Early stopping
        if cfg.training.early_stopping.enabled:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # # Evaluate a new graph
    # # --- Select a single sample graph from the test dataset ---
    # sample_idx = random.randint(0, len(test_dataset) - 1)
    # sample_data = test_dataset[sample_idx].to(device)
    #
    # # --- Get predictions ---
    # model.eval()
    # with torch.no_grad():
    #     predicted_distances = model(sample_data).cpu()
    # true_distances = sample_data.y.cpu()
    #
    # # Plot the graph
    # # --- Convert the PyG Data object to a NetworkX graph ---
    # edge_index_np = sample_data.edge_index.cpu().numpy()  # shape (2, num_edges)
    # edge_attr_np = sample_data.edge_attr.cpu().numpy()  # shape (num_edges, 1)
    # G_nx = nx.Graph()
    # num_nodes = sample_data.num_nodes
    # G_nx.add_nodes_from(range(num_nodes))
    #
    # # --- Add edges (each undirected edge appears only once) and record edge weights ---
    # edge_labels = {}
    # added_edges = set()
    # for i, (u, v) in enumerate(zip(edge_index_np[0], edge_index_np[1])):
    #     key = tuple(sorted((u, v)))
    #     if key not in added_edges:
    #         G_nx.add_edge(u, v)
    #         # Since each edge weight is stored as a one-element list, extract its value.
    #         edge_labels[key] = f"{edge_attr_np[i][0]:.2f}"
    #         added_edges.add(key)
    #
    # # --- Create node labels showing both true (T) and predicted (P) distances ---
    # node_labels = {
    #     i: f"T: {true_distances[i]:.2f}\nP: {predicted_distances[i]:.2f}"
    #     for i in range(num_nodes)
    # }
    #
    # # --- Compute positions for visualization ---
    # pos = nx.spring_layout(G_nx)
    #
    # # --- Draw the graph ---
    # plt.figure(figsize=(10, 10))
    # nx.draw_networkx_nodes(G_nx, pos, node_color="lightblue", node_size=500)
    # nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.7)
    # nx.draw_networkx_labels(G_nx, pos, labels=node_labels, font_size=10)
    # nx.draw_networkx_edge_labels(
    #     G_nx, pos, edge_labels=edge_labels, font_color="red", font_size=8
    # )
    #
    # plt.title("Graph Visualization: True vs Predicted Distances with Edge Weights")
    # plt.axis("off")
    # figure_dir = "./figures"
    # if not os.path.exists(figure_dir):
    #     os.makedirs(figure_dir)
    # plt.savefig(os.path.join(figure_dir, "gcn_example_pred.png"))
    # plt.show()

    # # Plot errors and distributions
    # # --- Compute errors ---
    # errors = predicted_distances - true_distances

    # # --- Plotting ---
    # plt.figure(figsize=(12, 5))

    # # Histogram of prediction errors
    # plt.subplot(1, 2, 1)
    # plt.hist(errors.numpy(), bins=20, edgecolor='k', color='skyblue')
    # plt.title("Histogram of Prediction Errors")
    # plt.xlabel("Error (Predicted - True)")
    # plt.ylabel("Frequency")

    # # Distribution of distances (true and predicted)
    # plt.subplot(1, 2, 2)
    # plt.hist(true_distances.numpy(), bins=20, alpha=0.6, label="True Distances", edgecolor='k')
    # plt.hist(predicted_distances.numpy(), bins=20, alpha=0.6, label="Predicted Distances", edgecolor='k')
    # plt.title("Distribution of Distances")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()

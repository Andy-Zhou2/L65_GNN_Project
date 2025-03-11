import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import random
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from graph_gen import SSSPDataset
from token_graph_transformer import TokenGT
from graph_learner_main import evaluate_graph

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print the loaded configuration.
    print(OmegaConf.to_yaml(cfg))

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

    model.load_state_dict(torch.load('models/model_140.pth', map_location=device))

    evaluate_graph(model, test_dataset, device)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import random
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

from .graph_gen import SSSPDataset, collate_fn
from .token_graph_transformer import TokenGT
from .evaluate_model import evaluate_on_graph


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
    node_id_encode = cfg.dataset.node_id_encode
    in_feat_dim = cfg.dataset.in_feat_dim
    token_in_dim = in_feat_dim + 2 * d_p + d_e

    # Create the dataset.
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
        pin_memory=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=False,
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

    model.load_state_dict(torch.load("models/2025-03-29/20-50-57/model_1400.pth", map_location=device))

    sample_data = test_dataset[0]
    evaluate_on_graph(model, sample_data, device, cfg.model.intermediate_supervision)


if __name__ == "__main__":
    main()

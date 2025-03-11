import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import random
import matplotlib.pyplot as plt

from graph_gen import SSSPDataset
from token_graph_transformer import TokenGT
from graph_learner_main import evaluate_graph

# ---------------------------
# Training with Train/Test Split and Evaluation
# ---------------------------
if __name__ == '__main__':
    # Set random seeds for reproducibility.
    torch.manual_seed(42)
    random.seed(42)

    # Parameters for token generation.
    d_p = 10      # dimension for node identifiers
    d_e = 10      # dimension for type embeddings
    in_feat_dim = 1  # node/edge feature dimension (e.g., source flag or weight)
    token_in_dim = in_feat_dim + 2 * d_p + d_e

    # Create the dataset.
    dataset = SSSPDataset(num_graphs=1000, d_p=d_p, d_e=d_e)
    print(f"Total graphs in dataset: {len(dataset)}")

    # Split dataset into train and test (80/20 split).
    num_train = int(0.8 * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    # Device configuration.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters.
    d_model = 128
    nhead = 8
    num_layers = 3
    model = TokenGT(token_in_dim=token_in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)

    model.load_state_dict(torch.load('models/model_140.pth', map_location=device))

    evaluate_graph(model, test_dataset, device)
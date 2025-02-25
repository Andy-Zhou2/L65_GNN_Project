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


from graph_gen import SSSPDataset
from model import ShortestPathModel


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # Predicted distances, shape (total_num_nodes_in_batch,)
        # Compute Mean Squared Error loss over all nodes in the batch.
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.x.size(0)
    return total_loss / total_nodes_train


@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y)
        total_loss += loss.item() * data.x.size(0)
    return total_loss / total_nodes_test


if __name__ == '__main__':
    # Check compatibility: The dataset provides in_dim=1 and edge_dim=1,
    # so we initialize the model accordingly.
    dataset = SSSPDataset(root='sssp_dataset', num_graphs=100)
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])

    # Split dataset into train and test (e.g., 80/20 split).
    train_dataset = dataset[:80]
    test_dataset = dataset[80:]

    # Create loaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Calculate total number of nodes in training and testing sets for averaging.
    total_nodes_train = sum(data.x.size(0) for data in train_dataset)
    total_nodes_test = sum(data.x.size(0) for data in test_dataset)

    # Set up device, model, and optimizer.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShortestPathModel(num_layers=10, emb_dim=64, in_dim=1, edge_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop.
    num_epochs = 1750
    for epoch in range(1, num_epochs + 1):
        loss = train()
        test_loss = test(test_loader)
        print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")


    # Evaluate a new graph
    # --- Select a single sample graph from the test dataset ---
    sample_idx = random.randint(0, len(test_dataset) - 1)
    sample_data = test_dataset[sample_idx].to(device)

    # --- Get predictions ---
    model.eval()
    with torch.no_grad():
        predicted_distances = model(sample_data).cpu()
    true_distances = sample_data.y.cpu()

    # Plot the graph
    # --- Convert the PyG Data object to a NetworkX graph ---
    edge_index_np = sample_data.edge_index.cpu().numpy()  # shape (2, num_edges)
    edge_attr_np = sample_data.edge_attr.cpu().numpy()  # shape (num_edges, 1)
    G_nx = nx.Graph()
    num_nodes = sample_data.num_nodes
    G_nx.add_nodes_from(range(num_nodes))

    # --- Add edges (each undirected edge appears only once) and record edge weights ---
    edge_labels = {}
    added_edges = set()
    for i, (u, v) in enumerate(zip(edge_index_np[0], edge_index_np[1])):
        key = tuple(sorted((u, v)))
        if key not in added_edges:
            G_nx.add_edge(u, v)
            # Since each edge weight is stored as a one-element list, extract its value.
            edge_labels[key] = f"{edge_attr_np[i][0]:.2f}"
            added_edges.add(key)

    # --- Create node labels showing both true (T) and predicted (P) distances ---
    node_labels = {
        i: f"T: {true_distances[i]:.2f}\nP: {predicted_distances[i]:.2f}"
        for i in range(num_nodes)
    }

    # --- Compute positions for visualization ---
    pos = nx.spring_layout(G_nx)

    # --- Draw the graph ---
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.7)
    nx.draw_networkx_labels(G_nx, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Graph Visualization: True vs Predicted Distances with Edge Weights")
    plt.axis('off')
    plt.show()

    # Plot errors and distributions
    # --- Compute errors ---
    errors = predicted_distances - true_distances

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Histogram of prediction errors
    plt.subplot(1, 2, 1)
    plt.hist(errors.numpy(), bins=20, edgecolor='k', color='skyblue')
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Frequency")

    # Distribution of distances (true and predicted)
    plt.subplot(1, 2, 2)
    plt.hist(true_distances.numpy(), bins=20, alpha=0.6, label="True Distances", edgecolor='k')
    plt.hist(predicted_distances.numpy(), bins=20, alpha=0.6, label="Predicted Distances", edgecolor='k')
    plt.title("Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import wandb

from graph_gen import SSSPDataset
from token_graph_transformer import TokenGT


# Helper function to move a dictionary of tensors to the device.
def to_device(data, device):
    data["tokens"] = data["tokens"].to(device)
    data["y"] = data["y"].to(device)
    return data


def main():
    wandb.init(project="transformer-graph-learner")
    # Set random seeds for reproducibility.
    torch.manual_seed(42)
    random.seed(42)

    # Parameters for token generation.
    d_p = 10  # dimension for node identifiers
    d_e = 10  # dimension for type embeddings
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

    # Create DataLoaders.
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device configuration.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters.
    d_model = 128
    nhead = 8
    num_layers = 4
    model = TokenGT(token_in_dim=token_in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)

    # Define optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.MSELoss()

    # Evaluation function.
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_nodes = 0
        with torch.no_grad():
            for batch in loader:
                # Since batch_size=1, extract the single sample from each batch.
                data = {k: v[0] for k, v in batch.items()}
                data = to_device(data, device)
                pred = model(data)
                loss = criterion(pred, data["y"])
                total_loss += loss.item() * data["node_count"]
                total_nodes += data["node_count"]
        return total_loss / total_nodes

    # Training loop.
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # Extract the single sample (dictionary) from the batch.
            data = {k: v[0] for k, v in batch.items()}
            data = to_device(data, device)
            optimizer.zero_grad()
            pred = model(data)  # forward pass: predict distances for each node.
            loss = criterion(pred, data["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate(test_loader)
        scheduler.step(test_loss)

        # Save the model every 10 epochs.
        if (epoch + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/model_{epoch + 1}.pth")

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "test_loss": test_loss}, step=epoch)

    evaluate_graph(model, test_dataset, device)


def evaluate_graph(model, test_dataset, device):
    # Evaluate a new graph
    # --- Select a single sample graph from the test dataset ---
    sample_idx = random.randint(0, len(test_dataset) - 1)
    sample_data = to_device(test_dataset[sample_idx], device)

    # --- Get predictions ---
    model.eval()
    with torch.no_grad():
        predicted_distances = model(sample_data).cpu()
    true_distances = sample_data['y'].cpu()

    # Plot the graph
    # --- Convert the PyG Data object to a NetworkX graph ---
    edge_index_np = sample_data['edge_index'].cpu().numpy()  # shape (2, num_edges)
    edge_attr_np = sample_data['edge_attr'].cpu().numpy()  # shape (num_edges, 1)
    G_nx = nx.Graph()
    num_nodes = sample_data['node_count']
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


if __name__ == "__main__":
    main()

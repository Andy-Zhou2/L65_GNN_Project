import random
import torch
import networkx as nx
import matplotlib.pyplot as plt

from utils import to_device


def evaluate(loader, model, criterion, device):
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


def evaluate_on_graph(model, test_dataset, device):
    # Select a random graph from the test dataset.
    sample_idx = random.randint(0, len(test_dataset) - 1)
    sample_data = to_device(test_dataset[sample_idx], device)

    model.eval()
    with torch.no_grad():
        predicted_distances = model(sample_data).cpu()
    true_distances = sample_data['y'].cpu()

    # Convert PyG Data object to a NetworkX graph.
    edge_index_np = sample_data['edge_index'].cpu().numpy()
    edge_attr_np = sample_data['edge_attr'].cpu().numpy()
    G_nx = nx.Graph()
    num_nodes = sample_data['node_count']
    G_nx.add_nodes_from(range(num_nodes))

    # Add edges (ensuring each undirected edge appears only once) and record edge weights.
    edge_labels = {}
    added_edges = set()
    for i, (u, v) in enumerate(zip(edge_index_np[0], edge_index_np[1])):
        key = tuple(sorted((u, v)))
        if key not in added_edges:
            G_nx.add_edge(u, v)
            edge_labels[key] = f"{edge_attr_np[i][0]:.2f}"
            added_edges.add(key)

    # Create node labels with true (T) and predicted (P) distances.
    node_labels = {
        i: f"T: {true_distances[i]:.2f}\nP: {predicted_distances[i]:.2f}"
        for i in range(num_nodes)
    }

    pos = nx.spring_layout(G_nx)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.7)
    nx.draw_networkx_labels(G_nx, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Graph Visualization: True vs Predicted Distances with Edge Weights")
    plt.axis('off')
    plt.show()

    # Plot errors and distributions.
    errors = predicted_distances - true_distances

    plt.figure(figsize=(12, 5))

    # Histogram of prediction errors.
    plt.subplot(1, 2, 1)
    plt.hist(errors.numpy(), bins=20, edgecolor='k')
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Frequency")

    # Distribution of true and predicted distances.
    plt.subplot(1, 2, 2)
    plt.hist(true_distances.numpy(), bins=20, alpha=0.6, label="True Distances", edgecolor='k')
    plt.hist(predicted_distances.numpy(), bins=20, alpha=0.6, label="Predicted Distances", edgecolor='k')
    plt.title("Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

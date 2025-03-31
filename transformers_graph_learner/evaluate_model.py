import random
import torch
import networkx as nx
import matplotlib.pyplot as plt

from .utils import to_device
from .graph_gen import SSSPDataset, collate_fn


def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    results = []
    with torch.no_grad():
        for data in loader:
            # Since batch_size=1, extract the single sample from each batch.
            data = to_device(data, device)
            pred = model(data)
            loss = criterion(pred, data["y"])
            total_loss += loss.item() * data["node_count"].sum()
            total_nodes += data["node_count"].sum().item()
    return total_loss / total_nodes

def evaluate_ood(loaders, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    results = []
    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                # Since batch_size=1, extract the single sample from each batch.
                data = to_device(data, device)
                pred = model(data)
                loss = criterion(pred, data["y"])
                total_loss += loss.item() * data["node_count"].sum()
                total_nodes += data["node_count"].sum().item()
                results.append(total_loss / total_nodes)
    return results

def plot_predicted_graph(edge_index_np, edge_attr_np, true_distances, predicted_distances, num_nodes, source_node, layer_num=None):
    G_nx = nx.Graph()
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

    pos = nx.spring_layout(G_nx, seed=42)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G_nx, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.7)
    nx.draw_networkx_labels(G_nx, pos, labels=node_labels, font_size=10)

    nx.draw_networkx_nodes(
        G_nx,
        pos,
        nodelist=[source_node],
        node_color="orange",
        node_size=600,
        label="Source",
    )

    nx.draw_networkx_edge_labels(
        G_nx, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    if layer_num is not None:
        plt.title(f"Graph Visualization: Layer {layer_num} - True vs Predicted Distances")
    else:
        plt.title("Graph Visualization: True vs Predicted Distances")
    plt.axis("off")
    plt.show()

def evaluate_on_graph(model, sample_data, device, intermediate_supervision=False):
    collate_data = collate_fn([sample_data])
    collate_data = to_device(collate_data, device)

    model.eval()
    with torch.no_grad():
        predicted_distances = model(collate_data, intermediate_supervision=intermediate_supervision).cpu()

    # Convert PyG Data object to a NetworkX graph.
    edge_index_np = sample_data["edge_index"].cpu().numpy()
    edge_attr_np = sample_data["edge_attr"].cpu().numpy()
    num_nodes = sample_data["node_count"]
    source_node = sample_data["x"].argmax().item()

    if intermediate_supervision:
       for i in range(len(predicted_distances)):
            print('predicted_distances', predicted_distances.shape)
            predicted_distance_layer = predicted_distances[i][0]
            true_distances = collate_data["intermediate_ys"].cpu()[-1][0]
            plot_predicted_graph(edge_index_np, edge_attr_np, true_distances, predicted_distance_layer, num_nodes, source_node, layer_num=i)
    else:
        true_distances = collate_data["intermediate_ys"].cpu()[-1][0]
        predicted_distances = predicted_distances[-1]
        plot_predicted_graph(edge_index_np, edge_attr_np, true_distances, predicted_distances, num_nodes, source_node)

    # # Plot errors and distributions.
    # errors = predicted_distances - true_distances
    #
    # plt.figure(figsize=(12, 5))
    #
    # # Histogram of prediction errors.
    # plt.subplot(1, 2, 1)
    # plt.hist(errors.numpy(), bins=20, edgecolor="k")
    # plt.title("Histogram of Prediction Errors")
    # plt.xlabel("Error (Predicted - True)")
    # plt.ylabel("Frequency")
    #
    # # Distribution of true and predicted distances.
    # plt.subplot(1, 2, 2)
    # plt.hist(
    #     true_distances.numpy(),
    #     bins=20,
    #     alpha=0.6,
    #     label="True Distances",
    #     edgecolor="k",
    # )
    # plt.hist(
    #     predicted_distances.numpy(),
    #     bins=20,
    #     alpha=0.6,
    #     label="Predicted Distances",
    #     edgecolor="k",
    # )
    # plt.title("Distribution of Distances")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

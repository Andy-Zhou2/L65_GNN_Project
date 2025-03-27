import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from .graph_gen import SSSPDataset


def visualize_graph(sample):
    edge_index = sample["edge_index"]
    edge_attr = sample["edge_attr"]
    node_count = sample["node_count"]
    source_node = sample["x"].argmax().item()

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(node_count))

    # Add edges with weights
    added_edges = set()
    for (u, v), weight in zip(edge_index.t().numpy(), edge_attr.numpy()):
        edge = tuple(sorted((u, v)))
        if edge not in added_edges:
            G.add_edge(u, v, weight=weight.item())
            added_edges.add(edge)

    # Plot
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source_node],
        node_color="orange",
        node_size=600,
        label="Source",
    )

    # Draw edges and edge labels
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    for key in edge_labels:
        edge_labels[key] = f'{edge_labels[key]:.2f}'
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    # Draw node labels
    node_labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_weight="bold")

    plt.title(
        f"Graph Visualization: Number Nodes: {node_count}, "
        f"Source Node: {source_node}, Diameter: {nx.diameter(G)}, "
        f"Eccentricity: {nx.eccentricity(G, v=source_node)}"
    )
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.show()


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

    dataset = SSSPDataset(num_graphs=cfg.dataset.num_graphs, d_p=d_p, n_nodes_range=(32, 40), max_hops=4)
    print(f"Total graphs in dataset: {len(dataset)}")

    # Split dataset into train and test (e.g., 80/20 split).
    num_train = int(cfg.dataset.split * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    for _ in range(5):
        sample = random.choice(dataset)
        visualize_graph(sample)

    # find the maximum diameter
    max_diameter = 0
    max_diameter_graph = None
    for sample in dataset:
        edge_index = sample["edge_index"]
        node_count = sample["node_count"]
        G = nx.Graph()
        G.add_nodes_from(range(node_count))
        added_edges = set()
        for u, v in edge_index.t().numpy():
            edge = tuple(sorted((u, v)))
            if edge not in added_edges:
                G.add_edge(u, v)
                added_edges.add(edge)
        diameter = nx.diameter(G)
        if diameter > max_diameter:
            max_diameter = diameter
            max_diameter_graph = sample
    print(f"Maximum diameter in the dataset: {max_diameter}")
    visualize_graph(max_diameter_graph)


if __name__ == "__main__":
    main()

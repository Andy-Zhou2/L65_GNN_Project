from copy import deepcopy
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import random


class SSSPDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=100, transform=None, pre_transform=None):
        self.num_graphs = num_graphs
        super(SSSPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for i in range(self.num_graphs):
            # --- Generate a random connected graph ---
            num_nodes = random.randint(5, 20)
            # Watts–Strogatz graphs require k to be even.
            k = min(4, num_nodes - 1)
            if k % 2 == 1:
                k += 1
            G = nx.connected_watts_strogatz_graph(num_nodes, k, 0.3)
            UG = deepcopy(G)

            # --- Assign random weights to edges ---
            for u, v in G.edges():
                G[u][v]['weight'] = float(random.randint(1, 5))

            # --- Choose a random source node ---
            source = random.choice(list(G.nodes()))

            # --- Compute shortest path distances from the source ---
            path_lengths = nx.single_source_dijkstra_path_length(G, source)
            y = [path_lengths.get(i, float('inf')) for i in range(num_nodes)]

            # Compute max hop
            hop_lengths = nx.single_source_dijkstra_path_length(UG, source)
            hops = torch.tensor(
                [hop_lengths.get(i, float("inf")) for i in range(num_nodes)],
                dtype=torch.int,
            )

            # --- Create node features ---
            # Here, the only node feature is the binary source flag.
            x = torch.zeros((num_nodes, 1))
            x[source] = 1.0

            # --- Create edge_index and edge_attr ---
            edge_index = []
            edge_attr = []
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                # For undirected graphs, add both directions.
                edge_index.append([u, v])
                edge_index.append([v, u])
                edge_attr.append([weight])
                edge_attr.append([weight])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # --- Build the Data object ---
            data_obj = Data(
                x=x,  # Node features (source flag)
                edge_index=edge_index,  # Graph connectivity
                edge_attr=edge_attr,  # Edge weights
                y=torch.tensor(y, dtype=torch.float),  # Ground-truth distances
                hops=hops,  # Max hops from the source
            )
            data_list.append(data_obj)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = SSSPDataset(root='sssp_dataset', num_graphs=100)
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])

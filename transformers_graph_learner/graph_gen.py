import torch
import networkx as nx
import random


class SSSPDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs=100, d_p=8, d_e=8):
        """
        Args:
            num_graphs (int): Number of graphs in the dataset.
            d_p (int): Dimensionality of the orthonormal node identifiers.
            d_e (int): Dimensionality of the fixed type embedding.
                        (Here we use fixed embeddings: zeros for nodes, ones for edges.)
        """
        self.num_graphs = num_graphs
        self.d_p = d_p
        self.d_e = d_e
        self.in_feat_dim = 1  # Node and edge features are 1-dimensional (e.g. source flag and weight)
        self.data_list = [self.generate_graph() for _ in range(num_graphs)]

    def generate_graph(self):
        # --- Generate a random connected graph ---
        num_nodes = random.randint(5, 5)
        k = min(4, num_nodes - 1)
        if k % 2 == 1:
            k += 1
        G = nx.connected_watts_strogatz_graph(num_nodes, k, 0.3)

        # --- Assign random weights to edges ---
        for u, v in G.edges():
            G[u][v]['weight'] = float(random.randint(1, 5))

        # --- Choose a random source node and compute shortest paths ---
        source = random.choice(list(G.nodes()))
        path_lengths = nx.single_source_dijkstra_path_length(G, source)
        y = torch.tensor(
            [path_lengths.get(i, float('inf')) for i in range(num_nodes)],
            dtype=torch.float
        )

        # --- Create node features (source flag) ---
        x = torch.zeros((num_nodes, self.in_feat_dim))
        x[source] = 1.0

        # --- Generate orthonormal features for nodes ---
        # # Create a random matrix and use QR decomposition.
        # A = torch.randn(num_nodes, self.d_p)
        # Q, _ = torch.linalg.qr(A)
        # P = Q  # shape: [num_nodes, d_p]
        # if num_nodes < self.d_p:
        #     pad = torch.zeros(num_nodes, self.d_p - num_nodes)
        #     P = torch.cat([P, pad], dim=-1)

        # Use one-hot encoding for nodes
        P = torch.eye(num_nodes, self.d_p, dtype=torch.float)

        # --- Construct node tokens ---
        # Each node token: [node feature, P[node], P[node], fixed node type embedding]
        node_type = torch.zeros((num_nodes, self.d_e))  # fixed type embedding for nodes
        node_tokens = torch.cat([x, P, P, node_type], dim=-1)  # shape: [num_nodes, 1 + 2*d_p + d_e]

        # --- Construct edge tokens ---
        edge_index = []
        edge_attr = []
        for u, v in G.edges():
            weight = G[u][v]['weight']
            # Since the graph is undirected, add both directions.
            edge_index.append([u, v])
            edge_index.append([v, u])
            edge_attr.append([weight])
            edge_attr.append([weight])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        num_edges = edge_attr.size(0)

        # For each edge (u, v): token = [edge feature, P[u], P[v], fixed edge type embedding]
        edge_source = edge_index[0]
        edge_target = edge_index[1]
        P_source = P[edge_source]  # shape: [num_edges, d_p]
        P_target = P[edge_target]  # shape: [num_edges, d_p]
        edge_type = torch.ones((num_edges, self.d_e))  # fixed type embedding for edges
        edge_tokens = torch.cat([edge_attr, P_source, P_target, edge_type], dim=-1)  # shape: [num_edges, 1+2*d_p+d_e]

        # --- Combine tokens (nodes first, then edges) ---
        tokens = torch.cat([node_tokens, edge_tokens], dim=0)

        return {'tokens': tokens, 'node_count': num_nodes, 'y': y,
                'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]

import numpy as np
import torch
import networkx as nx
import random
from torch_geometric.utils import get_laplacian


class SSSPDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs=100, d_p=8, n_nodes_range=(20, 20), node_identifier_encoding="one-hot"):
        """
        Args:
            num_graphs (int): Number of graphs in the dataset.
            d_p (int): Dimensionality of the orthonormal node identifiers.
            node_identifier_encoding (str): Encoding method of node identifier P.
              One of ["one-hot", "orf", "laplacian"].
        """
        self.num_graphs = num_graphs
        self.d_p = d_p
        self.in_feat_dim = (
            1  # Node and edge features are 1-dimensional (e.g. source flag and weight)
        )
        self.n_nodes_range = n_nodes_range  # Must be defined before generating graphs
        assert node_identifier_encoding in ["one-hot", "orf", "laplacian"], f'Unsupported node_identifier_encoding {node_identifier_encoding}, should be one of ["one-hot", "orf", "laplacian"]'
        self.node_identifier_encoding = node_identifier_encoding

        self.data_list = [self.generate_graph(self.node_identifier_encoding) for _ in range(num_graphs)]

    @torch.no_grad()
    def generate_graph(self, node_identifier_encoding):
        # --- Generate a random connected graph ---
        num_nodes = random.randint(*self.n_nodes_range)
        k = min(4, num_nodes - 1)
        if k % 2 == 1:
            k += 1
        G = nx.connected_watts_strogatz_graph(num_nodes, k, 0.3)

        # --- Assign random weights to edges ---
        for u, v in G.edges():
            G[u][v]["weight"] = float(random.randint(1, 5))

        # --- Choose a random source node and compute shortest paths ---
        source = random.choice(list(G.nodes()))
        path_lengths = nx.single_source_dijkstra_path_length(G, source)
        y = torch.tensor(
            [path_lengths.get(i, float("inf")) for i in range(num_nodes)],
            dtype=torch.float,
        )

        # --- Create node features (source flag) ---
        x = torch.zeros((num_nodes, self.in_feat_dim))
        x[source] = 1.0

        # --- Generate orthonormal features for nodes ---
        match node_identifier_encoding:
            case "one-hot":
                P = torch.eye(num_nodes, self.d_p, dtype=torch.float)
            case "orf":
                A = torch.randn(num_nodes, self.d_p)
                Q, _ = torch.linalg.qr(A)
                P = Q  # shape: [num_nodes, d_p]
                if num_nodes < self.d_p:
                    pad = torch.zeros(num_nodes, self.d_p - num_nodes)
                    P = torch.cat([P, pad], dim=-1)
            case "laplacian":
                A = torch.full((num_nodes, num_nodes), float("inf"))
                for u, v, w in G.edges.data("weight"):
                    A[u, v] = w
                in_degree = A.isfinite().sum(dim=1).view(-1)
                N = torch.diag(in_degree.clip(1) ** -0.5)
                L = torch.eye(num_nodes) - N @ A @ N
                _, eigvec = torch.linalg.eigh(L)  # shape: [num_nodes, num_nodes], eigvec has columns as eigenvectors
                # randomly flip eigvec's signs
                signs = torch.where(torch.randint(0, 2, (eigvec.size(1),)) == 0, -1, 1)  # shape: [num_nodes]
                P = eigvec.to(torch.float) * signs[None, :]
                if num_nodes < self.d_p:
                    pad = torch.zeros(num_nodes, self.d_p - num_nodes)
                    P = torch.cat([P, pad], dim=-1)
                elif num_nodes > self.d_p:
                    P = torch.cat([P[:, :self.d_p//2], P[:, (self.d_p-self.d_p//2):]], dim=-1)
            case _:
                raise ValueError(f"Invalid node identifier encoding {node_identifier_encoding}")
            
        assert P.shape == (num_nodes, self.d_p)

        # --- Construct node tokens ---
        # Each node token: [node feature, P[node], P[node], fixed node type embedding]
        # node_type = torch.zeros((num_nodes, self.d_e))  # fixed type embedding for nodes
        node_tokens = torch.cat(
            [x, P, P], dim=-1
        )  # shape: [num_nodes, 1 + 2*d_p + d_e]

        # --- Construct edge tokens ---
        edge_index = []
        edge_attr = []
        for u, v in G.edges():
            weight = G[u][v]["weight"]
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
        # edge_type = torch.ones((num_edges, self.d_e))  # fixed type embedding for edges
        edge_tokens = torch.cat(
            [edge_attr, P_source, P_target], dim=-1
        )  # shape: [num_edges, 1+2*d_p+d_e]

        # --- Combine tokens (nodes first, then edges) ---
        tokens = torch.cat([node_tokens, edge_tokens], dim=0)

        return {
            "tokens": tokens,
            "node_count": num_nodes,
            "y": y,
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    # batch is a list of dictionaries from __getitem__
    # Extract tokens and other items you need
    tokens_list = [d["tokens"] for d in batch]
    y_list = [d["y"] for d in batch]
    node_counts = [d["node_count"] for d in batch]

    # Determine the maximum token sequence length in this batch
    max_len = max(token.shape[0] for token in tokens_list)

    padded_tokens = []
    attn_masks = []
    for tokens in tokens_list:
        pad_len = max_len - tokens.shape[0]
        if pad_len > 0:
            # pad with zeros (or a dedicated pad value)
            pad = torch.zeros(pad_len, tokens.shape[1])
            tokens_padded = torch.cat([tokens, pad], dim=0)
        else:
            tokens_padded = tokens
        padded_tokens.append(tokens_padded)

        # Create an attention mask: 1 for real tokens, 0 for padding
        mask = torch.cat([torch.ones(tokens.shape[0]), torch.zeros(pad_len)])
        attn_masks.append(mask)

    # Stack padded tokens and masks: shapes [batch_size, max_len, feature_dim] and [batch_size, max_len]
    batch_tokens = torch.stack(padded_tokens)
    batch_masks = torch.stack(attn_masks)

    # Optionally, pad other sequence targets (e.g., y)
    # Here we assume y is a 1D tensor per graph; adjust if necessary.
    batch_y = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True, padding_value=0)

    # You can also collate node_counts, edge information, etc.
    batch_node_counts = torch.tensor(node_counts)

    return {
        "tokens": batch_tokens,
        "attn_mask": batch_masks,
        "y": batch_y,
        "node_count": batch_node_counts,
    }

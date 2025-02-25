import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

# ---------------------------
# Dataset Generation (SSSP)
# ---------------------------
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
            num_nodes = random.randint(5, 15)
            k = min(4, num_nodes - 1)
            if k % 2 == 1:
                k += 1
            G = nx.connected_watts_strogatz_graph(num_nodes, k, 0.3)

            # --- Assign random weights to edges ---
            for u, v in G.edges():
                G[u][v]['weight'] = float(random.randint(1, 5))

            # --- Choose a random source node ---
            source = random.choice(list(G.nodes()))

            # --- Compute shortest path distances from the source ---
            path_lengths = nx.single_source_dijkstra_path_length(G, source)
            y = [path_lengths.get(i, float('inf')) for i in range(num_nodes)]

            # --- Create node features (only the source flag) ---
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
                x=x,                    # Node features (source flag)
                edge_index=edge_index,  # Graph connectivity
                edge_attr=edge_attr,    # Edge weights
                y=torch.tensor(y, dtype=torch.float)  # Ground-truth distances
            )
            data_list.append(data_obj)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ---------------------------
# Tokenized Graph Transformer (Graph Learner)
# ---------------------------
class TokenGT(nn.Module):
    def __init__(self, in_feat_dim, d_model, d_p, d_e, nhead, num_layers, dropout=0.1):
        """
        Args:
            in_feat_dim (int): Dimensionality of input node/edge features (here 1).
            d_model (int): Hidden dimension of the Transformer.
            d_p (int): Dimension of the node identifier.
            d_e (int): Dimension of the trainable type embeddings.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
        """
        super(TokenGT, self).__init__()
        self.d_p = d_p
        self.d_e = d_e

        # Compute the input token dimension:
        # For both nodes and edges: feature dim + 2*d_p + d_e
        token_in_dim = in_feat_dim + 2 * d_p + d_e

        # Type embeddings: row 0 for nodes, row 1 for edges.
        self.type_embeddings = nn.Parameter(torch.randn(2, d_e))

        # Linear projection for tokens.
        self.token_proj = nn.Linear(token_in_dim, d_model)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head for node-level output (predicts a scalar per node).
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): A graph with attributes:
                - x: node features, shape [num_nodes, in_feat_dim]
                - edge_index: connectivity, shape [2, num_edges]
                - edge_attr: edge features, shape [num_edges, in_feat_dim]
                - y: ground truth distances for nodes, shape [num_nodes]
        Returns:
            pred (torch.Tensor): Predicted distances for each node, shape [num_nodes]
        """
        x = data.x  # [num_nodes, in_feat_dim]
        edge_index = data.edge_index  # [2, num_edges]
        edge_attr = data.edge_attr  # [num_edges, in_feat_dim]

        num_nodes = x.size(0)
        num_edges = edge_attr.size(0)

        # --- Generate node identifiers (using orthonormal random features) ---
        P = self.orthonormal_features(num_nodes, self.d_p)  # [num_nodes, d_p]
        # Zero pad P if needed to get shape [num_nodes, d_p]
        if num_nodes < self.d_p:
            pad = torch.zeros(num_nodes, self.d_p - num_nodes, device=P.device)
            P = torch.cat([P, pad], dim=-1)

        # --- Construct node tokens ---
        # For each node, token = [node feature, P[node], P[node], type_embedding for nodes]
        node_type = self.type_embeddings[0].unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, d_e]
        node_tokens = torch.cat([x, P, P, node_type], dim=-1)  # [num_nodes, in_feat_dim + 2*d_p + d_e]

        # --- Construct edge tokens ---
        # For each edge (u, v): token = [edge feature, P[u], P[v], type_embedding for edges]
        edge_source = edge_index[0]  # [num_edges]
        edge_target = edge_index[1]  # [num_edges]
        P_source = P[edge_source]  # [num_edges, d_p]
        P_target = P[edge_target]  # [num_edges, d_p]
        edge_type = self.type_embeddings[1].unsqueeze(0).expand(num_edges, -1)  # [num_edges, d_e]
        edge_tokens = torch.cat([edge_attr, P_source, P_target, edge_type], dim=-1)  # [num_edges, in_feat_dim + 2*d_p + d_e]

        # --- Combine tokens (nodes first, then edges) ---
        tokens = torch.cat([node_tokens, edge_tokens], dim=0)  # [num_nodes + num_edges, token_in_dim]

        # Project tokens to model dimension.
        tokens = self.token_proj(tokens)  # [num_tokens, d_model]

        # Add batch dimension (batch size = 1).
        tokens = tokens.unsqueeze(0)  # [1, num_tokens, d_model]

        # Apply Transformer encoder.
        tokens = self.transformer(tokens)  # [1, num_tokens, d_model]
        tokens = tokens.squeeze(0)  # [num_tokens, d_model]

        # --- Extract node token outputs and predict ---
        node_out = tokens[:num_nodes]  # first num_nodes tokens correspond to nodes
        pred = self.pred_head(node_out).squeeze(-1)  # [num_nodes]
        return pred

    def orthonormal_features(self, n, d):
        """
        Generates an [n x d] matrix with orthonormal columns.
        Assumes n >= d.
        """
        A = torch.randn(n, d, device=self.type_embeddings.device)
        Q, _ = torch.linalg.qr(A)
        return Q

# ---------------------------
# Training with Train/Test Split and Evaluation
# ---------------------------
if __name__ == '__main__':
    # Set random seed for reproducibility.
    torch.manual_seed(42)
    random.seed(42)

    # Create the dataset.
    dataset = SSSPDataset(root='sssp_dataset', num_graphs=100)
    print(f"Total graphs in dataset: {len(dataset)}")

    # Split dataset into train and test (80/20 split).
    num_train = int(0.8 * len(dataset))
    train_dataset = dataset[:num_train]
    test_dataset = dataset[num_train:]
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Device configuration.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model.
    model = TokenGT(
        in_feat_dim=1,    # node and edge feature dimension
        d_model=128,      # Transformer hidden dimension
        d_p=16,           # dimension of node identifier
        d_e=16,           # dimension of type identifier
        nhead=8,          # number of attention heads
        num_layers=10,     # number of Transformer layers
        dropout=0.1
    ).to(device)

    # Define optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # Define an evaluation function.
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_nodes = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                pred = model(data)
                loss = criterion(pred, data.y)
                total_loss += loss.item() * data.x.size(0)
                total_nodes += data.x.size(0)
        return total_loss / total_nodes

    model.load_state_dict(torch.load('models_1e-5/model_1000.pth', map_location=device))


    # EVALUATE
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

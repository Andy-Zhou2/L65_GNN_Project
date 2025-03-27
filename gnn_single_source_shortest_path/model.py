import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max


# ---------------------------
# 1. Define the MPNNLayer
# ---------------------------
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=1):
        """
        Message Passing Layer using max aggregation.

        Args:
            emb_dim (int): Dimension of node embeddings.
            edge_dim (int): Dimension of edge features (should match dataset, e.g. 1).
        """
        super().__init__(aggr="max")
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP for message computation: input dim = 2*emb_dim + edge_dim, output = emb_dim
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + edge_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
        )

        # MLP for updating node features: input dim = 2*emb_dim, output = emb_dim
        self.mlp_update = Sequential(
            Linear(2 * emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
        )

    def forward(self, h, edge_index, edge_attr):
        """
        Perform one round of message passing.

        Args:
            h (Tensor): Node embeddings, shape (num_nodes, emb_dim)
            edge_index (LongTensor): Edge indices in COO format, shape (2, num_edges)
            edge_attr (Tensor): Edge features, shape (num_edges, edge_dim)
        Returns:
            Tensor: Updated node embeddings, shape (num_nodes, emb_dim)
        """
        return self.propagate(edge_index, h=h, edge_attr=edge_attr)

    def message(self, h_i, h_j, edge_attr):
        # Concatenate destination, source node embeddings and edge features
        msg_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg_input)

    def aggregate(self, inputs, index):
        # Use max aggregation; scatter_max returns a tuple (values, argmax)
        aggr_out, _ = scatter_max(inputs, index, dim=self.node_dim)
        # Replace any -infs (from nodes that did not receive messages) with 0.
        aggr_out[aggr_out == -float("inf")] = 0
        return aggr_out

    def update(self, aggr_out, h):
        # Combine the original node embedding with the aggregated message.
        update_input = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_update(update_input)

    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim})"


# ---------------------------
# 2. Define the SSSP Model
# ---------------------------
class ShortestPathModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=1, edge_dim=1):
        """
        Model to predict node-level shortest path distances.

        Args:
            num_layers (int): Number of message passing layers.
            emb_dim (int): Hidden (embedding) dimension.
            in_dim (int): Dimension of the initial node features (here, 1: the source flag).
            edge_dim (int): Dimension of the edge features (here, 1: the edge weight).
        """
        super().__init__()
        # Project the initial node features to the embedding dimension.
        self.lin_in = Linear(in_dim, emb_dim)

        # Create a stack of message passing layers.
        self.convs = ModuleList(
            [MPNNLayer(emb_dim, edge_dim) for _ in range(num_layers)]
        )

        # Final MLP to output a scalar distance for each node.
        self.mlp_out = Sequential(
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), Linear(emb_dim, 1)
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data (Data): PyG data object with attributes:
                - x: node features, shape (num_nodes, in_dim)
                - edge_index: edge connectivity, shape (2, num_edges)
                - edge_attr: edge features, shape (num_edges, edge_dim)
                - y: ground-truth distances, shape (num_nodes,)
        Returns:
            Tensor: Predicted distances, shape (num_nodes,)
        """
        h = self.lin_in(data.x)  # (num_nodes, emb_dim)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)  # residual connection
        distances = self.mlp_out(h)  # (num_nodes, 1)
        return distances.squeeze(-1)  # (num_nodes,)

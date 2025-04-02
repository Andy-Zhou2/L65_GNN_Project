import torch

from ..gnn_single_source_shortest_path.model import ShortestPathModel


class EarlyExitGCN(torch.nn.Module):
    def __init__(self, model: ShortestPathModel):
        super().__init__()
        self.model = model
        self.num_layers = len(model.convs)

    def forward(self, data, early_exit_layer=None):
        if early_exit_layer is None:
            early_exit_layer = self.num_layers

        h = self.model.lin_in(data.x)  # (num_nodes, emb_dim)
        for i, conv in enumerate(self.model.convs):
            if i >= early_exit_layer:
                break
            h = h + conv(h, data.edge_index, data.edge_attr)  # residual connection
        distances = self.model.mlp_out(h)  # (num_nodes, 1)
        return distances.squeeze(-1)  # (num_nodes,)

    def get_hidden_embedding(self, data, early_exit_layer=None):
        if early_exit_layer is None:
            early_exit_layer = self.num_layers

        h = self.model.lin_in(data.x)  # (num_nodes, emb_dim)
        for i, conv in enumerate(self.model.convs):
            if i >= early_exit_layer:
                break
            h = h + conv(h, data.edge_index, data.edge_attr)  # residual connection
        return h  # (num_nodes, emb_dim)

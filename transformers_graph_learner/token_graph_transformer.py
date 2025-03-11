import torch.nn as nn

class TokenGT(nn.Module):
    def __init__(self, token_in_dim, d_model, nhead, num_layers, dropout=0.1):
        """
        Args:
            token_in_dim (int): Dimensionality of the input tokens
                                (should equal 1 + 2*d_p + d_e).
            d_model (int): Hidden dimension of the Transformer.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
        """
        super(TokenGT, self).__init__()
        self.token_proj = nn.Linear(token_in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, data):
        """
        Args:
            data (dict): A dictionary with keys:
                - 'tokens': precomputed tokens, tensor of shape [num_tokens, token_in_dim]
                - 'node_count': number of nodes in the graph (first tokens correspond to nodes)
                - 'y': ground-truth distances (unused in forward but useful for loss computation)
        Returns:
            pred (torch.Tensor): Predicted distances for each node, shape [num_nodes]
        """
        tokens = data['tokens']  # shape: [num_tokens, token_in_dim]
        num_nodes = data['node_count']
        tokens = self.token_proj(tokens)  # [num_tokens, d_model]
        tokens = tokens.unsqueeze(0)  # add batch dimension
        tokens = self.transformer(tokens)  # [1, num_tokens, d_model]
        tokens = tokens.squeeze(0)  # [num_tokens, d_model]

        # Extract node tokens (first num_nodes tokens) for prediction.
        node_out = tokens[:num_nodes]
        pred = self.pred_head(node_out).squeeze(-1)  # [num_nodes]
        return pred

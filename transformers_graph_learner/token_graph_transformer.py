import torch.nn as nn
import torch

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
            data (dict): Contains:
                - 'tokens': Tensor of shape [B, num_tokens, token_in_dim]
                - 'attn_mask': Tensor of shape [B, num_tokens] (1 for valid, 0 for padding)
                - 'node_count': Tensor of shape [B] with number of node tokens (first tokens in each sample)
                - 'y': Ground-truth (unused here)
        Returns:
            pred (torch.Tensor): Predicted distances for each node, shape [B, num_tokens]
                                 with padded positions zeroed out.
        """
        tokens = data["tokens"]  # [B, num_tokens, token_in_dim]
        attn_mask = data["attn_mask"]  # [B, num_tokens]
        node_count = data["node_count"]  # [B]

        # Project input tokens to model dimension.
        tokens = self.token_proj(tokens)  # [B, num_tokens, d_model]

        # Create a src_key_padding_mask: True for padded positions.
        src_key_padding_mask = (attn_mask == 0)  # [B, num_tokens]

        # Pass tokens through the transformer.
        tokens = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        # tokens: [B, num_tokens, d_model]

        # Apply the prediction head.
        pred_all = self.pred_head(tokens).squeeze(-1)  # [B, num_tokens]

        # Create a mask based on node_count to zero out predictions for non-node (padded) positions.
        # For each sample i, we want to keep only the first node_count[i] tokens.
        B, T = pred_all.size()
        device = pred_all.device
        node_mask = (torch.arange(T, device=device).unsqueeze(0) < node_count.unsqueeze(1)).float()  # [B, T]

        # Multiply elementwise so that positions beyond the actual nodes are zeroed out.
        pred = pred_all * node_mask  # [B, num_tokens]
        return pred[:, :node_count.max().item()]


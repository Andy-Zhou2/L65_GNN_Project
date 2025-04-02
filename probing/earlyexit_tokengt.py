import torch

from ..transformers_graph_learner.token_graph_transformer import TokenGT
from .earlyexit_transformer_encoder import EarlyExitTransformerEncoder


class EarlyExitTokenGT(torch.nn.Module):
    def __init__(self, model: TokenGT):
        super().__init__()
        self.model = model
        self.num_layers = model.transformer.num_layers
        # Copy encoder state dict to early-exit-encoder
        original_encoder = model.transformer
        early_exit_encoder = EarlyExitTransformerEncoder(
            original_encoder.layers[0],
            original_encoder.num_layers,
            original_encoder.norm,
            original_encoder.enable_nested_tensor,
            original_encoder.mask_check,
        )
        early_exit_encoder.load_state_dict(original_encoder.state_dict())
        self.model.transformer = early_exit_encoder

    def forward(self, data, early_exit_layer=None):
        if early_exit_layer is None:
            early_exit_layer = self.num_layers

        tokens = data["tokens"]  # shape: [num_tokens, token_in_dim]
        num_nodes = data["node_count"]
        tokens = self.model.token_proj(tokens)  # [num_tokens, d_model]
        tokens = tokens.unsqueeze(0)  # add batch dimension
        tokens = self.model.transformer(
            tokens, early_exit_layer_num=early_exit_layer
        )  # [1, num_tokens, d_model]
        tokens = tokens.squeeze(0)  # [num_tokens, d_model]

        # Extract node tokens (first num_nodes tokens) for prediction.
        node_out = tokens[:num_nodes]
        pred = self.pred_head(node_out).squeeze(-1)  # [num_nodes]
        return pred

    def get_hidden_embedding(self, data, early_exit_layer=None):
        if early_exit_layer is None:
            early_exit_layer = self.num_layers

        tokens = data["tokens"]  # shape: [num_tokens, token_in_dim]
        num_nodes = data["node_count"]
        tokens = self.model.token_proj(tokens)  # [num_tokens, d_model]
        tokens = tokens.unsqueeze(0)  # add batch dimension
        tokens = self.model.transformer(
            tokens, early_exit_layer_num=early_exit_layer
        )  # [1, num_tokens, d_model]
        tokens = tokens.squeeze(0)  # [num_tokens, d_model]

        # Return node and edge tokens
        return tokens

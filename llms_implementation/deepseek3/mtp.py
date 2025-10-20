import torch
import torch.nn as nn


class MultiTokenPrediction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        self.proj_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        predictions = []
        current_hidden = hidden_states

        for d in range(self.depth):
            # Project the hidden states
            projected = self.proj_layers[d](current_hidden)
            # Apply dropout
            projected = self.dropout(projected)
            normalized = self.norm(projected)
            # Apply output head directly here
            logits = self.output_head(normalized)
            predictions.append(logits)
            current_hidden = projected

        return torch.stack(predictions, dim=1)

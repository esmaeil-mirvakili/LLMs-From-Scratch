import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from llms_implementation.rope import RotaryPositionalEmbedding


class MultiHeadLatentAttention(nn.Module):
    """Implements Multi-head Latent Attention (MLA) for efficient inference."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_compression_dim: int,
        query_compression_dim: int,
        rope_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Key/Value compression layers
        self.kv_down = nn.Linear(hidden_dim, kv_compression_dim)
        self.key_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.value_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.key_rope = nn.Linear(kv_compression_dim, num_heads * rope_dim)

        # Query compression layers
        self.query_down = nn.Linear(hidden_dim, query_compression_dim)
        self.query_up = nn.Linear(query_compression_dim, num_heads * head_dim)
        self.query_rope = nn.Linear(query_compression_dim, num_heads * rope_dim)

        self.output_proj = nn.Linear(num_heads * head_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of Multi-head Latent Attention.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]

        Returns:
            Output tensor after attention [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compress and project keys/values
        kv_compressed = self.kv_down(x)
        keys_c = (
            self.key_up(kv_compressed)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.value_up(kv_compressed)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys_r = (
            self.key_rope(kv_compressed)
            .view(batch_size, seq_len, self.num_heads, self.rope_dim)
            .transpose(1, 2)
        )
        keys_r = RotaryPositionalEmbedding.apply(keys_r, cos, sin)

        # Compress and project queries
        query_compressed = self.query_down(x)
        queries_c = (
            self.query_up(query_compressed)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries_r = (
            self.query_rope(query_compressed)
            .view(batch_size, seq_len, self.num_heads, self.rope_dim)
            .transpose(1, 2)
        )
        queries_r = RotaryPositionalEmbedding.apply(queries_r, cos, sin)

        # Concatenate rotary and non-rotary parts
        queries = torch.cat([queries_c, queries_r], dim=-1)
        keys = torch.cat([keys_c, keys_r], dim=-1)

        # Compute attention scores - shape: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(
            self.head_dim + self.rope_dim
        )

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

        # Process attention mask if provided
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to 4D if needed
            if attention_mask.dim() == 2:
                # Convert to boolean mask first (1 -> True, 0 -> False)
                attention_mask = attention_mask.bool()
                # Then expand to 4D: [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Expand if needed to match attention score dims
                attention_mask = attention_mask.expand(-1, -1, seq_len, -1)

                # Invert the mask - we want True where tokens should be ignored
                attention_mask = ~attention_mask
            elif attention_mask.dim() == 4:
                # If already 4D, just ensure it's boolean
                attention_mask = attention_mask.bool()

            # Combine with causal mask - we want to apply both masks
            # (True wherever we want to mask out)
            combined_mask = causal_mask | attention_mask
        else:
            # Just use causal mask
            combined_mask = causal_mask

        # Apply mask - set masked positions to negative infinity
        attn_scores = attn_scores.masked_fill(combined_mask, float("-1e9"))

        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply dropout to attention probabilities
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        context = torch.matmul(
            attn_probs, values
        )  # [batch_size, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply output projection with dropout
        output = self.output_proj(context)
        output = self.dropout(output)

        return output

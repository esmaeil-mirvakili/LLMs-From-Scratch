import torch
import torch.nn as nn
import torch.nn.functional as F

from llms_implementation.rope import RotaryPositionalEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped query attention that shares key/value projections across head groups.
    """

    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        num_kv_groups,
        dtype=None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // self.num_heads
        self.att_scaling = self.head_dim**-0.5
        self.num_kv_groups = num_kv_groups  # 1 → MQA, num_heads → MHA
        self.num_repeat = self.num_heads // self.num_kv_groups
        self.w_queries = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.w_keys = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.w_values = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        queries = self.w_queries(x)  # shape (b, s, d_out)
        keys = self.w_keys(x)  # K and V shapes (b, s, num_kv_groups * head_dim)
        values = self.w_values(x)

        b, seq_len, _ = x.shape
        # Reshape into (batch, seq, heads, head_dim) for queries and groups for keys/values.
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_kv_groups, -1)
        values = values.view(b, seq_len, self.num_kv_groups, -1)

        # Move heads before sequence length for batched matmul.
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Add rotary position encodings.
        queries = RotaryPositionalEmbedding.apply(queries, cos, sin)
        keys = RotaryPositionalEmbedding.apply(keys, cos, sin)
        # Repeat keys/values so each query head sees its assigned group.
        keys = keys.repeat_interleave(self.num_repeat, dim=1)
        values = values.repeat_interleave(self.num_repeat, dim=1)
        att_scores = queries @ keys.mT  # shape (b, num_heads, seq_len, seq_len)
        current_mask = mask[:seq_len, :seq_len]
        scaled_att_scores = att_scores * self.att_scaling
        scaled_att_scores.masked_fill_(current_mask, -torch.inf)
        att_weights = F.softmax(scaled_att_scores, dim=-1)

        ctx_tensor = att_weights @ values
        # Restore (batch, seq, heads, head_dim).
        ctx_tensor = ctx_tensor.transpose(1, 2)
        # Merge heads back into the last dimension.
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)
        ctx_tensor = self.out_proj(ctx_tensor)

        return ctx_tensor

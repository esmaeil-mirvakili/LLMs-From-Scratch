import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim, num_head, seq_len, dropout_p=0.0, use_fused=True, **kwargs
    ):
        super().__init__()
        assert embed_dim % num_head == 0, "embed_dim must be divisible by num_head"
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scale_factor = self.head_dim**0.5
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_p)
        self.use_fused = use_fused and hasattr(F, "scaled_dot_product_attention")
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        setattr(self.o_proj, "SCALE_INIT", 1)
        self.register_buffer(
            "mask",
            (
                None
                if self.use_fused
                else torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(
                    1, 1, seq_len, seq_len
                )
            ),
        )

    def _get_causal_mask(self, T):
        assert not self.use_fused, "Causal mask not needed for fused attention"
        assert self.mask is not None, "Causal mask not initialized"
        assert T <= self.seq_len, f"T={T} exceeds configured seq_len={self.seq_len}"
        return self.mask[:, :, :T, :T]

    def _get_qkv(self, x):
        qkv = self.qkv_proj(x)  # B, T, 3 * embed_dim
        return qkv.split(self.embed_dim, dim=-1)

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        assert (
            x.size(-1) == self.embed_dim
        ), "Last dimension of input tensor must match embed_dim"
        B, T, _ = x.size()
        q, k, v = self._get_qkv(x)
        q, k, v = (
            q.view(B, T, self.num_head, self.head_dim),
            k.view(B, T, self.num_head, self.head_dim),
            v.view(B, T, self.num_head, self.head_dim),
        )  # B, T, num_head, head_dim
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # B, num_head, T, head_dim
        if self.use_fused:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # dot_prod = q @ k.transpose(-2, -1)  # B, num_head, T, T
            dot_prod = torch.einsum("bhte,bhse->bhts", q, k)
            scaled_dot_prod = dot_prod / self.scale_factor
            causal_mask = self._get_causal_mask(T)
            scaled_dot_prod = scaled_dot_prod.masked_fill(~causal_mask, float("-inf"))
            attn_scores = torch.softmax(scaled_dot_prod, dim=-1)
            attn_scores = self.dropout(attn_scores)
            # attn_output = attn_scores @ v
            attn_output = torch.einsum("bhst,bhte->bhse", attn_scores, v)
        attn_output = attn_output.transpose(1, 2)  # B, T, num_head, head_dim
        attn_output = attn_output.contiguous()
        attn_output = attn_output.view(B, T, self.embed_dim)  # B, T, embed_dim
        attn_output = self.o_proj(attn_output)  # B, T, embed_dim
        return attn_output

import torch
import torch.nn as nn

from llms_implementation.llama3.attention import GroupedQueryAttention
from llms_implementation.rope import RotaryPositionalEmbedding


class RMSNorm(nn.Module):
    """
    Root mean square normalization with a single scale parameter.
    """

    def __init__(self, emb_dim, dtype=None):
        super().__init__()
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(emb_dim, dtype=dtype))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        norm_x = x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps)
        return self.scale * norm_x.to(input_dtype)


class SiLU(nn.Module):
    """
    Sigmoid linear unit (SiLU), also known as Swish with β = 1.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * 1 / (1 + torch.exp(-x))


class FFN(nn.Module):
    """
    Feed-forward block with a SwiGLU-style gated activation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.lin_gate = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.silu_activ = SiLU()
        self.lin2 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.silu_activ(x_gate)
        x1 = self.lin1(x)

        return self.lin2(x1 * x_gate)


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-norm attention and SwiGLU feed-forward.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["num_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.norm_1 = RMSNorm(cfg["emb_dim"])
        self.norm_2 = RMSNorm(cfg["emb_dim"])
        self.ffn = FFN(cfg)

    def forward(self, x, mask, cos, sin):
        # Pre-norm attention
        residual = x
        x = self.norm_1(x)
        x = self.att(x, mask, cos, sin)
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class Llama3Model(nn.Module):
    """
    Llama 3-style transformer with tied output and embedding weights.
    """

    def __init__(self, config):
        super().__init__()

        self.emb_dict = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["emb_dim"],
            dtype=config["dtype"],
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config) for layer in range(config["n_layers"])],
        )
        self.final_norm = RMSNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False, dtype=config["dtype"]
        )

        mask = torch.triu(
            torch.ones(config["context_length"], config["context_length"], dtype=torch.bool),
            diagonal=1,
        )
        head_dim = config["emb_dim"] // config["n_heads"]
        cos, sin = RotaryPositionalEmbedding.compute_angles(
            base=config["rope_base"], head_dim=head_dim, ctx_len=config["context_length"]
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        assert (
            self.emb_dict.weight.shape == self.out_head.weight.shape
        ), "Shape mismatch for weight tying"
        self.out_head.weight = self.emb_dict.weight

    def forward(self, x, attn_mask=None):
        # Embed tokens then pass through stacked blocks.
        x = self.emb_dict(x)

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

import math

import torch
import torch.nn as nn
from torch import Tensor

from llms_implementation.gpt2.attention import MultiHeadAttention


class GELU(nn.Module):
    def __init__(self, use_erf: bool = True):
        super().__init__()
        self.use_erf = use_erf

    def forward(self, x: Tensor) -> Tensor:
        if self.use_erf:
            return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        z_scores = (x - mu) / (std - self.eps)
        return self.gamma * z_scores + self.beta


class FeedForwardNetwork(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        setattr(self.fc2, "SCALE_INIT", 1)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_head, seq_len, attn_config=None):
        super().__init__()
        self.causal_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_head=num_head,
            seq_len=seq_len,
            **attn_config if attn_config is not None else {},
        )
        self.ffn = FeedForwardNetwork(embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        out = self.ln1(x)
        x = x + self.causal_attn(out)
        out = self.ln2(x)
        return x + self.ffn(x)


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.get("embed_dim", 768)
        self.seq_len = config.get("seq_len", 1024)
        self.num_head = config.get("num_head", 12)
        self.num_layers = config.get("num_layers", 12)
        self.vocab_size = config.get("vocab_size", 50257)
        attn_config = config.get("attn_config", {})
        self.init_scale = (2.0 * self.num_layers) ** -0.5
        self.wte = nn.Embedding(self.vocab_size, self.seq_len)
        self.wpe = nn.Embedding(self.seq_len, self.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.embed_dim, self.num_head, self.seq_len, attn_config
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_f = LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "SCALE_INIT"):
            with torch.no_grad():
                module.weight.mul_(self.init_scale)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.seq_len, f"Cannot forward sequence of length {T}."
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)  # T
        pos_embed = self.wpe(positions)  # T, embed_dim
        tok_embed = self.wte(input_ids)  # B, T, embed_dim
        x = pos_embed + tok_embed  # B, T, embed_dim
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

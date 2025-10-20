import torch
import torch.nn as nn
from typing import Optional

from llms_implementation.rope import RotaryPositionalEmbedding
from llms_implementation.deepseek3.attention import MultiHeadLatentAttention
from llms_implementation.deepseek3.moe import DeepSeekMoE
from llms_implementation.deepseek3.mtp import MultiTokenPrediction


class DeepSeekV3(nn.Module):
    """DeepSeek-V3: A Mixture-of-Experts Transformer with Multi-Token Prediction."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Extract dropout rate from config or use default
        dropout_rate = config.get("dropout_rate", 0.1)

        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_dim"])
        # Add embedding dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)

        head_dim = config["hidden_dim"] // config["num_heads"]

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_norm": nn.LayerNorm(config["hidden_dim"]),
                        "attention": MultiHeadLatentAttention(
                            hidden_dim=config["hidden_dim"],
                            num_heads=config["num_heads"],
                            head_dim=head_dim,
                            kv_compression_dim=config["kv_compression_dim"],
                            query_compression_dim=config["query_compression_dim"],
                            rope_dim=config["rope_dim"],
                            dropout_rate=dropout_rate,
                        ),
                        "moe_norm": nn.LayerNorm(config["hidden_dim"]),
                        "moe": DeepSeekMoE(
                            hidden_dim=config["hidden_dim"],
                            num_experts=config["num_experts"],
                            top_k=config["activated_experts"],
                            dropout_rate=dropout_rate,
                        ),
                    }
                )
                for _ in range(config["num_layers"])
            ]
        )

        self.final_norm = nn.LayerNorm(config["hidden_dim"])
        # Add final dropout
        self.final_dropout = nn.Dropout(dropout_rate)

        self.output_head = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.mtp = MultiTokenPrediction(
            config["hidden_dim"],
            config["vocab_size"],
            depth=1,
            dropout_rate=dropout_rate,
        )

        mask = torch.triu(
            torch.ones(
                config["context_length"], config["context_length"], dtype=torch.bool
            ),
            diagonal=1,
        )
        cos, sin = RotaryPositionalEmbedding.compute_angles(
            base=config["rope_base"],
            head_dim=config["rope_dim"],
            ctx_len=config["context_length"],
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embedding layer
        x = self.embedding(input_ids)
        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Process through transformer layers
        for layer in self.layers:
            # Attention block
            attn_input = layer["attn_norm"](x)
            # Center and shift normalization
            attn_input = attn_input - attn_input.mean(dim=-1, keepdim=True) + 1.0
            attn_output = layer["attention"](attn_input, self.cos, self.sin, attention_mask)
            x = x + attn_output

            # MoE block
            moe_input = layer["moe_norm"](x)
            moe_output = layer["moe"](moe_input)
            x = x + moe_output

        # Final normalization
        x = self.final_norm(x)
        # Apply final dropout
        x = self.final_dropout(x)

        # Main logits from final hidden state
        logits = self.output_head(x)

        # During training or when explicitly requested, also compute MTP predictions
        if (self.training and target_ids is not None) or not self.training:
            # Get multi-token predictions
            mtp_outputs = self.mtp(x)
            return logits, mtp_outputs

        return logits

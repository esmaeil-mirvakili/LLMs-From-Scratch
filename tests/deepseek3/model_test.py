import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import pytest

from llms_implementation.deepseek3.model import DeepSeekV3
from llms_implementation.deepseek3.attention import MultiHeadLatentAttention
from llms_implementation.deepseek3.moe import DeepSeekMoE
from llms_implementation.deepseek3.mtp import MultiTokenPrediction


def _small_config():
    return {
        "vocab_size": 48,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "context_length": 16,
        "kv_compression_dim": 16,
        "query_compression_dim": 16,
        "rope_dim": 8,
        "num_experts": 4,
        "activated_experts": 2,
        "dropout_rate": 0.0,
        "rope_base": 10000,
    }


def test_initialization_builds_expected_components():
    cfg = _small_config()
    model = DeepSeekV3(cfg)

    assert model.embedding.num_embeddings == cfg["vocab_size"]
    assert model.embedding.embedding_dim == cfg["hidden_dim"]
    assert isinstance(model.embedding_dropout, torch.nn.Dropout)

    assert len(model.layers) == cfg["num_layers"]
    for layer in model.layers:
        assert isinstance(layer["attention"], MultiHeadLatentAttention)
        assert isinstance(layer["moe"], DeepSeekMoE)
        assert isinstance(layer["attn_norm"], torch.nn.LayerNorm)
        assert isinstance(layer["moe_norm"], torch.nn.LayerNorm)

    assert isinstance(model.final_norm, torch.nn.LayerNorm)
    assert isinstance(model.final_dropout, torch.nn.Dropout)
    assert isinstance(model.output_head, torch.nn.Linear)
    assert isinstance(model.mtp, MultiTokenPrediction)

    assert model.mask.shape == (
        cfg["context_length"],
        cfg["context_length"],
    )
    assert model.cos.shape[0] == cfg["context_length"]
    assert model.sin.shape == model.cos.shape


def test_forward_output_shapes_train_mode():
    torch.manual_seed(0)
    cfg = _small_config()
    model = DeepSeekV3(cfg)
    model.train()

    batch, seq_len = 2, 8
    input_ids = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    logits, mtp_logits = model(input_ids, attention_mask=None, target_ids=input_ids)

    assert logits.shape == (batch, seq_len, cfg["vocab_size"])
    assert mtp_logits.shape == (batch, model.mtp.depth, seq_len, cfg["vocab_size"])


def test_forward_output_shapes_eval_mode():
    torch.manual_seed(1)
    cfg = _small_config()
    model = DeepSeekV3(cfg)
    model.eval()

    batch, seq_len = 1, 6
    input_ids = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    logits, mtp_logits = model(input_ids)

    assert logits.shape == (batch, seq_len, cfg["vocab_size"])
    assert mtp_logits.shape == (batch, model.mtp.depth, seq_len, cfg["vocab_size"])


def test_attention_mask_is_respected():
    torch.manual_seed(2)
    cfg = _small_config()
    model = DeepSeekV3(cfg)
    model.eval()

    batch, seq_len = 1, 5
    input_ids = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    # mask the final token entirely
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    attention_mask[:, -1] = False

    logits_masked, _ = model(input_ids, attention_mask=attention_mask)
    logits_full, _ = model(input_ids, attention_mask=None)

    assert not torch.allclose(logits_masked, logits_full)


def test_requires_target_ids_for_mtp_in_train_mode():
    torch.manual_seed(3)
    cfg = _small_config()
    model = DeepSeekV3(cfg)
    model.train()

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 4))

    logits_only = model(input_ids, attention_mask=None, target_ids=None)
    assert logits_only.shape == (1, 4, cfg["vocab_size"])


def test_backward_pass_updates_parameters():
    torch.manual_seed(4)
    cfg = _small_config()
    model = DeepSeekV3(cfg)

    input_ids = torch.randint(0, cfg["vocab_size"], (2, 6))
    logits, mtp_logits = model(input_ids, target_ids=input_ids)
    loss = logits.mean() + mtp_logits.mean()
    loss.backward()

    assert model.embedding.weight.grad is not None
    for layer in model.layers:
        assert layer["attention"].query_up.weight.grad is not None
        assert layer["attention"].value_up.weight.grad is not None
        for param in layer["moe"].parameters():
            if param.requires_grad:
                assert param.grad is not None


@pytest.mark.parametrize("dropout_rate", [0.0, 0.3])
def test_dropout_rate_reflected_in_submodules(dropout_rate):
    cfg = _small_config()
    cfg["dropout_rate"] = dropout_rate
    model = DeepSeekV3(cfg)

    assert model.embedding_dropout.p == dropout_rate
    assert model.final_dropout.p == dropout_rate
    for layer in model.layers:
        assert layer["attention"].dropout.p == dropout_rate
        assert layer["moe"].dropout.p == dropout_rate
    assert model.mtp.dropout.p == dropout_rate


def test_context_length_buffer_slicing():
    torch.manual_seed(5)
    cfg = _small_config()
    model = DeepSeekV3(cfg)
    model.eval()

    shorter_seq = 5
    input_ids = torch.randint(0, cfg["vocab_size"], (1, shorter_seq))
    logits, mtp_logits = model(input_ids)

    assert logits.shape[1] == shorter_seq
    assert mtp_logits.shape[2] == shorter_seq

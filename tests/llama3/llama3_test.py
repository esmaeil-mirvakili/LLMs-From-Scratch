import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

from llms_implementation.llama3.model import (
    Llama3Model,
    RMSNorm,
    SiLU,
    TransformerBlock,
)
from llms_implementation.rope import RotaryPositionalEmbedding


def make_cfg():
    return {
        "vocab_size": 32,
        "emb_dim": 16,
        "hidden_dim": 64,
        "n_layers": 2,
        "context_length": 8,
        "n_heads": 4,
        "num_kv_groups": 2,
        "rope_base": 10_000,
        "dtype": torch.float32,
    }


def test_llama3_initialization_registers_expected_components():
    cfg = make_cfg()
    model = Llama3Model(cfg)

    assert isinstance(model.final_norm, RMSNorm)
    assert isinstance(model.trf_blocks, torch.nn.ModuleList)
    assert len(model.trf_blocks) == cfg["n_layers"]
    assert all(isinstance(block, TransformerBlock) for block in model.trf_blocks)

    context = cfg["context_length"]
    head_dim = cfg["emb_dim"] // cfg["n_heads"]

    expected_mask = torch.triu(
        torch.ones(context, context, dtype=torch.bool), diagonal=1
    )
    assert torch.equal(model.mask, expected_mask)
    assert model.cos.shape == (context, head_dim)
    assert model.sin.shape == (context, head_dim)

    ref_cos, ref_sin = RotaryPositionalEmbedding.compute_angles(
        base=cfg["rope_base"],
        head_dim=head_dim,
        ctx_len=context,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=True,
        rotation_factor=1.0,
        dtype=torch.float32,
    )
    assert torch.allclose(model.cos, ref_cos)
    assert torch.allclose(model.sin, ref_sin)
    assert model.out_head.weight.data_ptr() == model.emb_dict.weight.data_ptr()


def test_llama3_forward_output_shape_and_weight_tying():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = Llama3Model(cfg)

    batch = 3
    seq_len = 8
    tokens = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    logits = model(tokens)

    assert logits.shape == (batch, seq_len, cfg["vocab_size"])
    assert logits.dtype == torch.float32
    assert model.out_head.weight.data_ptr() == model.emb_dict.weight.data_ptr()


def test_llama3_backward_pass_computes_gradients():
    torch.manual_seed(1)
    cfg = make_cfg()
    model = Llama3Model(cfg)

    batch = 2
    seq_len = 8
    tokens = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    targets = torch.randint(0, cfg["vocab_size"], (batch, seq_len))

    logits = model(tokens)
    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
    loss.backward()

    assert model.emb_dict.weight.grad is not None
    first_block = model.trf_blocks[0]
    assert first_block.att.w_queries.weight.grad is not None
    assert first_block.ffn.lin1.weight.grad is not None
    assert model.final_norm.scale.grad is not None


def test_llama3_handles_variable_sequence_length():
    torch.manual_seed(2)
    cfg = make_cfg()
    model = Llama3Model(cfg)

    shorter_seq = 5
    tokens = torch.randint(0, cfg["vocab_size"], (1, shorter_seq))
    logits = model(tokens)

    assert logits.shape == (1, shorter_seq, cfg["vocab_size"])
    # verify the model only used the matching slice of buffers
    assert torch.allclose(
        model.cos[:shorter_seq],
        RotaryPositionalEmbedding.compute_angles(
            base=cfg["rope_base"],
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            ctx_len=cfg["context_length"],
            smooth_scaling_cfg=None,
            ntk_aware_scaling=True,
            rotation_factor=1.0,
            dtype=torch.float32,
        )[0][:shorter_seq],
    )


def test_rmsnorm_matches_manual_rms_normalization():
    torch.manual_seed(3)
    emb_dim = 6
    module = RMSNorm(emb_dim)
    x = torch.randn(4, emb_dim)

    out = module(x)

    expected = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + module.eps)
    expected = module.scale * expected
    assert torch.allclose(out, expected, atol=1e-6)


def test_silu_behaviour():
    module = SiLU()
    x = torch.tensor([-2.0, 0.0, 2.0])
    expected = x * torch.sigmoid(x)
    assert torch.allclose(module(x), expected, atol=1e-6)


def test_transformer_block_runs_attention_and_ffn_paths():
    torch.manual_seed(5)
    cfg = {
        "emb_dim": 16,
        "hidden_dim": 32,
        "dtype": torch.float32,
        "n_heads": 4,
        "num_kv_groups": 2,
        "context_length": 6,
    }

    block = TransformerBlock(cfg)
    x = torch.randn(2, cfg["context_length"], cfg["emb_dim"])
    mask = torch.triu(
        torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool),
        diagonal=1,
    )
    cos, sin = RotaryPositionalEmbedding.compute_angles(
        base=10_000,
        head_dim=cfg["emb_dim"] // cfg["n_heads"],
        ctx_len=cfg["context_length"],
        smooth_scaling_cfg=None,
        ntk_aware_scaling=True,
        rotation_factor=1.0,
        dtype=torch.float32,
    )

    out = block(x, mask, cos, sin)

    assert out.shape == x.shape
    (out.sum()).backward()
    assert block.att.w_queries.weight.grad is not None
    assert block.ffn.lin1.weight.grad is not None

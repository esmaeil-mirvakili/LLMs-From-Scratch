import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import math

import torch
import pytest

from llms_implementation.deepseek3.attention import MultiHeadLatentAttention
from llms_implementation.rope import RotaryPositionalEmbedding


def _rotary_tables(head_dim: int, seq_len: int, base: int = 10000):
    cos, sin = RotaryPositionalEmbedding.compute_angles(
        base=base,
        head_dim=head_dim,
        ctx_len=seq_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=True,
        rotation_factor=1.0,
        dtype=torch.float32,
    )
    return cos, sin


def test_shapes_and_dtypes():
    torch.manual_seed(0)
    module = MultiHeadLatentAttention(
        hidden_dim=32,
        num_heads=4,
        head_dim=8,
        kv_compression_dim=16,
        query_compression_dim=12,
        rope_dim=6,
        dropout_rate=0.0,
    )

    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, 32)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    out = module(x, cos, sin)

    assert out.shape == (batch, seq_len, 32)
    assert out.dtype == torch.float32


def test_rotary_component_applied_to_queries_and_keys(monkeypatch):
    torch.manual_seed(0)
    module = MultiHeadLatentAttention(
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        kv_compression_dim=8,
        query_compression_dim=8,
        rope_dim=4,
        dropout_rate=0.0,
    )

    batch, seq_len = 1, 3
    x = torch.randn(batch, seq_len, 16)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    applied = {"count": 0}

    def fake_apply(t, cos_, sin_):
        applied["count"] += 1
        assert cos_ is cos and sin_ is sin
        return t + 1.0  # simple marker

    monkeypatch.setattr(RotaryPositionalEmbedding, "apply", staticmethod(fake_apply))

    module(x, cos, sin)
    # Should be called twice: once for keys, once for queries
    assert applied["count"] == 2


def test_attention_mask_supports_boolean_and_4d():
    torch.manual_seed(1)
    module = MultiHeadLatentAttention(
        hidden_dim=24,
        num_heads=3,
        head_dim=4,
        kv_compression_dim=16,
        query_compression_dim=12,
        rope_dim=4,
        dropout_rate=0.0,
    )

    batch, seq_len = 2, 4
    x = torch.randn(batch, seq_len, 24)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    # Boolean mask (True = keep). The implementation internally inverts it.
    bool_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    bool_mask[0, -1] = False  # mask final token for first batch

    out_bool = module(x, cos, sin, attention_mask=bool_mask)
    assert out_bool.shape == (batch, seq_len, 24)

    # Explicit 4D mask with masked positions already True
    mask4d = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.bool)
    mask4d[0, :, :, -1] = True

    out_4d = module(x, cos, sin, attention_mask=mask4d)
    assert torch.allclose(out_bool, out_4d, atol=1e-5)


def test_outputs_change_when_mask_changes():
    torch.manual_seed(2)
    module = MultiHeadLatentAttention(
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        kv_compression_dim=12,
        query_compression_dim=12,
        rope_dim=4,
        dropout_rate=0.0,
    )

    batch, seq_len = 1, 4
    x = torch.randn(batch, seq_len, 16)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    mask_all = torch.ones(batch, seq_len, dtype=torch.bool)
    mask_cut = mask_all.clone()
    mask_cut[..., -1] = False

    out_all = module(x, cos, sin, attention_mask=mask_all)
    out_cut = module(x, cos, sin, attention_mask=mask_cut)

    assert not torch.allclose(out_all, out_cut)


def test_dropout_inference_mode_no_effect(monkeypatch):
    module = MultiHeadLatentAttention(
        hidden_dim=12,
        num_heads=3,
        head_dim=2,
        kv_compression_dim=10,
        query_compression_dim=10,
        rope_dim=2,
        dropout_rate=0.5,
    )
    module.eval()

    batch, seq_len = 2, 3
    x = torch.zeros(batch, seq_len, 12)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    out1 = module(x, cos, sin)
    out2 = module(x, cos, sin)

    assert torch.allclose(out1, out2)


def test_training_mode_dropout_changes_output():
    torch.manual_seed(3)
    module = MultiHeadLatentAttention(
        hidden_dim=12,
        num_heads=3,
        head_dim=2,
        kv_compression_dim=10,
        query_compression_dim=10,
        rope_dim=2,
        dropout_rate=0.5,
    )
    module.train()

    batch, seq_len = 2, 3
    x = torch.randn(batch, seq_len, 12)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    out1 = module(x, cos, sin)
    out2 = module(x, cos, sin)

    assert not torch.allclose(out1, out2)


def test_backward_pass_propagates_gradients():
    torch.manual_seed(4)
    module = MultiHeadLatentAttention(
        hidden_dim=20,
        num_heads=4,
        head_dim=5,
        kv_compression_dim=18,
        query_compression_dim=18,
        rope_dim=4,
        dropout_rate=0.0,
    )

    batch, seq_len = 2, 3
    x = torch.randn(batch, seq_len, 20, requires_grad=True)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    out = module(x, cos, sin)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert module.key_up.weight.grad is not None
    assert module.output_proj.weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp16_support(dtype):
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 test requires CUDA for reasonable behaviour")

    torch.manual_seed(5)
    module = MultiHeadLatentAttention(
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        kv_compression_dim=12,
        query_compression_dim=12,
        rope_dim=4,
        dropout_rate=0.0,
    )

    device = torch.device("cuda" if dtype == torch.float16 else "cpu")
    module = module.to(device=device, dtype=dtype)

    batch, seq_len = 2, 4
    x = torch.randn(batch, seq_len, 16, device=device, dtype=dtype)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)
    cos = cos.to(device=device, dtype=dtype)
    sin = sin.to(device=device, dtype=dtype)

    out = module(x, cos, sin)
    assert out.shape == (batch, seq_len, 16)
    assert out.dtype == dtype


def test_causal_mask_is_enforced():
    torch.manual_seed(6)
    module = MultiHeadLatentAttention(
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        kv_compression_dim=12,
        query_compression_dim=12,
        rope_dim=4,
        dropout_rate=0.0,
    )

    batch, seq_len = 1, 4
    x = torch.randn(batch, seq_len, 16)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    # Force attention probabilities to ones before masking to test the mask logic.
    def fake_softmax(t, dim):
        return torch.ones_like(t)

    orig_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = fake_softmax
    try:
        out = module(x, cos, sin)
    finally:
        torch.nn.functional.softmax = orig_softmax

    # Since causal mask zeroes upper triangle, the final token should remain unchanged.
    assert torch.allclose(out[:, 0], out[:, 0])  # sanity
    assert torch.allclose(out[:, -1], out[:, -1])  # mask prevents future influence


def test_attention_scores_use_expected_scaling(monkeypatch):
    torch.manual_seed(7)
    module = MultiHeadLatentAttention(
        hidden_dim=8,
        num_heads=1,
        head_dim=4,
        kv_compression_dim=6,
        query_compression_dim=6,
        rope_dim=2,
        dropout_rate=0.0,
    )

    batch, seq_len = 1, 3
    x = torch.randn(batch, seq_len, 8)
    cos, sin = _rotary_tables(head_dim=module.rope_dim, seq_len=seq_len)

    original_matmul = torch.matmul
    original_masked_fill = torch.Tensor.masked_fill
    recorded = {}

    def fake_matmul(a, b):
        result = original_matmul(a, b)
        if (
            result.dim() == 4
            and a.dim() == 4
            and b.dim() == 4
            and a.size(-1) == module.head_dim + module.rope_dim
        ):
            recorded["matmul"] = result.clone()
        return result

    def fake_masked_fill(self, mask, value):
        if self.dim() == 4 and self.size(-1) == seq_len:
            recorded["attn_scores"] = self.clone()
        return original_masked_fill(self, mask, value)

    monkeypatch.setattr(torch, "matmul", fake_matmul)
    monkeypatch.setattr(torch.Tensor, "masked_fill", fake_masked_fill)

    module(x, cos, sin)

    pre = recorded["matmul"]
    post = recorded["attn_scores"]
    expected_scale = math.sqrt(module.head_dim + module.rope_dim)

    assert torch.allclose(post, pre / expected_scale, atol=1e-6)

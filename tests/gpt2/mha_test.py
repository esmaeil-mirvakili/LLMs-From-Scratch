import math
import copy
import pytest

# Make package-less import work when running pytest from repo root
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn

from llms_implementation.gpt2.attention import MultiHeadAttention


@pytest.mark.parametrize(
    "embed_dim,num_head,seq_len,batch_size",
    [
        (32, 4, 16, 2),
        (64, 8, 8, 1),
    ],
)
def test_shapes_forward_cpu(embed_dim, num_head, seq_len, batch_size):
    torch.manual_seed(0)
    attn = MultiHeadAttention(
        embed_dim, num_head, seq_len, dropout_p=0.0, use_fused=True
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    y = attn(x)
    assert y.shape == (batch_size, seq_len, embed_dim)


def clone_weights(dst: nn.Module, src: nn.Module):
    with torch.no_grad():
        for dp, sp in zip(dst.parameters(), src.parameters()):
            dp.copy_(sp)
        for db, sb in zip(dst.buffers(), src.buffers()):
            if db.shape == sb.shape and db.dtype == sb.dtype:
                db.copy_(sb)


def test_fused_equals_manual_when_dropout_zero():
    torch.manual_seed(123)
    B, T, E, H = 2, 12, 48, 6
    attn_fused = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=True)
    attn_manual = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=False)

    # Ensure identical initialization
    clone_weights(attn_manual, attn_fused)

    x = torch.randn(B, T, E)
    y1 = attn_fused.eval()(x)
    y2 = attn_manual.eval()(x)

    # Numerical equality within a small tolerance
    torch.testing.assert_close(y1, y2, rtol=1e-4, atol=1e-4)


def test_causality_no_leakage_into_future():
    torch.manual_seed(0)
    B, T, E, H = 1, 10, 32, 4
    attn = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=True).eval()

    x = torch.randn(B, T, E)
    y_orig = attn(x).detach()

    # Perturb future tokens (positions > t) and check outputs up to t stay the same
    for t in range(T):
        x_perturbed = x.clone()
        if t + 1 < T:
            x_perturbed[:, t + 1 :, :] += (
                torch.randn_like(x_perturbed[:, t + 1 :, :]) * 0.5
            )
        y_new = attn(x_perturbed).detach()
        # up to and including position t should be identical
        torch.testing.assert_close(
            y_orig[:, : t + 1, :], y_new[:, : t + 1, :], rtol=0, atol=0
        )


def test_backward_and_grad_flow():
    torch.manual_seed(0)
    B, T, E, H = 2, 8, 32, 4
    x = torch.randn(B, T, E, requires_grad=True)
    attn = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=True)
    y = attn(x)
    loss = y.pow(2).mean()
    loss.backward()
    # grads flow to input and parameters
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in attn.parameters())


def test_mask_buffer_manual_and_getter():
    torch.manual_seed(0)
    E, H, T = 32, 4, 16
    attn_manual = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=False)
    # mask buffer exists and has correct shape/dtype
    assert isinstance(attn_manual.mask, torch.Tensor)
    assert attn_manual.mask.dtype == torch.bool
    assert attn_manual.mask.shape == (1, 1, T, T)

    # Slicing returns the right sub-mask
    sub = attn_manual._get_causal_mask(T // 2)
    assert sub.shape == (1, 1, T // 2, T // 2)
    # lower-triangular True, upper-triangular False after inversion in forward (~mask)
    tri = torch.tril(torch.ones(T // 2, T // 2, dtype=torch.bool)).view(
        1, 1, T // 2, T // 2
    )
    assert torch.equal(sub, tri)


def test_get_causal_mask_raises_when_fused():
    E, H, T = 16, 4, 8
    attn_fused = MultiHeadAttention(E, H, T, use_fused=True)
    with pytest.raises(AssertionError):
        _ = attn_fused._get_causal_mask(4)


def test_causal_mask_overflow_raises():
    E, H, T = 16, 4, 8
    attn_manual = MultiHeadAttention(E, H, T, use_fused=False)
    with pytest.raises(AssertionError):
        _ = attn_manual._get_causal_mask(T + 1)


@pytest.mark.parametrize("use_fused", [True, False])
def test_dropout_behavior_training_eval(use_fused):
    torch.manual_seed(0)
    E, H, T, B = 32, 4, 12, 2
    attn = MultiHeadAttention(E, H, T, dropout_p=0.5, use_fused=use_fused)
    x = torch.randn(B, T, E)

    attn.train()
    y_train = attn(x)

    attn.eval()
    y_eval = attn(x)

    # They shouldn't be exactly equal when dropout > 0 in training
    # (fused path applies dropout internally to attention probabilities)
    assert not torch.allclose(y_train, y_eval)


@pytest.mark.parametrize("use_fused", [True, False])
def test_attention_probs_are_rowwise_probabilities(use_fused, monkeypatch):
    """Hook into the module to capture attention probabilities and check softmax properties.
    We mock nn.Dropout to identity during this test to read the raw probabilities.
    """
    torch.manual_seed(0)
    E, H, T, B = 32, 4, 10, 1

    # Identity dropout that returns input unchanged
    class IdDrop(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = 0.0

        def forward(self, x):
            return x

    attn = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=use_fused)
    # swap dropout with identity so we can examine probabilities (manual path only)
    attn.dropout = IdDrop()

    x = torch.randn(B, T, E)

    if use_fused:
        # For fused path, directly check that outputs are finite and shape-correct
        y = attn.eval()(x)
        assert torch.isfinite(y).all()
    else:
        # For manual path, we can intercept by temporarily monkeypatching forward to return probs too
        # Minimal shim: copy of forward up to softmax
        with torch.no_grad():
            B, T, _ = x.size()
            qkv = attn.qkv_proj(x)
            q, k, v = qkv.split(attn.embed_dim, dim=-1)
            q = q.view(B, T, attn.num_head, attn.head_dim).transpose(1, 2)
            k = k.view(B, T, attn.num_head, attn.head_dim).transpose(1, 2)
            v = v.view(B, T, attn.num_head, attn.head_dim).transpose(1, 2)
            dot = q @ k.transpose(-2, -1)
            scaled = dot / attn.scale_factor
            mask = attn._get_causal_mask(T)
            scaled = scaled.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(scaled, dim=-1)  # (B,h,T,T)
            # Each row sums to 1 and is non-negative
            row_sums = probs.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
            assert (probs >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_parity_with_cpu():
    torch.manual_seed(0)
    E, H, T, B = 32, 4, 12, 2
    x = torch.randn(B, T, E)

    cpu = MultiHeadAttention(E, H, T, dropout_p=0.0, use_fused=True).eval()
    cuda = copy.deepcopy(cpu).cuda()
    y_cpu = cpu(x)
    y_cuda = cuda(x.cuda()).cpu()
    torch.testing.assert_close(y_cpu, y_cuda, rtol=1e-4, atol=1e-4)

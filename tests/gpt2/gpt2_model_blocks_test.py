# tests/model_blocks_test.py
import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from llms_implementation.gpt2.model import (
    GELU,
    LayerNorm,
    FeedForwardNetwork,
    TransformerBlock,
)


# ----------------------------- GELU -----------------------------
@pytest.mark.parametrize("use_erf", [True, False])
def test_gelu_matches_torch_reference(use_erf):
    torch.manual_seed(0)
    gelu = GELU(use_erf=use_erf)
    x = torch.randn(128, 64)
    y = gelu(x)
    # PyTorch reference
    if use_erf:
        y_ref = torch.nn.functional.gelu(x)  # exact/erf variant
    else:
        y_ref = torch.nn.functional.gelu(x, approximate="tanh")
    torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_erf", [True, False])
def test_gelu_gradcheck(use_erf):
    gelu = GELU(use_erf=use_erf).double()
    x = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
    assert gradcheck(lambda t: gelu(t), (x,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("dim,eps", [(8, 1e-5)])
def test_layernorm_gradcheck(dim, eps):
    ln = LayerNorm(dim, eps=eps).double()
    x = torch.randn(3, 4, dim, dtype=torch.double, requires_grad=True)

    def fn(t):
        return ln(t)

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-6)


# --------------------------- LayerNorm ---------------------------
@pytest.mark.parametrize("dim,eps", [(32, 1e-5), (64, 1e-6)])
def test_layernorm_zero_mean_unit_var_with_learnable_params(dim, eps):
    torch.manual_seed(0)
    ln = LayerNorm(dim, eps=eps)
    x = torch.randn(4, 10, dim)
    y = ln(x)

    # Per-position stats over the last dimension
    mean = y.mean(dim=-1)
    std = y.std(dim=-1)

    # Close to zero mean and unit variance
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)

    # Parameters exist and are learnable
    assert isinstance(ln.gamma, nn.Parameter) and ln.gamma.requires_grad
    assert isinstance(ln.beta, nn.Parameter) and ln.beta.requires_grad

    # Gradient check
    y.sum().backward()
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None


# ---------------------- FeedForwardNetwork -----------------------
@pytest.mark.parametrize("E", [16, 48])
def test_ffn_shape_grad_and_marker(E):
    torch.manual_seed(0)
    ffn = FeedForwardNetwork(E)
    x = torch.randn(2, 7, E, requires_grad=True)
    y = ffn(x)

    # Shape preservation (project back to embed dim)
    assert y.shape == x.shape

    # Gradients flow
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Attribute marker present as in the given code
    assert hasattr(ffn.fc2, "SCALE_INIT") and ffn.fc2.SCALE_INIT == 1


# ------------------------ TransformerBlock ----------------------
class ProbeFFN(nn.Module):
    """FFN stub that records its input and returns zeros of same shape."""

    def __init__(self, E):
        super().__init__()
        self.last_in = None
        self.E = E

    def forward(self, x):
        self.last_in = x
        return torch.zeros_like(x)


def test_transformerblock_shapes_and_ffn_input_matches_codepath():
    """
    According to the provided code, the FFN receives `x` AFTER attention (not ln2(x)).
    This test asserts that behavior exactly.
    """
    torch.manual_seed(0)
    E, H, T = 32, 4, 8
    # Turn off attention dropout (if any) for determinism via config passthrough
    attn_config = {"attn_pdrop": 0.0, "use_fused": True}
    block = TransformerBlock(E, H, T, attn_config=attn_config)

    # Replace FFN with a probe to inspect its input
    probe = ProbeFFN(E)
    block.ffn = probe

    x = torch.randn(2, T, E)
    out = block(x)

    # Output shape is preserved by residual connection
    assert out.shape == x.shape

    # Expected FFN input per the given code:
    # out = ln1(x); x = x + attn(out); out = ln2(x); return x + ffn(x)
    with torch.no_grad():
        expected_ffn_input = x + block.causal_attn(block.ln1(x))

    assert probe.last_in is not None, "FFN was not called"
    torch.testing.assert_close(probe.last_in, expected_ffn_input, rtol=0, atol=0)


def test_transformerblock_backward_grads_flow():
    torch.manual_seed(0)
    E, H, T = 24, 4, 6
    attn_config = {"attn_pdrop": 0.0, "use_fused": True}
    block = TransformerBlock(E, H, T, attn_config=attn_config)
    x = torch.randn(3, T, E, requires_grad=True)

    y = block(x)
    loss = (y**2).mean()
    loss.backward()

    # Gradients reach the input
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Some parameter grads exist
    has_grads = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()
    )
    assert has_grads


def test_block_determinism_with_dropout_off():
    torch.manual_seed(123)
    E, H, T = 32, 4, 8
    block = TransformerBlock(
        E, H, T, attn_config={"attn_pdrop": 0.0, "use_fused": True}
    ).eval()
    x = torch.randn(2, T, E)
    y1 = block(x)
    y2 = block(x)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)


def test_block_causality_invariant():
    torch.manual_seed(0)
    E, H, T = 32, 4, 10
    block = TransformerBlock(
        E, H, T, attn_config={"attn_pdrop": 0.0, "use_fused": True}
    ).eval()
    x = torch.randn(1, T, E)
    y = block(x).detach()
    # Perturb future tokens and ensure outputs up to t are unchanged
    for t in range(T):
        x2 = x.clone()
        if t + 1 < T:
            x2[:, t + 1 :, :] += torch.randn_like(x2[:, t + 1 :, :]) * 0.3
        y2 = block(x2).detach()
        torch.testing.assert_close(y[:, : t + 1, :], y2[:, : t + 1, :], rtol=0, atol=0)

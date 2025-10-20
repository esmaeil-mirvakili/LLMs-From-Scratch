import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import pytest

from llms_implementation.deepseek3.mtp import MultiTokenPrediction


def _manual_forward(module: MultiTokenPrediction, hidden_states: torch.Tensor):
    preds = []
    current = hidden_states
    for linear in module.proj_layers:
        projected = linear(current)
        projected = module.dropout(projected)
        normalized = module.norm(projected)
        logits = module.output_head(normalized)
        preds.append(logits)
        current = projected
    return torch.stack(preds, dim=1)


def test_shapes_and_depth_stack():
    torch.manual_seed(0)
    module = MultiTokenPrediction(hidden_dim=16, vocab_size=32, depth=3, dropout_rate=0.0)
    module.eval()

    batch, seq_len = 2, 4
    hidden = torch.randn(batch, seq_len, 16)
    out = module(hidden)

    assert out.shape == (batch, module.depth, seq_len, 32)


def test_forward_matches_manual_computation():
    torch.manual_seed(1)
    module = MultiTokenPrediction(hidden_dim=8, vocab_size=10, depth=2, dropout_rate=0.0)
    module.eval()

    hidden = torch.randn(3, 5, 8)
    out = module(hidden)
    manual = _manual_forward(module, hidden)

    assert torch.allclose(out, manual, atol=1e-6)


def test_dropout_behavior():
    torch.manual_seed(2)
    module = MultiTokenPrediction(hidden_dim=8, vocab_size=10, depth=2, dropout_rate=0.5)
    hidden = torch.randn(2, 3, 8)

    module.train()
    out1 = module(hidden)
    out2 = module(hidden)
    assert not torch.allclose(out1, out2)

    module.eval()
    out3 = module(hidden)
    out4 = module(hidden)
    assert torch.allclose(out3, out4)


def test_gradients_flow_through_all_layers():
    torch.manual_seed(3)
    module = MultiTokenPrediction(hidden_dim=6, vocab_size=9, depth=3, dropout_rate=0.0)
    hidden = torch.randn(2, 4, 6, requires_grad=True)

    out = module(hidden)
    loss = out.mean()
    loss.backward()

    assert hidden.grad is not None
    for linear in module.proj_layers:
        assert linear.weight.grad is not None
    assert module.output_head.weight.grad is not None


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_depth_parameter_changes_output_axis(depth):
    torch.manual_seed(4)
    module = MultiTokenPrediction(hidden_dim=4, vocab_size=7, depth=depth, dropout_rate=0.0)
    hidden = torch.randn(1, 2, 4)

    out = module(hidden)
    assert out.shape[1] == depth


def test_layernorm_shared_across_depth():
    torch.manual_seed(5)
    module = MultiTokenPrediction(hidden_dim=4, vocab_size=6, depth=2, dropout_rate=0.0)
    hidden = torch.randn(1, 3, 4)

    _ = module(hidden)
    norm_id = id(module.norm)
    projected_ids = [id(module.proj_layers[i]) for i in range(module.depth)]

    assert len(set(projected_ids)) == module.depth
    for _ in range(3):
        module(hidden)
        assert id(module.norm) == norm_id

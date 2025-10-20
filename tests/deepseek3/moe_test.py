import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import pytest

from llms_implementation.deepseek3.moe import ExpertFFN, DeepSeekMoE


class ScaleExpert(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def test_expert_ffn_matches_manual_ops():
    torch.manual_seed(0)
    hidden = 6
    module = ExpertFFN(hidden_dim=hidden, expansion_factor=3, dropout_rate=0.0)
    module.eval()

    x = torch.randn(4, hidden)
    with torch.no_grad():
        manual = module.down(module.gelu(module.up(x)))

    out = module(x)
    assert torch.allclose(out, manual, atol=1e-6)
    assert out.shape == x.shape


def test_expert_ffn_backward_updates_parameters():
    torch.manual_seed(1)
    module = ExpertFFN(hidden_dim=4, expansion_factor=2, dropout_rate=0.0)
    x = torch.randn(3, 4, requires_grad=True)

    out = module(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert module.up.weight.grad is not None
    assert module.down.weight.grad is not None


def _manual_moe_forward(module: DeepSeekMoE, x: torch.Tensor, bias, expert_load):
    hidden_dim = x.size(-1)
    x_flat = x.view(-1, hidden_dim)

    scores = torch.sigmoid(module.gate(x_flat) + bias)
    top_scores, top_indices = scores.topk(module.top_k, dim=-1)
    norm_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)

    combined = torch.zeros_like(x_flat)
    for i in range(module.top_k):
        expert_indices = top_indices[:, i]
        coeff = norm_scores[:, i].unsqueeze(-1)
        for expert_idx, expert in enumerate(module.experts):
            mask = expert_indices == expert_idx
            if mask.any():
                expert_inputs = x_flat[mask]
                expert_outputs = expert(expert_inputs) * coeff[mask]
                combined.index_add_(0, torch.where(mask)[0], expert_outputs)

    shared_out = module.shared_expert(x_flat) * 0.1
    combined = combined + shared_out
    combined = module.dropout(combined)

    mask_counts = torch.nn.functional.one_hot(
        top_indices, module.num_experts
    ).sum(dim=1).float()
    expert_load_counts = mask_counts.sum(dim=0)

    bias_new = bias + module.bias_update_speed * (expert_load_counts - expert_load)
    expert_load_new = expert_load + 0.1 * (expert_load_counts - expert_load)

    return combined.view_as(x), bias_new, expert_load_new


def test_deepseek_moe_matches_manual_computation():
    torch.manual_seed(2)
    module = DeepSeekMoE(hidden_dim=2, num_experts=3, top_k=2, dropout_rate=0.0)

    module.shared_expert = ScaleExpert(4.0)
    module.experts = torch.nn.ModuleList(
        [ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(3.0)]
    )
    module.dropout = torch.nn.Dropout(0.0)

    with torch.no_grad():
        w = module.gate.weight
        w.zero_()
        w[0, 0] = 10
        w[0, 1] = -10
        w[1, 0] = -10
        w[1, 1] = 10
        w[2, 0] = 10
        w[2, 1] = 10
        module.bias.fill_(0.0)
    module.bias_update_speed = 0.001

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    bias_before = module.bias.detach().clone()
    load_before = module.expert_load.detach().clone()

    expected, bias_after, load_after = _manual_moe_forward(
        module, x, bias_before, load_before
    )

    out = module(x)

    assert torch.allclose(out, expected, atol=1e-6)
    assert out.shape == x.shape

    assert torch.allclose(module.bias, bias_after, atol=1e-6)
    assert torch.allclose(module.expert_load, load_after, atol=1e-6)


def test_deepseek_moe_dropout_effects():
    torch.manual_seed(3)
    module = DeepSeekMoE(hidden_dim=4, num_experts=2, top_k=1, dropout_rate=0.5)
    x = torch.randn(2, 3, 4)

    module.train()
    out1 = module(x)
    out2 = module(x)
    assert not torch.allclose(out1, out2)

    module.eval()
    out3 = module(x)
    out4 = module(x)
    assert torch.allclose(out3, out4)


def test_deepseek_moe_backward_through_routing():
    torch.manual_seed(4)
    module = DeepSeekMoE(hidden_dim=3, num_experts=4, top_k=2, dropout_rate=0.0)
    x = torch.randn(2, 5, 3, requires_grad=True)

    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert module.gate.weight.grad is not None
    for expert in module.experts:
        for param in expert.parameters():
            assert param.grad is not None

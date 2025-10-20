import torch
import pytest

from llms_implementation.llama3.attention import GroupedQueryAttention


def _reference_forward(module, x, mask, cos, sin):
    queries = module.w_queries(x)
    keys = module.w_keys(x)
    values = module.w_values(x)

    b, seq_len, _ = x.shape
    head_dim = module.head_dim

    queries = queries.view(b, seq_len, module.num_heads, head_dim).transpose(1, 2)
    keys = keys.view(b, seq_len, module.num_kv_groups, head_dim).transpose(1, 2)
    values = values.view(b, seq_len, module.num_kv_groups, head_dim).transpose(1, 2)

    # cos=1, sin=0 in tests, so rotary application is a no-op
    keys = keys.repeat_interleave(module.num_repeat, dim=1)
    values = values.repeat_interleave(module.num_repeat, dim=1)

    att_scores = queries @ keys.mT
    att_scores = att_scores * module.att_scaling
    att_scores = att_scores.masked_fill(mask[:seq_len, :seq_len], -torch.inf)
    att_weights = torch.softmax(att_scores, dim=-1)

    ctx_tensor = att_weights @ values
    ctx_tensor = ctx_tensor.transpose(1, 2).contiguous().view(b, seq_len, module.d_out)
    return module.out_proj(ctx_tensor), att_weights


def test_invalid_configuration_raises_assertion():
    with pytest.raises(AssertionError):
        GroupedQueryAttention(d_in=8, d_out=10, num_heads=3, num_kv_groups=1)
    with pytest.raises(AssertionError):
        GroupedQueryAttention(d_in=8, d_out=8, num_heads=4, num_kv_groups=3)


def test_forward_matches_reference_implementation():
    torch.manual_seed(0)
    module = GroupedQueryAttention(
        d_in=6,
        d_out=6,
        num_heads=3,
        num_kv_groups=1,
        dtype=torch.float32,
    )

    x = torch.randn(2, 4, 6)
    seq_len = x.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    cos = torch.ones(seq_len, module.head_dim)
    sin = torch.zeros(seq_len, module.head_dim)

    out = module(x, mask, cos, sin)
    ref_out, att_weights = _reference_forward(module, x, mask, cos, sin)

    assert out.shape == (2, 4, 6)
    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(
        att_weights * torch.triu(torch.ones_like(att_weights), diagonal=1),
        torch.zeros_like(att_weights),
        atol=1e-6,
    )


def test_key_value_groups_are_repeated_per_head():
    torch.manual_seed(1)
    module = GroupedQueryAttention(
        d_in=4,
        d_out=8,
        num_heads=4,
        num_kv_groups=2,
        dtype=torch.float32,
    )

    x = torch.randn(1, 3, 4)
    seq_len = x.shape[1]

    keys = module.w_keys(x)
    keys = keys.view(1, seq_len, module.num_kv_groups, module.head_dim).transpose(1, 2)
    repeated_keys = keys.repeat_interleave(module.num_repeat, dim=1)

    for group_idx in range(module.num_kv_groups):
        expected = keys[:, group_idx]
        for repeat_idx in range(module.num_repeat):
            head_idx = group_idx * module.num_repeat + repeat_idx
            assert torch.allclose(repeated_keys[:, head_idx], expected, atol=1e-6)

    values = module.w_values(x)
    values = values.view(1, seq_len, module.num_kv_groups, module.head_dim).transpose(1, 2)
    repeated_values = values.repeat_interleave(module.num_repeat, dim=1)
    for group_idx in range(module.num_kv_groups):
        expected = values[:, group_idx]
        for repeat_idx in range(module.num_repeat):
            head_idx = group_idx * module.num_repeat + repeat_idx
            assert torch.allclose(repeated_values[:, head_idx], expected, atol=1e-6)

import math

import pytest
import torch

from llms_implementation.rope import RotaryPositionalEmbedding


def test_partial_rotation_scales_head_dim():
    assert RotaryPositionalEmbedding.partial_rotation(128, 1.0) == 128
    assert RotaryPositionalEmbedding.partial_rotation(128, 0.5) == 64
    assert RotaryPositionalEmbedding.partial_rotation(128, 0.49) == int(128 * 0.49)


def test_partial_rotation_rejects_out_of_range_factor():
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding.partial_rotation(64, 0.0)
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding.partial_rotation(64, 1.1)


def test_ntk_aware_base_scaling_matches_formula():
    theta_base = 10_000
    head_dim = 16
    ctx_len = 4096
    old_ctx_len = 2048

    scaled = RotaryPositionalEmbedding.ntk_aware_base_scaling(
        theta_base, head_dim, ctx_len, old_ctx_len
    )
    expected = theta_base * (ctx_len / old_ctx_len) ** (head_dim / (head_dim - 2))

    assert math.isclose(scaled, expected, rel_tol=1e-6)


def test_wavelength_scaling_respects_frequency_bands():
    base = 10_000
    head_dim = 8
    freq_cfg = {
        "ctx_len": 256,
        "og_ctx_len": 80,
        "alpha": 0.5,
        "beta": 1.5,
        "factor": 2.0,
    }

    scaled_theta = RotaryPositionalEmbedding.wavelength_scaling(
        base,
        head_dim,
        freq_cfg,
        ntk_aware_scaling=False,
        dtype=torch.float32,
    )

    indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
    base_theta = 1.0 / base ** (2 * indices / head_dim)
    ratio = freq_cfg["og_ctx_len"] / (2 * torch.pi / base_theta)

    expected = torch.where(
        ratio < freq_cfg["alpha"], base_theta / freq_cfg["factor"], base_theta
    )

    smooth = (
        (ratio - freq_cfg["alpha"]) / (freq_cfg["beta"] - freq_cfg["alpha"])
    ).clamp(0, 1)
    smooth_theta = (1 - smooth) * (
        base_theta / freq_cfg["factor"]
    ) + smooth * base_theta
    medium_mask = (ratio >= freq_cfg["alpha"]) & (ratio <= freq_cfg["beta"])
    expected = torch.where(medium_mask, smooth_theta, expected)

    assert torch.allclose(scaled_theta, expected, atol=1e-6)


def test_compute_angles_matches_manual_construction():
    base = 1_000
    head_dim = 6
    ctx_len = 4

    cos, sin = RotaryPositionalEmbedding.compute_angles(
        base=base,
        head_dim=head_dim,
        ctx_len=ctx_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=False,
        dtype=torch.float32,
    )

    indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
    theta = 1.0 / base ** (2 * indices / head_dim)
    positions = torch.arange(0, ctx_len, dtype=torch.float32)
    angles = torch.outer(positions, theta)
    angles = torch.cat([angles, angles], dim=-1)

    assert cos.shape == (ctx_len, head_dim)
    assert sin.shape == (ctx_len, head_dim)
    assert torch.allclose(cos, torch.cos(angles), atol=1e-6)
    assert torch.allclose(sin, torch.sin(angles), atol=1e-6)


def test_apply_matches_manual_full_rotation():
    torch.manual_seed(0)

    batch, n_head, seq_len, head_dim = 2, 3, 4, 8
    x = torch.randn(batch, n_head, seq_len, head_dim)

    cos, sin = RotaryPositionalEmbedding.compute_angles(
        base=5_000,
        head_dim=head_dim,
        ctx_len=seq_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=False,
        dtype=torch.float32,
    )

    result = RotaryPositionalEmbedding.apply(x.clone(), cos, sin)

    cos_trim = cos[:seq_len].to(x.dtype)
    sin_trim = sin[:seq_len].to(x.dtype)
    rotated = torch.cat(
        (-x[..., head_dim // 2 :], x[..., : head_dim // 2]),
        dim=-1,
    )
    expected = (
        cos_trim.unsqueeze(0).unsqueeze(0) * x
        + sin_trim.unsqueeze(0).unsqueeze(0) * rotated
    )

    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_partial_rotation_only_affects_rotated_slice():
    torch.manual_seed(1)

    batch, n_head, seq_len, head_dim = 1, 2, 3, 8
    rotation_factor = 0.5
    x = torch.randn(batch, n_head, seq_len, head_dim)

    cos, sin = RotaryPositionalEmbedding.compute_angles(
        base=3_000,
        head_dim=head_dim,
        ctx_len=seq_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=False,
        rotation_factor=rotation_factor,
        dtype=torch.float32,
    )

    result = RotaryPositionalEmbedding.apply(x.clone(), cos, sin)

    rotation_dim = cos.shape[-1]
    x_rot = x[..., :rotation_dim]
    x_rest = x[..., rotation_dim:]
    cos_trim = cos[:seq_len].to(x.dtype)
    sin_trim = sin[:seq_len].to(x.dtype)
    rotated = torch.cat(
        (-x_rot[..., rotation_dim // 2 :], x_rot[..., : rotation_dim // 2]),
        dim=-1,
    )
    expected_rot = (
        cos_trim.unsqueeze(0).unsqueeze(0) * x_rot
        + sin_trim.unsqueeze(0).unsqueeze(0) * rotated
    )
    expected = torch.cat([expected_rot, x_rest], dim=-1)

    assert torch.allclose(result, expected, atol=1e-6)
    assert torch.allclose(result[..., rotation_dim:], x[..., rotation_dim:], atol=1e-6)

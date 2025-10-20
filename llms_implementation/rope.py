import torch


class RotaryPositionalEmbedding:
    @staticmethod
    def partial_rotation(head_dim, factor):
        """
        Downscale the head dimension for partial rotation.
        Args:
            head_dim (int): Size of each attention head (must be even).
            factor (float): Fraction in (0, 1] used to shrink the head size.

        Returns:
            int: Even-sized slice of the head dimension to rotate.
        """
        assert (
            0 < factor <= 1.0
        ), "rotation factor must be greater than 0 and less than or equal to 1.0"
        return int(head_dim * factor)

    @staticmethod
    def ntk_aware_base_scaling(theta_base, head_dim, ctx_len, old_ctx_len):
        """
        Apply the NTK-aware base scaling described in the YaRN work.
        """
        return theta_base * (ctx_len / old_ctx_len) ** (head_dim / (head_dim - 2))

    @staticmethod
    def wavelength_scaling(
        base, head_dim, freq_cfg, ntk_aware_scaling=True, dtype=torch.float32
    ):
        """
        Smoothly adjust frequencies across three wavelength bands.

        Args:
            base (int): Base for frequency scaling.
            head_dim (int): Even-sized attention head dimension.
            freq_cfg (dict): Frequency scaling settings.
            ntk_aware_scaling (bool, optional): Use NTK-aware base scaling if True.
            dtype (torch.dtype, optional): Data type for the angle math.


        Returns:
            torch.Tensor: Updated theta values after banded scaling.
        """
        # Base frequencies
        if ntk_aware_scaling:
            base = RotaryPositionalEmbedding.ntk_aware_base_scaling(
                base, head_dim, freq_cfg["ctx_len"], freq_cfg["og_ctx_len"]
            )
        theta = 1 / base ** (
            2 * (torch.arange(0, head_dim // 2, dtype=dtype)) / head_dim
        )

        wavelen = 2 * torch.pi / theta

        # Ratio of context to wavelength
        ratio = freq_cfg["og_ctx_len"] / wavelen

        # Low-frequency region gets fully scaled
        scaled_theta = torch.where(
            ratio < freq_cfg["alpha"],
            theta / freq_cfg["factor"],
            theta,
        )

        # Mid frequencies blend between scaled and original values
        smooth_factor = (
            (ratio - freq_cfg["alpha"]) / (freq_cfg["beta"] - freq_cfg["alpha"])
        ).clamp(0, 1)

        # Blend both ends based on the smooth factor
        smoothed_theta = (1 - smooth_factor) * (
            theta / freq_cfg["factor"]  # scaled component
        ) + smooth_factor * theta  # unscaled component

        # Apply smoothing only to medium frequencies
        is_medium_freq = (ratio >= freq_cfg["alpha"]) & (ratio <= freq_cfg["beta"])
        final_theta = torch.where(is_medium_freq, smoothed_theta, scaled_theta)

        return final_theta

    @staticmethod
    def compute_angles(
        base,
        head_dim,
        ctx_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=True,
        rotation_factor=1.0,
        dtype=torch.float32,
    ):
        """
        Compute RoPE cosine and sine tables, optionally with YaRN scaling.

        Args:
            base (int): Base used for frequency generation.
            head_dim (int): Even-sized head dimension.
            ctx_len (int): Maximum sequence length.
            smooth_scaling_cfg (dict, optional): YaRN scaling config; None keeps vanilla RoPE.
            ntk_aware_scaling (bool, optional): Apply NTK-aware base scaling when True.
            rotation_factor (float, optional): Portion of the head to rotate.
            dtype (torch.dtype, optional): Data type for all angle work.

        Returns:
            tuple (torch.Tensor, torch.Tensor): Cosine and sine lookup tables of size (ctx_len, head_dim).
        """

        # Head dimension must split cleanly into two halves
        assert (
            head_dim % 2 == 0
        ), "head dim must be divisible by 2 as we have d/2 pairs of angles θi"
        assert (
            dtype == torch.float32
        ), "for now enforcing dtype as float32 as arg rather than .float() again"

        if rotation_factor != 1.0:
            head_dim = RotaryPositionalEmbedding.partial_rotation(
                head_dim, rotation_factor
            )

        # Choose between YaRN scaling and the standard formulation
        if smooth_scaling_cfg is not None:
            theta = RotaryPositionalEmbedding.wavelength_scaling(
                base,
                head_dim,
                smooth_scaling_cfg,
                ntk_aware_scaling,
                dtype,
            )
        else:
            theta = 1.0 / base ** (
                2 * (torch.arange(0, head_dim // 2, dtype=dtype)) / head_dim
            )

        positions = torch.arange(0, ctx_len, dtype=dtype)  # absolute position index

        # Outer product injects each position into every frequency
        angles = torch.outer(positions, theta)  # shape (ctx_len, head_dim //2)

        # Duplicate angles so the halves line up with our split dimensions
        angles = torch.cat([angles, angles], dim=-1)  # shape (ctx_len , head_dim)

        # Final lookup tables
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Needed for rotating the head dimensions later on
        return cos, sin

    @staticmethod
    def _apply_partial_rope(x, cos, sin):
        """
        Apply RoPE only to the rotated portion of x.
        """
        seq_length = x.shape[2]
        rotation_dim = cos.shape[-1]
        # Split the rotated and untouched chunks
        x_rot, x_rest = x[..., :rotation_dim], x[..., rotation_dim:]

        # Split the rotated chunk into two halves
        h1 = x_rot[..., : rotation_dim // 2]
        h2 = x_rot[..., rotation_dim // 2 :]

        # Prepare the swapped half for the sine term
        rotated = torch.concat((-h2, h1), dim=-1)
        # Trim cos and sin to the right sequence length and cast
        cos, sin = cos[:seq_length, :].to(x.dtype), sin[:seq_length, :].to(x.dtype)

        # Apply the rotation
        roped = cos * x_rot + sin * rotated

        # Reattach the untouched channels
        res = torch.cat((roped, x_rest), dim=-1)

        return res

    @staticmethod
    def apply(x, cos, sin):
        """
        Reshape and rotate x with the precomputed cos and sin tables.
        """
        _, _, seq_length, head_dim = x.shape
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we need pairs"

        # If cos shape doesn't match x's head_dim, we infer that a partial RoPE should be returned
        if head_dim != cos.shape[-1]:
            return RotaryPositionalEmbedding._apply_partial_rope(x, cos, sin)
        # Split the head into two halves
        h1 = x[..., : head_dim // 2]
        h2 = x[..., head_dim // 2 :]

        # Prepare the swapped half for the sine term
        rotated = torch.concat((-h2, h1), dim=-1)
        # Trim cos and sin to the sequence length and cast
        cos, sin = cos[:seq_length, :].to(x.dtype), sin[:seq_length, :].to(x.dtype)
        # Apply the rotation in vectorized form
        res = cos * x + sin * rotated
        return res

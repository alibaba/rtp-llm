"""DeepSeek-V4 partial RoPE with YaRN scaling.

Direct port of `inference/model.py:precompute_freqs_cis / apply_rotary_emb`.
V4 applies RoPE only to the LAST `rope_head_dim` dims of each head; the
non-RoPE dims pass through unchanged.

Two RoPE bases per model:
  - rope_theta = 10000          (main, used by SWA-only layers)
  - compress_rope_theta = 160000 (used by CSA/HCA layers' compressed branch)
"""

import math

import torch


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Returns complex cis tensor `[seqlen, dim/2]`.
    When `original_seq_len > 0`, applies YaRN frequency interpolation."""

    def find_correction_dim(num_rotations, dim_, base_, max_seq_len_):
        return (
            dim_
            * math.log(max_seq_len_ / (num_rotations * 2 * math.pi))
            / (2 * math.log(base_))
        )

    def find_correction_range(low_rot, high_rot, dim_, base_, max_seq_len_):
        low = math.floor(find_correction_dim(low_rot, dim_, base_, max_seq_len_))
        high = math.ceil(find_correction_dim(high_rot, dim_, base_, max_seq_len_))
        return max(low, 0), min(high, dim_ - 1)

    def linear_ramp_factor(min_, max_, dim_):
        if min_ == max_:
            max_ = max_ + 0.001
        linear_func = (
            torch.arange(dim_, dtype=torch.float32, device=device) - min_
        ) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)
    # vLLM DSV4 builds a fp32 cos/sin cache directly. Construct the complex
    # view from the same fp32 cos/sin values instead of torch.polar, which can
    # differ by one ULP and then persist as different bf16 RoPE cache bytes.
    freqs_cis = torch.complex(freqs.cos(), freqs.sin())
    return freqs_cis


def apply_rotary_emb_batched(
    x: torch.Tensor, freqs_cis_per_b: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place partial RoPE with PER-REQUEST freqs_cis (Stage 3B).

    Identical to :func:`apply_rotary_emb` except ``freqs_cis_per_b`` is
    ``[B, freqs_dim]`` (one row per batch entry) rather than ``[S, freqs_dim]``
    (broadcast across batch). Used by the CUDA-graph decode path where each
    request has its own ``start_pos`` and therefore its own RoPE row.

    Shapes:
        x                : ``[B, S, ..., 2k]`` (S typically 1 for decode)
        freqs_cis_per_b  : ``[B, k]`` complex64
    """
    if x.numel() == 0 or freqs_cis_per_b.numel() == 0:
        return x
    y = x
    B = x.size(0)
    S = x.size(1)
    last = x.size(-1)
    x = torch.view_as_complex(x.float().unflatten(-1, (last // 2, 2)))
    if inverse:
        freqs_cis_per_b = freqs_cis_per_b.conj()
    # freqs_cis_per_b: [B, k] -> broadcastable shape [B, S, ..., k]
    if x.ndim == 3:
        freqs_cis = freqs_cis_per_b.view(B, 1, last // 2).expand(B, S, last // 2)
    else:
        # x.ndim == 4 (B, S, H, k)
        freqs_cis = freqs_cis_per_b.view(B, 1, 1, last // 2).expand(
            B, S, x.size(2), last // 2
        )
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place partial RoPE. x: [..., S, (..., 2k)]; rotates last dim only.
    `inverse=True` applies the conjugate rotation (used on attention output).

    Empty-batch safe: if either ``x`` or the sliced ``freqs_cis`` has zero
    elements (DP rank with no local tokens, or start_pos past max_seq_len
    during warmup), return ``x`` unchanged rather than crashing in ``.view``.
    """
    if x.numel() == 0 or freqs_cis.numel() == 0:
        return x
    y = x
    # Use explicit size (last_dim // 2, 2) rather than (-1, 2).  Some
    # torch paths (dynamo/fakemode, certain warmup shapes) reject the
    # ``-1`` inferred dim to ``unflatten`` with "unknown parameter type".
    x = torch.view_as_complex(x.float().unflatten(-1, (x.size(-1) // 2, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(x.size(0), x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def apply_rotary_emb_gptj_native(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place GPT-J RoPE using the same real-valued formula as vLLM.

    ``apply_rotary_emb`` implements the same rotation via complex multiply.
    The two forms can differ by one bf16 ULP. DSv4 FP8 KV cache persists the
    RoPE tail as bf16 bytes, so the precision alignment path uses this helper
    where bit-for-bit vLLM comparison matters.
    """
    if x.numel() == 0 or freqs_cis.numel() == 0:
        return x
    half = x.size(-1) // 2
    cos = freqs_cis.real
    sin = freqs_cis.imag
    if inverse:
        sin = -sin

    rope = x.float().unflatten(-1, (half, 2))
    even = rope[..., 0]
    odd = rope[..., 1]
    if x.ndim == 3:
        cos = cos.view(x.size(0), x.size(1), half)
        sin = sin.view(x.size(0), x.size(1), half)
    elif x.ndim == 4:
        cos = cos.view(x.size(0), x.size(1), 1, half)
        sin = sin.view(x.size(0), x.size(1), 1, half)
    else:
        raise ValueError(f"unsupported RoPE tensor shape: {tuple(x.shape)}")

    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    x.copy_(rotated)
    return x

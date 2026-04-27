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
        linear_func = (torch.arange(dim_, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
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

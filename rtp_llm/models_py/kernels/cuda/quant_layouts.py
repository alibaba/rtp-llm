"""CUDA quantization layout and reference conversion helpers.

Two block sizes used across the FP8/FP4 MoE pipeline:
  - ``FP8_BLOCK = 128``: per-token-group block size for FP8 (E4M3) activation
    quantization (uses UE8M0 scale-factor packing on SM100).
  - ``FP4_BLOCK = 32``: per-row block size for FP4 weight scale factors
    (DeepGEMM ``m_grouped_fp8_fp4_*`` recipe).

The activation cast is a CUDA-graph-safe replacement
for ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True, use_packed_ue8m0=True)``
(the upstream helper does a ``.all()`` debug assertion that triggers a
CUDA->CPU sync illegal during stream capture).
"""

from typing import Optional, Tuple

import torch

FP4_BLOCK = 32
FP8_BLOCK = 128

# FP4 E2M1 lookup: raw nibble -> fp32 value.
_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def prepare_fp4_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert FP4 UE8M0 weight scale to DeepGEMM's SM100 layout.

    Routed expert checkpoints store weight scale as raw UE8M0
    ``float8_e8m0fnu``. DeepGEMM's FP8xFP4 kernels on SM100 consume the
    TMA-aligned packed ``int32`` layout. Do this once while binding weights,
    not in the GEMM hot path.
    """
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected FP4 UE8M0 scale, got {scale.dtype}")

    import deep_gemm

    scale_fp32 = scale.float()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (1, FP4_BLOCK)
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32, mn, k, (1, FP4_BLOCK), num_groups
    )


def dequantize_fp4_weight(
    weight_int8: torch.Tensor, scale_ue8m0: torch.Tensor
) -> torch.Tensor:
    """Dequantize packed FP4 weight and raw UE8M0 scale to fp32."""
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2
    weight_uint8 = weight_int8.to(torch.int32) & 0xFF
    low = weight_uint8 & 0x0F
    high = (weight_uint8 >> 4) & 0x0F
    interleaved = torch.empty(
        out_dim, in_dim, dtype=torch.int64, device=weight_int8.device
    )
    interleaved[:, 0::2] = low.long()
    interleaved[:, 1::2] = high.long()
    weight_fp32 = _FP4_LUT.to(weight_int8.device)[interleaved]
    scale_fp32 = scale_ue8m0.to(torch.float32).repeat_interleave(FP4_BLOCK, 1)
    return weight_fp32 * scale_fp32[:, :in_dim]


def dequantize_fp8_weight(
    weight_fp8: torch.Tensor, scale_ue8m0: torch.Tensor
) -> torch.Tensor:
    """Dequantize FP8 weight with raw or DeepGEMM-packed UE8M0 scale to fp32."""
    out_dim, in_dim = weight_fp8.shape
    weight_fp32 = weight_fp8.to(torch.float32)
    if scale_ue8m0.dtype == torch.int32:
        n_pad, k_block_div_4 = scale_ue8m0.shape
        k_block = k_block_div_4 * 4
        scale_bytes = scale_ue8m0.contiguous().view(torch.uint8).reshape(n_pad, k_block)
        scale_per_row = ((scale_bytes.to(torch.int32) - 127).to(torch.float32).exp2())[
            :out_dim
        ]
        scale_full = scale_per_row.repeat_interleave(FP8_BLOCK, 1)[:, :in_dim]
    else:
        scale_full = (
            scale_ue8m0.to(torch.float32)
            .repeat_interleave(FP8_BLOCK, 0)
            .repeat_interleave(FP8_BLOCK, 1)
        )[:out_dim, :in_dim]
    return weight_fp32 * scale_full


def per_token_cast_to_fp8_packed_ue8m0(
    x: torch.Tensor,
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inline ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True,
    use_packed_ue8m0=True)`` without the ``pack_ue8m0_to_int`` ``.all()``
    debug assertion. That assertion does a CUDA->CPU sync, which is illegal
    during ``cudaStreamCapture``.
    """
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    if padded_n != n:
        x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
        x_padded[:, :n] = x
    else:
        x_padded = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    bits = sf.abs().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    sf_u = (exp.clamp(1, 254) << 23).view(torch.float)
    x_fp8 = (
        (x_view * (1.0 / sf_u.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    sf_packed = (sf_u.view(torch.int) >> 23).to(torch.uint8).view(torch.int)
    return x_fp8, sf_packed


__all__ = [
    "FP4_BLOCK",
    "FP8_BLOCK",
    "dequantize_fp4_weight",
    "dequantize_fp8_weight",
    "per_token_cast_to_fp8_packed_ue8m0",
    "prepare_fp4_weight_scale_for_deepgemm",
]

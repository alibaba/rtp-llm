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
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(f"expected FP4 UE8M0 or float32 scale, got {scale.dtype}")

    import deep_gemm

    expected_shape = (mn, (k + FP4_BLOCK - 1) // FP4_BLOCK)
    if num_groups is not None:
        expected_shape = (num_groups, *expected_shape)
    if tuple(scale.shape) != expected_shape:
        raise ValueError(f"FP4 scale shape {tuple(scale.shape)} != {expected_shape}")
    scale_fp32 = scale.float()
    if scale.dtype == torch.float32:
        mantissa, exponent = torch.frexp(scale_fp32)
        if not bool(((mantissa == 0.5) & (exponent >= -126) & (exponent <= 128)).all()):
            raise ValueError(
                "FP4 float32 scales must be positive powers of two representable in UE8M0"
            )
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (1, FP4_BLOCK)
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32, mn, k, (1, FP4_BLOCK), num_groups
    )


def prepare_fp8_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert FP8 128x128 per-block scales to DeepGEMM's packed layout."""
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(
            f"expected FP8 per-block scale or packed int32, got {scale.dtype}"
        )

    import deep_gemm

    if scale.dim() < 2:
        raise ValueError(
            f"FP8 per-block scale must have at least 2 dims, got shape={tuple(scale.shape)}"
        )
    expected_m_blocks = (mn + FP8_BLOCK - 1) // FP8_BLOCK
    expected_k_blocks = (k + FP8_BLOCK - 1) // FP8_BLOCK
    if tuple(scale.shape[-2:]) != (expected_m_blocks, expected_k_blocks):
        raise ValueError(
            "FP8 mega_moe weight scale must be 128x128 per-block or already "
            f"packed int32. Got shape={tuple(scale.shape)}, expected trailing "
            f"dims=({expected_m_blocks}, {expected_k_blocks}) for mn={mn}, k={k}."
        )

    scale_fp32 = scale.float()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (FP8_BLOCK, FP8_BLOCK)
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32, mn, k, (FP8_BLOCK, FP8_BLOCK), num_groups
    )


def quantize_moe_weight_to_fp4_ue8m0(
    weight: torch.Tensor,
    block_size: int = FP4_BLOCK,
):
    """Quantize a stacked MoE weight ``[E, N, K]`` (BF16/FP16/FP32) to FP4
    + UE8M0 scale for load-time and runtime MegaMoE weight preparation.

    Returns:
        packed: int8 ``[E, N, K // 2]``
        sf    : float32 ``[E, N, K // block_size]``
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    assert weight.dim() == 3, f"expected 3D, got {tuple(weight.shape)}"
    E, N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    device = weight.device

    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=device)
    sf = torch.empty((E, N, K // block_size), dtype=torch.float32, device=device)
    for i in range(E):
        packed_i, sf_i = per_token_cast_to_fp4(
            weight[i], use_ue8m0=True, gran_k=block_size
        )
        packed[i].copy_(packed_i)
        sf[i].copy_(sf_i)
    return packed, sf


def convert_fp8_moe_to_fp4_ue8m0(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: int = FP4_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert stacked FP8 weights using raw per-row or 128x128 block scales.

    Accepts either layout flattened as well. Packed DeepGEMM scale tensors
    are not raw scale values and must not enter this conversion.
    """
    assert weight_fp8.dim() == 3, f"expected 3D, got {tuple(weight_fp8.shape)}"
    E, N, K = weight_fp8.shape
    if K % FP8_BLOCK != 0:
        raise ValueError(
            f"FP8 MoE K dimension must be divisible by {FP8_BLOCK}, got {K}"
        )
    if not torch.is_floating_point(weight_scale):
        raise ValueError(
            "MegaMoE FP8 conversion requires raw floating-point checkpoint scales"
        )
    k_blocks = K // FP8_BLOCK
    n_blocks = (N + FP8_BLOCK - 1) // FP8_BLOCK
    if weight_scale.numel() == E * N * k_blocks:
        scales = weight_scale.reshape(E, N, k_blocks)
        per_row = True
    elif weight_scale.numel() == E * n_blocks * k_blocks:
        scales = weight_scale.reshape(E, n_blocks, k_blocks)
        per_row = False
    else:
        raise ValueError(
            f"Cannot interpret scale shape {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight_fp8.shape)}"
        )
    # Dequantize one expert at a time. Holding all experts as FP32 and BF16
    # while their FP8 originals remain live can exhaust startup GPU memory.
    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=weight_fp8.device)
    sf = torch.empty(
        (E, N, K // block_size), dtype=torch.float32, device=weight_fp8.device
    )
    for expert in range(E):
        scale = scales[expert].float()
        if not per_row:
            scale = scale.repeat_interleave(FP8_BLOCK, dim=0)[:N]
        weight = weight_fp8[expert].float().reshape(N, k_blocks, FP8_BLOCK)
        weight = (weight * scale.unsqueeze(-1)).reshape(1, N, K).to(torch.bfloat16)
        packed_i, sf_i = quantize_moe_weight_to_fp4_ue8m0(weight, block_size)
        packed[expert].copy_(packed_i[0])
        sf[expert].copy_(sf_i[0])
        del weight, scale, packed_i, sf_i
    return packed, sf


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
    "convert_fp8_moe_to_fp4_ue8m0",
    "quantize_moe_weight_to_fp4_ue8m0",
    "dequantize_fp4_weight",
    "dequantize_fp8_weight",
    "per_token_cast_to_fp8_packed_ue8m0",
    "prepare_fp4_weight_scale_for_deepgemm",
    "prepare_fp8_weight_scale_for_deepgemm",
]

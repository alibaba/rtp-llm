from typing import Tuple

import torch


def quantize_weight_to_int4b(
    weight: torch.Tensor, group_size: int, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a two-dimensional weight into the Cutlass W4A8 layout."""
    if weight.dim() != 2:
        raise ValueError(f"W4A8 input must be two-dimensional, got {weight.shape}")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if not weight.is_floating_point():
        raise TypeError("W4A8 online quantization requires floating weights")
    n, k = weight.shape
    if n == 0 or k == 0:
        raise ValueError(f"W4A8 input dimensions must be non-zero, got {weight.shape}")
    if group_size <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    if k % 2 != 0:
        raise ValueError(f"W4A8 packing requires an even K dimension, got {k}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("W4A8 online quantization requires finite weights")

    n_groups = k // group_size
    input_grouped = weight.contiguous().view(n, n_groups, group_size)

    amax = input_grouped.abs().amax(dim=2, keepdim=True)
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (amax / 7.0).clamp(min=eps, max=finfo.max / 8.0)
    scale_for_quant = scale.to(torch.float8_e4m3fn).to(weight.dtype)

    output_int8 = (
        torch.round(input_grouped / scale).clamp_(min=-8, max=7).to(torch.int8)
    )
    output_int8_flat = output_int8.flatten()

    first_int4 = output_int8_flat[::2] & 0x0F
    second_int4 = output_int8_flat[1::2] & 0x0F
    output_int4 = ((second_int4 << 4) | first_int4).reshape(n, k // 2).contiguous()

    scale_transposed = (
        scale_for_quant.to(torch.float8_e4m3fn).squeeze(-1).t().contiguous()
    )
    from rtp_kernel.w4a8_group_gemm import (
        pack_scale_fp8,
        reorder_tensor,
        unified_encode_int4b,
    )

    output_unified_int4 = unified_encode_int4b(output_int4)
    output_unified_int4 = reorder_tensor(output_unified_int4)
    scale_packed = pack_scale_fp8(scale_transposed)
    return output_unified_int4, scale_packed


def repack_compressed_int4_to_cutlass(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert compressed-tensors INT4 weights into the Cutlass W4A8 layout."""
    if weight_packed.dim() != 2:
        raise ValueError(f"unexpected packed shape: {weight_packed.shape}")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if weight_packed.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
    ):
        raise TypeError(
            f"compressed INT4 weights require integer storage, got "
            f"{weight_packed.dtype}"
        )
    if not weight_scale.is_floating_point():
        raise TypeError(
            f"compressed INT4 scales must be floating point, got "
            f"{weight_scale.dtype}"
        )

    if weight_packed.dtype not in (torch.int8, torch.uint8):
        weight_packed = weight_packed.contiguous().view(torch.int8)

    n, packed_k = weight_packed.shape
    if n == 0 or packed_k == 0:
        raise ValueError(
            f"compressed INT4 dimensions must be non-zero, got {weight_packed.shape}"
        )
    k = packed_k * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    expected_scale_shape = (n, k // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape mismatch: {tuple(weight_scale.shape)} vs "
            f"{expected_scale_shape}"
        )
    if not bool(torch.isfinite(weight_scale).all()) or bool((weight_scale <= 0).any()):
        raise ValueError("compressed INT4 scales must be finite and positive")

    # compressed-tensors stores offset-binary nibbles. Flipping each nibble's
    # sign bit converts them to the two's-complement convention used by Cutlass.
    packed_int8 = (
        (weight_packed.to(torch.uint8).contiguous() ^ 0x88)
        .view(torch.int8)
        .contiguous()
    )
    scale_fp8 = weight_scale.to(torch.float8_e4m3fn)
    scale_roundtrip = scale_fp8.float()
    if not bool(torch.isfinite(scale_roundtrip).all()) or bool(
        (scale_roundtrip <= 0).any()
    ):
        raise ValueError(
            "compressed INT4 scales must be representable as float8_e4m3fn"
        )
    scale_fp8 = scale_fp8.t().contiguous()
    from rtp_kernel.w4a8_group_gemm import (
        pack_scale_fp8,
        reorder_tensor,
        unified_encode_int4b,
    )

    scale_packed = pack_scale_fp8(scale_fp8)
    output = reorder_tensor(unified_encode_int4b(packed_int8))
    return output, scale_packed

from typing import Dict, List

import torch


def calculate_k_for_swizzling(dtype: torch.dtype):
    if dtype == torch.float32:
        MiK, MiKv = 4, 1
    elif dtype in (torch.float16, torch.half, torch.bfloat16):
        MiK, MiKv = 16, 4
    elif dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ):
        MiK, MiKv = 32, 8
    else:
        raise ValueError(f"unsupported datatype in calculateKforSwizzling: {dtype}")
    elem_size = torch.zeros((), dtype=dtype).element_size()
    PackK = 16 // MiKv // elem_size
    return MiK, MiKv, PackK


def swizzle_tensor(
    src: torch.Tensor, col_maj: bool = False, MiM: int = 16
) -> torch.Tensor:
    tmp = src.clone()

    if col_maj:
        k, m = src.shape
        tmp = tmp.view(k, m).permute(1, 0).contiguous()
    else:
        m, k = src.shape

    MiK, MiKv, PackK = calculate_k_for_swizzling(src.dtype)

    if MiK == 16:
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 32 == 0, f"swizzle shape k = {k} must be divisible by 32"
    elif MiK == 32:
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 64 == 0, f"swizzle shape k = {k} must be divisible by 64"

    tmp = tmp.view(m // MiM, MiM, k // (MiK * PackK), MiK // MiKv, MiKv * PackK)
    tmp = tmp.permute(0, 2, 3, 1, 4).contiguous()

    dst = tmp.clone()
    return dst.view(src.shape)


def can_swizzle_kn(weight: torch.Tensor, dtype: torch.dtype = None) -> bool:
    """Whether a (k, n) = (hidden, out) weight can be swizzled via
    swizzle_tensor(weight.t(), col_maj=False).

    That call transposes to (n, k), so ``n`` must be divisible by 16 and ``k``
    must satisfy the dtype-specific packing divisor. Used by both the data side
    (device_impl layout rewrite) and dispatch side (linear strategy fallback)
    so the two stay consistent for the same weight.
    """
    if weight.dim() != 2:
        return False
    dt = dtype if dtype is not None else weight.dtype
    try:
        MiK, _, PackK = calculate_k_for_swizzling(dt)
    except ValueError:
        return False
    k_div = MiK * PackK
    k, n = weight.shape
    return (n % 16 == 0) and (k % k_div == 0)


def can_fuse_swizzled_kn(
    first_weight: torch.Tensor, second_weight: torch.Tensor
) -> bool:
    """Whether two (k, n) weights can share one swizzled GEMM.

    Each source weight must already be eligible for the same swizzle layout,
    and concatenating their output dimensions must remain eligible. Checking
    shapes directly avoids allocating the potentially very large fused weight.
    """
    if first_weight.dim() != 2 or second_weight.dim() != 2:
        return False
    if first_weight.dtype != second_weight.dtype:
        return False
    if first_weight.shape[0] != second_weight.shape[0]:
        return False
    if not can_swizzle_kn(first_weight) or not can_swizzle_kn(second_weight):
        return False

    k = first_weight.shape[0]
    fused_n = first_weight.shape[1] + second_weight.shape[1]
    MiK, _, PackK = calculate_k_for_swizzling(first_weight.dtype)
    k_div = MiK * PackK
    return (fused_n % 16 == 0) and (k % k_div == 0)


def should_swizzle_linear_attn_ba(weight: torch.Tensor) -> bool:
    """Whether a TP-local Qwen3Next BA weight supports the swizzled layout."""
    return can_swizzle_kn(weight)

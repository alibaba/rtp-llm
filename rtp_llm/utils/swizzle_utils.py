from typing import Dict, List, Set

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


def can_swizzle_tensor(src: torch.Tensor, col_maj: bool = False) -> bool:
    """Check if a 2D tensor meets the alignment requirements for swizzle."""
    if src.dim() != 2:
        return False
    if col_maj:
        k, m = src.shape
    else:
        m, k = src.shape
    MiK, _, _ = calculate_k_for_swizzling(src.dtype)
    if MiK == 16:
        return m % 16 == 0 and k % 32 == 0
    elif MiK == 32:
        return m % 16 == 0 and k % 64 == 0
    return False


# Track which tensors have been swizzled via their data_ptr
_swizzled_data_ptrs: Set[int] = set()


def is_swizzled(tensor: torch.Tensor) -> bool:
    """Check if a tensor was swizzled during weight loading."""
    return tensor.data_ptr() in _swizzled_data_ptrs


def mark_swizzled(tensor: torch.Tensor):
    """Mark a tensor as having been swizzled."""
    _swizzled_data_ptrs.add(tensor.data_ptr())


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

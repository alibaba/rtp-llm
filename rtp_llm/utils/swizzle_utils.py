import torch
from typing import List, Dict

def calculate_k_for_swizzling(dtype: torch.dtype):
    if dtype == torch.float32:
        MiK, MiKv = 4, 1
    elif dtype in (torch.float16, torch.half, torch.bfloat16):
        MiK, MiKv = 16, 4
    elif dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz):
        MiK, MiKv = 32, 8
    else:
        raise ValueError(f"unsupported datatype in calculateKforSwizzling: {dtype}")
    elem_size = torch.zeros((), dtype=dtype).element_size()
    PackK = 16 // MiKv // elem_size
    return MiK, MiKv, PackK

def swizzle_tensor(
    src: torch.Tensor,
    col_maj: bool = False,
    MiM: int = 16) -> torch.Tensor:
    tmp = src.clone()

    if col_maj:
        k, m = src.shape
        tmp = tmp.view(k, m).permute(1, 0).contiguous()
    else:
        m, k = src.shape

    MiK, MiKv, PackK = calculate_k_for_swizzling(src.dtype)

    if (MiK == 16):
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 32 == 0, f"swizzle shape k = {k} must be divisible by 32"
    elif (MiK == 32):
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 64 == 0, f"swizzle shape k = {k} must be divisible by 64"

    tmp = tmp.view(m // MiM, MiM, k // (MiK * PackK), MiK // MiKv, MiKv * PackK)
    tmp = tmp.permute(0, 2, 3, 1, 4).contiguous()

    dst = tmp.clone()
    return dst.view(src.shape)

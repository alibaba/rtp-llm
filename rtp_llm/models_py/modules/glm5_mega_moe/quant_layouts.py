"""Quant constants and helpers for GLM-5 MegaMoE.

Ported from dsv4/quant_layouts.py. Handles FP4/FP8 block sizes and
the activation cast for UE8M0 packed scale factors.
"""

import os
import tempfile
from typing import Optional, Tuple

import torch

FP4_BLOCK = 32
FP8_BLOCK = 128
MXFP8_BLOCK = 32


def prepare_fp4_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert FP4 UE8M0 weight scale to DeepGEMM's SM100 layout.

    Converts raw float8_e8m0fnu or float32 scale tensors into the
    TMA-aligned packed int32 layout that DeepGEMM's FP8xFP4 kernels expect.
    """
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(f"expected FP4 UE8M0 or float32 scale, got {scale.dtype}")

    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
    )
    os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

    import deep_gemm

    scale_fp32 = scale.float()
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
    recipe: Tuple[int, int] = (FP8_BLOCK, FP8_BLOCK),
) -> torch.Tensor:
    """Convert FP8 UE8M0 scales to DeepGEMM's packed layout.

    Supports both GLM-style ``128x128`` block FP8 scales and MiniMax-M3
    MXFP8 ``1x32`` microscaling scales.
    """
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(f"expected FP8 UE8M0 scale or packed int32, got {scale.dtype}")

    gran_mn, gran_k = recipe
    if gran_mn <= 0 or gran_k <= 0:
        raise ValueError(f"invalid FP8 scale recipe={recipe}")

    expected_m = (mn + gran_mn - 1) // gran_mn
    expected_k = (k + gran_k - 1) // gran_k
    if tuple(scale.shape[-2:]) != (expected_m, expected_k):
        raise ValueError(
            "FP8 mega_moe weight scale has unexpected shape. Got "
            f"shape={tuple(scale.shape)}, expected trailing dims="
            f"({expected_m}, {expected_k}) for mn={mn}, k={k}, recipe={recipe}."
        )

    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
    )
    os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

    import deep_gemm

    scale_fp32 = scale.float().contiguous()

    def _transform():
        if num_groups is None:
            return deep_gemm.transform_sf_into_required_layout(
                scale_fp32, mn, k, recipe
            )
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, recipe, num_groups
        )

    if scale_fp32.is_cuda:
        with torch.cuda.device(scale_fp32.device):
            return _transform()
    return _transform()


def per_token_cast_to_fp8_packed_ue8m0(
    x: torch.Tensor,
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast BF16 activation to FP8 E4M3 with packed UE8M0 scale factors.

    CUDA-graph-safe version (no .all() assertion that triggers CPU sync).
    Returns (x_fp8, scale_packed_int32).
    """
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    if padded_n != n:
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
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

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


def convert_fp8_weights_to_fp4(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: int = FP4_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert FP8 per-block weights to FP4 format for mega kernel.

    Dequantizes FP8 weights to BF16/FP32, then requantizes to FP4 (int8 packed)
    with UE8M0 scale factors suitable for DeepGEMM mega_moe.

    Args:
        weight_fp8: [N, K] or [E, N, K] float8_e4m3fn weights
        weight_scale: [N, K//128] or [E, N, K//128] float32 per-block scales
        block_size: FP4 quantization block size (default 32)

    Returns:
        (packed_fp4, scale_ue8m0): packed int8 [.., K//2] and ue8m0 scale [.., K//block_size]
    """
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    original_shape = weight_fp8.shape
    if weight_fp8.dim() == 3:
        E, N, K = weight_fp8.shape
        # Dequantize: fp8 * scale -> bf16
        fp8_block = 128
        w_float = weight_fp8.float()
        scale_expanded = weight_scale.unsqueeze(-1).expand(
            E, N, K // fp8_block, fp8_block
        )
        w_float = (
            w_float.view(E, N, K // fp8_block, fp8_block) * scale_expanded
        ).reshape(E, N, K)
        w_bf16 = w_float.to(torch.bfloat16)

        # Quantize to FP4 per expert
        packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=weight_fp8.device)
        sf = torch.empty(
            (E, N, K // block_size), dtype=torch.float, device=weight_fp8.device
        )
        for i in range(E):
            packed[i], sf[i] = per_token_cast_to_fp4(
                w_bf16[i], use_ue8m0=True, gran_k=block_size
            )
        return packed, sf
    else:
        N, K = weight_fp8.shape
        fp8_block = 128
        w_float = weight_fp8.float()
        scale_expanded = weight_scale.unsqueeze(-1).expand(N, K // fp8_block, fp8_block)
        w_float = (w_float.view(N, K // fp8_block, fp8_block) * scale_expanded).reshape(
            N, K
        )
        w_bf16 = w_float.to(torch.bfloat16)
        packed, sf = per_token_cast_to_fp4(w_bf16, use_ue8m0=True, gran_k=block_size)
        return packed, sf

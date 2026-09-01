"""MXFP8 (1x32 microscaling FP8) linear primitives.

Weights are e4m3 with a UE8M0 scale on fixed ``[1, 32]`` micro-blocks.
Activations are dynamically quantized to the same format, then the GEMM uses
DeepGEMM's ``fp8_fp4_gemm_nt`` with ``recipe=(1, 32)``. SM100 only.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt

MX_BLOCK = 32
_FLASHINFER_CUTE_DSL_MAX_NUMEL = 2**31 - 1


@triton.jit
def _pack_flashinfer_mxfp8_scale_kernel(
    scale_u8_ptr,
    packed_ptr,
    M: tl.constexpr,
    K_GROUPS: tl.constexpr,
    K_PACKED: tl.constexpr,
    ALIGNED_MN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kp = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
    shifts = tl.arange(0, 4) * 8
    offs_g = offs_kp[:, None] * 4 + tl.arange(0, 4)[None, :]
    mask = (offs_m[:, None, None] < M) & (offs_g[None, :, :] < K_GROUPS)
    vals = tl.load(
        scale_u8_ptr + offs_m[:, None, None] * K_GROUPS + offs_g[None, :, :],
        mask=mask,
        other=0,
    ).to(tl.int32)
    packed = tl.sum(vals << shifts[None, None, :], axis=2).to(tl.int32)
    tl.store(
        packed_ptr + offs_m[:, None] + offs_kp[None, :] * ALIGNED_MN,
        packed,
        mask=(offs_m[:, None] < M) & (offs_kp[None, :] < K_PACKED),
    )


def _pack_flashinfer_mxfp8_scale(
    scale_u8: torch.Tensor, m: int, k: int
) -> torch.Tensor:
    """Pack FlashInfer uint8 UE8M0 scales into DeepGEMM's int32 TMA layout."""
    assert scale_u8.dtype == torch.uint8
    assert scale_u8.numel() == m * (k // MX_BLOCK)
    import deep_gemm

    k_groups = k // MX_BLOCK
    assert k_groups % 4 == 0
    k_packed = k_groups // 4
    aligned_mn = deep_gemm.get_tma_aligned_size(m, 4)
    storage = torch.empty(
        (k_packed, aligned_mn), device=scale_u8.device, dtype=torch.int32
    )
    packed = storage.transpose(0, 1)
    grid = (triton.cdiv(m, 64), triton.cdiv(k_packed, 32))
    with torch.cuda.device(scale_u8.device):
        _pack_flashinfer_mxfp8_scale_kernel[grid](
            scale_u8,
            packed,
            M=m,
            K_GROUPS=k_groups,
            K_PACKED=k_packed,
            ALIGNED_MN=aligned_mn,
            BLOCK_M=64,
            BLOCK_K_PACKED=32,
            num_warps=8,
        )
    return packed[:m, :]


def _mxfp8_quant_flashinfer_backend(x: torch.Tensor) -> str:
    # cute-dsl uses 32-bit flattened offsets.
    if x.numel() > _FLASHINFER_CUTE_DSL_MAX_NUMEL:
        return "cuda"
    return "cute-dsl"


def mxfp8_quant_act_packed(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation and return e4m3 data plus packed UE8M0 scale."""
    assert x.dim() == 2, f"expected 2D activation, got {x.shape}"
    k = x.shape[1]
    assert k % MX_BLOCK == 0, f"K={k} must be a multiple of {MX_BLOCK}"
    assert x.is_cuda, "FlashInfer MXFP8 quant requires CUDA input"
    assert x.is_contiguous(), "input must be contiguous"

    import flashinfer

    q, scale_u8 = flashinfer.mxfp8_quantize(
        x,
        is_sf_swizzled_layout=False,
        alignment=MX_BLOCK,
        backend=_mxfp8_quant_flashinfer_backend(x),
    )
    return q, _pack_flashinfer_mxfp8_scale(scale_u8, x.shape[0], k)


def pack_mxfp8_scale(
    scale_fp32: torch.Tensor,
    mn: int,
    k: int,
) -> torch.Tensor:
    """Pack power-of-two FP32 scales into DeepGEMM's int32 TMA layout."""
    import deep_gemm

    kwargs = dict(mn=mn, k=k, recipe=(1, MX_BLOCK))
    scale = scale_fp32.contiguous()
    if scale.is_cuda:
        with torch.cuda.device(scale.device):
            return deep_gemm.transform_sf_into_required_layout(scale, **kwargs)
    return deep_gemm.transform_sf_into_required_layout(scale, **kwargs)


def mxfp8_linear(
    x: torch.Tensor,
    weight_e4m3: torch.Tensor,
    weight_scale_packed: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` with MXFP8 activations and weights."""
    m, n = x.shape[0], weight_e4m3.shape[0]
    a_q, a_s_packed = mxfp8_quant_act_packed(x)
    out = torch.empty(m, n, device=x.device, dtype=out_dtype)
    with torch.cuda.device(x.device):
        fp8_fp4_gemm_nt(
            (a_q, a_s_packed),
            (weight_e4m3, weight_scale_packed),
            out,
            recipe_a=(1, MX_BLOCK),
            recipe_b=(1, MX_BLOCK),
            disable_ue8m0_cast=True,
        )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out

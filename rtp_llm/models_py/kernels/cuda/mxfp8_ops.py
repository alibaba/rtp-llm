"""MXFP8 (1x32 microscaling FP8) linear primitives.

Weights are e4m3 with a UE8M0 scale on fixed ``[1, 32]`` micro-blocks.
Activations are dynamically quantized to the same format, then the GEMM uses
DeepGEMM's ``fp8_fp4_gemm_nt`` with ``recipe=(1, 32)``. SM100 only.
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt

MX_BLOCK = 32
_FLASHINFER_CUTE_DSL_MAX_NUMEL = 2**31 - 1
_FUSED_QUANT_ENV = "RTP_LLM_MXFP8_FUSED_QUANT"
_FUSED_QUANT_AUTO_MAX_M = 64


@triton.jit
def _float_to_ue8m0(value):
    """Match FlashInfer's round-toward-+inf UE8M0 conversion exactly."""
    bits = value.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    bump = tl.where(mantissa != 0, 1, 0)
    tiny_subnormal = (exponent == 0) & (mantissa <= 0x400000)
    bump = tl.where(tiny_subnormal, 0, bump)
    result = tl.minimum(exponent + bump, 254)
    return tl.where(value <= 0.0, 0, result)


@triton.jit
def _ue8m0_to_inv_scale(exponent):
    """Construct FlashInfer's exact reciprocal power-of-two scale."""
    inv_exponent = tl.maximum(254 - exponent, 0)
    inv_bits = inv_exponent << 23
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    return tl.where(exponent == 0, 0.0, inv_scale)


@triton.jit
def _mxfp8_quant_act_packed_kernel(
    x_ptr,
    q_ptr,
    packed_scale_ptr,
    M,
    stride_x_m,
    stride_q_m,
    stride_scale_m,
    stride_scale_k,
    GROUP_SIZE: tl.constexpr,
    K_PACKED: tl.constexpr,
):
    """Quantize four adjacent 32-value MX groups and pack their scales."""
    row = tl.program_id(0).to(tl.int64)
    packed_group = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, GROUP_SIZE)
    row_valid = row < M
    packed_scale: tl.int32 = 0

    for group_in_pack in tl.static_range(4):
        group = packed_group * 4 + group_in_pack
        columns = group * GROUP_SIZE + offsets
        values = tl.load(
            x_ptr + row * stride_x_m + columns,
            mask=row_valid,
            other=0.0,
        ).to(tl.float32)

        # FlashInfer MXFP8 uses max(abs(x)) * (1 / 448), followed by the
        # hardware UE8M0 round-toward-+inf conversion.  In particular, it
        # does not clamp zero/small groups to 1e-4 like the generic RTP FP8
        # quantizer does.
        absmax = tl.max(tl.abs(values), axis=0)
        normalized_max = absmax * tl.full((), 1.0 / 448.0, tl.float32)
        scale_exponent = _float_to_ue8m0(normalized_max)
        inv_scale = _ue8m0_to_inv_scale(scale_exponent)

        scaled = values * tl.full(values.shape, inv_scale, tl.float32)
        quantized = tl.clamp(scaled, -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(
            q_ptr + row * stride_q_m + columns,
            quantized,
            mask=row_valid,
        )
        packed_scale = packed_scale | (scale_exponent << (group_in_pack * 8))

    tl.store(
        packed_scale_ptr
        + row * stride_scale_m
        + packed_group * stride_scale_k,
        packed_scale,
        mask=row_valid & (packed_group < K_PACKED),
    )


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


def create_mxfp8_packed_scale(m: int, k: int, device: torch.device) -> torch.Tensor:
    """Allocate DeepGEMM's column-major packed UE8M0 activation layout."""
    import deep_gemm

    k_groups = k // MX_BLOCK
    k_packed = k_groups // 4
    aligned_m = deep_gemm.get_tma_aligned_size(m, 4)
    storage = torch.empty((k_packed, aligned_m), device=device, dtype=torch.int32)
    return storage.transpose(0, 1)[:m, :]


def mxfp8_quant_act_packed_fused(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-launch MXFP8 activation quantization with DeepGEMM-ready scales.

    Its numerical contract follows ``flashinfer.mxfp8_quantize`` rather than
    RTP's generic per-token FP8 quantizer: scale groups are 32 values, zero
    groups use UE8M0 byte 0, and scale rounding is toward positive infinity.
    """
    assert x.dim() == 2, f"expected 2D activation, got {x.shape}"
    m, k = x.shape
    assert k % (4 * MX_BLOCK) == 0, f"K={k} must be a multiple of {4 * MX_BLOCK}"
    assert x.is_cuda, "fused MXFP8 quant requires CUDA input"
    assert x.is_contiguous(), "input must be contiguous"
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"fused MXFP8 quant expects bf16/fp16 input, got {x.dtype}"
    )

    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    packed_scale = create_mxfp8_packed_scale(m, k, x.device)
    if m == 0:
        return q, packed_scale

    k_packed = k // (4 * MX_BLOCK)
    with torch.cuda.device(x.device):
        _mxfp8_quant_act_packed_kernel[(m, k_packed)](
            x,
            q,
            packed_scale,
            m,
            x.stride(0),
            q.stride(0),
            packed_scale.stride(0),
            packed_scale.stride(1),
            GROUP_SIZE=MX_BLOCK,
            K_PACKED=k_packed,
            num_warps=1,
            num_stages=1,
        )
    return q, packed_scale


def _use_fused_quant(x: torch.Tensor) -> bool:
    requested = os.environ.get(_FUSED_QUANT_ENV, "auto").strip().lower()
    if requested in ("0", "false", "off", "no"):
        return False
    if requested not in ("", "auto", "1", "true", "on", "yes"):
        raise ValueError(
            f"invalid {_FUSED_QUANT_ENV}={requested!r}; expected auto, 0, or 1"
        )
    supported = (
        x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[1] % (4 * MX_BLOCK) == 0
    )
    if not supported:
        return False
    return requested not in ("", "auto") or x.shape[0] <= _FUSED_QUANT_AUTO_MAX_M


def mxfp8_quant_act_packed(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation and return e4m3 data plus packed UE8M0 scale."""
    assert x.dim() == 2, f"expected 2D activation, got {x.shape}"
    k = x.shape[1]
    assert k % MX_BLOCK == 0, f"K={k} must be a multiple of {MX_BLOCK}"
    assert x.is_cuda, "FlashInfer MXFP8 quant requires CUDA input"
    assert x.is_contiguous(), "input must be contiguous"

    if _use_fused_quant(x):
        return mxfp8_quant_act_packed_fused(x)

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

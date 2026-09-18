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
def _silu_mul_mxfp8_quant_act_packed_kernel(
    input_ptr,
    output_q_ptr,
    output_scale_ptr,
    M,
    input_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    N: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    PACKS_PER_ROW: tl.constexpr,
    PACKS_PER_CTA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """vLLM SiLU+MXFP8 fusion with FlashInfer UE8M0 scale semantics.

    This is adapted from vLLM's
    ``_silu_mul_quant_fp8_packed_kernel``.  The execution/persistence shape is
    retained; its generic ``max(absmax / 448, 1e-10)`` scale floor is replaced
    by the FlashInfer-compatible ``_float_to_ue8m0`` conversion used by the
    existing RTP MXFP8 activation quantizer.
    """
    GROUPS_PER_PACK: tl.constexpr = 4
    HIDDEN_SIZE: tl.constexpr = N // 2

    pack_tile = tl.program_id(0)
    row_start = tl.program_id(1).to(tl.int64) * BLOCK_M
    row_step = tl.num_programs(1).to(tl.int64) * BLOCK_M

    groups_per_cta: tl.constexpr = PACKS_PER_CTA * GROUPS_PER_PACK
    elements_per_cta: tl.constexpr = groups_per_cta * GROUP_SIZE
    col_start = pack_tile * elements_per_cta
    col_offsets = tl.arange(0, elements_per_cta)
    row_offsets = tl.arange(0, BLOCK_M)
    pack_offsets = tl.arange(0, PACKS_PER_CTA)
    col_mask = (col_start + col_offsets) < (GROUPS_PER_ROW * GROUP_SIZE)

    while row_start < M:
        rows = row_start + row_offsets
        row_mask = rows < M
        input_row_start = rows[:, None] * input_stride_m
        output_row_start = rows[:, None] * output_q_stride_m

        gate_flat = tl.load(
            input_ptr + input_row_start + col_start + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        up_flat = tl.load(
            input_ptr
            + input_row_start
            + HIDDEN_SIZE
            + col_start
            + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        gate = tl.reshape(gate_flat, (BLOCK_M, groups_per_cta, GROUP_SIZE)).to(
            tl.float32
        )
        up = tl.reshape(up_flat, (BLOCK_M, groups_per_cta, GROUP_SIZE)).to(
            tl.float32
        )

        # Match the unfused path: SiLU/multiply followed by a BF16 materialized
        # activation, then groupwise MXFP8 quantization.
        y = (gate / (1.0 + tl.exp(-gate))) * up
        y = y.to(tl.bfloat16).to(tl.float32)

        absmax = tl.max(tl.abs(y), axis=2)
        scale_exponent = _float_to_ue8m0(
            absmax * tl.full(absmax.shape, 1.0 / 448.0, tl.float32)
        )
        inv_scale = _ue8m0_to_inv_scale(scale_exponent)
        y_q = tl.clamp(y * inv_scale[:, :, None], -448.0, 448.0)

        y_q_flat = tl.reshape(y_q, (BLOCK_M, elements_per_cta))
        tl.store(
            output_q_ptr + output_row_start + col_start + col_offsets[None, :],
            y_q_flat.to(output_q_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

        scale_bytes = tl.reshape(
            scale_exponent, (BLOCK_M, PACKS_PER_CTA, GROUPS_PER_PACK)
        )
        shifts = tl.arange(0, GROUPS_PER_PACK) * 8
        packed_scale = tl.sum(scale_bytes << shifts[None, None, :], axis=2)
        scale_pack = pack_tile * PACKS_PER_CTA + pack_offsets
        tl.store(
            output_scale_ptr
            + scale_pack[None, :] * output_scale_stride_k
            + rows[:, None],
            packed_scale,
            mask=row_mask[:, None] & (scale_pack[None, :] < PACKS_PER_ROW),
        )
        row_start += row_step


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


def silu_mul_mxfp8_quant_act_packed_fused(
    gate_up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU/mul and HY4's 1x32 MXFP8 activation quantization.

    The Triton scheduling and layout are adapted from vLLM's
    ``silu_mul_quant_fp8_packed_triton`` for group size 32.  Unlike vLLM's
    generic FP8 helper, output scales precisely follow FlashInfer's UE8M0
    contract, including a zero byte for all-zero groups and upward exponent
    rounding for tiny nonzero groups.
    """
    assert gate_up.dim() == 2, f"expected 2D activation, got {gate_up.shape}"
    assert gate_up.is_cuda, "fused SiLU/MXFP8 quant requires CUDA input"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    # HY4's MXFP8 down-projection input is BF16.  The fused path intentionally
    # reproduces the legacy BF16 materialization before FlashInfer quantizes;
    # do not silently route FP16 callers through a numerically different path.
    assert gate_up.dtype == torch.bfloat16, (
        f"fused SiLU/MXFP8 quant expects BF16 input, got {gate_up.dtype}"
    )
    m, n = gate_up.shape
    assert n % 2 == 0, f"expected [gate|up] input, got last dimension {n}"
    hidden_size = n // 2
    assert hidden_size % (4 * MX_BLOCK) == 0, (
        f"hidden size {hidden_size} must be a multiple of {4 * MX_BLOCK}"
    )

    output_q = torch.empty(
        (m, hidden_size), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    output_scale = create_mxfp8_packed_scale(m, hidden_size, gate_up.device)
    if m == 0:
        return output_q, output_scale

    groups_per_row = hidden_size // MX_BLOCK
    packs_per_row = groups_per_row // 4
    # Keep vLLM's tuned group-32 persistent layout: 8 packed-scale columns
    # (1024 values) per CTA, with one row per CTA for large prefill.
    packs_per_cta = 8
    block_m = 1
    grid = (
        triton.cdiv(packs_per_row, packs_per_cta),
        min(triton.cdiv(m, block_m), 4096),
    )
    with torch.cuda.device(gate_up.device):
        _silu_mul_mxfp8_quant_act_packed_kernel[grid](
            gate_up,
            output_q,
            output_scale,
            m,
            gate_up.stride(0),
            output_q.stride(0),
            output_scale.stride(1),
            N=n,
            GROUPS_PER_ROW=groups_per_row,
            PACKS_PER_ROW=packs_per_row,
            PACKS_PER_CTA=packs_per_cta,
            BLOCK_M=block_m,
            GROUP_SIZE=MX_BLOCK,
            num_warps=4,
            num_stages=2,
        )
    return output_q, output_scale


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

"""Fused add-residual + RMSNorm + per-token-group FP8 quantization.

Combines three operations into one kernel launch:
1. residual += hidden_states  (in-place update)
2. normed = rmsnorm(residual, weight, eps)
3. (fp8_out, scale) = per_token_group_fp8_quant(normed, group_size=128)

Two scale layouts:
  - SCALE_UE8M0=True  : int32 packed (4 UE8M0 exponents per int32), Blackwell
  - SCALE_UE8M0=False : float32 unpacked, H20

Single-pass design: ``BLOCK_N = next_power_of_2(H)`` so the entire row fits in
one Triton tile. Non-power-of-2 H is handled by masking loads/stores. For
``H > MAX_INREG_H`` (8192) the function falls back to optimized baseline CUDA
kernels (fused_add_rmsnorm + per_token_group_quant) — the fallback is internal
and transparent to callers.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)

MAX_INREG_H = 8192
MX_BLOCK = 32


def _create_mxfp8_packed_scale_output(
    tokens: int, hidden_size: int, device: torch.device
) -> torch.Tensor:
    """Allocate the exact DeepGEMM MXFP8 activation-scale layout.

    One int32 packs four adjacent group-32 UE8M0 bytes, so the logical K
    dimension is ``hidden_size / 128``.  The generic FP8 allocator assumes
    one UE8M0 byte per group-128 and therefore cannot be used for MXFP8.
    """
    import deep_gemm

    packed_k = hidden_size // (4 * MX_BLOCK)
    aligned_tokens = deep_gemm.get_tma_aligned_size(tokens, 4)
    storage = torch.empty((packed_k, aligned_tokens), device=device, dtype=torch.int32)
    return storage.transpose(0, 1)[:tokens, :]


def _canonical_mxfp8_quant_act_packed(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Import MXFP8 only when an MXFP8 fallback is actually selected."""
    from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

    return mxfp8_quant_act_packed(value)


@triton.jit
def _mxfp8_float_to_ue8m0(value):
    """Match FlashInfer's round-toward-+inf UE8M0 conversion."""
    bits = value.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    bump = tl.where(mantissa != 0, 1, 0)
    tiny_subnormal = (exponent == 0) & (mantissa <= 0x400000)
    bump = tl.where(tiny_subnormal, 0, bump)
    result = tl.minimum(exponent + bump, 254)
    return tl.where(value <= 0.0, 0, result)


@triton.jit
def _mxfp8_ue8m0_to_inv_scale(exponent):
    """Construct FlashInfer's reciprocal power-of-two MXFP8 scale."""
    inv_exponent = tl.maximum(254 - exponent, 0)
    inv_bits = inv_exponent << 23
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    return tl.where(exponent == 0, 0.0, inv_scale)


@triton.jit
def _ieee_rn_div_f32(x, y):
    """IEEE round-to-nearest-even fp32 division.

    Triton's default fp32 ``/`` lowers to ``div.approx.f32`` (~1 ULP off true
    IEEE-RNE). This helper forces ``div.rn.f32`` via inline asm for callers
    that need byte-exact alignment with ``sgl_per_token_group_quant_fp8``.

    NOTE: not used by the kernels in this file anymore — they switched to
    the default approx-div + reciprocal-multiply path after empirical
    verification that the 1 ULP difference is absorbed by UE8M0 power-of-2
    rounding and E4M3 3-mantissa quant (bit-identical fp8/bf16/scale outputs,
    ~20% wall-time savings). Kept here because other kernels still import it.
    """
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ue8m0_pow2_round(s_init):
    """Round a positive fp32 value up to the nearest power of 2 via bit hack.

    Cheaper than ``tl.exp2(tl.ceil(tl.log2(s_init)))`` (3 transcendentals)."""
    bits = s_init.to(tl.int32, bitcast=True)
    mantissa_nz = (bits & 0x7FFFFF) != 0
    exp_field = (bits >> 23) & 0xFF
    exp_field = exp_field + tl.where(mantissa_nz, 1, 0)
    s_int = exp_field << 23
    return s_int.to(tl.float32, bitcast=True), exp_field & 0xFF


@triton.jit
def _fused_add_rmsnorm_fp8_quant_singlepass_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    fp8_out_ptr,
    scale_out_ptr,
    H: tl.constexpr,
    eps,
    fp8_max,
    fp8_min,
    stride_h_t,
    stride_r_t,
    stride_o_t,
    stride_scale_t,
    stride_scale_g,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    MXFP8_SEMANTICS: tl.constexpr,
):
    """Single-pass: load whole row → r_new in registers → reuse for normalize+quant.

    Requires BLOCK_N >= H. Handles non-power-of-2 H via masking.
    """
    token_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < H

    h = tl.load(hidden_ptr + token_id * stride_h_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    r = tl.load(residual_ptr + token_id * stride_r_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    # Match production ``RMSResNorm`` (= ``rtp_llm_ops.fused_add_rmsnorm`` =
    # flashinfer single-pass): r + h is computed in fp32 and used DIRECTLY
    # for the rmsnorm reduction WITHOUT a bf16 round-trip. Only the residual
    # store rounds to bf16. A round-trip on r_new introduces a ~6e-2 max
    # bf16 diff vs production and causes per-layer cascading divergence.
    r_new = r + h
    tl.store(
        residual_ptr + token_id * stride_r_t + offs,
        r_new.to(tl.bfloat16),
        mask=mask,
    )
    sq_sum = tl.sum(r_new * r_new)
    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Production stores the normed result as bf16 (RMSResNorm output). The
    # consumer (Linear's internal sgl quant) reads bf16 → fp32. Round-trip
    # here so absmax/scale and fp8 cast see the same bf16-rounded value.
    normed = (r_new * rsqrt_val * w).to(tl.bfloat16).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_N // GROUP_SIZE
    actual_num_groups: tl.constexpr = H // GROUP_SIZE
    normed_2d = tl.reshape(normed, (num_groups, GROUP_SIZE))
    abs_2d = tl.abs(normed_2d)
    # NOTE: do NOT clamp absmax to a Python-float floor like
    # ``tl.maximum(..., 1e-10)``: the Python literal becomes fp64 and
    # promotes the whole expression to fp64 division, which produces a
    # 1-ULP-different fp32 scale vs the baseline sgl_per_token_group_quant_fp8
    # CUDA kernel (which does pure fp32 ``local_absmax / max_8bit``). The fp8
    # cast below already clamps to [fp8_min, fp8_max] so a zero absmax yields
    # NaN/inf that gets safely clamped to 0 (matching baseline behaviour).
    absmax = tl.max(abs_2d, axis=1)
    if not MXFP8_SEMANTICS:
        absmax = tl.maximum(absmax, 1e-4)

    # Use default fp32 `/` (div.approx.f32) + reciprocal-multiply for the
    # quant divisions. Empirically bit-identical to the prior div.rn.f32
    # path for UE8M0+E4M3 (the ~1 ULP fp32 difference is absorbed by UE8M0's
    # power-of-2 rounding and E4M3's 3-mantissa quant). Saves ~20% wall time
    # on the kernel by avoiding the inline-asm div.rn.f32.
    if SCALE_UE8M0:
        if MXFP8_SEMANTICS:
            normalized_max = absmax * tl.full(absmax.shape, 1.0 / 448.0, tl.float32)
            exp_field = _mxfp8_float_to_ue8m0(normalized_max)
            inv_scale = _mxfp8_ue8m0_to_inv_scale(exp_field)
            inv_scale_full = tl.broadcast_to(
                tl.reshape(inv_scale, (num_groups, 1)),
                (num_groups, GROUP_SIZE),
            )
            fp8_2d = tl.clamp(
                normed_2d * inv_scale_full,
                fp8_min,
                fp8_max,
            ).to(fp8_out_ptr.dtype.element_ty)
        else:
            s_init = absmax / fp8_max
            s, exp_field = _ue8m0_pow2_round(s_init)
            s_bcast = tl.reshape(s, (num_groups, 1))
            s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
            inv_s = 1.0 / s_full
            fp8_2d = tl.clamp(
                normed_2d * inv_s,
                fp8_min,
                fp8_max,
            ).to(fp8_out_ptr.dtype.element_ty)
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)

        num_packed: tl.constexpr = num_groups // 4
        actual_packed: tl.constexpr = actual_num_groups // 4
        g_idx = tl.arange(0, num_groups)
        shift_amt = (g_idx % 4) * 8
        shifted = tl.where(g_idx < actual_num_groups, exp_field << shift_amt, 0)
        shifted_2d = tl.reshape(shifted, (num_packed, 4))
        packed = tl.sum(shifted_2d, axis=1)
        pack_offs = tl.arange(0, num_packed)
        pack_mask = pack_offs < actual_packed
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + pack_offs * stride_scale_g,
            packed,
            mask=pack_mask,
        )
    else:
        s = absmax / fp8_max
        s_bcast = tl.reshape(s, (num_groups, 1))
        s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
        inv_s = 1.0 / s_full
        fp8_2d = tl.clamp(
            normed_2d * inv_s,
            fp8_min,
            fp8_max,
        ).to(fp8_out_ptr.dtype.element_ty)
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        g_offs = tl.arange(0, num_groups)
        g_mask = g_offs < actual_num_groups
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + g_offs * stride_scale_g,
            s,
            mask=g_mask,
        )


@triton.jit
def _fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    bf16_out_ptr,
    fp32_out_ptr,
    raw_gate_clear_ptr,
    fp8_out_ptr,
    scale_out_ptr,
    mega_mxfp8_out_ptr,
    mega_mxfp8_scale_out_ptr,
    H: tl.constexpr,
    eps,
    fp8_max,
    fp8_min,
    stride_h_t,
    stride_r_t,
    stride_b_t,
    stride_fp32_t,
    stride_o_t,
    stride_scale_t,
    stride_scale_g,
    stride_mega_mxfp8_t,
    stride_mega_mxfp8_scale_t,
    stride_mega_mxfp8_scale_g,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    MXFP8_SEMANTICS: tl.constexpr,
    HAS_MEGA_MOE_OUTPUT: tl.constexpr,
    HAS_FP32_OUTPUT: tl.constexpr,
    HAS_RAW_GATE_CLEAR: tl.constexpr,
    ROUND_RESIDUAL_BF16: tl.constexpr,
):
    """Single-pass dual-output: also stores bf16 normed alongside fp8.

    Handles non-power-of-2 H via masking.
    """
    token_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < H

    h = tl.load(hidden_ptr + token_id * stride_h_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    r = tl.load(residual_ptr + token_id * stride_r_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    # Ordinary fused add/RMSNorm retains the FP32 sum for the reduction.
    # HY4 MTP post-attention instead has an explicit BF16 add before norm.
    r_new = r + h
    if ROUND_RESIDUAL_BF16:
        r_new = r_new.to(tl.bfloat16).to(tl.float32)
    tl.store(
        residual_ptr + token_id * stride_r_t + offs,
        r_new.to(tl.bfloat16),
        mask=mask,
    )
    sq_sum = tl.sum(r_new * r_new)
    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # bf16 round-trip on normed: production stores normed as bf16 in
    # ``bf16_out`` AND re-reads it as fp32 for fp8 quant. Both consumers
    # must see the same bf16-rounded value.
    normed_bf16 = (r_new * rsqrt_val * w).to(tl.bfloat16)
    tl.store(
        bf16_out_ptr + token_id * stride_b_t + offs,
        normed_bf16,
        mask=mask,
    )
    normed = normed_bf16.to(tl.float32)
    if HAS_FP32_OUTPUT:
        tl.store(
            fp32_out_ptr + token_id * stride_fp32_t + offs,
            normed,
            mask=mask,
        )
    if HAS_RAW_GATE_CLEAR:
        raw_gate_offsets = tl.arange(0, 32)
        tl.store(raw_gate_clear_ptr + token_id * 32 + raw_gate_offsets, 0.0)

    num_groups: tl.constexpr = BLOCK_N // GROUP_SIZE
    actual_num_groups: tl.constexpr = H // GROUP_SIZE
    normed_2d = tl.reshape(normed, (num_groups, GROUP_SIZE))
    abs_2d = tl.abs(normed_2d)
    # NOTE: do NOT clamp absmax to a Python-float floor like
    # ``tl.maximum(..., 1e-10)``: the Python literal becomes fp64 and
    # promotes the whole expression to fp64 division, which produces a
    # 1-ULP-different fp32 scale vs the baseline sgl_per_token_group_quant_fp8
    # CUDA kernel (which does pure fp32 ``local_absmax / max_8bit``). The fp8
    # cast below already clamps to [fp8_min, fp8_max] so a zero absmax yields
    # NaN/inf that gets safely clamped to 0 (matching baseline behaviour).
    absmax = tl.max(abs_2d, axis=1)
    if not MXFP8_SEMANTICS:
        absmax = tl.maximum(absmax, 1e-4)

    # Use default fp32 `/` (div.approx.f32) + reciprocal-multiply for the
    # quant divisions. Empirically bit-identical to the prior div.rn.f32
    # path for UE8M0+E4M3 (the ~1 ULP fp32 difference is absorbed by UE8M0's
    # power-of-2 rounding and E4M3's 3-mantissa quant). Saves ~20% wall time
    # on the kernel by avoiding the inline-asm div.rn.f32.
    if SCALE_UE8M0:
        if MXFP8_SEMANTICS:
            normalized_max = absmax * tl.full(absmax.shape, 1.0 / 448.0, tl.float32)
            exp_field = _mxfp8_float_to_ue8m0(normalized_max)
            inv_scale = _mxfp8_ue8m0_to_inv_scale(exp_field)
            inv_scale_full = tl.broadcast_to(
                tl.reshape(inv_scale, (num_groups, 1)),
                (num_groups, GROUP_SIZE),
            )
            fp8_2d = tl.clamp(
                normed_2d * inv_scale_full,
                fp8_min,
                fp8_max,
            ).to(fp8_out_ptr.dtype.element_ty)
        else:
            s_init = absmax / fp8_max
            s, exp_field = _ue8m0_pow2_round(s_init)
            s_bcast = tl.reshape(s, (num_groups, 1))
            s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
            inv_s = 1.0 / s_full
            fp8_2d = tl.clamp(
                normed_2d * inv_s,
                fp8_min,
                fp8_max,
            ).to(fp8_out_ptr.dtype.element_ty)
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)

        num_packed: tl.constexpr = num_groups // 4
        actual_packed: tl.constexpr = actual_num_groups // 4
        g_idx = tl.arange(0, num_groups)
        shift_amt = (g_idx % 4) * 8
        shifted = tl.where(g_idx < actual_num_groups, exp_field << shift_amt, 0)
        shifted_2d = tl.reshape(shifted, (num_packed, 4))
        packed = tl.sum(shifted_2d, axis=1)
        pack_offs = tl.arange(0, num_packed)
        pack_mask = pack_offs < actual_packed
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + pack_offs * stride_scale_g,
            packed,
            mask=pack_mask,
        )
    else:
        s = absmax / fp8_max
        s_bcast = tl.reshape(s, (num_groups, 1))
        s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
        inv_s = 1.0 / s_full
        fp8_2d = tl.clamp(
            normed_2d * inv_s,
            fp8_min,
            fp8_max,
        ).to(fp8_out_ptr.dtype.element_ty)
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        g_offs = tl.arange(0, num_groups)
        g_mask = g_offs < actual_num_groups
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + g_offs * stride_scale_g,
            s,
            mask=g_mask,
        )

    if HAS_MEGA_MOE_OUTPUT:
        # Plain MegaMoE consumes the same normalized BF16 values as the
        # shared expert, but owns a distinct activation ABI: group-32 E4M3,
        # four packed UE8M0 bytes, contiguous row-major scales, and a 1e-4
        # absmax floor.  Produce that ABI directly in the caller-owned CUDA
        # Graph buffers while the normalized row is still resident.
        mega_absmax = tl.maximum(absmax, 1.0e-4)
        mega_scale_exponent = _mxfp8_float_to_ue8m0(mega_absmax / 448.0)
        mega_inv_scale = _mxfp8_ue8m0_to_inv_scale(mega_scale_exponent)
        mega_inv_scale_full = tl.broadcast_to(
            tl.reshape(mega_inv_scale, (num_groups, 1)),
            (num_groups, GROUP_SIZE),
        )
        mega_fp8_2d = tl.clamp(
            normed_2d * mega_inv_scale_full,
            fp8_min,
            fp8_max,
        ).to(mega_mxfp8_out_ptr.dtype.element_ty)
        tl.store(
            mega_mxfp8_out_ptr + token_id * stride_mega_mxfp8_t + offs,
            tl.reshape(mega_fp8_2d, (BLOCK_N,)),
            mask=mask,
        )

        mega_num_packed: tl.constexpr = num_groups // 4
        mega_actual_packed: tl.constexpr = actual_num_groups // 4
        mega_group_offsets = tl.arange(0, num_groups)
        mega_shifted = tl.where(
            mega_group_offsets < actual_num_groups,
            mega_scale_exponent << ((mega_group_offsets % 4) * 8),
            0,
        )
        mega_packed = tl.sum(tl.reshape(mega_shifted, (mega_num_packed, 4)), axis=1)
        mega_packed_offsets = tl.arange(0, mega_num_packed)
        tl.store(
            mega_mxfp8_scale_out_ptr
            + token_id * stride_mega_mxfp8_scale_t
            + mega_packed_offsets * stride_mega_mxfp8_scale_g,
            mega_packed,
            mask=mega_packed_offsets < mega_actual_packed,
        )


def _select_num_warps(H: int) -> int:
    if H <= 512:
        return 2
    if H <= 2048:
        return 4
    return 8


def _baseline_add_rmsnorm_fp8_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    scale_ue8m0: bool,
    mxfp8_semantics: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: baseline CUDA kernels for H > MAX_INREG_H."""
    import flashinfer.norm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    residual.add_(hidden_states)
    normed = flashinfer.norm.rmsnorm(residual, weight, eps=eps)
    if mxfp8_semantics:
        return _canonical_mxfp8_quant_act_packed(normed)
    return sgl_per_token_group_quant_fp8(
        normed,
        group_size=group_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )


def _baseline_add_rmsnorm_fp8_quant_with_bf16_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    scale_ue8m0: bool,
    mxfp8_semantics: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback: baseline CUDA kernels for H > MAX_INREG_H (dual output)."""
    import flashinfer.norm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    residual.add_(hidden_states)
    bf16_out = flashinfer.norm.rmsnorm(residual, weight, eps=eps)
    if mxfp8_semantics:
        fp8_out, scale = _canonical_mxfp8_quant_act_packed(bf16_out)
        return bf16_out, fp8_out, scale
    fp8_out, scale = sgl_per_token_group_quant_fp8(
        bf16_out,
        group_size=group_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return bf16_out, fp8_out, scale


def fused_add_rmsnorm_fp8_quant_with_bf16_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
    mxfp8_semantics: bool = False,
    mega_mxfp8_out: torch.Tensor | None = None,
    mega_mxfp8_scale_out: torch.Tensor | None = None,
    emit_fp32_output: bool = False,
    raw_gate_clear_out: torch.Tensor | None = None,
    round_residual_bf16: bool = False,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Same as ``fused_add_rmsnorm_fp8_quant`` but also returns bf16 normed."""
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    assert residual.shape == hidden_states.shape
    assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1]
    T, H = hidden_states.shape
    assert H % group_size == 0
    if mxfp8_semantics:
        assert group_size == MX_BLOCK and scale_ue8m0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    emit_mega_moe = mega_mxfp8_out is not None or mega_mxfp8_scale_out is not None
    if emit_mega_moe:
        if mega_mxfp8_out is None or mega_mxfp8_scale_out is None:
            raise ValueError(
                "mega_mxfp8_out and mega_mxfp8_scale_out must be provided together"
            )
        expected_scale_shape = (T, H // (4 * MX_BLOCK))
        if (
            not mxfp8_semantics
            or group_size != MX_BLOCK
            or H % (4 * MX_BLOCK) != 0
            or tuple(mega_mxfp8_out.shape) != (T, H)
            or mega_mxfp8_out.dtype != torch.float8_e4m3fn
            or not mega_mxfp8_out.is_contiguous()
            or tuple(mega_mxfp8_scale_out.shape) != expected_scale_shape
            or mega_mxfp8_scale_out.dtype != torch.int32
            or not mega_mxfp8_scale_out.is_contiguous()
            or mega_mxfp8_out.device != hidden_states.device
            or mega_mxfp8_scale_out.device != hidden_states.device
        ):
            raise ValueError("invalid HY4 MegaMoE activation/scale output ABI")
    clear_raw_gate = raw_gate_clear_out is not None
    if clear_raw_gate and (
        not mxfp8_semantics
        or group_size != MX_BLOCK
        or H != 6144
        or tuple(raw_gate_clear_out.shape) != (T, 32)
        or raw_gate_clear_out.dtype != torch.float32
        or not raw_gate_clear_out.is_contiguous()
        or raw_gate_clear_out.device != hidden_states.device
    ):
        raise ValueError("invalid HY4 raw head-gate clear output ABI")

    if out is not None and (H != 6144 or not mxfp8_semantics or emit_fp32_output):
        raise ValueError("preallocated outputs require the HY4 BF16/MXFP8 producer")

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H:
        if clear_raw_gate:
            raise ValueError(
                "raw head-gate clear requires the single-pass RMSNorm producer"
            )
        result = _baseline_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden_states,
            residual,
            weight,
            eps,
            group_size,
            scale_ue8m0,
            mxfp8_semantics,
        )
        if emit_mega_moe:
            from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts import (
                per_token_cast_to_fp8_packed_ue8m0,
            )

            mega_fp8, mega_scale = per_token_cast_to_fp8_packed_ue8m0(
                result[0].contiguous(), gran_k=MX_BLOCK
            )
            mega_mxfp8_out.copy_(mega_fp8)
            mega_mxfp8_scale_out.copy_(mega_scale)
        return (*result, result[0].float()) if emit_fp32_output else result

    fp32_out = (
        torch.empty((T, H), dtype=torch.float32, device=hidden_states.device)
        if emit_fp32_output
        else None
    )
    if out is None:
        bf16_out = torch.empty(
            (T, H), dtype=torch.bfloat16, device=hidden_states.device
        )
        fp8_out = torch.empty(
            (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
        )
        if mxfp8_semantics:
            scale_out = _create_mxfp8_packed_scale_output(T, H, hidden_states.device)
        else:
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=(T, H),
                device=hidden_states.device,
                group_size=group_size,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=scale_ue8m0,
            )
    else:
        bf16_out, fp8_out, scale_out = out
        assert mxfp8_semantics and not emit_fp32_output
        assert bf16_out.shape == fp8_out.shape == (T, H)
        assert bf16_out.dtype == torch.bfloat16 and fp8_out.dtype == torch.float8_e4m3fn
        assert bf16_out.is_contiguous() and fp8_out.is_contiguous()
        assert scale_out.shape == (T, H // 128) and scale_out.dtype == torch.int32
        assert scale_out.stride() == (1, (T + 3) // 4 * 4)
        assert (
            bf16_out.device
            == fp8_out.device
            == scale_out.device
            == hidden_states.device
        )
    if T == 0:
        if emit_fp32_output:
            assert fp32_out is not None
            return bf16_out, fp8_out, scale_out, fp32_out
        return bf16_out, fp8_out, scale_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    grid = (T,)

    _fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel[grid](
        hidden_states,
        residual,
        weight,
        bf16_out,
        fp32_out if emit_fp32_output else bf16_out,
        raw_gate_clear_out if clear_raw_gate else bf16_out,
        fp8_out,
        scale_out,
        mega_mxfp8_out if emit_mega_moe else fp8_out,
        mega_mxfp8_scale_out if emit_mega_moe else scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        hidden_states.stride(0),
        residual.stride(0),
        bf16_out.stride(0),
        fp32_out.stride(0) if emit_fp32_output else 0,
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        mega_mxfp8_out.stride(0) if emit_mega_moe else 0,
        mega_mxfp8_scale_out.stride(0) if emit_mega_moe else 0,
        mega_mxfp8_scale_out.stride(1) if emit_mega_moe else 0,
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_UE8M0=scale_ue8m0,
        MXFP8_SEMANTICS=mxfp8_semantics,
        HAS_MEGA_MOE_OUTPUT=emit_mega_moe,
        HAS_FP32_OUTPUT=emit_fp32_output,
        HAS_RAW_GATE_CLEAR=clear_raw_gate,
        ROUND_RESIDUAL_BF16=round_residual_bf16,
        num_warps=_select_num_warps(H),
    )
    if emit_fp32_output:
        assert fp32_out is not None
        return bf16_out, fp8_out, scale_out, fp32_out
    return bf16_out, fp8_out, scale_out


def fused_add_rmsnorm_fp8_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
    mxfp8_semantics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add-residual + RMSNorm + per-token-group FP8 quant.

    Modifies ``residual`` in-place (``residual += hidden_states``).
    Returns ``(fp8_output, scale)`` matching DeepGEMM's expected layout.
    """
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    assert residual.shape == hidden_states.shape
    assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1]
    T, H = hidden_states.shape
    assert H % group_size == 0
    if mxfp8_semantics:
        assert group_size == MX_BLOCK and scale_ue8m0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H:
        return _baseline_add_rmsnorm_fp8_quant(
            hidden_states,
            residual,
            weight,
            eps,
            group_size,
            scale_ue8m0,
            mxfp8_semantics,
        )

    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    if mxfp8_semantics:
        scale_out = _create_mxfp8_packed_scale_output(T, H, hidden_states.device)
    else:
        scale_out = create_per_token_group_quant_fp8_output_scale(
            x_shape=(T, H),
            device=hidden_states.device,
            group_size=group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )
    if T == 0:
        return fp8_out, scale_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    grid = (T,)

    _fused_add_rmsnorm_fp8_quant_singlepass_kernel[grid](
        hidden_states,
        residual,
        weight,
        fp8_out,
        scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        hidden_states.stride(0),
        residual.stride(0),
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_UE8M0=scale_ue8m0,
        MXFP8_SEMANTICS=mxfp8_semantics,
        num_warps=_select_num_warps(H),
    )
    return fp8_out, scale_out

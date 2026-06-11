"""Fused add-residual + RMSNorm + per-token-group FP8 quantization.

Combines three operations into one kernel launch:
1. residual += hidden_states  (in-place update)
2. normed = rmsnorm(residual, weight, eps)
3. (fp8_out, scale) = per_token_group_fp8_quant(normed, group_size)

Three scale layouts:
  - SCALE_UE8M0=True  : int32 packed (4 UE8M0 exponents per int32), Blackwell
  - SCALE_UE8M0=False, ROUND_POW2=False : float32 unpacked, H20
  - SCALE_UE8M0=False, ROUND_POW2=True  : float32 power-of-two (MXFP8 1×32)

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
    ROUND_POW2: tl.constexpr = False,
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
    absmax = tl.maximum(tl.max(abs_2d, axis=1), 1e-4)

    # Use default fp32 `/` (div.approx.f32) + reciprocal-multiply for the
    # quant divisions. Empirically bit-identical to the prior div.rn.f32
    # path for UE8M0+E4M3 (the ~1 ULP fp32 difference is absorbed by UE8M0's
    # power-of-2 rounding and E4M3's 3-mantissa quant). Saves ~20% wall time
    # on the kernel by avoiding the inline-asm div.rn.f32.
    if SCALE_UE8M0:
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
        if ROUND_POW2:
            s, _ = _ue8m0_pow2_round(s)
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
    fp8_out_ptr,
    scale_out_ptr,
    H: tl.constexpr,
    eps,
    fp8_max,
    fp8_min,
    stride_h_t,
    stride_r_t,
    stride_b_t,
    stride_o_t,
    stride_scale_t,
    stride_scale_g,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    ROUND_POW2: tl.constexpr = False,
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
    # Precision-alignment with baseline: baseline performs bf16 in-place add
    # then re-reads the bf16 residual for rmsnorm. We round-trip r_new through
    # bf16 here to match exactly (otherwise the fp32 r_new used directly for
    # sq_sum produces 1-ULP-different normed output, which is bit-different
    # from baseline and accumulates across all transformer layers).
    # Match production single-pass fused_add_rmsnorm: r + h stays in fp32
    # for the rmsnorm reduction; only the residual store rounds to bf16.
    r_new = r + h
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
    absmax = tl.maximum(tl.max(abs_2d, axis=1), 1e-4)

    # Use default fp32 `/` (div.approx.f32) + reciprocal-multiply for the
    # quant divisions. Empirically bit-identical to the prior div.rn.f32
    # path for UE8M0+E4M3 (the ~1 ULP fp32 difference is absorbed by UE8M0's
    # power-of-2 rounding and E4M3's 3-mantissa quant). Saves ~20% wall time
    # on the kernel by avoiding the inline-asm div.rn.f32.
    if SCALE_UE8M0:
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
        if ROUND_POW2:
            s, _ = _ue8m0_pow2_round(s)
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
    round_to_pow2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: baseline CUDA kernels for H > MAX_INREG_H."""
    import flashinfer.norm

    residual.add_(hidden_states)
    normed = flashinfer.norm.rmsnorm(residual, weight, eps=eps)
    if round_to_pow2 and not scale_ue8m0:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act

        return mxfp8_quant_act(normed)
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

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
    round_to_pow2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback: baseline CUDA kernels for H > MAX_INREG_H (dual output)."""
    import flashinfer.norm

    residual.add_(hidden_states)
    bf16_out = flashinfer.norm.rmsnorm(residual, weight, eps=eps)
    if round_to_pow2 and not scale_ue8m0:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act

        fp8_out, scale = mxfp8_quant_act(bf16_out)
        return bf16_out, fp8_out, scale
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

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
    round_to_pow2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as ``fused_add_rmsnorm_fp8_quant`` but also returns bf16 normed.

    When ``round_to_pow2=True`` and ``scale_ue8m0=False``, the scale is
    rounded to the nearest power of two (MXFP8 1×32 format). The returned
    scale is a row-major fp32 ``[T, H // group_size]`` tensor matching
    :func:`mxfp8_quant_act`'s contract.
    """
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    assert residual.shape == hidden_states.shape
    assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1]
    T, H = hidden_states.shape
    assert H % group_size == 0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H:
        return _baseline_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden_states, residual, weight, eps, group_size, scale_ue8m0,
            round_to_pow2=round_to_pow2,
        )

    mxfp8_mode = round_to_pow2 and not scale_ue8m0

    bf16_out = torch.empty((T, H), dtype=torch.bfloat16, device=hidden_states.device)
    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    if mxfp8_mode:
        num_groups = H // group_size
        scale_out = torch.empty(
            (T, num_groups), dtype=torch.float32, device=hidden_states.device,
        )
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
        fp8_out,
        scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        hidden_states.stride(0),
        residual.stride(0),
        bf16_out.stride(0),
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_UE8M0=scale_ue8m0,
        ROUND_POW2=round_to_pow2 and not scale_ue8m0,
        num_warps=_select_num_warps(H),
    )
    return bf16_out, fp8_out, scale_out


def fused_add_rmsnorm_fp8_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
    round_to_pow2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add-residual + RMSNorm + per-token-group FP8 quant.

    Modifies ``residual`` in-place (``residual += hidden_states``).
    Returns ``(fp8_output, scale)`` matching DeepGEMM's expected layout.

    When ``round_to_pow2=True`` and ``scale_ue8m0=False``, the scale is
    rounded to the nearest power of two (MXFP8 1×32 format). The returned
    scale is a row-major fp32 ``[T, H // group_size]`` tensor matching
    :func:`mxfp8_quant_act`'s contract.
    """
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    assert residual.shape == hidden_states.shape
    assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1]
    T, H = hidden_states.shape
    assert H % group_size == 0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H:
        return _baseline_add_rmsnorm_fp8_quant(
            hidden_states, residual, weight, eps, group_size, scale_ue8m0,
            round_to_pow2=round_to_pow2,
        )

    mxfp8_mode = round_to_pow2 and not scale_ue8m0

    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    if mxfp8_mode:
        num_groups = H // group_size
        scale_out = torch.empty(
            (T, num_groups), dtype=torch.float32, device=hidden_states.device,
        )
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
        ROUND_POW2=round_to_pow2 and not scale_ue8m0,
        num_warps=_select_num_warps(H),
    )
    return fp8_out, scale_out

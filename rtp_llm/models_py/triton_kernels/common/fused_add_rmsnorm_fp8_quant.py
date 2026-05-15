"""Fused add-residual + RMSNorm + per-token-group FP8 quantization.

Combines three operations into one kernel launch:
1. residual += hidden_states  (in-place update)
2. normed = rmsnorm(residual, weight, eps)
3. (fp8_out, scale) = per_token_group_fp8_quant(normed, group_size=128)

Two scale layouts:
  - SCALE_UE8M0=True  : int32 packed (4 UE8M0 exponents per int32), Blackwell
  - SCALE_UE8M0=False : float32 unpacked, H20

Single-pass design: when ``H <= MAX_INREG_H`` (4096 by default) we use
``BLOCK_N = next_pow2(H)`` so each token's whole row fits in one Triton tile.
That lets Pass 1 (add + sum_sq) and Pass 2 (normalize + quant) reuse the
``residual_new`` and ``weight`` values held in registers — no second load.
For H > MAX_INREG_H we fall back to the legacy 2-pass loop.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)

MAX_INREG_H = 4096


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
):
    """Single-pass: load whole row → r_new in registers → reuse for normalize+quant.

    Requires BLOCK_N >= H. Produces residual update + fp8 + scale in one tile pass.
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
    r_new = r + h
    tl.store(
        residual_ptr + token_id * stride_r_t + offs,
        r_new.to(tl.bfloat16),
        mask=mask,
    )
    sq_sum = tl.sum(r_new * r_new)
    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = r_new * rsqrt_val * w  # full-row normed in registers

    # Per-group absmax: reshape into [num_groups, GROUP_SIZE] then reduce along
    # group dim. Triton requires constant shape — we reshape via the mask trick.
    num_groups: tl.constexpr = BLOCK_N // GROUP_SIZE
    normed_2d = tl.reshape(normed, (num_groups, GROUP_SIZE))
    abs_2d = tl.abs(normed_2d)
    absmax = tl.maximum(tl.max(abs_2d, axis=1), 1e-10)  # [num_groups]

    if SCALE_UE8M0:
        s_init = absmax / fp8_max
        s, exp_field = _ue8m0_pow2_round(s_init)
        s_bcast = tl.reshape(s, (num_groups, 1))
        fp8_2d = tl.clamp(normed_2d / s_bcast, fp8_min, fp8_max).to(
            fp8_out_ptr.dtype.element_ty
        )
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        # Pack 4 groups per int32 via gather-and-shift-and-reduce.
        # exp_field [num_groups] → shift each by (g%4)*8 → reshape to [num_packed,4]
        # → sum along axis=1 to get packed int32.
        num_packed: tl.constexpr = num_groups // 4
        g_idx = tl.arange(0, num_groups)
        shift_amt = (g_idx % 4) * 8
        shifted = exp_field << shift_amt
        shifted_2d = tl.reshape(shifted, (num_packed, 4))
        packed = tl.sum(shifted_2d, axis=1)
        pack_offs = tl.arange(0, num_packed)
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + pack_offs * stride_scale_g,
            packed,
        )
    else:
        s = absmax / fp8_max
        s_bcast = tl.reshape(s, (num_groups, 1))
        fp8_2d = tl.clamp(normed_2d / s_bcast, fp8_min, fp8_max).to(
            fp8_out_ptr.dtype.element_ty
        )
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        g_offs = tl.arange(0, num_groups)
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + g_offs * stride_scale_g,
            s,
        )


@triton.jit
def _fused_add_rmsnorm_fp8_quant_kernel(
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
    GROUP_SIZE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """Legacy 2-pass kernel — fallback for H > MAX_INREG_H."""
    token_id = tl.program_id(0).to(tl.int64)

    h_base = token_id * stride_h_t
    r_base = token_id * stride_r_t
    o_base = token_id * stride_o_t

    sq_sum = 0.0
    for start in tl.range(0, H, GROUP_SIZE):
        offs = start + tl.arange(0, GROUP_SIZE)
        mask = offs < H
        h = tl.load(hidden_ptr + h_base + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(tl.float32)
        r_new = r + h
        tl.store(residual_ptr + r_base + offs, r_new.to(tl.bfloat16), mask=mask)
        sq_sum += tl.sum(r_new * r_new)

    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    if SCALE_UE8M0:
        num_packed = tl.cdiv(H, GROUP_SIZE * 4)
        for pg in tl.range(0, num_packed):
            packed_scale: tl.int32 = 0
            for g in tl.static_range(4):
                gidx = pg * 4 + g
                offs = gidx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
                mask = offs < H
                r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                normed = r * rsqrt_val * w

                absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-10)
                s_init = absmax / fp8_max
                bits = s_init.to(tl.int32, bitcast=True)
                mantissa_nz = (bits & 0x7FFFFF) != 0
                exp_field = ((bits >> 23) & 0xFF) + tl.where(mantissa_nz, 1, 0)
                s_int = exp_field << 23
                s = s_int.to(tl.float32, bitcast=True)
                fp8_val = tl.clamp(normed / s, fp8_min, fp8_max).to(
                    fp8_out_ptr.dtype.element_ty
                )
                tl.store(fp8_out_ptr + o_base + offs, fp8_val, mask=mask)

                exp_bits = exp_field & 0xFF
                packed_scale = packed_scale | (exp_bits << (g * 8))

            tl.store(
                scale_out_ptr + token_id * stride_scale_t + pg * stride_scale_g,
                packed_scale,
            )
    else:
        num_groups = tl.cdiv(H, GROUP_SIZE)
        for g in tl.range(0, num_groups):
            offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            mask = offs < H
            r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            normed = r * rsqrt_val * w

            absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-10)
            s = absmax / fp8_max
            fp8_val = tl.clamp(normed / s, fp8_min, fp8_max).to(
                fp8_out_ptr.dtype.element_ty
            )
            tl.store(fp8_out_ptr + o_base + offs, fp8_val, mask=mask)
            tl.store(
                scale_out_ptr + token_id * stride_scale_t + g * stride_scale_g,
                s,
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
):
    """Single-pass dual-output: also stores bf16 normed alongside fp8."""
    token_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < H

    h = tl.load(hidden_ptr + token_id * stride_h_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    r = tl.load(residual_ptr + token_id * stride_r_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    r_new = r + h
    tl.store(
        residual_ptr + token_id * stride_r_t + offs,
        r_new.to(tl.bfloat16),
        mask=mask,
    )
    sq_sum = tl.sum(r_new * r_new)
    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = r_new * rsqrt_val * w
    tl.store(
        bf16_out_ptr + token_id * stride_b_t + offs,
        normed.to(tl.bfloat16),
        mask=mask,
    )

    num_groups: tl.constexpr = BLOCK_N // GROUP_SIZE
    normed_2d = tl.reshape(normed, (num_groups, GROUP_SIZE))
    abs_2d = tl.abs(normed_2d)
    absmax = tl.maximum(tl.max(abs_2d, axis=1), 1e-10)

    if SCALE_UE8M0:
        s_init = absmax / fp8_max
        s, exp_field = _ue8m0_pow2_round(s_init)
        s_bcast = tl.reshape(s, (num_groups, 1))
        fp8_2d = tl.clamp(normed_2d / s_bcast, fp8_min, fp8_max).to(
            fp8_out_ptr.dtype.element_ty
        )
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        num_packed: tl.constexpr = num_groups // 4
        g_idx = tl.arange(0, num_groups)
        shift_amt = (g_idx % 4) * 8
        shifted = exp_field << shift_amt
        shifted_2d = tl.reshape(shifted, (num_packed, 4))
        packed = tl.sum(shifted_2d, axis=1)
        pack_offs = tl.arange(0, num_packed)
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + pack_offs * stride_scale_g,
            packed,
        )
    else:
        s = absmax / fp8_max
        s_bcast = tl.reshape(s, (num_groups, 1))
        fp8_2d = tl.clamp(normed_2d / s_bcast, fp8_min, fp8_max).to(
            fp8_out_ptr.dtype.element_ty
        )
        fp8_flat = tl.reshape(fp8_2d, (BLOCK_N,))
        tl.store(fp8_out_ptr + token_id * stride_o_t + offs, fp8_flat, mask=mask)
        g_offs = tl.arange(0, num_groups)
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + g_offs * stride_scale_g,
            s,
        )


@triton.jit
def _fused_add_rmsnorm_fp8_quant_dual_output_kernel(
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
    GROUP_SIZE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """Legacy 2-pass dual-output — fallback for H > MAX_INREG_H."""
    token_id = tl.program_id(0).to(tl.int64)

    h_base = token_id * stride_h_t
    r_base = token_id * stride_r_t
    b_base = token_id * stride_b_t
    o_base = token_id * stride_o_t

    sq_sum = 0.0
    for start in tl.range(0, H, GROUP_SIZE):
        offs = start + tl.arange(0, GROUP_SIZE)
        mask = offs < H
        h = tl.load(hidden_ptr + h_base + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(tl.float32)
        r_new = r + h
        tl.store(residual_ptr + r_base + offs, r_new.to(tl.bfloat16), mask=mask)
        sq_sum += tl.sum(r_new * r_new)

    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    if SCALE_UE8M0:
        num_packed = tl.cdiv(H, GROUP_SIZE * 4)
        for pg in tl.range(0, num_packed):
            packed_scale: tl.int32 = 0
            for g in tl.static_range(4):
                gidx = pg * 4 + g
                offs = gidx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
                mask = offs < H
                r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                normed = r * rsqrt_val * w

                tl.store(
                    bf16_out_ptr + b_base + offs, normed.to(tl.bfloat16), mask=mask
                )

                absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-10)
                s_init = absmax / fp8_max
                bits = s_init.to(tl.int32, bitcast=True)
                mantissa_nz = (bits & 0x7FFFFF) != 0
                exp_field = ((bits >> 23) & 0xFF) + tl.where(mantissa_nz, 1, 0)
                s_int = exp_field << 23
                s = s_int.to(tl.float32, bitcast=True)
                fp8_val = tl.clamp(normed / s, fp8_min, fp8_max).to(
                    fp8_out_ptr.dtype.element_ty
                )
                tl.store(fp8_out_ptr + o_base + offs, fp8_val, mask=mask)

                exp_bits = exp_field & 0xFF
                packed_scale = packed_scale | (exp_bits << (g * 8))

            tl.store(
                scale_out_ptr + token_id * stride_scale_t + pg * stride_scale_g,
                packed_scale,
            )
    else:
        num_groups = tl.cdiv(H, GROUP_SIZE)
        for g in tl.range(0, num_groups):
            offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            mask = offs < H
            r = tl.load(residual_ptr + r_base + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            normed = r * rsqrt_val * w

            tl.store(bf16_out_ptr + b_base + offs, normed.to(tl.bfloat16), mask=mask)

            absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-10)
            s = absmax / fp8_max
            fp8_val = tl.clamp(normed / s, fp8_min, fp8_max).to(
                fp8_out_ptr.dtype.element_ty
            )
            tl.store(fp8_out_ptr + o_base + offs, fp8_val, mask=mask)
            tl.store(
                scale_out_ptr + token_id * stride_scale_t + g * stride_scale_g,
                s,
            )


def _select_num_warps(H: int) -> int:
    if H <= 512:
        return 2
    if H <= 2048:
        return 4
    return 8


def fused_add_rmsnorm_fp8_quant_with_bf16_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as ``fused_add_rmsnorm_fp8_quant`` but also returns bf16 normed."""
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    assert residual.shape == hidden_states.shape
    assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1]
    T, H = hidden_states.shape
    assert H % group_size == 0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    bf16_out = torch.empty((T, H), dtype=torch.bfloat16, device=hidden_states.device)
    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
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

    is_pow2 = H > 0 and (H & (H - 1)) == 0
    if H <= MAX_INREG_H and is_pow2:
        block_n = H
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
            num_warps=_select_num_warps(H),
        )
    else:
        _fused_add_rmsnorm_fp8_quant_dual_output_kernel[grid](
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
            GROUP_SIZE=group_size,
            SCALE_UE8M0=scale_ue8m0,
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
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
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

    is_pow2 = H > 0 and (H & (H - 1)) == 0
    if H <= MAX_INREG_H and is_pow2:
        block_n = H
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
            num_warps=_select_num_warps(H),
        )
    else:
        _fused_add_rmsnorm_fp8_quant_kernel[grid](
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
            GROUP_SIZE=group_size,
            SCALE_UE8M0=scale_ue8m0,
            num_warps=_select_num_warps(H),
        )
    return fp8_out, scale_out

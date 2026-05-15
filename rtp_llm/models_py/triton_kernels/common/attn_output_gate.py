"""Fused sigmoid-gate-mul kernel for Qwen3.5 attention output gate.

Replaces:
    attn_output = attn_output * torch.sigmoid(gate)
which launches two kernels (sigmoid + mul) with one fused triton kernel
that mutates attn_output in place.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)

_MIN_TOTAL_PROGRAMS = 512
_MIN_BLOCK_H = 128
_MAX_BLOCK_H = 4096


@triton.jit
def _SigmoidMulInplace_kernel(
    out_ptr,  # [T, H]  — attn_output, modified in-place
    gate_ptr,  # [T, H]  — gate output from the attn output-gate linear
    T,
    H,
    stride_out_t,
    stride_gate_t,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[t, :] = out[t, :] * sigmoid(gate[t, :])

    Grid: (T, ceil(H / BLOCK_H))
    """
    tid = tl.program_id(axis=0)
    hid = tl.program_id(axis=1)

    h_offsets = hid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    out_base = out_ptr + tid * stride_out_t
    gate_base = gate_ptr + tid * stride_gate_t

    out_vec = tl.load(out_base + h_offsets, mask=mask, other=0.0)
    gate_vec = tl.load(gate_base + h_offsets, mask=mask, other=0.0)

    sig = tl.sigmoid(gate_vec.to(tl.float32))
    result = out_vec.to(tl.float32) * sig
    tl.store(out_base + h_offsets, result.to(out_vec.dtype), mask=mask)


def _select_block_h(T: int, H: int) -> int:
    target_h_blocks = max(1, _MIN_TOTAL_PROGRAMS // max(T, 1))
    ideal = triton.next_power_of_2(max(1, H // target_h_blocks))
    return max(_MIN_BLOCK_H, min(_MAX_BLOCK_H, ideal))


def sigmoid_mul_inplace_triton(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Compute in-place on *attn_output*:
        attn_output[t, :] = attn_output[t, :] * sigmoid(gate[t, :])

    Args:
        attn_output: [T, H] — modified in-place.
        gate:        [T, H] — same shape as attn_output.

    Returns:
        attn_output (same object).
    """
    assert (
        attn_output.shape == gate.shape
    ), f"shape mismatch: {attn_output.shape} vs {gate.shape}"
    assert attn_output.is_cuda and gate.is_cuda
    assert attn_output.dim() == 2

    T, H = attn_output.shape
    if T == 0 or H == 0:
        return attn_output

    BLOCK_H = _select_block_h(T, H)
    grid = (T, triton.cdiv(H, BLOCK_H))
    _SigmoidMulInplace_kernel[grid](
        attn_output,
        gate,
        T,
        H,
        attn_output.stride(0),
        gate.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return attn_output


@triton.jit
def _ue8m0_pow2_round_scalar(s_init):
    bits = s_init.to(tl.int32, bitcast=True)
    mantissa_nz = (bits & 0x7FFFFF) != 0
    exp_field = (bits >> 23) & 0xFF
    exp_field = exp_field + tl.where(mantissa_nz, 1, 0)
    s_int = exp_field << 23
    return s_int.to(tl.float32, bitcast=True), exp_field & 0xFF


@triton.jit
def _sigmoid_mul_fp8_quant_kernel(
    attn_ptr,
    gate_ptr,
    fp8_out_ptr,
    scale_out_ptr,
    H,
    fp8_max,
    fp8_min,
    stride_attn_t,
    stride_gate_t,
    stride_fp8_t,
    stride_scale_t,
    stride_scale_g,
    BLOCK_N: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """Fused sigmoid-mul + per-token-group FP8 quant.

    Grid: (num_blocks, T).
    Each program handles one (token, group_block) tile.
    """
    block_id = tl.program_id(0)
    token_id = tl.program_id(1).to(tl.int64)

    attn_base = attn_ptr + token_id * stride_attn_t
    gate_base = gate_ptr + token_id * stride_gate_t
    fp8_base = fp8_out_ptr + token_id * stride_fp8_t

    if SCALE_UE8M0:
        base_group_idx = block_id * 4
        packed_scale: tl.int32 = 0
        for g in tl.static_range(4):
            group_idx = base_group_idx + g
            offs = group_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < H
            a = tl.load(attn_base + offs, mask=mask, other=0.0).to(tl.float32)
            g_val = tl.load(gate_base + offs, mask=mask, other=0.0).to(tl.float32)
            result = a * tl.sigmoid(g_val)
            _absmax = tl.maximum(tl.max(tl.abs(result)), 1e-10)
            s_init = _absmax / fp8_max
            s, exp_bits = _ue8m0_pow2_round_scalar(s_init)
            fp8_val = tl.clamp(result / s, fp8_min, fp8_max).to(
                fp8_out_ptr.dtype.element_ty
            )
            tl.store(fp8_base + offs, fp8_val, mask=mask)
            packed_scale = packed_scale | (exp_bits << (g * 8))
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + block_id * stride_scale_g,
            packed_scale,
        )
    else:
        group_idx = block_id
        offs = group_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < H
        a = tl.load(attn_base + offs, mask=mask, other=0.0).to(tl.float32)
        g_val = tl.load(gate_base + offs, mask=mask, other=0.0).to(tl.float32)
        result = a * tl.sigmoid(g_val)
        _absmax = tl.maximum(tl.max(tl.abs(result)), 1e-10)
        s = _absmax / fp8_max
        fp8_val = tl.clamp(result / s, fp8_min, fp8_max).to(
            fp8_out_ptr.dtype.element_ty
        )
        tl.store(fp8_base + offs, fp8_val, mask=mask)
        tl.store(
            scale_out_ptr + token_id * stride_scale_t + group_idx * stride_scale_g,
            s,
        )


_SIGMOID_MUL_FP8_QUANT_M_THRESHOLD = 1024


def sigmoid_mul_fp8_quant_fwd(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid-mul + per-token-group FP8 quantization.

    Computes: result = attn_output * sigmoid(gate), then quantizes to fp8.
    Falls back to unfused path for large T (prefill) where the fused kernel
    is slower than baseline.

    Returns:
        (fp8_output, scale) matching DeepGEMM's expected layout.
    """
    assert attn_output.shape == gate.shape
    assert attn_output.dim() == 2
    T, H = attn_output.shape
    assert H % quant_group_size == 0
    num_groups = H // quant_group_size

    if T >= _SIGMOID_MUL_FP8_QUANT_M_THRESHOLD:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        sigmoid_mul_inplace_triton(attn_output, gate)
        return sgl_per_token_group_quant_fp8(
            attn_output,
            group_size=quant_group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )

    fp8_out = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=attn_output.device)
    scale_out = create_per_token_group_quant_fp8_output_scale(
        x_shape=(T, H),
        device=attn_output.device,
        group_size=quant_group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if T == 0:
        return fp8_out, scale_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    if scale_ue8m0:
        assert num_groups % 4 == 0
        num_blocks = num_groups // 4
    else:
        num_blocks = num_groups

    grid = (num_blocks, T)
    _sigmoid_mul_fp8_quant_kernel[grid](
        attn_output,
        gate,
        fp8_out,
        scale_out,
        H,
        fp8_max,
        fp8_min,
        attn_output.stride(0),
        gate.stride(0),
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=quant_group_size,
        SCALE_UE8M0=scale_ue8m0,
    )
    return fp8_out, scale_out

"""Fused RmsNormGated + per-token-group FP8 quantization.

Combines RmsNormGated (RMSNorm + silu gating) and fp8 quantization into one
kernel, eliminating the bf16 intermediate between norm and out_proj GEMM.

Input layout:  x, gate are [M, head_v_dim] where M = T * num_heads.
Output layout: fp8_out is [T, num_heads * head_v_dim], scale matches DeepGEMM.

Grid choice:
  - Small M (decode, M ≤ DECODE_M_THRESHOLD): grid = (M,) — one program per
    head-row gives plenty of parallelism for SM utilisation. Each program
    only processes head_v_dim elements which is tiny but launch latency wins.
  - Large M (prefill): grid = (T,), one program per token covers all heads.
    Avoids the 32K-program blow-up of (M,) grid at prefill, where launching
    32K kernels of 256 bytes each leaves the GPU mostly idle.

UE8M0 packs 4 group-scales per int32. The (M,) decode kernel needs ``atomic_or``
to merge across head-programs writing the same packed int32; the (T,) prefill
kernel computes the full pack inside a single program — no atomics.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)

# Threshold: below this M (= T * num_heads) we use the per-row (M,) grid;
# above we use the per-token (T,) grid that processes all heads sequentially.
# Tuned so decode (M ~ batch * num_heads ≤ 256) stays on the parallel path.
DECODE_M_THRESHOLD = 1024


@triton.jit
def _ieee_rn_div_f32(x, y):
    """IEEE round-to-nearest-even fp32 division (matches sgl CUDA `/`)."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ue8m0_pow2_round_scalar(s_init):
    """Round positive fp32 up to nearest power of 2 via bit hack."""
    bits = s_init.to(tl.int32, bitcast=True)
    mantissa_nz = (bits & 0x7FFFFF) != 0
    exp_field = ((bits >> 23) & 0xFF) + tl.where(mantissa_nz, 1, 0)
    s_int = exp_field << 23
    return s_int.to(tl.float32, bitcast=True), exp_field & 0xFF


@triton.jit
def _fused_rmsnorm_gated_fp8_quant_perrow_kernel(
    X,
    Z,
    W,
    fp8_out,
    scale_out,
    HEAD_V_DIM: tl.constexpr,
    eps,
    fp8_max,
    fp8_min,
    num_heads,
    groups_per_head,
    stride_x_row,
    stride_z_token,
    stride_o_row,
    stride_scale_t,
    stride_scale_g,
    QUANT_GROUP: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """One program per (token, head). Used for small M (decode)."""
    row = tl.program_id(0).to(tl.int64)
    token_id = row // num_heads
    head_id = row % num_heads

    x_base = row * stride_x_row
    z_base = token_id * stride_z_token + head_id * HEAD_V_DIM
    o_base = row * stride_o_row

    sq_sum = 0.0
    for start in tl.range(0, HEAD_V_DIM, QUANT_GROUP):
        offs = start + tl.arange(0, QUANT_GROUP)
        mask = offs < HEAD_V_DIM
        x = tl.load(X + x_base + offs, mask=mask, other=0.0).to(tl.float32)
        sq_sum += tl.sum(x * x)
    # Match RmsNormGated baseline: ``rstd = 1 / tl.sqrt(var + eps)`` (NOT
    # ``tl.rsqrt`` which uses the lower-precision ``rsqrt.approx`` PTX op).
    rsqrt_val = 1.0 / tl.sqrt(sq_sum / HEAD_V_DIM + eps)

    if SCALE_UE8M0:
        for g in tl.range(0, groups_per_head):
            offs = g * QUANT_GROUP + tl.arange(0, QUANT_GROUP)
            mask = offs < HEAD_V_DIM
            x = tl.load(X + x_base + offs, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(Z + z_base + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

            # Match RmsNormGated baseline ordering exactly (non-associative fp):
            # baseline computes ``y = (x * rstd * w); y *= z * tl.sigmoid(z)``
            # so the gate term ``z * sigmoid(z)`` is computed FIRST then
            # multiplied into the normed value.
            normed = x * rsqrt_val * w
            gate_term = z * tl.sigmoid(z)
            normed = normed * gate_term
            # bf16 round-trip: baseline RmsNormGated stores fp32 result to a
            # bf16 buffer, sgl_per_token_group_quant_fp8 reads bf16 back. To
            # bit-match we must round to bf16 BEFORE quant.
            normed = normed.to(tl.bfloat16).to(tl.float32)

            # Match sgl_per_token_group_quant_fp8 byte-exact:
            #   absmax = max(eps, max(|val|))     # fp32, eps=1e-10 floor
            #   y_s    = absmax / fp8_max          # IEEE-RNE fp32 div
            #   q      = clamp(val / y_s,...).to(fp8)  # IEEE-RNE fp32 div per-elem
            # Triton's default fp32 `/` is ``div.approx.f32`` (~1 ULP off);
            # fp64-promote the divisions to get IEEE-RNE that bit-matches
            # sgl's CUDA-default IEEE-RNE division.
            absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-4)
            s_init = _ieee_rn_div_f32(absmax, fp8_max)
            s, exp_bits = _ue8m0_pow2_round_scalar(s_init)
            fp8_val = tl.clamp(
                _ieee_rn_div_f32(normed, tl.full(normed.shape, s, tl.float32)),
                fp8_min,
                fp8_max,
            ).to(fp8_out.dtype.element_ty)
            tl.store(fp8_out + o_base + offs, fp8_val, mask=mask)

            global_group = head_id * groups_per_head + g
            packed_idx = global_group // 4
            byte_idx = global_group % 4
            shift = byte_idx * 8
            packed_val = exp_bits << shift
            tl.atomic_or(
                scale_out + token_id * stride_scale_t + packed_idx * stride_scale_g,
                packed_val,
            )
    else:
        for g in tl.range(0, groups_per_head):
            offs = g * QUANT_GROUP + tl.arange(0, QUANT_GROUP)
            mask = offs < HEAD_V_DIM
            x = tl.load(X + x_base + offs, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(Z + z_base + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

            # Match RmsNormGated baseline ordering exactly (non-associative fp):
            # baseline computes ``y = (x * rstd * w); y *= z * tl.sigmoid(z)``
            # so the gate term ``z * sigmoid(z)`` is computed FIRST then
            # multiplied into the normed value.
            normed = x * rsqrt_val * w
            gate_term = z * tl.sigmoid(z)
            normed = normed * gate_term
            # See SCALE_UE8M0 branch above for bf16 round-trip rationale.
            normed = normed.to(tl.bfloat16).to(tl.float32)

            absmax = tl.maximum(tl.max(tl.abs(normed)), 1e-4)
            s = _ieee_rn_div_f32(absmax, fp8_max)
            fp8_val = tl.clamp(
                _ieee_rn_div_f32(normed, tl.full(normed.shape, s, tl.float32)),
                fp8_min,
                fp8_max,
            ).to(fp8_out.dtype.element_ty)
            tl.store(fp8_out + o_base + offs, fp8_val, mask=mask)
            global_group = head_id * groups_per_head + g
            tl.store(
                scale_out + token_id * stride_scale_t + global_group * stride_scale_g,
                s,
            )


def fused_rmsnorm_gated_fp8_quant(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_heads: int,
    quant_group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RmsNormGated + per-token-group FP8 quantization.

    Args:
        x:    [M, head_v_dim] bf16, M = T * num_heads.
        gate: [M, head_v_dim] bf16 OR [T, num_heads*head_v_dim] (strided OK).
              When gate is [T, H_total], the kernel reads it with the correct
              stride — no .contiguous() copy needed.
        weight: [head_v_dim] bf16, RMSNorm weight.
        eps:  RMSNorm epsilon.
        num_heads: number of attention heads per token.
        quant_group_size: fp8 quantization group size (default 128).
        scale_ue8m0: True for UE8M0 packed scale, False for fp32.

    Returns:
        (fp8_out, scale):
            fp8_out — [T, num_heads * head_v_dim] float8_e4m3fn
            scale   — layout from create_per_token_group_quant_fp8_output_scale
    """
    assert x.dim() == 2 and gate.dim() == 2
    M, N = x.shape
    assert M % num_heads == 0
    T = M // num_heads
    head_v_dim = N
    H_total = num_heads * head_v_dim

    # gate can be [M, head_v_dim] (same shape as x) or [T, H_total] (strided)
    if gate.shape[0] == M and gate.shape[1] == N:
        stride_z_token = gate.stride(0) * num_heads
    elif gate.shape[0] == T and gate.shape[1] == H_total:
        stride_z_token = gate.stride(0)
    else:
        raise ValueError(
            f"gate shape {gate.shape} not compatible with x shape {x.shape} "
            f"and num_heads={num_heads}"
        )

    assert N % quant_group_size == 0
    assert weight.dim() == 1 and weight.shape[0] == N

    groups_per_head = N // quant_group_size

    if scale_ue8m0:
        total_groups = num_heads * groups_per_head
        assert (
            total_groups % 4 == 0
        ), f"UE8M0 requires total groups per token ({total_groups}) divisible by 4"

    fp8_out = torch.empty((T, H_total), dtype=torch.float8_e4m3fn, device=x.device)
    scale_out = create_per_token_group_quant_fp8_output_scale(
        x_shape=(T, H_total),
        device=x.device,
        group_size=quant_group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if scale_ue8m0:
        scale_out.zero_()

    if T == 0:
        return fp8_out, scale_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = finfo.max, -finfo.max

    # Dispatch: per-row Triton grid wins for small M (decode); for large M
    # (prefill) the 32K-program launch is slower than baseline RmsNormGated
    # + per-token-group quant (CUDA, single launch each). Measured perrow
    # prefill ~308us/call vs baseline ~30us/call; pertoken (T,) grid was
    # also worse (~586us/call) due to per-program serial work.
    if M > DECODE_M_THRESHOLD:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        norm = RmsNormGated(weight, eps=eps, group_size=N)
        # RmsNormGated expects gate as [M, head_v_dim]; reshape if needed
        gate_for_norm = (
            gate.reshape(M, N).contiguous() if gate.shape != (M, N) else gate
        )
        normed = norm(x, gate_for_norm)
        flat = normed.reshape(T, H_total).contiguous()
        return sgl_per_token_group_quant_fp8(
            flat,
            group_size=quant_group_size,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )

    fp8_view = fp8_out.view(M, N)
    grid = (M,)
    _fused_rmsnorm_gated_fp8_quant_perrow_kernel[grid](
        x,
        gate,
        weight,
        fp8_view,
        scale_out,
        N,
        eps,
        fp8_max,
        fp8_min,
        num_heads,
        groups_per_head,
        x.stride(0),
        stride_z_token,
        fp8_view.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        QUANT_GROUP=quant_group_size,
        SCALE_UE8M0=scale_ue8m0,
        num_warps=4,
    )

    return fp8_out, scale_out

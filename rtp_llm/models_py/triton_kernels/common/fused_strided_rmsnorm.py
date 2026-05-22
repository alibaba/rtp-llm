"""Fused strided RMSNorm + (optional) per-token-group FP8 quantization.

Targets the DeepSeek-V3.2 MLA path where ``torch.split(fused_qkv, ...)``
produces strided (non-contiguous) views that today need an extra
``.contiguous()`` copy before RMSNorm. Three flavors are exposed:

  - ``fused_strided_rmsnorm``                : bf16 normed output
                                               (replaces F2 = compressed_kv path
                                               and F1a = q-path when q_b_proj/wq_b
                                               are bf16).
  - ``fused_strided_rmsnorm_per_token_fp8_quant`` : fp8 + scale only
                                               (F1b single-output: when both
                                               q_b_proj and wq_b are fp8 and
                                               share scale_ue8m0).
  - ``fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output``: bf16 + fp8 + scale
                                               (F1b dual-output: when one
                                               consumer wants bf16 (e.g. RoPE)
                                               and the other wants fp8).

All three accept a strided 2-D input (``stride_x_t = x.stride(0)``) so the
caller can pass a slice from ``torch.split`` directly without forcing
``.contiguous()``.

Single-pass kernel design: ``BLOCK_N = next_power_of_2(H)`` so the entire row
fits in one Triton tile. The wrappers route to the unfused baseline
(``.contiguous() + flashinfer.norm.rmsnorm``) only when shape constraints
genuinely don't fit the fast path:

  - ``H > MAX_INREG_H = 8192``         (row doesn't fit in registers)
  - ``x.stride(-1) != 1``              (last dim must be contiguous)
  - ``H % group_size != 0``            (fp8 quant variants only)
  - UE8M0 with ``num_groups % 4 != 0`` (fp8 quant variants only)

The Triton fast path is otherwise enabled for all shapes. Bench (idle H20)
shows ~1.0x at T=1 and 1.4-3.5x at T>=8 across H=256..6144 — basically
always tied or a win. Earlier observed 0.4x slowdowns turned out to be GPU
contention noise, not a real loss zone.
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
def _ue8m0_pow2_round(s_init):
    """Round a positive fp32 value up to the nearest power of 2 via bit hack."""
    bits = s_init.to(tl.int32, bitcast=True)
    mantissa_nz = (bits & 0x7FFFFF) != 0
    exp_field = (bits >> 23) & 0xFF
    exp_field = exp_field + tl.where(mantissa_nz, 1, 0)
    s_int = exp_field << 23
    return s_int.to(tl.float32, bitcast=True), exp_field & 0xFF


@triton.jit
def _fused_strided_rmsnorm_singlepass_kernel(
    x_ptr,
    weight_ptr,
    bf16_out_ptr,
    fp8_out_ptr,
    scale_out_ptr,
    H: tl.constexpr,
    eps,
    fp8_max,
    fp8_min,
    stride_x_t,
    stride_b_t,
    stride_o_t,
    stride_scale_t,
    stride_scale_g,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BF16_OUT: tl.constexpr,
    HAS_FP8_OUT: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """Single-pass strided RMSNorm with optional fp8 quant.

    Reads x via ``stride_x_t`` (token row stride). The fast path keeps the row
    in registers and emits bf16, fp8, or both depending on compile-time flags.
    """
    token_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < H

    x = tl.load(x_ptr + token_id * stride_x_t + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    sq_sum = tl.sum(x * x)
    rsqrt_val = tl.rsqrt(sq_sum / H + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Round normed through bf16 BEFORE the fp8 quant: baseline path is
    # ``flashinfer.norm.rmsnorm(...) → bf16 q_c → sgl_per_token_group_quant_fp8(q_c)``,
    # so the fp8 quant input is already bf16-rounded. Without this round-trip
    # we quantize the higher-precision fp32 ``x * rsqrt * w`` and produce ~2% of
    # bytes that mismatch the baseline fp8.
    normed_bf16 = (x * rsqrt_val * w).to(tl.bfloat16)
    normed = normed_bf16.to(tl.float32)

    if HAS_BF16_OUT:
        tl.store(
            bf16_out_ptr + token_id * stride_b_t + offs,
            normed_bf16,
            mask=mask,
        )

    if HAS_FP8_OUT:
        num_groups: tl.constexpr = BLOCK_N // GROUP_SIZE
        actual_num_groups: tl.constexpr = H // GROUP_SIZE
        normed_2d = tl.reshape(normed, (num_groups, GROUP_SIZE))
        abs_2d = tl.abs(normed_2d)
        # See ``fused_add_rmsnorm_fp8_quant.py`` for the full rationale: drop
        # the ``1e-10`` floor (fp64 promotion in Triton) and run both
        # divisions through fp64 to get IEEE-RNE-correct fp32 scale + fp8
        # bucket, bit-aligned with sgl_per_token_group_quant_fp8.
        absmax = tl.maximum(tl.max(abs_2d, axis=1), 1e-4)

        # Match sgl_per_token_group_quant_fp8 byte-exact: IEEE-RNE fp32 div
        # for both scale and per-element val/scale, fp64-promoted to escape
        # Triton's ``div.approx.f32`` default.
        if SCALE_UE8M0:
            s_init = _ieee_rn_div_f32(
                absmax, tl.full(absmax.shape, fp8_max, tl.float32)
            )
            s, exp_field = _ue8m0_pow2_round(s_init)
            s_bcast = tl.reshape(s, (num_groups, 1))
            s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
            fp8_2d = tl.clamp(
                _ieee_rn_div_f32(normed_2d, s_full),
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
            s = _ieee_rn_div_f32(absmax, tl.full(absmax.shape, fp8_max, tl.float32))
            s_bcast = tl.reshape(s, (num_groups, 1))
            s_full = tl.broadcast_to(s_bcast, (num_groups, GROUP_SIZE))
            fp8_2d = tl.clamp(
                _ieee_rn_div_f32(normed_2d, s_full),
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


def _baseline_strided_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fallback baseline: .contiguous() + flashinfer.norm.rmsnorm."""
    import flashinfer.norm

    return flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=eps)


def _baseline_strided_rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer.norm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    normed = flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=eps)
    return sgl_per_token_group_quant_fp8(
        normed,
        group_size=group_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )


def _baseline_strided_rmsnorm_fp8_quant_with_bf16_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer.norm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    bf16_out = flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=eps)
    fp8_out, scale = sgl_per_token_group_quant_fp8(
        bf16_out,
        group_size=group_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    return bf16_out, fp8_out, scale


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def fused_strided_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm over a (possibly strided) 2-D input. Returns bf16 normed output.

    Args:
        x:      [T, H] strided OK. ``x.stride(0)`` is read inside the kernel.
                Last dim must be contiguous (stride 1).
        weight: [H] bf16.
        eps:    RMSNorm epsilon.

    Returns:
        bf16 normed output, contiguous shape ``[T, H]``.

    Falls back to ``.contiguous() + flashinfer.norm.rmsnorm`` for ``H > 8192``
    or when ``x.stride(-1) != 1``.
    """
    assert x.dim() == 2
    assert weight.dim() == 1 and weight.shape[0] == x.shape[1]
    T, H = x.shape

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H or x.stride(-1) != 1:
        return _baseline_strided_rmsnorm(x, weight, eps)

    bf16_out = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return bf16_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    grid = (T,)

    _fused_strided_rmsnorm_singlepass_kernel[grid](
        x,
        weight,
        bf16_out,
        bf16_out,  # fp8_out_ptr unused
        bf16_out,  # scale_out_ptr unused
        H,
        eps,
        fp8_max,
        fp8_min,
        x.stride(0),
        bf16_out.stride(0),
        bf16_out.stride(0),  # stride_o_t unused
        0,
        0,
        BLOCK_N=block_n,
        GROUP_SIZE=128,
        HAS_BF16_OUT=True,
        HAS_FP8_OUT=False,
        SCALE_UE8M0=False,
        num_warps=_select_num_warps(H),
    )
    return bf16_out


def fused_strided_rmsnorm_per_token_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token-group fp8 quant on a strided input.

    Use when the only consumer of the normed output is an fp8 GEMM (or when
    multiple fp8 GEMMs share the same scale_ue8m0). Returns ``(fp8, scale)``
    in DeepGEMM's expected layout.

    Falls back to ``.contiguous() + rmsnorm + sgl_per_token_group_quant_fp8``
    for ``H > 8192`` or when ``x.stride(-1) != 1`` or shape constraints.
    """
    assert x.dim() == 2
    assert weight.dim() == 1 and weight.shape[0] == x.shape[1]
    T, H = x.shape
    assert H % group_size == 0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H or x.stride(-1) != 1:
        return _baseline_strided_rmsnorm_fp8_quant(
            x, weight, eps, group_size, scale_ue8m0
        )

    fp8_out = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x.device)
    scale_out = create_per_token_group_quant_fp8_output_scale(
        x_shape=(T, H),
        device=x.device,
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

    _fused_strided_rmsnorm_singlepass_kernel[grid](
        x,
        weight,
        fp8_out,  # bf16_out_ptr unused
        fp8_out,
        scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        x.stride(0),
        fp8_out.stride(0),  # stride_b_t unused
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        HAS_BF16_OUT=False,
        HAS_FP8_OUT=True,
        SCALE_UE8M0=scale_ue8m0,
        num_warps=_select_num_warps(H),
    )
    return fp8_out, scale_out


def fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token-group fp8 quant + bf16 normed output.

    Dual-output variant: when one consumer wants bf16 (e.g. q_b_proj is bf16
    or wq_b is bf16) and another wants fp8 (the other one).
    Returns ``(bf16_normed, fp8_out, scale)``.
    """
    assert x.dim() == 2
    assert weight.dim() == 1 and weight.shape[0] == x.shape[1]
    T, H = x.shape
    assert H % group_size == 0
    if scale_ue8m0:
        assert (H // group_size) % 4 == 0, "UE8M0 requires num_groups divisible by 4"

    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H or x.stride(-1) != 1:
        return _baseline_strided_rmsnorm_fp8_quant_with_bf16_output(
            x, weight, eps, group_size, scale_ue8m0
        )

    bf16_out = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
    fp8_out = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x.device)
    scale_out = create_per_token_group_quant_fp8_output_scale(
        x_shape=(T, H),
        device=x.device,
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

    _fused_strided_rmsnorm_singlepass_kernel[grid](
        x,
        weight,
        bf16_out,
        fp8_out,
        scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        x.stride(0),
        bf16_out.stride(0),
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        HAS_BF16_OUT=True,
        HAS_FP8_OUT=True,
        SCALE_UE8M0=scale_ue8m0,
        num_warps=_select_num_warps(H),
    )
    return bf16_out, fp8_out, scale_out

"""Fused SiLU + (optional clamp) + multiply + per-token-group FP8 quantization
with packed UE8M0 scale, dsv4-private.

Replaces the 5-step legacy chain in
``moe/strategies/grouped_fp4.py::GroupedFP4Strategy.forward``::

    gate = gate_up[:, :inter].float()
    up = gate_up[:, inter:].float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    hidden = (F.silu(gate) * up).to(bf16)
    h_fp8, h_scale = sgl_per_token_group_quant_fp8(hidden, ...)

with one Triton launch.

Why a dsv4-private port (vs. directly calling the framework's
``silu_and_mul_masked_post_quant_packed_fwd``):
  - the framework kernel is masked-only (input ``[E, max_m, 2*inter]`` 3D)
  - the framework kernel has no ``swiglu_limit`` clamp parameter (V4 needs it)

Adapted from
``vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py``
(``_silu_mul_quant_fp8_packed_kernel`` + ``silu_mul_quant_fp8_packed_triton``).

Output layout:
  - ``out_fp8: [M, inter]`` torch.float8_e4m3fn
  - ``out_scale: [M, num_packed_groups]`` torch.int32, COLUMN-MAJOR with
    TMA-aligned M (M rounded up to multiple of 4). Each int32 packs 4
    UE8M0 scales (8 bits each, exponent biased by 127). This matches what
    DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_contiguous`` expects with
    ``recipe_a=(1, 128)``, identical to the layout produced by
    ``sgl_per_token_group_quant_fp8(column_major_scales=True,
    scale_tma_aligned=True, scale_ue8m0=True)``.

"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "output_scale_stride_k"])
def _silu_mul_fp8_quant_packed_kernel(
    input_ptr,            # [M, N=2*inter] BF16 (gate_up)
    output_q_ptr,         # [M, N_2=inter]  FP8 e4m3fn
    output_scale_ptr,     # column-major [num_packed_groups, tma_aligned_M] int32 view
    M,
    input_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    clamp_limit,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    BF16_ACTIVATION: tl.constexpr,
):
    N_2: tl.constexpr = N // 2

    pid_pack = tl.program_id(0)   # which packed-int32 column (= 4 groups)
    pid_m = tl.program_id(1)      # which BLOCK_M row tile
    m_offset = pid_m * BLOCK_M

    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    base_row_offset = (m_offset + offs_m[:, None]) * input_stride_m
    base_out_offset = (m_offset + offs_m[:, None]) * output_q_stride_m

    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx

        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE

            # Load gate (first half, [:N_2]) and up (second half, [N_2:N])
            act_ptrs = input_ptr + base_row_offset + n_offset + offs_n[None, :]
            act_in = tl.load(act_ptrs, mask=row_mask[:, None], other=0.0)

            mul_ptrs = act_ptrs + N_2
            mul_in = tl.load(mul_ptrs, mask=row_mask[:, None], other=0.0)

            if BF16_ACTIVATION:
                # vLLM shared expert semantics: clamp and SiLU run from BF16
                # tensors and the activation output is BF16 before multiply.
                act_bf16 = act_in
                mul_bf16 = mul_in
                if HAS_CLAMP:
                    act_bf16 = tl.minimum(act_bf16.to(tl.float32), clamp_limit).to(tl.bfloat16)
                    mul_bf16 = tl.clamp(mul_bf16.to(tl.float32), -clamp_limit, clamp_limit).to(tl.bfloat16)
                act_f32 = act_bf16.to(tl.float32)
                silu_bf16 = (act_f32 / (1.0 + tl.exp(-act_f32))).to(tl.bfloat16)
                y = (silu_bf16 * mul_bf16).to(tl.bfloat16).to(tl.float32)
            else:
                act_f32 = act_in.to(tl.float32)
                mul_f32 = mul_in.to(tl.float32)

                # V4 SwiGLU clamp convention:
                #   gate (act): clamp(max=L)            ← upper-only
                #   up   (mul): clamp(-L, L)            ← symmetric
                if HAS_CLAMP:
                    act_f32 = tl.minimum(act_f32, clamp_limit)
                    mul_f32 = tl.clamp(mul_f32, -clamp_limit, clamp_limit)

                y = (act_f32 / (1.0 + tl.exp(-act_f32))) * mul_f32
                # Round through bf16 to match the legacy unfused path's precision
                # (legacy: hidden = (F.silu(gate) * up).to(bfloat16), then quant
                # reads from bf16 not fp32). Without this the outputs differ at
                # ~ulp level from legacy and confound smoke validation.
                y = y.to(tl.bfloat16).to(tl.float32)

            # Per-row absmax → fp8 scale (UE8M0 exponent quantization)
            absmax = tl.max(tl.abs(y), axis=1)
            scale_raw = tl.maximum(absmax / fp8_max, 1e-10)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)

            # Quantize and store
            y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max)
            out_q_ptrs = output_q_ptr + base_out_offset + n_offset + offs_n[None, :]
            tl.store(
                out_q_ptrs,
                y_q.to(output_q_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )

            # Pack the UE8M0 exponent (biased by 127) into the right byte
            # of the int32 (4 packs per int32 across the K dim).
            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

    # Write the packed scale once per BLOCK_M row tile, into column-major slot.
    scale_ptrs = output_scale_ptr + pid_pack * output_scale_stride_k + m_offset + offs_m
    tl.store(scale_ptrs, packed_scale, mask=row_mask)


@triton.jit(do_not_specialize=["M", "output_scale_stride_k"])
def _silu_mul_fp8_quant_packed_split_kernel(
    gate_ptr,             # [M, inter] BF16
    up_ptr,               # [M, inter] BF16
    output_q_ptr,         # [M, inter] FP8 e4m3fn
    output_scale_ptr,     # column-major [num_packed_groups, tma_aligned_M] int32 view
    M,
    gate_stride_m,
    up_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    clamp_limit,
    N_2: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    BF16_ACTIVATION: tl.constexpr,
):
    pid_pack = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offset = pid_m * BLOCK_M

    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    gate_base = (m_offset + offs_m[:, None]) * gate_stride_m
    up_base = (m_offset + offs_m[:, None]) * up_stride_m
    out_base = (m_offset + offs_m[:, None]) * output_q_stride_m

    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx
        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE
            cols = n_offset + offs_n
            mask = row_mask[:, None] & (cols[None, :] < N_2)
            gate_in = tl.load(gate_ptr + gate_base + cols[None, :], mask=mask, other=0.0)
            up_in = tl.load(up_ptr + up_base + cols[None, :], mask=mask, other=0.0)

            if BF16_ACTIVATION:
                gate_bf16 = gate_in
                up_bf16 = up_in
                if HAS_CLAMP:
                    gate_bf16 = tl.minimum(gate_bf16.to(tl.float32), clamp_limit).to(tl.bfloat16)
                    up_bf16 = tl.clamp(up_bf16.to(tl.float32), -clamp_limit, clamp_limit).to(tl.bfloat16)
                gate_f32 = gate_bf16.to(tl.float32)
                silu_bf16 = (gate_f32 / (1.0 + tl.exp(-gate_f32))).to(tl.bfloat16)
                y = (silu_bf16 * up_bf16).to(tl.bfloat16).to(tl.float32)
            else:
                gate = gate_in.to(tl.float32)
                up = up_in.to(tl.float32)

                if HAS_CLAMP:
                    gate = tl.minimum(gate, clamp_limit)
                    up = tl.clamp(up, -clamp_limit, clamp_limit)

                y = (gate / (1.0 + tl.exp(-gate))) * up
                y = y.to(tl.bfloat16).to(tl.float32)

            absmax = tl.max(tl.abs(y), axis=1)
            scale_raw = tl.maximum(absmax / fp8_max, 1e-10)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)

            y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max)
            tl.store(
                output_q_ptr + out_base + cols[None, :],
                y_q.to(output_q_ptr.dtype.element_ty),
                mask=mask,
            )

            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

    scale_ptrs = output_scale_ptr + pid_pack * output_scale_stride_k + m_offset + offs_m
    tl.store(scale_ptrs, packed_scale, mask=row_mask)


def silu_mul_fp8_quant_packed(
    gate_up: torch.Tensor,
    clamp_limit: float = 0.0,
    group_size: int = 128,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    bf16_activation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU + clamp + mul + per-token-group FP8 quant + UE8M0 packed scale.

    Args:
      gate_up: ``[M, 2*inter]`` BF16 contiguous. Gate in [:inter], up in [inter:].
      clamp_limit: V4 SwiGLU clamp threshold; ``0`` (or ≤0) disables clamp.
      group_size: per-token quant group size (V4 uses 128).
      output_q: optional pre-allocated FP8 output buffer.

    Returns:
      out_fp8: ``[M, inter]`` torch.float8_e4m3fn.
      out_scale: column-major TMA-aligned ``[M, num_packed_groups]`` int32,
                 packed 4 UE8M0 per int32. Layout matches
                 ``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
                 scale_tma_aligned=True, scale_ue8m0=True)``.
    """
    assert gate_up.dim() == 2, f"expected 2D, got {gate_up.shape}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    assert gate_up.dtype == torch.bfloat16, f"expected bf16, got {gate_up.dtype}"

    M, N = gate_up.shape
    N_2 = N // 2

    assert N_2 % group_size == 0, (
        f"inter ({N_2}) must be a multiple of group_size ({group_size})"
    )

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    if output_q is None:
        output_q = torch.empty((M, N_2), dtype=fp8_dtype, device=gate_up.device)
    else:
        assert output_q.shape == (M, N_2)
        assert output_q.dtype == fp8_dtype

    if output_scale is None:
        # Allocate as [num_packed_groups, tma_aligned_M] int32 row-major, then
        # transpose + slice to [M, num_packed_groups] giving the column-major
        # TMA-aligned layout DeepGEMM expects.
        output_scale_packed = torch.empty(
            (num_packed_groups, tma_aligned_M),
            dtype=torch.int32,
            device=gate_up.device,
        ).T[:M, :]
    else:
        assert output_scale.shape == (M, num_packed_groups)
        assert output_scale.dtype == torch.int32
        output_scale_packed = output_scale

    BLOCK_M = 8
    grid = (num_packed_groups, (M + BLOCK_M - 1) // BLOCK_M)

    num_warps = max(4, group_size // 32)
    num_stages = 2

    has_clamp = clamp_limit > 0
    _silu_mul_fp8_quant_packed_kernel[grid](
        gate_up,
        output_q,
        output_scale_packed,
        M,
        gate_up.stride(0),
        output_q.stride(0),
        output_scale_packed.stride(1),
        clamp_limit if has_clamp else 0.0,
        N=N,
        NUM_GROUPS=num_groups_per_row,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        HAS_CLAMP=has_clamp,
        BF16_ACTIVATION=bf16_activation,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_q, output_scale_packed


def silu_mul_fp8_quant_packed_from_parts(
    gate: torch.Tensor,
    up: torch.Tensor,
    clamp_limit: float = 0.0,
    group_size: int = 128,
    output_q: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    bf16_activation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same fused activation+quant path as :func:`silu_mul_fp8_quant_packed`,
    but reads gate/up from two contiguous BF16 GEMM outputs.
    """
    assert gate.dim() == 2 and up.dim() == 2
    assert gate.shape == up.shape, f"gate/up shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.is_contiguous() and up.is_contiguous(), "gate/up must be contiguous"
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16

    M, N_2 = gate.shape
    assert N_2 % group_size == 0, (
        f"inter ({N_2}) must be a multiple of group_size ({group_size})"
    )

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    if output_q is None:
        output_q = torch.empty((M, N_2), dtype=fp8_dtype, device=gate.device)
    else:
        assert output_q.shape == (M, N_2)
        assert output_q.dtype == fp8_dtype

    if output_scale is None:
        output_scale_packed = torch.empty(
            (num_packed_groups, tma_aligned_M),
            dtype=torch.int32,
            device=gate.device,
        ).T[:M, :]
    else:
        assert output_scale.shape == (M, num_packed_groups)
        assert output_scale.dtype == torch.int32
        output_scale_packed = output_scale

    if M == 0:
        return output_q, output_scale_packed

    BLOCK_M = 8
    grid = (num_packed_groups, (M + BLOCK_M - 1) // BLOCK_M)
    has_clamp = clamp_limit > 0
    _silu_mul_fp8_quant_packed_split_kernel[grid](
        gate,
        up,
        output_q,
        output_scale_packed,
        M,
        gate.stride(0),
        up.stride(0),
        output_q.stride(0),
        output_scale_packed.stride(1),
        clamp_limit if has_clamp else 0.0,
        N_2=N_2,
        NUM_GROUPS=num_groups_per_row,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        HAS_CLAMP=has_clamp,
        BF16_ACTIVATION=bf16_activation,
        num_warps=max(4, group_size // 32),
        num_stages=2,
    )

    return output_q, output_scale_packed

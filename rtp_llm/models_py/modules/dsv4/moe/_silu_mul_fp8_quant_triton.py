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

Router weight is NOT folded in here — it's applied by the downstream
``ep_gather`` (Phase 2 optimization 4) so we only output silu(gate)*up.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)


@triton.jit(do_not_specialize=["M", "output_scale_stride_k"])
def _silu_mul_fp8_quant_packed_kernel(
    input_ptr,  # [M, N=2*inter] BF16 (gate_up)
    output_q_ptr,  # [M, N_2=inter]  FP8 e4m3fn
    output_scale_ptr,  # column-major [num_packed_groups, tma_aligned_M] int32 view
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
):
    N_2: tl.constexpr = N // 2

    pid_pack = tl.program_id(0).to(tl.int64)  # which packed-int32 column (= 4 groups)
    pid_m = tl.program_id(1).to(tl.int64)  # which BLOCK_M row tile
    m_offset = pid_m * BLOCK_M

    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
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
    gate_ptr,  # [M, inter] BF16
    up_ptr,  # [M, inter] BF16
    output_q_ptr,  # [M, inter] FP8 e4m3fn
    output_scale_ptr,  # column-major [num_packed_groups, tma_aligned_M] int32 view
    M,
    active_indptr,
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
    HAS_ACTIVE_ROWS: tl.constexpr,
    ACTIVE_INDEX: tl.constexpr,
):
    pid_pack = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M

    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M
    if HAS_ACTIVE_ROWS:
        row_mask = row_mask & (
            (m_offset + offs_m) < tl.load(active_indptr + ACTIVE_INDEX)
        )

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
            gate = tl.load(
                gate_ptr + gate_base + cols[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            up = tl.load(up_ptr + up_base + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )

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

    assert (
        N_2 % group_size == 0
    ), f"inter ({N_2}) must be a multiple of group_size ({group_size})"

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
    active_indptr: Optional[torch.Tensor] = None,
    zero_inactive_q: bool = True,
    zero_inactive_scale: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bound activation work by the producer-owned current-call aligned prefix.

    Initialize scale backing storage separately: contiguous() reads even inactive scales.
    """
    assert gate.dim() == 2 and up.dim() == 2
    assert gate.shape == up.shape, f"gate/up shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.stride(1) == 1 and up.stride(1) == 1, "gate/up columns must be dense"
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16

    M, N_2 = gate.shape
    if active_indptr is not None:
        # Private current-call monotone align4 prefix, produced by route counts.
        # Validate host ABI only; NEVER convert its device last value to Python.
        if (
            active_indptr.dim() != 1
            or not 2 <= active_indptr.shape[0] <= 1025
            or active_indptr.dtype != torch.int32
            or active_indptr.device != gate.device
            or gate.device.type != "cuda"
            or up.device != gate.device
            or not active_indptr.is_contiguous()
            or M % 4
            or M + 127 * (active_indptr.shape[0] - 1) > 2**31 - 1
            or gate.stride(0) < N_2
            or up.stride(0) < N_2
        ):
            raise ValueError("invalid active-row activation metadata")
        if output_q is not None or output_scale is not None:
            raise ValueError("active-row activation owns its own scratch")
    if zero_inactive_scale is None:
        zero_inactive_scale = zero_inactive_q
    q_alloc = torch.empty
    scale_alloc = torch.empty
    if active_indptr is not None:
        if zero_inactive_q:
            q_alloc = torch.zeros
        if zero_inactive_scale:
            scale_alloc = torch.zeros
    assert (
        N_2 % group_size == 0
    ), f"inter ({N_2}) must be a multiple of group_size ({group_size})"

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    if output_q is None:
        output_q = q_alloc((M, N_2), dtype=fp8_dtype, device=gate.device)
    else:
        assert output_q.shape == (M, N_2)
        assert output_q.dtype == fp8_dtype

    if output_scale is None:
        output_scale_packed = scale_alloc(
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
        active_indptr,
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
        HAS_ACTIVE_ROWS=active_indptr is not None,
        ACTIVE_INDEX=0 if active_indptr is None else active_indptr.shape[0] - 1,
        num_warps=max(4, group_size // 32),
        num_stages=2,
    )

    return output_q, output_scale_packed


@triton.jit
def _silu_mul_masked_fp8_quant_packed_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    output_ptr,
    stride_output_0,
    stride_output_1,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    clamp_limit,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    """3-D masked SiLU+clamp+mul+FP8 quant. Gate in [:H], up in [H:]."""
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    packed_group_index = tl.program_id(0)
    block_num_per_expert = tl.num_programs(1)
    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)
    if token_id >= token_num_cur_expert:
        return

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    input_base = input_ptr + expert_id * stride_input_0
    output_base = output_ptr + expert_id * stride_output_0
    output_scale_base = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + packed_group_index * stride_output_scale_2
    )
    offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), 16), 16)
    base = packed_group_index * (4 * BLOCK_N)

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        packed_scale: tl.int32 = 0
        token_in = input_base + token_index * stride_input_1
        token_out = output_base + token_index * stride_output_1
        for g in tl.static_range(4):
            offs_in_d = base + g * BLOCK_N + offs
            gate = tl.load(token_in + offs_in_d).to(tl.float32)
            up = tl.load(token_in + offs_in_d + size_n).to(tl.float32)
            if HAS_CLAMP:
                gate = tl.minimum(gate, clamp_limit)
                up = tl.clamp(up, -clamp_limit, clamp_limit)
            y = (gate / (1.0 + tl.exp(-gate))) * up
            y = y.to(tl.bfloat16).to(tl.float32)
            absmax = tl.max(tl.abs(y))
            scale_raw = tl.maximum(absmax / fp8_max, 1e-10)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            output_q = tl.clamp(y / scale, fp8_min, fp8_max).to(
                output_ptr.dtype.element_ty
            )
            tl.store(token_out + offs_in_d, output_q)
            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (g * 8))
        tl.store(
            output_scale_base + token_index * stride_output_scale_1,
            packed_scale,
        )


def silu_mul_masked_fp8_quant_packed(
    gate_up: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    clamp_limit: float = 0.0,
    group_size: int = 128,
) -> None:
    """Masked ``[E, T, 2H]`` SiLU+clamp+mul+packed UE8M0 quant (gate-first)."""
    assert gate_up.dim() == 3 and gate_up.is_contiguous()
    hidden = gate_up.size(-1) // 2
    assert hidden % group_size == 0
    groups = hidden // group_size
    assert groups % 4 == 0, "UE8M0 packing needs groups % 4 == 0"
    expert_num = gate_up.size(0)
    token_pad = gate_up.size(1)
    packed = groups // 4
    # Pack-parallel, 1 warp. Decode T=128 with ~1-8 live tokens/expert: 8
    # token CTAs beat 32 empty ones; prefill keeps up to 32. NUM_STAGE=2
    # avoids the 2x regression STAGE=6 showed at live>=32.
    if token_pad <= 256:
        block_num = 8
    else:
        block_num = min(32, max(token_pad // 64, 8))
    grid = (packed, block_num, expert_num)
    has_clamp = clamp_limit > 0
    _silu_mul_masked_fp8_quant_packed_kernel[grid](
        gate_up,
        gate_up.stride(0),
        gate_up.stride(1),
        output_q,
        output_q.stride(0),
        output_q.stride(1),
        output_scale,
        output_scale.stride(0),
        output_scale.stride(1),
        output_scale.stride(2),
        masked_m,
        clamp_limit if has_clamp else 0.0,
        hidden,
        _FP8_INFO.max,
        _FP8_INFO.min,
        BLOCK_N=group_size,
        NUM_STAGE=2,
        HAS_CLAMP=has_clamp,
        num_warps=1,
    )

"""Fused MXFP8 (1x32 microscaling FP8) activation quant for MiniMax-M3.

Replaces the eager-PyTorch ``mxfp8_quant_act`` (``abs``/``amax``/``log2``/
``ceil``/``exp2``/``div``/``clamp``/``to(fp8)`` ~= 7 kernels) with a single
Triton kernel: dynamic per-(row, 32-col) max-abs -> UE8M0 power-of-two scale
-> e4m3 cast. The power-of-two rounding reuses :func:`_ue8m0_pow2_round` from
the existing FP8 quant kernels so the scale math is shared, and the output
(fp32 ``[M, K//32]`` scale) feeds DeepGEMM's native ``pack_mxfp8_scale``.

There is no existing drop-in for this: the FP8 group-quant kernels
(``sgl_per_token_group_quant_fp8`` / ``per_token_group_quant_fp8_v2`` /
``trt_fp8_quantize_128``) are hardwired to the 1x128 recipe and do not emit a
1x32 power-of-two scale.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    _ue8m0_pow2_round,
)

MX_BLOCK = 32
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


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


def pack_flashinfer_mxfp8_scale_triton(
    scale_u8: torch.Tensor, M: int, K: int
) -> torch.Tensor:
    """Pack FlashInfer uint8 UE8M0 scales into DeepGEMM's int32 TMA layout."""
    assert scale_u8.dtype == torch.uint8
    assert scale_u8.numel() == M * (K // MX_BLOCK)
    import deep_gemm

    k_groups = K // MX_BLOCK
    assert k_groups % 4 == 0
    k_packed = k_groups // 4
    aligned_mn = deep_gemm.get_tma_aligned_size(M, 4)
    storage = torch.empty(
        (k_packed, aligned_mn), device=scale_u8.device, dtype=torch.int32
    )
    packed = storage.transpose(0, 1)
    grid = (triton.cdiv(M, 64), triton.cdiv(k_packed, 32))
    with torch.cuda.device(scale_u8.device):
        _pack_flashinfer_mxfp8_scale_kernel[grid](
            scale_u8,
            packed,
            M=M,
            K_GROUPS=k_groups,
            K_PACKED=k_packed,
            ALIGNED_MN=aligned_mn,
            BLOCK_M=64,
            BLOCK_K_PACKED=32,
            num_warps=8,
        )
    return packed[:M, :]


@triton.jit
def _mxfp8_quant_act_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    n_groups,
    x_row_stride,
    q_row_stride,
    s_row_stride,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    NG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gblk = tl.program_id(1)
    g0 = gblk * NG
    g_idx = tl.arange(0, NG)
    groups = g0 + g_idx
    gmask = groups < n_groups
    col = tl.arange(0, GROUP)
    # [NG, GROUP] element offsets inside the row
    elem = groups[:, None] * GROUP + col[None, :]

    x = tl.load(
        x_ptr + row * x_row_stride + elem,
        mask=gmask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-20)
    # UE8M0 power-of-two scale == exp2(ceil(log2(amax / FP8_MAX))), shared with
    # the existing FP8 quant kernels via the bit-hack helper.
    scale, _ = _ue8m0_pow2_round(amax / FP8_MAX)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)

    tl.store(
        q_ptr + row * q_row_stride + elem,
        q.to(q_ptr.dtype.element_ty),
        mask=gmask[:, None],
    )
    tl.store(s_ptr + row * s_row_stride + groups, scale, mask=gmask)


def mxfp8_quant_act_triton(x: torch.Tensor):
    """Fused per-(row, 32) MXFP8 quant. Returns (e4m3 ``[M, K]``, fp32 scale ``[M, K//32]``)."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    x = x.contiguous()
    n_groups = K // MX_BLOCK
    q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, n_groups, device=x.device, dtype=torch.float32)
    if M == 0:
        return q, s
    NG = 16
    grid = (M, triton.cdiv(n_groups, NG))
    _mxfp8_quant_act_kernel[grid](
        x,
        q,
        s,
        n_groups,
        x.stride(0),
        q.stride(0),
        s.stride(0),
        FP8_MAX=float(_FP8_E4M3_MAX),
        GROUP=MX_BLOCK,
        NG=NG,
        num_warps=4,
    )
    return q, s


@triton.jit
def _mxfp8_quant_act_masked_packed_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    masked_m_ptr,
    x_stride_e,
    x_stride_m,
    x_stride_k,
    q_stride_e,
    q_stride_m,
    q_stride_k,
    s_stride_e,
    s_stride_m,
    s_stride_g,
    K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    packed_group = tl.program_id(2)
    valid = token < tl.load(masked_m_ptr + expert)
    # Most decode low-latency CTAs are padded rows because the graph-safe DeepEP
    # buffer uses M=512 while each expert usually receives only a few tokens.
    # Return before vector loads/reductions for those rows; the DeepGEMM masked
    # kernel never consumes q/scale for token >= masked_m[expert].
    if not valid:
        return

    offs = tl.arange(0, GROUP)
    base_group = packed_group * 4
    packed_scale: tl.int32 = 0

    for g in tl.static_range(4):
        group = base_group + g
        cols = group * GROUP + offs
        mask = cols < K
        vals = tl.load(
            x_ptr + expert * x_stride_e + token * x_stride_m + cols * x_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-20)
        scale, exp_bits = _ue8m0_pow2_round(amax / FP8_MAX)
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -FP8_MAX), FP8_MAX)
        tl.store(
            q_ptr + expert * q_stride_e + token * q_stride_m + cols * q_stride_k,
            q_vals.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        packed_scale = packed_scale | (exp_bits.to(tl.int32) << (g * 8))

    tl.store(
        s_ptr + expert * s_stride_e + token * s_stride_m + packed_group * s_stride_g,
        packed_scale,
        mask=valid,
    )


@triton.jit
def _mxfp8_build_active_expert_kernel(
    masked_m_ptr,
    active_expert_ptr,
    active_count_ptr,
    E: tl.constexpr,
):
    tl.store(active_count_ptr, 0)
    for expert in tl.static_range(0, E):
        count = tl.load(masked_m_ptr + expert)
        if count > 0:
            slot = tl.atomic_add(active_count_ptr, 1, sem="relaxed")
            tl.store(active_expert_ptr + slot, expert)


@triton.jit
def _mxfp8_quant_act_active_expert_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    active_expert_ptr,
    active_count_ptr,
    masked_m_ptr,
    x_stride_e,
    x_stride_m,
    x_stride_k,
    q_stride_e,
    q_stride_m,
    q_stride_k,
    s_stride_e,
    s_stride_m,
    s_stride_g,
    K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
):
    active_slot = tl.program_id(0)
    token = tl.program_id(1).to(tl.int64)
    packed_group = tl.program_id(2)
    if active_slot >= tl.load(active_count_ptr):
        return
    expert = tl.load(active_expert_ptr + active_slot).to(tl.int64)
    valid = token < tl.load(masked_m_ptr + expert)
    if not valid:
        return

    offs = tl.arange(0, GROUP)
    base_group = packed_group * 4
    packed_scale: tl.int32 = 0

    for g in tl.static_range(4):
        group = base_group + g
        cols = group * GROUP + offs
        mask = cols < K
        vals = tl.load(
            x_ptr + expert * x_stride_e + token * x_stride_m + cols * x_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-20)
        scale, exp_bits = _ue8m0_pow2_round(amax / FP8_MAX)
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -FP8_MAX), FP8_MAX)
        tl.store(
            q_ptr + expert * q_stride_e + token * q_stride_m + cols * q_stride_k,
            q_vals.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        packed_scale = packed_scale | (exp_bits.to(tl.int32) << (g * 8))

    tl.store(
        s_ptr + expert * s_stride_e + token * s_stride_m + packed_group * s_stride_g,
        packed_scale,
    )


@triton.jit
def _mxfp8_zero_i32_kernel(ptr):
    tl.store(ptr, 0)


@triton.jit
def _mxfp8_build_active_row_kernel(
    masked_m_ptr,
    row_expert_ptr,
    row_token_ptr,
    row_count_ptr,
    MAX_ACTIVE_ROWS: tl.constexpr,
):
    expert = tl.program_id(0)
    token = tl.program_id(1)
    if token < tl.load(masked_m_ptr + expert):
        slot = tl.atomic_add(row_count_ptr, 1, sem="relaxed")
        if slot < MAX_ACTIVE_ROWS:
            tl.store(row_expert_ptr + slot, expert)
            tl.store(row_token_ptr + slot, token)


@triton.jit
def _mxfp8_build_active_row_prefix_kernel(
    masked_m_ptr,
    row_expert_ptr,
    row_token_ptr,
    row_count_ptr,
    MAX_M: tl.constexpr,
    MAX_ACTIVE_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    E: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_M)

    prefix = tl.full((), 0, tl.int32)
    for prev_expert in tl.static_range(0, E):
        prev_count = tl.load(masked_m_ptr + prev_expert).to(tl.int32)
        prev_count = tl.minimum(prev_count, MAX_M)
        prefix += tl.where(prev_expert < expert, prev_count, 0)

    count = tl.load(masked_m_ptr + expert).to(tl.int32)
    count = tl.minimum(count, MAX_M)
    slots = prefix + offsets
    mask = (offsets < count) & (slots < MAX_ACTIVE_ROWS)
    tl.store(row_expert_ptr + slots, expert, mask=mask)
    tl.store(row_token_ptr + slots, offsets, mask=mask)

    total = tl.minimum(prefix + count, MAX_ACTIVE_ROWS)
    tl.store(row_count_ptr, total, mask=expert == E - 1)


@triton.jit
def _mxfp8_quant_act_active_row_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    row_expert_ptr,
    row_token_ptr,
    row_count_ptr,
    masked_m_ptr,
    x_stride_e,
    x_stride_m,
    x_stride_k,
    q_stride_e,
    q_stride_m,
    q_stride_k,
    s_stride_e,
    s_stride_m,
    s_stride_g,
    K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
):
    row_slot = tl.program_id(0)
    packed_group = tl.program_id(1)
    if row_slot >= tl.load(row_count_ptr):
        return
    expert = tl.load(row_expert_ptr + row_slot).to(tl.int64)
    token = tl.load(row_token_ptr + row_slot).to(tl.int64)
    if token >= tl.load(masked_m_ptr + expert):
        return

    offs = tl.arange(0, GROUP)
    base_group = packed_group * 4
    packed_scale: tl.int32 = 0

    for g in tl.static_range(4):
        group = base_group + g
        cols = group * GROUP + offs
        mask = cols < K
        vals = tl.load(
            x_ptr + expert * x_stride_e + token * x_stride_m + cols * x_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-20)
        scale, exp_bits = _ue8m0_pow2_round(amax / FP8_MAX)
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -FP8_MAX), FP8_MAX)
        tl.store(
            q_ptr + expert * q_stride_e + token * q_stride_m + cols * q_stride_k,
            q_vals.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        packed_scale = packed_scale | (exp_bits.to(tl.int32) << (g * 8))

    tl.store(
        s_ptr + expert * s_stride_e + token * s_stride_m + packed_group * s_stride_g,
        packed_scale,
    )


@triton.jit
def _mxfp8_swiglu_oai_quant_active_row_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    row_expert_ptr,
    row_token_ptr,
    row_count_ptr,
    masked_m_ptr,
    x_stride_e,
    x_stride_m,
    x_stride_k,
    q_stride_e,
    q_stride_m,
    q_stride_k,
    s_stride_e,
    s_stride_m,
    s_stride_g,
    K2: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
):
    row_slot = tl.program_id(0)
    packed_group = tl.program_id(1)
    if row_slot >= tl.load(row_count_ptr):
        return
    expert = tl.load(row_expert_ptr + row_slot).to(tl.int64)
    token = tl.load(row_token_ptr + row_slot).to(tl.int64)
    if token >= tl.load(masked_m_ptr + expert):
        return

    offs = tl.arange(0, GROUP)
    base_group = packed_group * 4
    packed_scale: tl.int32 = 0
    half_k: tl.constexpr = K2 // 2

    for g in tl.static_range(4):
        group = base_group + g
        cols = group * GROUP + offs
        mask = cols < half_k
        up = tl.load(
            x_ptr + expert * x_stride_e + token * x_stride_m + cols * x_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            x_ptr
            + expert * x_stride_e
            + token * x_stride_m
            + (cols + half_k) * x_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        vals = gate * tl.sigmoid(gate * ALPHA) * (up + 1.0)
        # Match the previous path, which materializes the activation as BF16
        # before MXFP8 quantization.
        vals = vals.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-20)
        scale, exp_bits = _ue8m0_pow2_round(amax / FP8_MAX)
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -FP8_MAX), FP8_MAX)
        tl.store(
            q_ptr + expert * q_stride_e + token * q_stride_m + cols * q_stride_k,
            q_vals.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        packed_scale = packed_scale | (exp_bits.to(tl.int32) << (g * 8))

    tl.store(
        s_ptr + expert * s_stride_e + token * s_stride_m + packed_group * s_stride_g,
        packed_scale,
    )


@triton.jit
def _mxfp8_quant_act_active_row_block_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    row_expert_ptr,
    row_token_ptr,
    row_count_ptr,
    masked_m_ptr,
    x_stride_e,
    x_stride_m,
    x_stride_k,
    q_stride_e,
    q_stride_m,
    q_stride_k,
    s_stride_e,
    s_stride_m,
    s_stride_g,
    K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row_block = tl.program_id(0) * BLOCK_R
    packed_group = tl.program_id(1)
    offs = tl.arange(0, GROUP)
    row_count = tl.load(row_count_ptr)
    if row_block >= row_count:
        return
    base_group = packed_group * 4

    for r in tl.static_range(0, BLOCK_R):
        row_slot = row_block + r
        valid_row = row_slot < row_count
        expert = tl.load(row_expert_ptr + row_slot, mask=valid_row, other=0).to(
            tl.int64
        )
        token = tl.load(row_token_ptr + row_slot, mask=valid_row, other=0).to(tl.int64)
        valid_row = valid_row & (token < tl.load(masked_m_ptr + expert))
        packed_scale: tl.int32 = 0

        for g in tl.static_range(4):
            group = base_group + g
            cols = group * GROUP + offs
            mask = valid_row & (cols < K)
            vals = tl.load(
                x_ptr + expert * x_stride_e + token * x_stride_m + cols * x_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-20)
            scale, exp_bits = _ue8m0_pow2_round(amax / FP8_MAX)
            q_vals = vals / scale
            q_vals = tl.minimum(tl.maximum(q_vals, -FP8_MAX), FP8_MAX)
            tl.store(
                q_ptr + expert * q_stride_e + token * q_stride_m + cols * q_stride_k,
                q_vals.to(q_ptr.dtype.element_ty),
                mask=mask,
            )
            packed_scale = packed_scale | (exp_bits.to(tl.int32) << (g * 8))

        tl.store(
            s_ptr
            + expert * s_stride_e
            + token * s_stride_m
            + packed_group * s_stride_g,
            packed_scale,
            mask=valid_row,
        )


def mxfp8_build_active_experts(masked_m: torch.Tensor, E: int):
    active_experts = torch.empty((E,), device=masked_m.device, dtype=torch.int32)
    active_count = torch.empty((1,), device=masked_m.device, dtype=torch.int32)
    _mxfp8_build_active_expert_kernel[(1,)](
        masked_m,
        active_experts,
        active_count,
        E=E,
        num_warps=1,
    )
    return active_experts, active_count


def mxfp8_build_active_rows(
    masked_m: torch.Tensor, E: int, max_m: int, max_active_rows: int
):
    row_experts = torch.empty(
        (max_active_rows,), device=masked_m.device, dtype=torch.int32
    )
    row_tokens = torch.empty(
        (max_active_rows,), device=masked_m.device, dtype=torch.int32
    )
    row_count = torch.empty((1,), device=masked_m.device, dtype=torch.int32)
    block_m = triton.next_power_of_2(max(1, min(int(max_m), int(max_active_rows))))
    _mxfp8_build_active_row_prefix_kernel[(E,)](
        masked_m,
        row_experts,
        row_tokens,
        row_count,
        MAX_M=max_m,
        MAX_ACTIVE_ROWS=max_active_rows,
        BLOCK_M=block_m,
        E=E,
        num_warps=8,
    )
    return row_experts, row_tokens, row_count


def mxfp8_quant_act_masked_packed_triton(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    max_m: int | None = None,
    active_experts: torch.Tensor | None = None,
    active_count: torch.Tensor | None = None,
    max_active_experts: int | None = None,
    active_row_experts: torch.Tensor | None = None,
    active_row_tokens: torch.Tensor | None = None,
    active_row_count: torch.Tensor | None = None,
    max_active_rows: int | None = None,
):
    """Masked MXFP8 quant for DeepEP low-latency [E, M, K] BF16 layout.

    Returns fp8 activations [E, M, K] and packed int32 UE8M0 scales [E, M, K//128].
    Only rows with token index < masked_m[expert] are read or written.
    ``max_m`` may cap the launched token slots when the caller already has a
    graph-safe upper bound such as DeepGEMM's ``expected_m``.
    """
    assert x.dim() == 3, f"expected [E, M, K], got {tuple(x.shape)}"
    E, M, K = x.shape
    assert K % (MX_BLOCK * 4) == 0, f"K={K} must be a multiple of {MX_BLOCK * 4}"
    assert masked_m.numel() == E
    q = torch.empty((E, M, K), device=x.device, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (E, K // (MX_BLOCK * 4), M), device=x.device, dtype=torch.int32
    )
    s_packed = scale_storage.transpose(1, 2)
    active_m = M if max_m is None else max(0, min(M, int(max_m)))
    if E == 0 or active_m == 0:
        return q, s_packed
    if (
        active_row_experts is not None
        and active_row_tokens is not None
        and active_row_count is not None
    ):
        active_rows = (
            E * active_m
            if max_active_rows is None
            else max(0, min(E * active_m, int(max_active_rows)))
        )
        block_r = 4
        grid = (triton.cdiv(active_rows, block_r), K // (MX_BLOCK * 4))
        _mxfp8_quant_act_active_row_block_kernel[grid](
            x,
            q,
            s_packed,
            active_row_experts,
            active_row_tokens,
            active_row_count,
            masked_m,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            s_packed.stride(0),
            s_packed.stride(1),
            s_packed.stride(2),
            K=K,
            FP8_MAX=float(_FP8_E4M3_MAX),
            GROUP=MX_BLOCK,
            BLOCK_R=block_r,
            num_warps=1,
        )
    elif active_experts is not None and active_count is not None:
        active_e = (
            E if max_active_experts is None else max(0, min(E, int(max_active_experts)))
        )
        grid = (active_e, active_m, K // (MX_BLOCK * 4))
        _mxfp8_quant_act_active_expert_kernel[grid](
            x,
            q,
            s_packed,
            active_experts,
            active_count,
            masked_m,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            s_packed.stride(0),
            s_packed.stride(1),
            s_packed.stride(2),
            K=K,
            FP8_MAX=float(_FP8_E4M3_MAX),
            GROUP=MX_BLOCK,
            num_warps=1,
        )
    else:
        grid = (E, active_m, K // (MX_BLOCK * 4))
        _mxfp8_quant_act_masked_packed_kernel[grid](
            x,
            q,
            s_packed,
            masked_m,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            s_packed.stride(0),
            s_packed.stride(1),
            s_packed.stride(2),
            K=K,
            FP8_MAX=float(_FP8_E4M3_MAX),
            GROUP=MX_BLOCK,
            num_warps=1,
        )
    return q, s_packed


def mxfp8_swiglu_oai_quant_active_row_packed_triton(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    alpha: float,
    limit: float,
    active_row_experts: torch.Tensor,
    active_row_tokens: torch.Tensor,
    active_row_count: torch.Tensor,
    max_active_rows: int | None = None,
):
    """Fuse MiniMax-M3 SwiGLU-OAI activation with MXFP8 active-row quant.

    ``x`` is the first MoE GEMM output in [up | gate] layout. The result is
    directly consumable by DeepGEMM masked MXFP8 GEMM.
    """
    assert x.dim() == 3, f"expected [E, M, 2K], got {tuple(x.shape)}"
    E, M, K2 = x.shape
    assert K2 % 2 == 0
    K = K2 // 2
    assert K % (MX_BLOCK * 4) == 0, f"K={K} must be a multiple of {MX_BLOCK * 4}"
    assert masked_m.numel() == E
    q = torch.empty((E, M, K), device=x.device, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (E, K // (MX_BLOCK * 4), M), device=x.device, dtype=torch.int32
    )
    s_packed = scale_storage.transpose(1, 2)
    active_rows = (
        E * M if max_active_rows is None else max(0, min(E * M, int(max_active_rows)))
    )
    if E == 0 or M == 0 or active_rows == 0:
        return q, s_packed
    grid = (active_rows, K // (MX_BLOCK * 4))
    _mxfp8_swiglu_oai_quant_active_row_kernel[grid](
        x,
        q,
        s_packed,
        active_row_experts,
        active_row_tokens,
        active_row_count,
        masked_m,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        s_packed.stride(0),
        s_packed.stride(1),
        s_packed.stride(2),
        K2=K2,
        ALPHA=float(alpha),
        LIMIT=float(limit),
        FP8_MAX=float(_FP8_E4M3_MAX),
        GROUP=MX_BLOCK,
        num_warps=1,
    )
    return q, s_packed

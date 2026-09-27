"""Fused MiniMax-M3.1 Q8KV4 sparse decode attention for RTP pages."""

import torch
import triton
import triton.language as tl

from .nvfp4_q8_math import (
    E2M1X4_ASM_HI,
    E2M1X4_ASM_LO,
    FP32_ROUND_INT,
    LN2_F32,
    LOG2E_F32,
    POLY_EX2_C1,
    POLY_EX2_C2,
    POLY_EX2_C3,
    ex2_emulated,
    ex2_ftz,
    lg2_ftz,
    packed4_to_fp8,
    rcp_ftz,
    rcp_rn,
    softmax_scale_log2,
    tree_sum_128,
)


@triton.jit
def _load_main_page_fp8(
    packed_ptr,
    scale_ptr,
    page,
    head,
    packed_page_stride,
    scale_page_stride,
    MMA_SCALE_LAYOUT: tl.constexpr,
):
    token = tl.arange(0, 128)
    byte = tl.arange(0, 64)
    group = tl.arange(0, 8)
    packed = tl.load(
        packed_ptr
        + page * packed_page_stride
        + head * (128 * 64)
        + token[:, None] * 64
        + byte[None, :]
    )
    if MMA_SCALE_LAYOUT:
        scale_offset = (
            (group[None, :] // 4) * 512
            + (token[:, None] % 32) * 16
            + (token[:, None] // 32) * 4
            + group[None, :] % 4
        )
    else:
        scale_offset = token[:, None] * 8 + group[None, :]
    scales = tl.load(
        scale_ptr + page * scale_page_stride + head * (128 * 8) + scale_offset
    )
    scales = tl.reshape(tl.broadcast_to(scales[:, :, None], [128, 8, 8]), [128, 64])
    low = tl.reshape(packed4_to_fp8(packed, scales, E2M1X4_ASM_LO), [128, 16, 4])
    high = tl.reshape(packed4_to_fp8(packed, scales, E2M1X4_ASM_HI), [128, 16, 4])
    values = tl.reshape(tl.permute(tl.join(low, high), (0, 1, 3, 2)), [128, 128])
    return values.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _q8kv4_selected_page_partial(
    q_ptr,
    packed_k_ptr,
    packed_v_ptr,
    scale_k_ptr,
    scale_v_ptr,
    block_table_ptr,
    topk_ptr,
    seq_lens_ptr,
    partial_ptr,
    lse_ptr,
    counts_ptr,
    packed_k_page_stride,
    packed_v_page_stride,
    scale_k_page_stride,
    scale_v_page_stride,
    q_batch_stride,
    q_head_stride,
    table_batch_stride,
    topk_head_stride,
    topk_batch_stride,
    num_phys_pages,
    max_pages,
    ALPHA: tl.constexpr,
    LN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    RINT: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    slot = tl.program_id(2)
    if slot == 0:
        tl.store(counts_ptr + batch * 4 + kv_head, 16)

    seq_len = tl.load(seq_lens_ptr + batch)
    num_blocks = (seq_len + 127) // 128
    block = tl.load(
        topk_ptr + kv_head * topk_head_stride + batch * topk_batch_stride + slot
    )
    valid = (block >= 0) & (block < num_blocks) & (block < max_pages)
    safe_block = tl.where(valid, block, 0)
    page = tl.load(
        block_table_ptr + batch * table_batch_stride + safe_block,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid = valid & (page >= 0) & (page < num_phys_pages)
    if not valid:
        group = tl.arange(0, 16)
        dim = tl.arange(0, 128)
        base = ((batch * 4 + kv_head) * 16 + slot) * 16 + group
        tl.store(partial_ptr + base[:, None] * 128 + dim[None, :], 0.0)
        tl.store(lse_ptr + base, float("-inf"))
        return

    # Only 16 GQA rows are live. M64 preserves the token reduction order while
    # reducing padded FP8 MMA work and register pressure versus M128.
    rows = tl.arange(0, 64)
    group = rows % 16
    active = rows // 16 == 0
    dim = tl.arange(0, 128)
    token = tl.arange(0, 128)
    q = tl.load(
        q_ptr
        + batch * q_batch_stride
        + (kv_head * 16 + group[:, None]) * q_head_stride
        + dim[None, :],
        mask=active[:, None],
        other=0.0,
    )
    key = _load_main_page_fp8(
        packed_k_ptr,
        scale_k_ptr,
        page,
        kv_head,
        packed_k_page_stride,
        scale_k_page_stride,
        MMA_SCALE_LAYOUT,
    )
    value = _load_main_page_fp8(
        packed_v_ptr,
        scale_v_ptr,
        page,
        kv_head,
        packed_v_page_stride,
        scale_v_page_stride,
        MMA_SCALE_LAYOUT,
    )
    score = tl.dot(q, tl.trans(key), out_dtype=tl.float32)
    score = tl.where((block * 128 + token)[None, :] < seq_len, score, float("-inf"))
    maximum = tl.max(score, axis=1)
    maximum_safe = tl.where(maximum == float("-inf"), 0.0, maximum)
    exponent = tl.fma(score, ALPHA, maximum_safe[:, None] * (-ALPHA))
    emulated_column = (token >= 32) & (token < 96) & ((token % 16) >= 12)
    probability = tl.where(
        emulated_column[None, :],
        ex2_emulated(exponent, C1, C2, C3, RINT),
        ex2_ftz(exponent),
    )
    probability_fp8 = probability.to(tl.float8e4nv)
    denominator = tree_sum_128(probability, 64)
    numerator = tl.dot(probability_fp8, value, out_dtype=tl.float32)
    inverse = rcp_ftz(tl.where(denominator != 0.0, denominator, 1.0))
    output = (numerator * inverse[:, None]).to(tl.bfloat16)
    lse = tl.where(
        denominator != 0.0,
        tl.fma(maximum, ALPHA, lg2_ftz(denominator)) * LN2,
        float("-inf"),
    )
    base = ((batch * 4 + kv_head) * 16 + slot) * 16 + group
    tl.store(
        partial_ptr + base[:, None] * 128 + dim[None, :],
        output,
        mask=active[:, None],
    )
    tl.store(lse_ptr + base, lse, mask=active)


@triton.jit
def _q8kv4_combine(
    partial_ptr,
    lse_ptr,
    counts_ptr,
    out_ptr,
    out_batch_stride,
    out_head_stride,
    LOG2E: tl.constexpr,
    valid_token_mask=None,
    HAS_VALID_TOKEN_MASK: tl.constexpr = False,
    valid_token_mask_stride=1,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    group = tl.arange(0, 16)
    dim = tl.arange(0, 128)
    slot = tl.arange(0, 16)
    row = batch * 4 + kv_head
    count = tl.load(counts_ptr + row)
    valid = slot < count
    lse_base = (row * 16 + slot)[:, None] * 16 + group[None, :]
    lse = tl.load(lse_ptr + lse_base, mask=valid[:, None], other=float("-inf"))
    finite = lse != float("-inf")
    has_finite = tl.sum(finite.to(tl.int32), axis=0) > 0
    maximum = tl.max(lse, axis=0)
    maximum_safe = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = ex2_ftz(tl.fma(lse, LOG2E, -(maximum_safe * LOG2E)[None, :]))
    weight_lanes = tl.reshape(weight, [4, 4, 16])
    lane_index = tl.arange(0, 4)[:, None, None]
    lane_sum = tl.zeros([4, 16], dtype=tl.float32)
    for i in tl.static_range(4):
        lane_sum += tl.sum(tl.where(lane_index == i, weight_lanes, 0.0), axis=0)
    lanes = tl.arange(0, 4)[:, None]
    s0 = tl.sum(tl.where(lanes == 0, lane_sum, 0.0), axis=0)
    s1 = tl.sum(tl.where(lanes == 1, lane_sum, 0.0), axis=0)
    s2 = tl.sum(tl.where(lanes == 2, lane_sum, 0.0), axis=0)
    s3 = tl.sum(tl.where(lanes == 3, lane_sum, 0.0), axis=0)
    denominator = (s0 + s2) + (s1 + s3)
    good = has_finite & (denominator != 0.0) & (denominator == denominator)
    inverse = tl.where(good, rcp_rn(tl.where(good, denominator, 1.0)), 0.0)
    weight *= inverse[None, :]
    accumulator = tl.zeros([16, 128], dtype=tl.float32)
    for i in tl.static_range(16):
        selected_weight = tl.sum(tl.where(slot[:, None] == i, weight, 0.0), axis=0)
        partial = tl.load(
            partial_ptr + ((row * 16 + i) * 16 + group)[:, None] * 128 + dim[None, :],
            mask=(i < count) & (group[:, None] >= 0),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.where(
            selected_weight[:, None] > 0.0,
            tl.fma(selected_weight[:, None], partial, accumulator),
            accumulator,
        )
    if HAS_VALID_TOKEN_MASK:
        # Clear padding even when the partial reduction produced NaN. Multiplying
        # by zero would retain NaN and would not match torch.where semantics.
        accumulator = tl.where(
            tl.load(valid_token_mask + batch * valid_token_mask_stride),
            accumulator,
            0.0,
        )
    tl.store(
        out_ptr
        + batch * out_batch_stride
        + (kv_head * 16 + group)[:, None] * out_head_stride
        + dim[None, :],
        accumulator.to(tl.bfloat16),
    )


@torch.no_grad()
def q8kv4_sparse_decode_attention(
    q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    block_table: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    out: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    counts: torch.Tensor,
    mma_scale_layout: bool = False,
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fixed M3.1 64Q/4KV/D128/TopK16 decode contract."""
    batch = q.shape[0]
    if valid_token_mask is not None and (
        valid_token_mask.dtype != torch.bool
        or tuple(valid_token_mask.shape) != (batch,)
        or valid_token_mask.device != q.device
    ):
        raise ValueError("valid_token_mask must be bool [batch] on Q device")
    if q.dtype != torch.float8_e4m3fn or tuple(q.shape[1:]) != (64, 128):
        raise ValueError("Q8KV4 attention requires E4M3 Q [batch,64,128]")
    if packed_k.dtype != torch.uint8 or packed_v.dtype != torch.uint8:
        raise ValueError("packed main K/V must be uint8")
    if packed_k.ndim != 2 or packed_v.ndim != 2:
        raise ValueError("packed main K/V must be [physical_page,bytes]")
    if scale_k.ndim != 2 or scale_v.ndim != 2:
        raise ValueError("main K/V scales must be [physical_page,bytes]")
    if tuple(topk_idx.shape) != (4, batch, 16) or topk_idx.dtype != torch.int32:
        raise ValueError("topk_idx must be int32 [4,batch,16]")
    if tuple(partial_out.shape) != (batch, 4, 16, 16, 128):
        raise ValueError("partial output has incompatible Q8KV4 shape")
    if tuple(partial_lse.shape) != (batch, 4, 16, 16):
        raise ValueError("partial LSE has incompatible Q8KV4 shape")
    if tuple(counts.shape) != (batch, 4) or counts.dtype != torch.int32:
        raise ValueError("counts must be int32 [batch,4]")
    if tuple(out.shape) != (batch, 64, 128) or out.dtype != torch.bfloat16:
        raise ValueError("output must be BF16 [batch,64,128]")
    for packed in (packed_k, packed_v):
        if packed.data_ptr() % 4 or packed.stride(0) % 4:
            raise ValueError("packed main page base/stride must be 4-byte aligned")
    _q8kv4_selected_page_partial[(batch, 4, 16)](
        q,
        packed_k,
        packed_v,
        scale_k.view(torch.uint8),
        scale_v.view(torch.uint8),
        block_table,
        topk_idx,
        seq_lens,
        partial_out,
        partial_lse,
        counts,
        packed_k.stride(0),
        packed_v.stride(0),
        scale_k.stride(0),
        scale_v.stride(0),
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        packed_k.shape[0],
        block_table.shape[1],
        ALPHA=softmax_scale_log2(sm_scale),
        LN2=LN2_F32,
        C1=POLY_EX2_C1,
        C2=POLY_EX2_C2,
        C3=POLY_EX2_C3,
        RINT=FP32_ROUND_INT,
        MMA_SCALE_LAYOUT=bool(mma_scale_layout),
        num_warps=4,
    )
    _q8kv4_combine[(batch, 4)](
        partial_out,
        partial_lse,
        counts,
        out,
        out.stride(0),
        out.stride(1),
        LOG2E=LOG2E_F32,
        valid_token_mask=valid_token_mask,
        HAS_VALID_TOKEN_MASK=valid_token_mask is not None,
        valid_token_mask_stride=(
            valid_token_mask.stride(0) if valid_token_mask is not None else 1
        ),
        num_warps=4,
    )
    return out

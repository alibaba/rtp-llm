"""V4.1-Flash FP4 GLOBAL decode attention with the SWA FlashMLA merge.

The FlashMLA wheel only accepts 584B FP8 ``extra_k_cache`` tensors, so the
FP4 GLOBAL (288B row-interleaved) pool cannot ride the dual-pool FlashMLA
call. Per the migration plan §5 step 6 the GLOBAL decode read is a Triton
compact reader (single path): FlashMLA scores the SWA pool (with the
attention sink), the Triton kernel scores the selected FP4 GLOBAL entries,
and the two softmaxes merge with the sink through FlashMLA's returned LSE.

The merge reproduces FlashMLA's in-kernel dual-pool semantics:
``out = (num_swa + num_global) / (Z_swa + Z_global + exp(sink))`` where
``num_swa = out_swa * (Z_swa + exp(sink))`` (FlashMLA normalizes its output
with the sink already applied; its LSE is the raw ``log Z_swa``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    FP4_GLOBAL_ENTRY_BYTES,
    FP4_GLOBAL_GROUP,
    FP4_GLOBAL_HEAD_DIM,
    _e2m1_to_float,
)


@triton.jit(do_not_specialize=["pool_stride", "lse_stride_b", "lse_stride_h"])
def _fp4_global_decode_attn_kernel(
    q_ptr,  # [B*S, H, 512] bf16 (post-RoPE)
    pool_ptr,  # uint8 [num_blocks, entries, 288] GLOBAL pool base
    slots_ptr,  # int32 [B*S, K] global slot ids (-1 = skip)
    sink_ptr,  # fp32 [H]
    swa_out_ptr,  # bf16 [B*S, H, 512] FlashMLA output (merged in place)
    swa_lse_ptr,  # fp32 [B, H, S] FlashMLA softmax LSE
    pool_stride,  # bytes per pool block
    lse_stride_b,  # LSE element strides for [B, H, S]
    lse_stride_h,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    ENTRIES: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)  # b * S + s
    head = tl.program_id(1)
    dims = tl.arange(0, DIM)
    q = tl.load(q_ptr + (row * H + head) * DIM + dims).to(tl.float32)
    sink = tl.load(sink_ptr + head)
    # A finite floor (not -inf) keeps the rescale factor well-defined when no
    # GLOBAL entry is valid; the sink guarantees a positive denominator.
    maximum = -1.0e30
    denominator = 0.0
    accumulator = tl.zeros((DIM,), tl.float32)
    for start in range(0, K, BLOCK_N):
        columns = start + tl.arange(0, BLOCK_N)
        slot = tl.load(slots_ptr + row * K + columns, columns < K, other=-1)
        valid = (columns < K) & (slot >= 0)
        safe = tl.maximum(slot, 0).to(tl.int64)
        base = pool_ptr + (safe // ENTRIES) * pool_stride + (safe % ENTRIES) * ROW_BYTES
        packed = tl.load(base[:, None] + dims[None, :] // 2, valid[:, None], other=0)
        code = (packed.to(tl.int32) >> ((dims[None, :] % 2) * 4)) & 15
        scale = tl.load(
            base[:, None] + DIM // 2 + dims[None, :] // GROUP,
            valid[:, None],
            other=0,
        )
        scale_value = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        values = _e2m1_to_float(code) * scale_value
        scores = tl.sum(values * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        rescale = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, axis=0)
        accumulator = accumulator * rescale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        maximum = next_maximum
    # Merge with the SWA softmax (FlashMLA already applied the sink to its
    # output; its LSE is the raw log Z). Empty SWA rows report +inf.
    request = row // S
    position = row % S
    lse = tl.load(
        swa_lse_ptr + request * lse_stride_b + head * lse_stride_h + position
    )
    lse = tl.where(lse > 1.0e30, -float("inf"), lse)
    reference = tl.maximum(tl.maximum(maximum, lse), sink)
    weight_global = denominator * tl.exp(maximum - reference)
    numerator_global = accumulator * tl.exp(maximum - reference)
    weight_swa = tl.exp(lse - reference) + tl.exp(sink - reference)
    swa = tl.load(swa_out_ptr + (row * H + head) * DIM + dims).to(tl.float32)
    numerator_swa = swa * weight_swa
    merged = (numerator_global + numerator_swa) / (
        weight_global + weight_swa
    )
    tl.store(
        swa_out_ptr + (row * H + head) * DIM + dims, merged.to(tl.bfloat16)
    )


def fp4_dual_decode_attention(
    *,
    q: torch.Tensor,  # [B, S, H, 512] bf16 post-RoPE
    swa_pool_3d: torch.Tensor,  # [num_blocks, eb_swa, 584] uint8
    global_pool_3d: torch.Tensor,  # [num_blocks, eb_global, 288] uint8
    attn_sink: torch.Tensor,  # [H] fp32
    swa_topk_3d: torch.Tensor,  # [B, S, win] int32 global slots into SWA pool
    global_topk_3d: torch.Tensor,  # [B, S, K] int32 global slots into GLOBAL pool
    swa_block_table: torch.Tensor,  # [B, max_blocks] int32 (unused by sparse path)
    sched_meta,  # FlashMLA sched_meta (SWA-only shape)
    fp8_op,  # SparseAttnV4DecodeFp8Op
) -> torch.Tensor:
    """Dual-pool V4.1 decode attention: FlashMLA SWA + Triton FP4 GLOBAL.

    Returns ``[B, S, H, 512]`` bf16 with FlashMLA's dual-pool sink semantics.
    The GLOBAL pool must be the contiguous uint8 ``[blocks, entries, 288]``
    view; slot ids index its flattened rows (``block * entries + offset``).
    """
    out_swa, lse_swa = fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=None,
        return_lse=True,
    )
    if lse_swa.stride(2) != 1:
        lse_swa = lse_swa.contiguous()
    batch, span, heads, head_dim = q.shape
    rows = batch * span
    topk = global_topk_3d.shape[-1]
    entries = global_pool_3d.shape[1]
    _fp4_global_decode_attn_kernel[(rows, heads)](
        q,
        global_pool_3d,
        global_topk_3d.contiguous(),
        attn_sink,
        out_swa,
        lse_swa,
        global_pool_3d.stride(0),
        lse_swa.stride(0),
        lse_swa.stride(1),
        SCALE=head_dim**-0.5,
        H=heads,
        S=span,
        K=topk,
        ENTRIES=entries,
        DIM=FP4_GLOBAL_HEAD_DIM,
        GROUP=FP4_GLOBAL_GROUP,
        ROW_BYTES=FP4_GLOBAL_ENTRY_BYTES,
        BLOCK_N=16,
        num_warps=8,
    )
    return out_swa
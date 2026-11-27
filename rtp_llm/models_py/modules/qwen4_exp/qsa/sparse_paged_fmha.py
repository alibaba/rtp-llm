"""Paged sparse GQA attention for Qwen4 decode and context-style prefill.

The QSA indexer returns request-local absolute token indices. The kernels
resolve these through the main MHA cache block table. The grouped path reads
HND pages directly. The SM90 graph decode path packs selected BF16 K/V for
varlen FlashAttention. Block id zero is the allocator sentinel.

The same B/S-shaped primitive serves ordinary decode, target verification, and
the correctness-first MTP draft incremental-prefill bridge (which invokes it
once per ragged request with ``B=1``).  Model-level routing remains explicitly
gated until indexer side-pool scoring/writes and prefix semantics are connected.
"""

from contextvars import ContextVar
from dataclasses import dataclass
import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@dataclass
class QsaFp8OutputRequest:
    v2: bool
    result: tuple[torch.Tensor, torch.Tensor] | None = None


_qsa_fp8_output_request: ContextVar[QsaFp8OutputRequest | None] = ContextVar(
    "_qsa_fp8_output_request", default=None
)

_BLOCK_K = 64


def _sparse_paged_indices_verdict(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
) -> torch.Tensor:
    """0-dim boolean verdict for _validate_sparse_paged_indices (True=valid).

    Computes the same combined predicate without any host synchronization so
    the production entry can consume it with a device-side assert.
    """
    table_width = int(block_table.shape[1])
    invalid_lengths = (kv_lens < 0) | (kv_lens > table_width * page_size)
    invalid_pool = block_table >= cache_blocks
    verdict = ~invalid_lengths.any() & ~invalid_pool.any()
    if selected.numel():
        invalid_negative = selected < -1
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        invalid_logical = logical_blocks >= table_width
        # An invalid selected index must never reach gather before its error
        # is reported. Clamp only the lookup; invalid_logical retains the
        # original value for the verdict and the diagnostic below.
        safe_logical_blocks = logical_blocks.clamp(max=table_width - 1)
        physical = torch.gather(
            block_table.unsqueeze(1).expand(-1, selected.shape[1], -1),
            2,
            safe_logical_blocks.to(torch.long),
        )
        invalid_missing = (selected >= 0) & (physical <= 0)
        invalid_selection = (
            invalid_negative | invalid_visible | invalid_logical | invalid_missing
        )
        verdict = verdict & ~invalid_selection.any()
    return verdict


def _validate_sparse_paged_indices(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
) -> None:
    """Check paged reads with one device-to-host verdict on valid inputs."""
    verdict = _sparse_paged_indices_verdict(
        block_table,
        kv_lens,
        selected,
        page_size=page_size,
        cache_blocks=cache_blocks,
    )
    if bool(verdict.item()):
        return

    # Error paths may synchronize again to preserve the existing specific
    # diagnostics. No main-cache write has happened at this point.
    invalid_lengths = (kv_lens < 0) | (kv_lens > int(block_table.shape[1]) * page_size)
    if bool(invalid_lengths.any().item()):
        raise ValueError("kv_lens exceed the block-table token capacity")
    if selected.numel():
        invalid_negative = selected < -1
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        invalid_logical = logical_blocks >= int(block_table.shape[1])
        safe_logical_blocks = logical_blocks.clamp(max=int(block_table.shape[1]) - 1)
        physical = torch.gather(
            block_table.unsqueeze(1).expand(-1, selected.shape[1], -1),
            2,
            safe_logical_blocks.to(torch.long),
        )
        invalid_missing = (selected >= 0) & (physical <= 0)
        if bool(invalid_negative.any().item()):
            raise ValueError("selected indices may only use -1 as padding")
        if bool(invalid_visible.any().item()):
            raise ValueError("selected contains an index outside its row's kv_len")
        if bool(invalid_logical.any().item()):
            raise ValueError("selected index exceeds the block-table width")
        if bool(invalid_missing.any().item()):
            raise ValueError("selected index resolves to an unallocated cache block")
    max_physical = int(block_table.max().item())
    raise ValueError(
        f"block table physical id {max_physical} exceeds cache blocks {cache_blocks}"
    )


def main_kv_scale_view(cache, scale, *, kv_heads, page_size):
    """Validate the native per-token/head K/V dequant scale layout.

    The fused native writer currently casts to E4M3 and writes unit scales.
    Readers consume the stored scale rather than assuming that value, so PD
    copies and future scaled writers preserve the same cache contract.
    No tensor contents are read back during capture.
    """
    if cache.dtype == torch.bfloat16:
        return None
    if cache.dtype != torch.float8_e4m3fn:
        raise ValueError("QSA main KV must be BF16 or FP8 E4M3")
    pages = int(cache.shape[0])
    if scale is None or scale.dtype != torch.float32 or scale.device != cache.device:
        raise ValueError("FP8 main KV requires device-local FP32 K/V scales")
    shape = (pages, 2, kv_heads, page_size)
    if scale.ndim == 2:
        width = 2 * kv_heads * page_size
        if scale.shape[0] != pages or scale.shape[1] < width or scale.stride(1) != 1:
            raise ValueError("FP8 main KV packed scale geometry is invalid")
        return torch.as_strided(
            scale, shape, (scale.stride(0), kv_heads * page_size, page_size, 1)
        )
    if tuple(scale.shape) != shape or tuple(scale.stride()[1:]) != (
        kv_heads * page_size,
        page_size,
        1,
    ):
        raise ValueError("FP8 main KV scale geometry must be HND per-token/head")
    return scale


@triton.jit
def _sparse_paged_online_split_kernel(
    q_ptr,
    cache_ptr,
    cache_scale_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    partial_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_cache_page,
    stride_cache_kv,
    stride_cache_h,
    stride_cache_token,
    stride_scale_page,
    stride_scale_kv,
    stride_scale_h,
    stride_scale_token,
    stride_bt_b,
    stride_bt_block,
    stride_len_b,
    stride_len_s,
    stride_sel_b,
    stride_sel_s,
    selected_width,
    max_blocks,
    cache_blocks,
    HAS_COUNT: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    # QUERY_LEN/BLOCKS_PER_SPLIT stay runtime args: making them constexpr
    # recompiles the kernel for every new request length.
    QUERY_LEN,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCKS_PER_SPLIT,
):
    partition = tl.program_id(0)
    batch = partition // (QUERY_LEN * NUM_SPLITS)
    head = tl.program_id(1)
    query = (partition // NUM_SPLITS) % QUERY_LEN
    split = partition % NUM_SPLITS
    kv_head = head // (H_Q // H_KV)
    rd = tl.arange(0, D)
    q = tl.load(
        q_ptr + batch * stride_q_b + head * stride_q_h + query * stride_q_s + rd
    ).to(tl.float32)
    row = selected_ptr + batch * stride_sel_b + query * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch * stride_len_b + query * stride_len_s)
    maximum = tl.full((), float("-inf"), tl.float32)
    normalizer = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    count = (
        tl.minimum(tl.maximum(tl.load(row + selected_width), 0), selected_width)
        if HAS_COUNT
        else selected_width
    )
    limit = tl.maximum(
        0,
        tl.minimum(
            BLOCKS_PER_SPLIT, tl.cdiv(count - split * BLOCK_K, NUM_SPLITS * BLOCK_K)
        ),
    )
    for block in range(limit):
        # Interleave chunks so a short valid prefix is not concentrated in
        # one partition when the selected tensor has a fixed wide capacity.
        offset = (block * NUM_SPLITS + split) * BLOCK_K + tl.arange(0, BLOCK_K)
        index = tl.load(row + offset, mask=offset < count, other=-1)
        logical = index // PAGE_SIZE
        candidate = (offset < count) & (index >= 0) & (index < kv_len)
        physical = tl.load(
            block_table_ptr + batch * stride_bt_b + logical * stride_bt_block,
            mask=candidate & (logical < max_blocks),
            other=0,
        )
        valid = (
            candidate
            & (logical < max_blocks)
            & (physical > 0)
            & (physical < cache_blocks)
        )
        key_ptr = (
            cache_ptr
            + physical[:, None] * stride_cache_page
            + kv_head * stride_cache_h
            + (index % PAGE_SIZE)[:, None] * stride_cache_token
            + rd[None, :]
        )
        key = tl.load(key_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        if FP8_CACHE:
            scale_ptr = (
                cache_scale_ptr
                + physical * stride_scale_page
                + kv_head * stride_scale_h
                + (index % PAGE_SIZE) * stride_scale_token
            )
            key *= tl.load(scale_ptr, mask=valid, other=1.0)[:, None]
        score = tl.sum(q[None, :] * key, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=0))
        alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
        weights = tl.where(valid, tl.exp(score - next_max), 0.0)
        value = tl.load(key_ptr + stride_cache_kv, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        if FP8_CACHE:
            value *= tl.load(scale_ptr + stride_scale_kv, mask=valid, other=1.0)[
                :, None
            ]
        acc = acc * alpha + tl.sum(weights[:, None] * value, axis=0)
        normalizer = normalizer * alpha + tl.sum(weights)
        maximum = next_max
    output_row = (batch * H_Q + head) * QUERY_LEN + query
    if NUM_SPLITS == 1:
        out = tl.where(normalizer > 0, acc / normalizer, 0.0)
        tl.store(partial_ptr + output_row * D + rd, out.to(q_ptr.dtype.element_ty))
    else:
        base = partial_ptr + (output_row * NUM_SPLITS + split) * (D + 2)
        tl.store(base + rd, acc)
        tl.store(base + D, maximum)
        tl.store(base + D + 1, normalizer)


def _grouped_split_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    selected_width: int,
) -> bool:
    """Gate the tensor-core grouped kernel to proven SM90 geometry.

    Shape-only checks; no tensor contents are read and no device synchronisation
    happens, so the gate is safe during CUDA graph capture.
    """
    if torch.version.hip is not None or not q.is_cuda:
        return False
    if cached_device_capability(q.device) != (9, 0):
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4:
        return False
    if kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
        return False
    heads, kv_heads = int(q.shape[1]), int(kv_cache.shape[2])
    if kv_heads <= 0 or heads % kv_heads:
        return False
    group = heads // kv_heads
    # tl.dot needs M >= 16; keep the register tile bounded.
    block_m = max(16, triton.next_power_of_2(group))
    if group > 32 or block_m > 32:
        return False
    if not (16 <= q.shape[3] <= 256) or q.shape[3] & (q.shape[3] - 1):
        return False
    if kv_cache.shape[3] != page_size or kv_cache.shape[4] != q.shape[3]:
        return False
    if page_size <= 0 or page_size & (page_size - 1):
        return False
    if selected.dim() != 3 or selected.dtype != torch.int32:
        return False
    if selected_width <= 0:
        return False
    return kv_cache.stride(-1) == 1


def _grouped_split_config(rows: int, kv_heads: int, selected_width: int) -> tuple:
    """(BLOCK_N, num_warps, num_stages, num_splits) for the grouped kernel.

    rows = batch * query_len. Splits keep small decode/verify batches parallel
    across SMs; prefill rows already saturate the device with one split.
    Starting table ported from the vLLM sm90 tuning shape; retuned per-benchmark.
    """
    base = rows * kv_heads
    if base > 4096:
        splits = 1
    elif base > 512:
        splits = 2
    elif base > 64:
        splits = 4
    else:
        splits = 8
    # BLOCK_N=128 measured fastest for prefill on SM90: fewer tiles amortise
    # the per-tile gather setup, and the bf16 dot operands still fit in shared
    # memory with two pipeline stages.
    block_n = 128 if base > 256 else 64 if base > 64 else 32
    tiles = triton.cdiv(selected_width, block_n)
    # The merge kernel indexes splits with tl.arange, so the split count itself
    # must be a power of two.
    splits = min(splits, 1 << (tiles.bit_length() - 1))
    return block_n, 4, 2, splits


@triton.jit
def _sparse_paged_grouped_split_kernel(
    q_ptr,
    cache_ptr,
    cache_scale_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    partial_ptr,
    gate_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_cache_page,
    stride_cache_kv,
    stride_cache_h,
    stride_cache_token,
    stride_scale_page,
    stride_scale_kv,
    stride_scale_h,
    stride_scale_token,
    stride_bt_b,
    stride_bt_block,
    stride_len_b,
    stride_len_s,
    stride_sel_b,
    stride_sel_s,
    scores_ptr,
    stride_score_row,
    score_width,
    selected_width,
    max_blocks,
    cache_blocks,
    HAS_COUNT: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    H_Q: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # QUERY_LEN/NUM_TILES stay runtime args: they vary with every request
    # length, and constexpr keys recompile the kernel per new length.
    QUERY_LEN,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WITH_GATE: tl.constexpr = False,
    OUT_BTHD: tl.constexpr = False,
    PV_BF16_RESIDUAL: tl.constexpr = False,
    BLOCK_IDS: tl.constexpr = False,
    FINITE_ONLY: tl.constexpr = True,
):
    """Tensor-core grouped sparse paged GQA.

    One program owns a full GQA group (all query heads of one query row sharing
    one KV head), so each gathered K/V tile is loaded once per group instead of
    once per head, and QK^T / PV run on tensor cores via tl.dot. The group is
    the MMA M dimension; both dot operands keep the wgmma-friendly sourcing
    (A from registers, B from the pipelined gather tiles).
    """
    partition = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = partition // (QUERY_LEN * NUM_SPLITS)
    query = (partition // NUM_SPLITS) % QUERY_LEN
    split = partition % NUM_SPLITS

    heads = tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    live = heads < GROUP_SIZE
    q = tl.load(
        q_ptr
        + batch * stride_q_b
        + (kv_head * GROUP_SIZE + heads)[:, None] * stride_q_h
        + query * stride_q_s
        + rd[None, :],
        mask=live[:, None],
        other=0.0,
    )

    row = selected_ptr + batch * stride_sel_b + query * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch * stride_len_b + query * stride_len_s)
    count = (
        tl.minimum(tl.maximum(tl.load(row + selected_width), 0), selected_width)
        if HAS_COUNT
        else selected_width
    )
    tiles = tl.cdiv(count, BLOCK_N)
    tile_start = split * NUM_TILES // NUM_SPLITS
    tile_end = tl.minimum((split + 1) * NUM_TILES // NUM_SPLITS, tiles)

    maximum = tl.full([BLOCK_M], float("-inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    columns = tl.arange(0, BLOCK_N)
    out_rows = (batch * H_Q + kv_head * GROUP_SIZE + heads) * QUERY_LEN + query
    if BLOCK_IDS:
        complete = (kv_len - tl.where(kv_len < 0, 3, 0)) // 4
        row_scores = scores_ptr + (batch * QUERY_LEN + query) * stride_score_row
        BLOCK_TILES: tl.constexpr = 2048 // BLOCK_N
        BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // 4
        PAGE_BLOCKS: tl.constexpr = PAGE_SIZE // 4
    for tile in range(tile_start, tile_end):
        if BLOCK_IDS:
            if tile < BLOCK_TILES:
                b_off = tile * BLOCKS_PER_TILE + tl.arange(0, BLOCKS_PER_TILE)
                blk = tl.load(row + b_off, mask=b_off < 512, other=-1)
                in_score = (b_off < 512) & (blk >= 0) & (blk < score_width)
                sval = tl.load(row_scores + blk, mask=in_score, other=float("-inf"))
                if FINITE_ONLY:
                    finite = (sval == sval) & (tl.abs(sval) != float("inf"))
                else:
                    finite = sval > float("-inf")
                b_cand = in_score & finite & (blk < complete)
                b_log = blk // PAGE_BLOCKS
                b_phys = tl.load(
                    block_table_ptr + batch * stride_bt_b + b_log * stride_bt_block,
                    mask=b_cand & (b_log < max_blocks),
                    other=0,
                )
                b_valid = (
                    b_cand
                    & (b_log < max_blocks)
                    & (b_phys > 0)
                    & (b_phys < cache_blocks)
                )
                b_safe_page = tl.maximum(b_phys, 0).to(tl.int64)
                b_slot_base = (tl.maximum(blk, 0) % PAGE_BLOCKS) * 4
                safe_page = tl.broadcast_to(
                    b_safe_page[:, None], [BLOCKS_PER_TILE, 4]
                ).reshape(BLOCK_N)
                page_slot = (
                    b_slot_base[:, None] + tl.arange(0, 4)[None, :]
                ).reshape(BLOCK_N)
                valid = tl.broadcast_to(
                    b_valid[:, None], [BLOCKS_PER_TILE, 4]
                ).reshape(BLOCK_N)
            else:
                tail_count = tl.minimum(tl.maximum(kv_len - complete * 4, 0), 3)
                t_cand = columns < tail_count
                t_base = complete * 4
                t_log = t_base // PAGE_SIZE
                t_phys = tl.load(
                    block_table_ptr + batch * stride_bt_b + t_log * stride_bt_block,
                    mask=(tail_count > 0) & (t_base >= 0) & (t_log < max_blocks),
                    other=0,
                )
                valid = (
                    t_cand
                    & (t_base >= 0)
                    & (t_log < max_blocks)
                    & (t_phys > 0)
                    & (t_phys < cache_blocks)
                )
                safe_page = (
                    tl.zeros([BLOCK_N], dtype=tl.int64)
                    + tl.maximum(t_phys, 0).to(tl.int64)
                )
                page_slot = (tl.maximum(t_base, 0) % PAGE_SIZE) + tl.where(
                    t_cand, columns, 0
                )
        else:
            offset = tile * BLOCK_N + columns
            index = tl.load(row + offset, mask=offset < count, other=-1)
            logical = index // PAGE_SIZE
            candidate = (offset < count) & (index >= 0) & (index < kv_len)
            physical = tl.load(
                block_table_ptr + batch * stride_bt_b + logical * stride_bt_block,
                mask=candidate & (logical < max_blocks),
                other=0,
            )
            valid = (
                candidate
                & (logical < max_blocks)
                & (physical > 0)
                & (physical < cache_blocks)
            )
            safe_page = tl.maximum(physical, 0).to(tl.int64)
            page_slot = tl.maximum(index, 0) % PAGE_SIZE
        keys = tl.load(
            cache_ptr
            + safe_page[None, :] * stride_cache_page
            + kv_head * stride_cache_h
            + page_slot[None, :] * stride_cache_token
            + rd[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            cache_ptr
            + safe_page[:, None] * stride_cache_page
            + stride_cache_kv
            + kv_head * stride_cache_h
            + page_slot[:, None] * stride_cache_token
            + rd[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if FP8_CACHE:
            # Per-token scales factor out of both dots: the K scale multiplies
            # the score column after the QK dot, and the V scale folds into the
            # attention weights before the PV dot. This keeps the tile-wide
            # work to a single exact fp8->bf16 conversion.
            key_scale = tl.load(
                cache_scale_ptr
                + safe_page * stride_scale_page
                + kv_head * stride_scale_h
                + page_slot * stride_scale_token,
                mask=valid,
                other=0.0,
            )
            value_scale = tl.load(
                cache_scale_ptr
                + safe_page * stride_scale_page
                + stride_scale_kv
                + kv_head * stride_scale_h
                + page_slot * stride_scale_token,
                mask=valid,
                other=0.0,
            )
            keys = keys.to(tl.bfloat16)
            values = values.to(tl.bfloat16)
            scores = tl.dot(q, keys)
            scores = scores * (key_scale * SCALE)[None, :]
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
            weights = tl.where(valid[None, :], tl.exp(scores - next_max[:, None]), 0.0)
            # The softmax normalizer stays in raw probability space; only the PV
            # operand carries the per-token V scale.
            normalizer = normalizer * alpha + tl.sum(weights, 1)
            pv = weights * value_scale[None, :]
            acc = acc * alpha[:, None] + tl.dot(pv.to(tl.bfloat16), values)
            maximum = next_max
        else:
            keys = keys.to(tl.bfloat16)
            values = values.to(tl.bfloat16)
            scores = tl.dot(q, keys) * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
            weights = tl.where(valid[None, :], tl.exp(scores - next_max[:, None]), 0.0)
            rounded_weights = weights.to(tl.bfloat16)
            pv = tl.dot(rounded_weights, values)
            if PV_BF16_RESIDUAL:
                # Preserve the low part of the FP32 softmax weights for Decode.
                residual = (weights - rounded_weights.to(tl.float32)).to(tl.bfloat16)
                pv = tl.dot(residual, values, pv)
            acc = acc * alpha[:, None] + pv
            normalizer = normalizer * alpha + tl.sum(weights, 1)
            maximum = next_max

    if NUM_SPLITS == 1:
        out = tl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
        if WITH_GATE:
            # Same quantised sigmoid math as the merge kernel: the gate
            # rounds to bf16 before the fp32 multiply, so single-split and
            # multi-split requests stay bit-identical.
            gate_offset = (
                (batch * QUERY_LEN + query) * H_Q + kv_head * GROUP_SIZE + heads
            )
            gate = tl.load(
                gate_ptr + gate_offset[:, None] * D + rd[None, :],
                mask=live[:, None],
                other=0.0,
            ).to(tl.float32)
            sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
            out = out.to(tl.bfloat16).to(tl.float32) * sigmoid
        store_rows = (
            (batch * QUERY_LEN + query) * H_Q + kv_head * GROUP_SIZE + heads
            if OUT_BTHD
            else out_rows
        )
        tl.store(
            partial_ptr + store_rows[:, None] * D + rd[None, :],
            out.to(partial_ptr.dtype.element_ty),
            mask=live[:, None],
        )
    else:
        base = (out_rows * NUM_SPLITS + split) * (D + 2)
        tl.store(partial_ptr + base[:, None] + rd[None, :], acc, mask=live[:, None])
        tl.store(partial_ptr + base + D, maximum, mask=live)
        tl.store(partial_ptr + base + D + 1, normalizer, mask=live)


@triton.jit
def _merge_sparse_paged_splits_kernel(
    partial_ptr,
    out_ptr,
    gate_ptr,
    D: tl.constexpr,
    # NUM_SPLITS indexes tl.arange so it must stay constexpr (bounded
    # power-of-two set); QUERIES varies per request and stays runtime.
    NUM_SPLITS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERIES,
    WITH_GATE: tl.constexpr,
    OUT_BTHD: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    splits = tl.arange(0, NUM_SPLITS)
    rd = tl.arange(0, D)
    if WITH_GATE or OUT_BTHD:
        batch = row // (HEADS * QUERIES)
        head = (row // QUERIES) % HEADS
        query = row % QUERIES
        offset = ((batch * QUERIES + query) * HEADS + head) * D + rd
        if WITH_GATE:
            gate = tl.load(gate_ptr + offset).to(tl.float32)
    else:
        offset = row * D + rd
    if PDL:
        tl.extra.cuda.gdc_wait()
    base = partial_ptr + row * NUM_SPLITS * (D + 2)
    maxima = tl.load(base + splits * (D + 2) + D)
    sums = tl.load(base + splits * (D + 2) + D + 1)
    maximum = tl.max(maxima, axis=0)
    # Empty partitions contribute exactly zero, including an all-empty row.
    scale = tl.where(sums > 0, tl.exp(maxima - maximum), 0.0)
    partial = tl.load(base + splits[:, None] * (D + 2) + rd[None, :])
    numerator = tl.sum(partial * scale[:, None], axis=0)
    denominator = tl.sum(sums * scale, axis=0)
    out = tl.where(denominator > 0, numerator / denominator, 0.0)
    if WITH_GATE:
        sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
        out = out.to(tl.bfloat16).to(tl.float32) * sigmoid
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["QUERIES", "S_ROW_STRIDE", "S_COL_STRIDE"])
def _merge_sparse_paged_splits_fp8_kernel(
    partial_ptr,
    q_ptr,
    s_ptr,
    gate_ptr,
    S_ROW_STRIDE,
    S_COL_STRIDE,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERIES,
    V2: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    splits = tl.arange(0, NUM_SPLITS)
    rd = tl.arange(0, D)
    batch = row // (HEADS * QUERIES)
    head = (row // QUERIES) % HEADS
    query = row % QUERIES
    token = batch * QUERIES + query
    offset = (token * HEADS + head) * D + rd
    gate = tl.load(gate_ptr + offset).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    base = partial_ptr + row * NUM_SPLITS * (D + 2)
    maxima = tl.load(base + splits * (D + 2) + D)
    sums = tl.load(base + splits * (D + 2) + D + 1)
    maximum = tl.max(maxima, axis=0)
    scale = tl.where(sums > 0, tl.exp(maxima - maximum), 0.0)
    partial = tl.load(base + splits[:, None] * (D + 2) + rd[None, :])
    numerator = tl.sum(partial * scale[:, None], axis=0)
    denominator = tl.sum(sums * scale, axis=0)
    out = tl.where(denominator > 0, numerator / denominator, 0.0)
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    out_bf16 = (out.to(tl.bfloat16).to(tl.float32) * sigmoid).to(tl.bfloat16)

    GROUPS_PER_HEAD: tl.constexpr = D // 128
    val = out_bf16.to(tl.float32).reshape(GROUPS_PER_HEAD, 128)
    absolute = tl.abs(val)
    raw_max = tl.max(tl.where(absolute == absolute, absolute, 0.0), axis=1)
    amax = tl.maximum(raw_max, 1.0e-4)
    if V2:
        q_scale = amax * (1.0 / 448.0)
        code = val * tl.div_rn(448.0, amax)[:, None]
    else:
        q_scale = tl.div_rn(amax, 448.0)
        code = tl.div_rn(val, q_scale[:, None])
    code = tl.where(code == code, code, -448.0)
    code = tl.minimum(448.0, tl.maximum(-448.0, code)).to(tl.float8e4nv)

    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(q_ptr + offset, code.reshape(D))
    g_idx = tl.arange(0, GROUPS_PER_HEAD)
    tl.store(
        s_ptr + token * S_ROW_STRIDE + (head * GROUPS_PER_HEAD + g_idx) * S_COL_STRIDE,
        q_scale,
    )


@triton.jit
def _resolve_paged_slots(
    selected,
    block_table,
    cu_seqlens,
    lengths,
    slots,
    selected_row_stride,
    cu_stride,
    length_stride,
    table_row_stride,
    table_col_stride,
    batch,
    width,
    logical_blocks,
    cache_blocks,
    PAGE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    DECODE: tl.constexpr = False,
    PACKED_CONTEXT: tl.constexpr = False,
):
    # Resolve pages before the attention loop, as in SGLang's QSA paged reader.
    # The scan keeps the tail after the compressed selection, including on rows
    # with holes between those two regions.
    row = tl.program_id(0)
    if DECODE:
        request = row
        visible = tl.load(lengths + row * length_stride)
    else:
        req = tl.arange(0, REQUEST_BLOCK)
        end = tl.load(cu_seqlens + (req + 1) * cu_stride, req < batch, other=2147483647)
        request = tl.sum((row >= end).to(tl.int32), axis=0)
        start = tl.load(cu_seqlens + request * cu_stride)
        visible = tl.load(lengths + request * length_stride) + row - start + 1

    column = tl.arange(0, ROW_BLOCK)
    token = tl.load(
        selected + row * selected_row_stride + column, column < width, other=-1
    )
    logical = token // PAGE
    candidate = (token >= 0) & (token < visible) & (logical < logical_blocks)
    physical = tl.load(
        block_table + request * table_row_stride + logical * table_col_stride,
        candidate,
        other=0,
    )
    valid = candidate & (physical > 0) & (physical < cache_blocks)
    packed_offset = tl.cumsum(valid.to(tl.int32)) - 1
    slot_row = slots + row * (width + 1)
    slot = (
        request * logical_blocks * PAGE + token
        if PACKED_CONTEXT
        else physical * PAGE + token % PAGE
    )
    tl.store(slot_row + packed_offset, slot, valid)
    tl.store(slot_row + width, tl.sum(valid.to(tl.int32), axis=0))


def _main_cache_hnd_view(kv_cache, kv_head_num, page_size, head_dim):
    if kv_cache.dim() == 2:
        if kv_head_num is None or kv_head_num <= 0:
            raise ValueError("kv_head_num is required for a packed HybridPool view")
        required_width = 2 * kv_head_num * page_size * head_dim
        if int(kv_cache.shape[1]) < required_width:
            raise ValueError(
                f"packed kv_cache width {kv_cache.shape[1]} is smaller than "
                f"the required main-cache width {required_width}"
            )
        if kv_cache.stride(1) != 1:
            raise ValueError("packed HybridPool rows must have contiguous elements")
        kv_cache = torch.as_strided(
            kv_cache,
            (int(kv_cache.shape[0]), 2, kv_head_num, page_size, head_dim),
            (
                kv_cache.stride(0),
                kv_head_num * page_size * head_dim,
                page_size * head_dim,
                head_dim,
                1,
            ),
        )
    if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
        raise ValueError("kv_cache must be [pages, 2, H_kv, page_size, D]")
    _, _, kv_heads, cache_page_size, cache_head_dim = kv_cache.shape
    if kv_head_num is not None and kv_heads != kv_head_num:
        raise ValueError(
            f"kv_cache has {kv_heads} KV heads, expected kv_head_num={kv_head_num}"
        )
    if cache_page_size != page_size or cache_head_dim != head_dim:
        raise ValueError(
            "kv_cache page/head geometry does not match q: "
            f"cache={tuple(kv_cache.shape)}, page_size={page_size}, D={head_dim}"
        )
    expected_inner_strides = (
        kv_heads * page_size * head_dim,
        page_size * head_dim,
        head_dim,
        1,
    )
    if kv_cache.stride()[1:] != expected_inner_strides:
        raise ValueError(
            f"kv_cache must use HND inner layout; got strides {kv_cache.stride()}"
        )
    return kv_cache


def _pack_paged_slots(
    selected: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
    cu_seqlens: torch.Tensor | None = None,
    packed_context: bool = False,
) -> torch.Tensor:
    rows, width = selected.shape
    batch, logical_blocks = block_table.shape
    slots = torch.empty((rows, width + 1), device=selected.device, dtype=torch.int32)
    decode = cu_seqlens is None
    _resolve_paged_slots[(rows,)](
        selected,
        block_table,
        lengths if decode else cu_seqlens,
        lengths,
        slots,
        selected.stride(0),
        0 if decode else cu_seqlens.stride(0),
        lengths.stride(0),
        *block_table.stride(),
        batch,
        width,
        logical_blocks,
        cache_blocks,
        PAGE=page_size,
        ROW_BLOCK=triton.next_power_of_2(width),
        REQUEST_BLOCK=1 if decode else triton.next_power_of_2(batch),
        DECODE=decode,
        PACKED_CONTEXT=packed_context,
        num_warps=4,
    )
    return slots


def _supports_sm90_paged_packing(q, kv_cache, page_size, kv_head_num):
    return (
        torch.version.hip is None
        and q.is_cuda
        and q.dtype == kv_cache.dtype == torch.bfloat16
        and q.shape[-2:] == (12, 256)
        and kv_head_num == 1
        and page_size in (64, 128)
        and kv_cache.shape[0] * page_size <= 2**31
        and cached_device_capability(q.device) == (9, 0)
    )


@triton.jit
def _pack_prefill_context(
    cache,
    table,
    cu,
    prefixes,
    k,
    v,
    table_row_stride,
    table_col_stride,
    cu_stride,
    prefix_stride,
    cache_stride,
    cache_blocks,
    context_length,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, 256)
    end = tl.load(prefixes + request * prefix_stride)
    end += tl.load(cu + (request + 1) * cu_stride) - tl.load(cu + request * cu_stride)
    live = (token < end) & (token < context_length)
    physical = tl.load(
        table + request * table_row_stride + token // PAGE * table_col_stride,
        live,
        other=0,
    )
    live &= (physical > 0) & (physical < cache_blocks)
    source = physical.to(tl.int64) * cache_stride + token % PAGE * 256
    target = (request * context_length + token).to(tl.int64) * 256
    key = tl.load(cache + source[:, None] + dim[None, :], live[:, None], other=0.0)
    value = tl.load(
        cache + source[:, None] + PAGE * 256 + dim[None, :], live[:, None], other=0.0
    )
    tl.store(k + target[:, None] + dim[None, :], key, (token < context_length)[:, None])
    tl.store(
        v + target[:, None] + dim[None, :], value, (token < context_length)[:, None]
    )


def try_sm90_packed_paged_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefixes: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int,
    output_gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """Pack cached contexts and reuse the cold-prefill attention kernel.

    Like SGLang, each request's KV is copied once per layer rather than doing
    paged address calculations in every query's attention loop. Both metadata
    preparation and KV packing stay on-device, including ragged request splits.
    """
    rows, _, dim = q.shape
    context_length = block_table.shape[1] * page_size
    batch = block_table.shape[0]
    if (
        not 512 <= rows <= 32768
        or selected.shape != (rows, 2051)
        or selected.stride(-1) != 1
        or q.stride(-1) != 1
        or batch * context_length > 2**31 - 1
        or not _supports_sm90_paged_packing(q, kv_cache, page_size, kv_head_num)
    ):
        return None
    from .sparse_prefill_sm90 import _is_h20_device, _sparse_gqa_prefill_sm90

    kv_cache = _main_cache_hnd_view(kv_cache, kv_head_num, page_size, dim)
    cache_blocks = int(kv_cache.shape[0])
    slots = _pack_paged_slots(
        selected,
        block_table,
        prefixes,
        page_size=page_size,
        cache_blocks=cache_blocks,
        cu_seqlens=cu_seqlens,
        packed_context=True,
    )
    k, v = torch.empty((2, batch * context_length, dim), device=q.device, dtype=q.dtype)
    _pack_prefill_context[(batch, triton.cdiv(context_length, 32))](
        kv_cache,
        block_table,
        cu_seqlens,
        prefixes,
        k,
        v,
        *block_table.stride(),
        cu_seqlens.stride(0),
        prefixes.stride(0),
        kv_cache.stride(0),
        cache_blocks,
        context_length,
        PAGE=page_size,
        BLOCK=32,
        num_warps=4,
    )
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _sparse_gqa_prefill_sm90[(rows, 1)](
        q,
        k,
        v,
        out,
        slots,
        dim**-0.5,
        0,
        q.stride(1),
        q.stride(0),
        0,
        k.stride(0),
        0,
        v.stride(0),
        0,
        out.stride(1),
        out.stride(0),
        S=rows,
        T=batch * context_length,
        gate=output_gate if output_gate is not None else out,
        WIDTH=2051,
        BLOCK_M=16,
        BLOCK_N=32,
        HEAD_DIM=dim,
        WITH_GATE=output_gate is not None,
        num_warps=4,
        num_stages=2,
        # On H20, capping registers keeps five CTAs resident per SM for the 32-key tile.
        maxnreg=96 if _is_h20_device(torch.cuda.get_device_name(q.device)) else None,
    )
    return out


@triton.jit
def _paged_decode_cu_seqlens(slots, cu_q, cu_k, batch, width, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    counts = tl.load(slots + rows * (width + 1) + width, rows < batch, other=0)
    tl.store(cu_q, 0)
    tl.store(cu_k, 0)
    tl.store(cu_q + rows + 1, rows + 1, rows < batch)
    tl.store(cu_k + rows + 1, tl.cumsum(counts, 0), rows < batch)


@triton.jit
def _gather_paged_decode_kv(
    cache,
    slots,
    cu_k,
    k,
    v,
    width,
    cache_stride,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, D)
    start = tl.load(cu_k + row)
    valid = column < tl.load(cu_k + row + 1) - start
    slot = tl.load(slots + row * (width + 1) + column, valid, other=0)
    src = (slot // PAGE).to(tl.int64) * cache_stride + (slot % PAGE) * D
    dst = (start + column) * D
    key = tl.load(cache + src[:, None] + dim[None, :], valid[:, None], other=0)
    value = tl.load(
        cache + src[:, None] + PAGE * D + dim[None, :], valid[:, None], other=0
    )
    tl.store(k + dst[:, None] + dim[None, :], key, valid[:, None])
    tl.store(v + dst[:, None] + dim[None, :], value, valid[:, None])


def try_sm90_packed_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int,
    output_gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """Gather the selected paged KV, then run native varlen FlashAttention.

    This follows SGLang's SM90 decode packing. All lengths and page lookups
    stay on-device so each graph replay can change the selected KV.
    """
    batch, heads, dim = q.shape
    if (
        not 16 <= batch <= 64
        or selected.shape != (batch, 1, 2051)
        or selected.stride(-1) != 1
        or not _supports_sm90_paged_packing(q, kv_cache, page_size, kv_head_num)
        or not torch.backends.cuda.is_flash_attention_available()
    ):
        return None
    kv_cache = _main_cache_hnd_view(kv_cache, kv_head_num, page_size, dim)
    width = selected.shape[-1]
    slots = _pack_paged_slots(
        selected[:, 0],
        block_table,
        kv_lens,
        page_size=page_size,
        cache_blocks=kv_cache.shape[0],
    )
    cu_q, cu_k = torch.empty((2, batch + 1), device=q.device, dtype=torch.int32)
    _paged_decode_cu_seqlens[(1,)](
        slots,
        cu_q,
        cu_k,
        batch,
        width,
        BLOCK=triton.next_power_of_2(batch),
        num_warps=1,
    )
    k, v = torch.empty((2, batch * width, 1, dim), device=q.device, dtype=q.dtype)
    block = 16
    _gather_paged_decode_kv[(batch, triton.cdiv(width, block))](
        kv_cache,
        slots,
        cu_k,
        k,
        v,
        width,
        kv_cache.stride(0),
        PAGE=page_size,
        D=dim,
        BLOCK=block,
        num_warps=4,
    )
    # Each request has one query and all packed KV are already causal.
    output = torch.ops.aten._flash_attention_forward(
        q.contiguous(), k, v, cu_q, cu_k, 1, width, 0.0, False, False, scale=dim**-0.5
    )[0]
    output = output.view(batch, heads * dim)
    return output if output_gate is None else output * torch.sigmoid(output_gate)


def supports_bthd_paged_output(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int | None = None,
    selected_has_count: bool = False,
    block_selection=None,
) -> bool:
    if q.dim() != 4 or q.dtype != torch.bfloat16 or not q.is_cuda or q.stride(-1) != 1:
        return False
    try:
        kv_view = _main_cache_hnd_view(kv_cache, kv_head_num, page_size, int(q.shape[3]))
    except ValueError:
        return False
    selected_width = (
        2051
        if block_selection is not None
        else int(selected.shape[-1]) - int(selected_has_count)
    )
    return _grouped_split_supported(
        q, kv_view, selected, page_size=page_size, selected_width=selected_width
    )


def try_h20_direct_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int,
    output_gate: torch.Tensor | None,
    block_selection=None,
) -> torch.Tensor | None:
    """Read sparse BF16 KV directly in the existing grouped split kernel."""
    batch, heads, dim = q.shape
    expected_width = 512 if block_selection is not None else 2051
    if (
        not 1 <= batch <= 512
        or page_size != 64
        or selected.shape != (batch, 1, expected_width)
        or q.stride(-1) != 1
        or selected.stride(-1) != 1
        or not _supports_sm90_paged_packing(q, kv_cache, page_size, kv_head_num)
    ):
        return None
    output = sparse_paged_gqa_attn(
        q.unsqueeze(2),
        kv_cache,
        block_table,
        kv_lens,
        selected,
        page_size=page_size,
        kv_head_num=kv_head_num,
        graph_capture=torch.cuda.is_current_stream_capturing(),
        output_gate=(
            None if output_gate is None else output_gate.view(batch, 1, heads, dim)
        ),
        validate_selected=False,
        bthd_output=True,
        _h20_direct_decode=True,
        block_selection=block_selection,
    )
    return output.view(batch, heads * dim)


def sparse_paged_gqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int | None = None,
    kv_scale: torch.Tensor | None = None,
    block_k: int = _BLOCK_K,
    graph_capture: bool = False,
    output_gate: torch.Tensor | None = None,
    selected_has_count: bool = False,
    fp16_pv: bool = False,
    validate_selected: bool = True,
    bthd_output: bool = False,
    _h20_direct_decode: bool = False,
    block_selection=None,
) -> torch.Tensor:
    """Attend to request-local token indices in a paged main MHA cache.

    Args:
        q: ``[B, H_q, S, D]`` BF16 CUDA queries after RoPE.
        kv_cache: HND cache ``[pages, 2, H_kv, page_size, D]`` or its packed
            two-dimensional HybridPool view.
        block_table: ``[B, max_blocks]`` int32 CUDA logical-to-physical map.
        kv_lens: ``[B, S]`` int32 CUDA visible main-KV lengths.
        selected: ``[B, S, K]`` int32 CUDA request-local token indices; ``-1``
            is padding.
        page_size: tokens per main MHA cache page.
        kv_head_num: local KV head count. Required for a packed HybridPool view,
            whose row stride may be padded by a larger cache group and therefore
            cannot be inferred from ``kv_cache.shape[1]``.
        graph_capture: capture only. The caller must validate the initial
            metadata before capture; replayed values are masked by the kernel.
        validate_selected: run the per-selection fail-fast preflight. Internal
            callers whose selections come from this model's causal QSA indexer
            pass ``False`` and trust the in-kernel masking (invalid ids,
            out-of-range counts, and unallocated blocks are masked by the
            split/merge kernels on either path), matching the SM120 sparse
            paths' trust-and-mask contract. Public and injected selections
            retain the preflight. Strict validation
            (``RTP_LLM_QSA_STRICT_VALIDATION=1``, tests and CI) restores the
            full per-check validation regardless of this flag.
        bthd_output: write the result as contiguous ``[B, S, H, D]`` instead of
            ``[B, H, S, D]``. Only the SM90 tensor-core grouped path honours
            this; other geometries raise. Combined with a strided BTHD query
            view this removes both per-request layout clones.
    """
    if q.dim() != 4 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be a rank-4 BF16 CUDA tensor")
    batch, q_heads, query_len, head_dim = q.shape
    if output_gate is not None and not (
        output_gate.shape == (batch, query_len, q_heads, head_dim)
        and output_gate.dtype == torch.bfloat16
        and output_gate.device == q.device
        and output_gate.is_contiguous()
    ):
        raise ValueError(
            "QSA output gate must be contiguous BF16 [B,S,H,D] on the query device"
        )
    if q_heads <= 0 or head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("query heads must be positive and head_dim a power of two")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a positive power of two, got {page_size}")
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a positive power of two, got {block_k}")
    if (
        block_table.dim() != 2
        or tuple(block_table.shape[:1]) != (batch,)
        or block_table.dtype != torch.int32
    ):
        raise ValueError(f"block_table must be int32 [{batch}, max_blocks]")
    if kv_lens.shape != (batch, query_len) or kv_lens.dtype != torch.int32:
        raise ValueError(f"kv_lens must be int32 [{batch}, {query_len}]")
    if selected.dim() != 3 or selected.shape[:2] != (batch, query_len):
        raise ValueError(f"selected must have leading shape [{batch}, {query_len}]")
    if selected.dtype != torch.int32:
        raise ValueError("selected must be int32")

    if selected_has_count and selected.shape[-1] != 2052:
        raise ValueError("packed QSA requires 2051 token slots and one valid_count")

    kv_cache = _main_cache_hnd_view(kv_cache, kv_head_num, page_size, head_dim)
    cache_blocks, _, kv_heads, _, _ = kv_cache.shape
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"query heads {q_heads} must be divisible by KV heads {kv_heads}"
        )
    tensors = (kv_cache, block_table, kv_lens, selected)
    if any(not tensor.is_cuda or tensor.device != q.device for tensor in tensors):
        raise ValueError(
            "q, kv_cache, block_table, kv_lens, and selected must share CUDA"
        )
    scale_view = main_kv_scale_view(
        kv_cache, kv_scale, kv_heads=kv_heads, page_size=page_size
    )
    scale_pointer = kv_cache if scale_view is None else scale_view
    scale_strides = (0, 0, 0, 0) if scale_view is None else scale_view.stride()
    if int(block_table.shape[1]) == 0:
        raise ValueError("block_table must contain at least one logical block")
    if graph_capture:
        warmup = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
        if not torch.cuda.is_current_stream_capturing() and not warmup:
            raise RuntimeError(
                "qwen4 sparse paged GQA graph_capture requires an active capture or graph warmup"
            )
    elif validate_selected or qsa_strict_validation_enabled():
        # Fail-fast preflight for public and injected selections. Trusted
        # internal selections skip it: the split/merge kernels already mask
        # invalid ids, out-of-range counts, and unallocated blocks, so a
        # trusted caller pays zero per-selection launches.
        if selected_has_count:
            count = selected[..., -1]
            slots = torch.arange(2051, device=selected.device)
            tokens = selected[..., :-1]
            invalid = (count < 0) | (count > 2051) | ((tokens >= 0).sum(-1) != count)
            invalid = (
                invalid.any() | ((tokens >= 0) != (slots < count.unsqueeze(-1))).any()
            )
            if qsa_strict_validation_enabled():
                if bool(invalid.item()):
                    raise ValueError("invalid packed QSA prefix/count")
            else:
                qsa_assert_valid(~invalid, "invalid packed QSA prefix/count")
        paged_selected = selected[..., :-1] if selected_has_count else selected
        if qsa_strict_validation_enabled():
            _validate_sparse_paged_indices(
                block_table,
                kv_lens,
                paged_selected,
                page_size=page_size,
                cache_blocks=cache_blocks,
            )
        else:
            # FMHA-entry validation (the kept half of the pair with the
            # per-request planner checks): one enqueue-only device verdict.
            qsa_assert_valid(
                _sparse_paged_indices_verdict(
                    block_table,
                    kv_lens,
                    paged_selected,
                    page_size=page_size,
                    cache_blocks=cache_blocks,
                ),
                "qwen4_exp sparse paged GQA selection, kv_lens or block table "
                "is invalid",
            )

    if q.stride(-1) != 1:
        # The kernels index the head dim contiguously; every other layout
        # mismatch is carried by the runtime stride arguments, so a strided
        # BTHD query view flows through without a copy.
        q = q.contiguous()
    block_table = block_table.contiguous()
    kv_lens = kv_lens.contiguous()
    selected = selected.contiguous()
    use_block_ids = (
        block_selection is not None
        and not selected_has_count
        and int(selected.shape[2]) == 512
        and page_size % 4 == 0
        and block_selection.scores.is_cuda
        and block_selection.scores.device == q.device
        and block_selection.scores.dtype == torch.float32
        and block_selection.scores.dim() == 2
        and int(block_selection.scores.shape[0]) == batch * query_len
        and block_selection.scores.stride(1) == 1
    )
    if batch == 0 or query_len == 0:
        return torch.empty(
            (
                (batch, query_len, q_heads, head_dim)
                if bthd_output
                else (batch, q_heads, query_len, head_dim)
            ),
            device=q.device,
            dtype=q.dtype,
        )

    selected_width = (
        2051 if use_block_ids else (int(selected.shape[2]) - int(selected_has_count))
    )
    if selected_width == 0:
        output = torch.zeros(
            (
                (batch, query_len, q_heads, head_dim)
                if bthd_output
                else (batch, q_heads, query_len, head_dim)
            ),
            device=q.device,
            dtype=q.dtype,
        )
        if output_gate is not None:
            gate = output_gate if bthd_output else output_gate.transpose(1, 2)
            output = output * torch.sigmoid(gate)
        return output
    # SM90 single-token decode needs the online kernel's FP32 PV weights.
    # Explicit BTHD output is the grouped prefill layout.
    grouped_ok = (query_len > 1 or bthd_output) and _grouped_split_supported(
        q, kv_cache, selected, page_size=page_size, selected_width=selected_width
    )
    if bthd_output and not grouped_ok:
        raise ValueError(
            "bthd_output requires the SM90 tensor-core grouped split geometry"
        )
    if grouped_ok:
        kv_heads = int(kv_cache.shape[2])
        group = q_heads // kv_heads
        if _h20_direct_decode:
            # Balance smaller batches with finer splits across H20 SMs.
            block_n, warps = 64, 2
            stages, splits = (1, 32) if batch <= 16 else (2, 16)
        else:
            block_n, warps, stages, splits = _grouped_split_config(
                batch * query_len, kv_heads, selected_width
            )
        fp8_req = _qsa_fp8_output_request.get()
        use_fp8_merge = (
            fp8_req is not None
            and fp8_req.result is None
            and splits > 1
            and output_gate is not None
            and bthd_output
            and head_dim == 256
        )
        output = (
            None
            if use_fp8_merge
            else torch.empty(
                (
                    (batch, query_len, q_heads, head_dim)
                    if bthd_output
                    else (batch, q_heads, query_len, head_dim)
                ),
                device=q.device,
                dtype=q.dtype,
            )
        )
        partial = (
            torch.empty(
                (batch, q_heads, query_len, splits, head_dim + 2),
                dtype=torch.float32,
                device=q.device,
            )
            if splits > 1
            else output
        )
        pdl = sm90_pdl_enabled(q.device)
        scores_arg = block_selection.scores if use_block_ids else selected
        scores_row_stride = (
            block_selection.scores.stride(0) if use_block_ids else 0
        )
        scores_width = int(block_selection.scores.shape[1]) if use_block_ids else 0
        _sparse_paged_grouped_split_kernel[(batch * query_len * splits, kv_heads)](
            q,
            kv_cache,
            scale_pointer,
            block_table,
            kv_lens,
            selected,
            partial,
            output_gate if output_gate is not None else partial,
            *q.stride()[:3],
            *kv_cache.stride()[:4],
            *scale_strides,
            *block_table.stride(),
            *kv_lens.stride(),
            *selected.stride()[:2],
            scores_arg,
            scores_row_stride,
            scores_width,
            selected_width,
            int(block_table.shape[1]),
            cache_blocks,
            HAS_COUNT=selected_has_count,
            FP8_CACHE=scale_view is not None,
            H_Q=q_heads,
            GROUP_SIZE=group,
            QUERY_LEN=query_len,
            PAGE_SIZE=page_size,
            D=head_dim,
            SCALE=1.0 / float(head_dim) ** 0.5,
            NUM_SPLITS=splits,
            NUM_TILES=triton.cdiv(selected_width, block_n),
            BLOCK_M=max(16, triton.next_power_of_2(group)),
            BLOCK_N=block_n,
            WITH_GATE=output_gate is not None,
            OUT_BTHD=bthd_output,
            PV_BF16_RESIDUAL=_h20_direct_decode,
            BLOCK_IDS=use_block_ids,
            FINITE_ONLY=(
                bool(block_selection.finite_only) if use_block_ids else True
            ),
            num_warps=warps,
            num_stages=stages,
        )
        if use_fp8_merge:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                create_per_token_group_quant_fp8_output_scale,
            )

            tokens = batch * query_len
            q_fp8 = torch.empty(
                (tokens, q_heads * head_dim),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            s_col = create_per_token_group_quant_fp8_output_scale(
                (tokens, q_heads * head_dim), q.device, 128, True, True, False
            )
            _merge_sparse_paged_splits_fp8_kernel[(batch * q_heads * query_len,)](
                partial,
                q_fp8,
                s_col,
                output_gate,
                s_col.stride(0),
                s_col.stride(1),
                D=head_dim,
                NUM_SPLITS=splits,
                HEADS=q_heads,
                QUERIES=query_len,
                V2=fp8_req.v2,
                PDL=pdl,
                enable_fp_fusion=False,
                launch_pdl=pdl,
            )
            fp8_req.result = (q_fp8, s_col)
            return q_fp8.view(batch, query_len, q_heads, head_dim)
        if splits > 1:
            if not bthd_output and output_gate is not None:
                output = torch.empty(
                    (batch, query_len, q_heads, head_dim),
                    device=q.device,
                    dtype=q.dtype,
                ).transpose(1, 2)
            _merge_sparse_paged_splits_kernel[(batch * q_heads * query_len,)](
                partial,
                output,
                output_gate if output_gate is not None else output,
                D=head_dim,
                NUM_SPLITS=splits,
                HEADS=q_heads,
                QUERIES=query_len,
                WITH_GATE=output_gate is not None,
                OUT_BTHD=bthd_output,
                PDL=pdl,
                launch_pdl=pdl,
            )
        return output
    if use_block_ids:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
            _expand_tokens,
        )

        flat_len = kv_lens.reshape(-1)
        expanded = torch.empty(
            (batch * query_len, 2051), device=q.device, dtype=torch.int32
        )
        _expand_tokens[(batch * query_len,)](
            block_selection.scores,
            selected.reshape(batch * query_len, 512),
            flat_len,
            expanded,
            K=512,
            LENGTH_STRIDE=flat_len.stride(0),
            FINITE_ONLY=bool(block_selection.finite_only),
            SCORE_WIDTH=int(block_selection.scores.shape[1]),
            SCORE_ROW_STRIDE=block_selection.scores.stride(0),
            num_warps=4,
        )
        selected = expanded.view(batch, query_len, 2051)
    output = torch.empty(
        (batch, q_heads, query_len, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    splits = min(8, triton.next_power_of_2(triton.cdiv(selected_width, block_k)))
    from .grouped_sparse_paged_fmha import preferred_splits

    tuned_splits = preferred_splits(
        q, kv_cache, selected, page_size=page_size, block_k=block_k
    )
    if tuned_splits is not None:
        splits = tuned_splits

    partial = (
        torch.empty(
            (batch, q_heads, query_len, splits, head_dim + 2),
            dtype=torch.float32,
            device=q.device,
        )
        if splits > 1
        else output
    )
    grouped = False
    from rtp_llm.models_py.modules.qwen4_exp.qsa.grouped_sparse_paged_fmha import (
        grouped_sparse_paged_split,
        is_supported,
    )

    grouped = is_supported(
        q,
        kv_cache,
        block_table,
        kv_lens,
        selected,
        partial,
        page_size=page_size,
        splits=splits,
        block_k=block_k,
        scale=scale_view,
        selected_has_count=selected_has_count,
    )
    if grouped:
        grouped_sparse_paged_split(
            q,
            kv_cache,
            block_table,
            kv_lens,
            selected,
            partial,
            page_size=page_size,
            splits=splits,
            scale=scale_view,
            selected_has_count=selected_has_count,
            fp16_pv=fp16_pv,
        )
    if not grouped:
        # CUDA grid Y/Z are limited to 65535. Keep all query partitions on X.
        _sparse_paged_online_split_kernel[(batch * query_len * splits, q_heads)](
            q,
            kv_cache,
            scale_pointer,
            block_table,
            kv_lens,
            selected,
            partial,
            *q.stride()[:3],
            *kv_cache.stride()[:4],
            *scale_strides,
            *block_table.stride(),
            *kv_lens.stride(),
            *selected.stride()[:2],
            selected_width,
            int(block_table.shape[1]),
            cache_blocks,
            HAS_COUNT=selected_has_count,
            FP8_CACHE=scale_view is not None,
            H_Q=q_heads,
            H_KV=kv_heads,
            QUERY_LEN=query_len,
            PAGE_SIZE=page_size,
            D=head_dim,
            SCALE=1.0 / float(head_dim) ** 0.5,
            BLOCK_K=block_k,
            NUM_SPLITS=splits,
            BLOCKS_PER_SPLIT=triton.cdiv(selected_width, block_k * splits),
        )
    if splits > 1:
        if output_gate is not None:
            # The owner consumes [B,S,H,D]. Store that layout directly, while
            # retaining the public attention result's logical [B,H,S,D] shape.
            output = torch.empty(
                (batch, query_len, q_heads, head_dim), device=q.device, dtype=q.dtype
            ).transpose(1, 2)
        _merge_sparse_paged_splits_kernel[(batch * q_heads * query_len,)](
            partial,
            output,
            output_gate if output_gate is not None else output,
            D=head_dim,
            NUM_SPLITS=splits,
            HEADS=q_heads,
            QUERIES=query_len,
            WITH_GATE=output_gate is not None,
        )
    elif output_gate is not None:
        output = output * torch.sigmoid(output_gate.transpose(1, 2))
    return output

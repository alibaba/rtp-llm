"""Official paged Sparse MQA for V4.1 decode reindex layers.

The plan is shared by reindex layers for one forward, then rebuilt for the next
step. DeepGEMM requires sorted candidate blocks and uses BF16 head reduction and
BF16 logits, matching the current dense paged scorer's precision boundary.
"""

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

try:
    import deep_gemm
except ImportError:
    deep_gemm = None

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_deepselect as deepselect
from rtp_llm.models_py.modules.dsv41.fp8._v41_decode_indexer import (
    prepare_indexer_q_and_weights,
)
from rtp_llm.models_py.modules.dsv41.fp8.indexer import _get_topk_workspace
from rtp_llm.ops.compute_ops import rtp_llm_ops

PAGE_SIZE = 128
BLOCK_SIZE = 8
NUM_HEADS = 32
HEAD_DIM = 128
TOKEN_TOPK = 512


def is_supported(candidates):
    """Unsupported backends/layouts keep dense scoring; kernel errors propagate."""
    if (
        not candidates.is_cuda
        or candidates.ndim != 2
        or candidates.dtype != torch.int32
        or candidates.shape[1] not in (512, 1024, 2048)
        or not candidates.is_contiguous()
    ):
        return False
    return (
        callable(getattr(deep_gemm, "get_paged_sparse_mqa_logits_metadata", None))
        and callable(getattr(deep_gemm, "fp8_fp4_paged_sparse_mqa_logits", None))
        and callable(getattr(rtp_llm_ops, "topk_v3", None))
        and torch.cuda.get_device_capability(candidates.device)[0] == 10
    )


class SparseDecodePlan(NamedTuple):
    blocks: torch.Tensor
    active: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    metadata: torch.Tensor


@triton.jit
def _prepare_plan_kernel(
    candidates,
    lengths,
    table,
    sorted_blocks,
    active_blocks,
    row_table,
    requests,
    QUERY_LEN: tl.constexpr,
    K: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    SORT_I16: tl.constexpr = False,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    values = tl.load(candidates + row * K + lanes)
    if SORT_I16:
        # Logical block IDs fit exactly; reserve INT16_MAX for padding only.
        values = tl.sort(
            tl.where(values >= 0, values, 32767).to(tl.int16), descending=False
        ).to(tl.int32)
        values = tl.where(values == 32767, 2147483647, values)
    else:
        values = tl.sort(tl.where(values >= 0, values, 2147483647), descending=False)
    length = tl.load(lengths + row)
    expected = tl.minimum(tl.cdiv(length, BLOCK_SIZE), K)
    count = tl.sum((values != 2147483647).to(tl.int32), 0)
    active = lanes < count
    if count < expected:
        # DG infers its prefix size from context length, not candidate count.
        # Fill holes with distinct unused blocks from [0, expected), then
        # exclude these fillers before TopK. Normal complete rows skip this.
        lo = tl.full((K,), 0, tl.int32)
        hi = tl.full((K,), K, tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            value = tl.gather(values, tl.minimum(mid, K - 1), 0)
            right = (mid < K) & (value < lanes)
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        present = tl.gather(values, tl.minimum(lo, K - 1), 0) == lanes
        fillers = tl.sort(tl.where((lanes < expected) & ~present, lanes, 2147483647))
        filler = tl.gather(fillers, tl.maximum(lanes - count, 0), 0)
        # Encode the validity bit so sorting keeps it attached to its block.
        blocks = tl.where(active, values, filler)
        tagged = tl.sort(
            tl.where(
                lanes < expected,
                blocks.to(tl.int64) * 2 + ~active,
                9223372036854775807,
            )
        )
        values = (tagged // 2).to(tl.int32)
        active = (lanes < expected) & ((tagged & 1) == 0)
    last = tl.maximum(tl.cdiv(length, BLOCK_SIZE) - 1, 0)
    tl.store(sorted_blocks + row * K + lanes, tl.where(lanes < expected, values, last))
    tl.store(active_blocks + row * K + lanes, active & (lanes < expected))
    columns = tl.arange(0, COL_BLOCK)
    request = row // QUERY_LEN
    physical = tl.load(
        table + request * TABLE_STRIDE + columns, columns < COLS, other=0
    )
    tl.store(row_table + row * COLS + columns, tl.maximum(physical, 0), columns < COLS)
    tl.store(requests + row, request)


@triton.jit
def _prepare_flags_plan_kernel(
    flags, lengths, table, sorted_blocks, active_blocks, row_table, requests,
    QUERY_LEN: tl.constexpr, K: tl.constexpr, COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, COL_BLOCK: tl.constexpr,
    DOMAIN: tl.constexpr, DOMAIN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
):
    row = tl.program_id(0)
    ids = tl.arange(0, DOMAIN_BLOCK)
    present = tl.load(flags + row * DOMAIN + ids, ids < DOMAIN, other=0) != 0
    length = tl.load(lengths + row)
    expected = tl.minimum(tl.cdiv(length, BLOCK_SIZE), K)
    count = tl.sum(present.to(tl.int32), 0)
    selected = present
    if count < expected:
        # Same fillers as the sorter: smallest unused IDs in [0, expected).
        missing = (ids < expected) & ~present
        missing_rank = tl.cumsum(missing.to(tl.int32), 0)
        selected = present | (missing & (missing_rank <= expected - count))
    slot = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(sorted_blocks + row * K + slot, ids, selected)
    tl.store(active_blocks + row * K + slot, present, selected)
    lanes = tl.arange(0, K)
    last = tl.maximum(tl.cdiv(length, BLOCK_SIZE) - 1, 0)
    tl.store(sorted_blocks + row * K + lanes, last, lanes >= expected)
    tl.store(active_blocks + row * K + lanes, False, lanes >= expected)
    columns = tl.arange(0, COL_BLOCK)
    request = row // QUERY_LEN
    physical = tl.load(
        table + request * TABLE_STRIDE + columns, columns < COLS, other=0
    )
    tl.store(row_table + row * COLS + columns, tl.maximum(physical, 0), columns < COLS)
    tl.store(requests + row, request)


def _plan_candidate_flags(candidates, block_table, candidate_mask):
    """Reuse only the producer's mask bound to this forward's candidates.

    The producer must update candidates and flags together. Identity/shape
    checks do not inspect GPU contents or synchronize during graph capture.
    Larger domains retain sorting until their scan performance is validated.
    """
    if (
        candidates.shape[1] != 2048
        or candidate_mask is None
        or candidate_mask[0] is not candidates
        or candidate_mask[2] != BLOCK_SIZE
    ):
        return None
    flags = candidate_mask[1]
    if (
        flags.ndim != 2
        or flags.shape[0] != candidates.shape[0]
        or not 0 < flags.shape[1] <= min(
            4224, block_table.shape[1] * (PAGE_SIZE // BLOCK_SIZE)
        )
        or flags.dtype != torch.uint8
        or flags.device != candidates.device
        or not flags.is_contiguous()
    ):
        return None
    return flags


def make_plan(candidates, lengths, block_table, *, query_len, candidate_mask=None):
    """Build once per forward from unique block IDs and -1 padding.

    NaN/-inf source scores may leave fewer candidates than context implies.
    Metadata fillers remain distinct and are never eligible for token TopK.
    Queries of the same request are consecutive; block_table is request-level.
    Warm up on the actual capture stream: DG's scheduler workspace is per-stream.

    Candidate IDs must lie within the logical capacity of block_table. For
    K=2048, sort in int16 when that capacity leaves INT16_MAX free as padding;
    all downstream arithmetic and outputs remain int32. Dispatch uses shapes
    only (no device synchronization), so it is safe during CUDA Graph capture.
    DSV41_SPARSE_PLAN_I16=0 retains the original sorter for controlled A/B tests.
    When the producer supplies matching candidate flags (K=2048, <=4224
    logical blocks), compact those flags instead of sorting the IDs again.
    """
    rows, k = candidates.shape
    if k not in (512, 1024, 2048) or candidates.dtype != torch.int32:
        raise ValueError("Sparse decode requires int32 candidate TopK 512/1024/2048")
    if rows != block_table.shape[0] * query_len:
        raise ValueError("Sparse query rows must match request count * query_len")
    lengths = lengths.reshape(rows).to(torch.int32).contiguous()
    blocks = torch.empty_like(candidates)
    active = torch.empty_like(candidates, dtype=torch.bool)
    table = torch.empty(
        (rows, block_table.shape[1]), device=candidates.device, dtype=torch.int32
    )
    requests = torch.empty(rows, device=candidates.device, dtype=torch.int32)
    flags = _plan_candidate_flags(candidates, block_table, candidate_mask)
    common = (
        candidates,
        lengths,
        block_table,
        blocks,
        active,
        table,
        requests,
        query_len,
        k,
        table.shape[1],
        block_table.stride(0),
        triton.next_power_of_2(table.shape[1]),
    )
    if flags is not None:
        _prepare_flags_plan_kernel[(rows,)](
            flags, *common[1:], flags.shape[1],
            triton.next_power_of_2(flags.shape[1]),
            num_warps=4 if flags.shape[1] <= 4096 else 8,
        )
    else:
        _prepare_plan_kernel[(rows,)](
            *common, k.bit_length(), num_warps=8,
            SORT_I16=(
                k == 2048
                and block_table.shape[1] * (PAGE_SIZE // BLOCK_SIZE) <= 32767
                and os.environ.get("DSV41_SPARSE_PLAN_I16", "1") == "1"
            ),
        )
    metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
        lengths,
        table,
        requests,
        PAGE_SIZE,
        blocks,
        torch.int8,
        BLOCK_SIZE,
    )
    return SparseDecodePlan(blocks, active, table, lengths, metadata)


@triton.jit
def _clean_logits_kernel(
    raw,
    blocks,
    active,
    table,
    lengths,
    output,
    select_lengths,
    K: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    PAGE_SIZE: tl.constexpr = PAGE_SIZE,
):
    row = tl.program_id(0)
    slot = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block_slot = slot // BLOCK_SIZE
    logical_block = tl.load(blocks + row * K + block_slot, block_slot < K, other=0)
    selected = tl.load(active + row * K + block_slot, block_slot < K, other=False)
    token = logical_block * BLOCK_SIZE + slot % BLOCK_SIZE
    length = tl.load(lengths + row)
    valid = (slot < K * BLOCK_SIZE) & selected & (token < length)
    page = tl.load(table + row * TABLE_COLS + token // PAGE_SIZE, valid, other=0)
    # Never read DG's unwritten padding. Restore RTP's zero-K semantics for
    # unallocated pages, independently of bytes stored in the page-0 sentinel.
    score = tl.load(raw + row * RAW_STRIDE + slot, valid & (page > 0), other=0).to(
        tl.float32
    )
    tl.store(
        output + row * (K * BLOCK_SIZE) + slot,
        tl.where(valid, score, -float("inf")),
        slot < K * BLOCK_SIZE,
    )
    if tl.program_id(1) == 0:
        # Short contexts contain every historical block, including fillers;
        # long contexts scan all K blocks. Holes stay -inf in either case.
        tl.store(select_lengths + row, tl.minimum(length, K * BLOCK_SIZE))


@triton.jit
def _restore_indices_kernel(
    output,
    logits,
    blocks,
    lengths,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    TOKEN_TOPK: tl.constexpr = TOKEN_TOPK,
):
    row = tl.program_id(0)
    lane = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(output + row * TOKEN_TOPK + lane, lane < TOKEN_TOPK, other=-1)
    valid = (lane < TOKEN_TOPK) & (slot >= 0) & (slot < K * BLOCK_SIZE)
    score = tl.load(logits + row * (K * BLOCK_SIZE) + slot, valid, other=-float("inf"))
    block = tl.load(blocks + row * K + slot // BLOCK_SIZE, valid, other=0)
    token = block * BLOCK_SIZE + slot % BLOCK_SIZE
    keep = valid & (token < tl.load(lengths + row)) & (tl.abs(score) < float("inf"))
    tl.store(
        output + row * TOKEN_TOPK + lane, tl.where(keep, token, -1), lane < TOKEN_TOPK
    )


def select_tokens(logits, plan):
    """Select compact sparse logits and return logical compressed-token IDs."""
    rows, k = plan.blocks.shape
    width = k * BLOCK_SIZE
    # Both native selectors accept BF16. Preserve DG's score precision instead
    # of expanding the compact logits to FP32 before scanning them again.
    scores = torch.empty((rows, width), device=logits.device, dtype=logits.dtype)
    lengths = torch.empty_like(plan.lengths)
    tile = 2048
    _clean_logits_kernel[(rows, triton.cdiv(width, tile))](
        logits,
        plan.blocks,
        plan.active,
        plan.table,
        plan.lengths,
        scores,
        lengths,
        k,
        logits.stride(0),
        plan.table.shape[1],
        tile,
    )
    output = torch.empty((rows, TOKEN_TOPK), device=logits.device, dtype=torch.int32)
    selected = deepselect.try_select_tokens(
        scores,
        lengths,
        out=output,
        filter_finite=False,
        enabled=os.environ.get("DSV41_DECODE_DEEPSELECT", "1") == "1",
    )
    if selected is None:
        rtp_llm_ops.topk_v3(
            scores,
            lengths,
            output,
            _get_topk_workspace(logits.device),
            TOKEN_TOPK,
            width,
        )
    # Both selectors share finite filtering and logical-index restoration.
    _restore_indices_kernel[(rows, triton.cdiv(TOKEN_TOPK, 256))](
        output,
        scores,
        plan.blocks,
        plan.lengths,
        k,
        256,
    )
    return output


def select_decode(
    q, weights, freqs_cis, positions, pool, plan, *, rope_head_dim, prepared_query=None
):
    """Run fused Q preparation, official sparse score, and logical token TopK."""
    if prepared_query is None:
        prepared_query = prepare_indexer_q_and_weights(
            q,
            weights,
            freqs_cis,
            positions,
            rope_head_dim,
            weights_out_dtype=torch.bfloat16,
        )
    payload, scales, weights = prepared_query
    rows, k = plan.blocks.shape
    logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (
            payload.reshape(rows, 1, NUM_HEADS, HEAD_DIM // 2),
            scales.reshape(rows, 1, NUM_HEADS),
        ),
        pool.unsqueeze(2),
        weights.reshape(rows, NUM_HEADS),
        plan.metadata,
        k,
        BLOCK_SIZE,
    )
    return select_tokens(logits, plan)

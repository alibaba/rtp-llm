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
    compact_lengths: torch.Tensor | None = None
    repair_rows: torch.Tensor | None = None


@triton.jit
def _prepare_plan_kernel(
    candidates,
    lengths,
    table,
    sorted_blocks,
    active_blocks,
    row_table,
    requests,
    compact_lengths,
    repair_rows,
    QUERY_LEN: tl.constexpr,
    K: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    SORT_I16: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = PAGE_SIZE,
    valid_requests=None,
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
    planned = tl.where(lanes < expected, values, last)
    tl.store(sorted_blocks + row * K + lanes, planned)
    tl.store(active_blocks + row * K + lanes, active & (lanes < expected))
    columns = tl.arange(0, COL_BLOCK)
    request = row // QUERY_LEN
    physical = tl.load(
        table + request * TABLE_STRIDE + columns, columns < COLS, other=0
    )
    tl.store(row_table + row * COLS + columns, tl.maximum(physical, 0), columns < COLS)
    tl.store(requests + row, request)
    # Sorted blocks put the only possible partial block at the prefix's end.
    # Include metadata fillers in the span: interior fillers are repaired, not
    # removed by shortening this bound (which would discard later real blocks).
    token_counts = tl.minimum(tl.maximum(length - planned * BLOCK_SIZE, 0), BLOCK_SIZE)
    compact = tl.sum(tl.where(lanes < expected, token_counts, 0), 0)
    if valid_requests is not None:
        compact = tl.where(tl.load(valid_requests + request), compact, 0)
    tl.store(compact_lengths + row, compact)
    pages = tl.load(
        table + request * TABLE_STRIDE + planned // (PAGE_SIZE // BLOCK_SIZE),
        lanes < expected, other=0,
    )
    exceptional = (lanes < expected) & (~active | (pages <= 0))
    tl.store(repair_rows + row, (compact > 0) & (tl.sum(exceptional.to(tl.int32), 0) != 0))


@triton.jit
def _prepare_flags_plan_kernel(
    flags, lengths, table, sorted_blocks, active_blocks, row_table, requests,
    compact_lengths, repair_rows,
    QUERY_LEN: tl.constexpr, K: tl.constexpr, COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, COL_BLOCK: tl.constexpr,
    DOMAIN: tl.constexpr, DOMAIN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    PAGE_SIZE: tl.constexpr = PAGE_SIZE,
    valid_requests=None,
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
    token_counts = tl.minimum(tl.maximum(length - ids * BLOCK_SIZE, 0), BLOCK_SIZE)
    compact = tl.sum(tl.where(selected, token_counts, 0), 0)
    if valid_requests is not None:
        compact = tl.where(tl.load(valid_requests + request), compact, 0)
    tl.store(compact_lengths + row, compact)
    pages = tl.load(
        table + request * TABLE_STRIDE + ids // (PAGE_SIZE // BLOCK_SIZE),
        selected, other=0,
    )
    exceptional = selected & (~present | (pages <= 0))
    tl.store(repair_rows + row, (compact > 0) & (tl.sum(exceptional.to(tl.int32), 0) != 0))


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
        flags is None
        or flags.ndim != 2
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


def _has_complete_prefix(candidates, candidate_mask):
    """CPU provenance only: the current producer owns this sorted prefix."""
    return (
        isinstance(candidate_mask, tuple)
        and len(candidate_mask) == 4
        and candidate_mask[0] is candidates
        and candidate_mask[2] == BLOCK_SIZE
        and candidate_mask[3] == "sorted_prefix"
    )


@triton.jit
def _prepare_prefix_plan_kernel(
    candidates, lengths, table, sorted_blocks, active_blocks, row_table, requests,
    compact_lengths, repair_rows,
    QUERY_LEN: tl.constexpr, K: tl.constexpr, COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, COL_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE, PAGE_SIZE: tl.constexpr = PAGE_SIZE,
    valid_requests=None,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    length = tl.load(lengths + row)
    expected = tl.minimum(tl.maximum(tl.cdiv(length, BLOCK_SIZE), 0), K)
    values = tl.load(candidates + row * K + lanes, lanes < expected, other=0)
    last = tl.maximum(tl.cdiv(length, BLOCK_SIZE) - 1, 0)
    # The source guarantees distinct sorted IDs, with only trailing -1. There
    # is no sorting, membership scan or inactive filler insertion here.
    tl.store(sorted_blocks + row * K + lanes, tl.where(lanes < expected, values, last))
    tl.store(active_blocks + row * K + lanes, lanes < expected)
    columns = tl.arange(0, COL_BLOCK)
    request = row // QUERY_LEN
    physical = tl.load(table + request * TABLE_STRIDE + columns, columns < COLS, other=0)
    tl.store(row_table + row * COLS + columns, tl.maximum(physical, 0), columns < COLS)
    tl.store(requests + row, request)
    token_counts = tl.minimum(tl.maximum(length - values * BLOCK_SIZE, 0), BLOCK_SIZE)
    compact = tl.sum(tl.where(lanes < expected, token_counts, 0), 0)
    if valid_requests is not None:
        compact = tl.where(tl.load(valid_requests + request), compact, 0)
    tl.store(compact_lengths + row, compact)
    if repair_rows is not None:
        pages = tl.load(
            table + request * TABLE_STRIDE + values // (PAGE_SIZE // BLOCK_SIZE),
            lanes < expected, other=0,
        )
        tl.store(repair_rows + row, tl.sum(((lanes < expected) & (pages <= 0)).to(tl.int32), 0) != 0)


def make_plan(candidates, lengths, block_table, *, query_len, candidate_mask=None, valid_requests=None):
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
    Compact scan lengths and compatibility-row flags are produced in that same
    launch and rebuilt on every graph replay; no GPU predicate is read on CPU.
    The current L20 producer's sorted-prefix contract bypasses both sort/flags
    compaction and filler insertion. Unmarked/manual inputs keep the old path.
    ``valid_requests`` is the engine's replay-updated bool request mask, not a
    page-table heuristic. All paths give dummy rows compact end 0. Only with
    a complete source prefix may real requests be assumed to have allocated
    positive pages and score repair omitted. Context lengths/metadata stay unchanged.
    """
    rows, k = candidates.shape
    if k not in (512, 1024, 2048) or candidates.dtype != torch.int32:
        raise ValueError("Sparse decode requires int32 candidate TopK 512/1024/2048")
    if rows != block_table.shape[0] * query_len:
        raise ValueError("Sparse query rows must match request count * query_len")
    complete_prefix = _has_complete_prefix(candidates, candidate_mask)
    if valid_requests is not None and not (
        valid_requests.dtype == torch.bool
        and valid_requests.device == candidates.device
        and valid_requests.shape == (block_table.shape[0],)
        and valid_requests.is_contiguous()
    ):
        raise ValueError("Sparse request validity must be contiguous bool [requests] on the query device")
    lengths = lengths.reshape(rows).to(torch.int32).contiguous()
    blocks = torch.empty_like(candidates)
    active = torch.empty_like(candidates, dtype=torch.bool)
    table = torch.empty(
        (rows, block_table.shape[1]), device=candidates.device, dtype=torch.int32
    )
    requests = torch.empty(rows, device=candidates.device, dtype=torch.int32)
    compact_lengths = torch.empty_like(lengths)
    repair_rows = (
        None if complete_prefix and valid_requests is not None
        else torch.empty(rows, device=candidates.device, dtype=torch.bool)
    )
    flags = None if complete_prefix else _plan_candidate_flags(candidates, block_table, candidate_mask)
    common = (
        candidates,
        lengths,
        block_table,
        blocks,
        active,
        table,
        requests,
        compact_lengths,
        repair_rows,
        query_len,
        k,
        table.shape[1],
        block_table.stride(0),
        triton.next_power_of_2(table.shape[1]),
    )
    if complete_prefix:
        _prepare_prefix_plan_kernel[(rows,)](
            *common, valid_requests=valid_requests, num_warps=4,
        )
    elif flags is not None:
        _prepare_flags_plan_kernel[(rows,)](
            flags, *common[1:], flags.shape[1],
            triton.next_power_of_2(flags.shape[1]),
            valid_requests=valid_requests,
            num_warps=4 if flags.shape[1] <= 4096 else 8,
        )
    else:
        _prepare_plan_kernel[(rows,)](
            *common, k.bit_length(), num_warps=8,
            valid_requests=valid_requests,
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
    return SparseDecodePlan(
        blocks, active, table, lengths, metadata, compact_lengths, repair_rows
    )


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
    if select_lengths is not None:
        if tl.program_id(1) == 0:
            # Only legacy plans need this bound. Never overwrite a current
            # plan's compact lengths, especially its zero-ended dummy rows.
            tl.store(select_lengths + row, tl.minimum(length, K * BLOCK_SIZE))


@triton.jit
def _repair_compat_logits_kernel(
    raw, blocks, active, table, compact_lengths, repair_rows,
    K: tl.constexpr, RAW_STRIDE: tl.constexpr, TABLE_COLS: tl.constexpr,
    BLOCK_TILE: tl.constexpr = 128,
    BLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    PAGE_SIZE: tl.constexpr = PAGE_SIZE,
):
    """No score reads/copies: only exceptional rows write filler/page-0 slots."""
    row = tl.program_id(0)
    if tl.load(repair_rows + row):
        end = tl.load(compact_lengths + row)
        offsets = tl.arange(0, BLOCK_SIZE)
        for group in range(tl.cdiv(K, BLOCK_TILE)):
            block_slot = group * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
            in_prefix = (block_slot < K) & (block_slot * BLOCK_SIZE < end)
            selected = tl.load(active + row * K + block_slot, in_prefix, other=False)
            block = tl.load(blocks + row * K + block_slot, in_prefix, other=0)
            page = tl.load(
                table + row * TABLE_COLS + block // (PAGE_SIZE // BLOCK_SIZE),
                in_prefix & selected, other=0,
            )
            slot = block_slot[:, None] * BLOCK_SIZE + offsets[None, :]
            repair = in_prefix & (~selected | (page <= 0))
            # A selected unallocated page is zero-K, not an excluded candidate.
            value = tl.where(selected, 0.0, -float("inf"))
            tl.store(
                raw + row * RAW_STRIDE + slot,
                value[:, None], repair[:, None] & (slot < end),
            )


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
    SCORE_STRIDE: tl.constexpr = 0,
    select_lengths=None,
):
    row = tl.program_id(0)
    lane = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(output + row * TOKEN_TOPK + lane, lane < TOKEN_TOPK, other=-1)
    valid = (lane < TOKEN_TOPK) & (slot >= 0) & (slot < K * BLOCK_SIZE)
    if select_lengths is not None:
        valid = valid & (slot < tl.load(select_lengths + row))
    stride = SCORE_STRIDE if SCORE_STRIDE else K * BLOCK_SIZE
    block = tl.load(blocks + row * K + slot // BLOCK_SIZE, valid, other=0)
    token = block * BLOCK_SIZE + slot % BLOCK_SIZE
    score = tl.load(logits + row * stride + slot, valid, other=-float("inf"))
    keep = valid & (token < tl.load(lengths + row)) & (tl.abs(score) < float("inf"))
    tl.store(
        output + row * TOKEN_TOPK + lane, tl.where(keep, token, -1), lane < TOKEN_TOPK
    )


def select_tokens(logits, plan, *, inplace=False):
    """Select logical token IDs. ``inplace`` requires disposable scorer output.

    Complete engine plans skip clean/repair entirely: dummy compact ends are 0.
    Untrusted/manual plans keep filler/page-0 compatibility in disposable logits.
    The standard DeepSelect entry is unchanged.
    Callers retaining raw logits keep the original copy-based path by default.
    """
    rows, k = plan.blocks.shape
    width = k * BLOCK_SIZE
    output = torch.empty((rows, TOKEN_TOPK), device=logits.device, dtype=torch.int32)
    enabled = os.environ.get("DSV41_DECODE_DEEPSELECT", "1") == "1"
    if inplace and plan.compact_lengths is not None:
        scores, lengths = logits, plan.compact_lengths
        if plan.repair_rows is not None:
            _repair_compat_logits_kernel[(rows,)](
                scores, plan.blocks, plan.active, plan.table, lengths, plan.repair_rows,
                k, scores.stride(0), plan.table.shape[1], num_warps=4,
            )
    else:
        # Keep older/manual plans and borrowed logits non-mutating.
        scores = torch.empty((rows, width), device=logits.device, dtype=logits.dtype)
        legacy_lengths = (
            torch.empty_like(plan.lengths) if plan.compact_lengths is None else None
        )
        tile = 2048
        _clean_logits_kernel[(rows, triton.cdiv(width, tile))](
            logits, plan.blocks, plan.active, plan.table, plan.lengths,
            scores, legacy_lengths, k, logits.stride(0), plan.table.shape[1], tile,
        )
        lengths = plan.compact_lengths if plan.compact_lengths is not None else legacy_lengths
    selected = deepselect.try_select_tokens(
        scores, lengths, out=output, filter_finite=False, enabled=enabled,
        sorted_index=(
            scores.dtype == torch.bfloat16
            and os.environ.get("DSV41_DECODE_DEEPSELECT_SORTED", "1") == "1"
        ),
    )
    if selected is None:
        # topk_v3 also consumes scores.stride(0); preserve the scorer storage.
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
        SCORE_STRIDE=scores.stride(0),
        select_lengths=lengths,
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
    return select_tokens(logits, plan, inplace=True)

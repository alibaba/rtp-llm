# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only FP4 prefill scoring with request-local sparse columns.

Adapted from vLLM's model_executor/kernels/attention/dsa/sparse_mqa_logits.py.
RTP has already restored request-local logical K order in PrefillIndexerKeys,
so every row starts at zero; prefix tokens are included in ``visible``.
Candidate sorting additionally removes duplicates to satisfy the installed
DeepGEMM metadata contract. Valid slots are an ascending prefix, followed by
the last valid block, as in vLLM. The metadata builder receives the compact
``end``, so only the unique prefix participates in its merge-path schedule.
The independent logical ``row_ke`` retains causal bounds for remapping.

The sparse scorer performs BF16 intermediate head reduction, unlike the old
FP32 dense scorer. Its result must be validated against that BF16 contract;
casting dense FP32 logits at the end is not an exact reference.

Plans are immutable snapshots of one request/chunk's candidates and bounds.
The caller owns their forward-local, byte-bounded cache and must invalidate
it whenever candidates, request/prefix bounds, or logical K order change.
No automatic cache keyed only by shape or device pointers is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import PrefillIndexerKeys

MAX_CHUNK_ROWS = 4096
_WARMED_STREAMS: set[tuple[int, int]] = set()


@dataclass(frozen=True)
class SparsePrefillPlan:
    row_ks: torch.Tensor  # [M] int32 zeros, request-local packed K starts
    row_ke: torch.Tensor  # [M] int32 logical visible lengths, clamped to N
    sparse_indices: torch.Tensor  # [M, S] int32 sorted block IDs, last-ID padded
    end: torch.Tensor  # [M] int32 valid sparse-token columns, not logical length
    metadata: torch.Tensor  # DeepGEMM uint8 schedule; retains no input pointers
    key_count: int
    block_size: int

    @property
    def rows(self) -> int:
        return self.sparse_indices.shape[0]

    @property
    def sparse_columns(self) -> int:
        return self.sparse_indices.shape[1] * self.block_size

    @property
    def nbytes(self) -> int:
        """Owned plan buffers, including the relatively large DG schedule."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.row_ks,
                self.row_ke,
                self.sparse_indices,
                self.end,
                self.metadata,
            )
        )


@lru_cache(maxsize=1)
def _get_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    if not all(
        callable(getattr(deep_gemm, name, None))
        for name in ("get_sparse_mqa_logits_metadata", "fp8_fp4_sparse_mqa_logits")
    ):
        return None
    return deep_gemm


def _plan_supported(candidates, visible, key_count, block_size):
    if not (
        torch.version.hip is None
        and candidates.is_cuda
        and candidates.ndim == 2
        and candidates.dtype == torch.int32
        and 0 < candidates.shape[0] <= MAX_CHUNK_ROWS
        and 4 <= candidates.shape[1] <= 4096
        and candidates.shape[1] & (candidates.shape[1] - 1) == 0
        and candidates.stride(1) == 1
        and candidates.stride(0) >= candidates.shape[1]
        and visible.device == candidates.device
        and visible.ndim == 1
        and visible.numel() == candidates.shape[0]
        and visible.dtype in (torch.int32, torch.int64)
        and visible.stride(0) > 0
        and 0 < key_count <= 1 << 20
        and block_size == 8
    ):
        return False
    return (
        torch.cuda.get_device_capability(candidates.device)[0] == 10
        and _get_deep_gemm() is not None
    )


def _score_supported(q_payload, q_sf, keys, weights, rows, key_count, device):
    return (
        isinstance(keys, PrefillIndexerKeys)
        and q_payload.device == device
        and q_payload.dtype == torch.int8
        and q_payload.shape == (rows, 32, 64)
        and q_payload.is_contiguous()
        and q_sf.device == device
        and q_sf.dtype == torch.int32
        and q_sf.shape == (rows, 32)
        and q_sf.is_contiguous()
        and keys.quant.device == device
        and keys.quant.dtype == torch.int8
        and keys.quant.shape == (key_count, 64)
        and keys.quant.is_contiguous()
        and keys.scale.device == device
        and keys.scale.dtype == torch.int32
        and keys.scale.shape == (key_count,)
        and keys.scale.is_contiguous()
        and weights.device == device
        and weights.dtype in (torch.bfloat16, torch.float32)
        and weights.shape == (rows, 32)
        and weights.stride(1) == 1
        and weights.stride(0) >= 32
        and not (torch.is_grad_enabled() and weights.requires_grad)
    )


def is_supported(
    q_payload: torch.Tensor,
    q_sf: torch.Tensor,
    keys: PrefillIndexerKeys,
    weights: torch.Tensor,
    candidates: torch.Tensor,
    visible: torch.Tensor,
    block_size: int = 8,
    topk: int = 512,
) -> bool:
    """Whole scoring boundary gate; the caller separately gates DeepSelect.

    The production DeepSelect input stride needs 1024-byte alignment and its
    int32 output row needs 32-byte alignment. Plan-only tests may use smaller
    K through prepare_plan, but this gate rejects those selection layouts.
    """
    if not isinstance(keys, PrefillIndexerKeys) or not _plan_supported(
        candidates, visible, len(keys), block_size
    ):
        return False
    width = candidates.shape[1] * block_size
    return (
        0 < topk <= min(4096, width)
        and topk % 8 == 0
        and width * 2 % 1024 == 0
        and _score_supported(
            q_payload,
            q_sf,
            keys,
            weights,
            candidates.shape[0],
            len(keys),
            candidates.device,
        )
    )


# Prefix reuse changes the logical key count for every request. It is only a
# clamp bound; specializing it recompiles the expensive K2048 sorting network.
@triton.jit(do_not_specialize=["KEY_COUNT"])
def _prepare_sparse_prefill_plan_kernel(
    candidates,
    visible,
    row_ks,
    row_ke,
    sparse_indices,
    sparse_end,
    CANDIDATE_STRIDE: tl.constexpr,
    VISIBLE_STRIDE: tl.constexpr,
    K: tl.constexpr,
    KEY_COUNT,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, K)
    candidate = tl.load(candidates + row * CANDIDATE_STRIDE + column)
    length = tl.load(visible + row * VISIBLE_STRIDE)
    length = tl.minimum(tl.maximum(length, 0), KEY_COUNT).to(tl.int32)
    tl.store(row_ks + row, 0)
    tl.store(row_ke + row, length)
    max_block = tl.cdiv(length, BLOCK)
    valid = (candidate >= 0) & (candidate < max_block)
    sentinel: tl.constexpr = 2147483647
    sorted_blocks = tl.sort(tl.where(valid, candidate, sentinel), descending=False)
    previous = tl.gather(sorted_blocks, tl.maximum(column - 1, 0), axis=0)
    unique = (sorted_blocks != sentinel) & ((column == 0) | (sorted_blocks != previous))
    compact_column = tl.cumsum(unique.to(tl.int32)) - 1
    valid_blocks = tl.sum(unique.to(tl.int32))
    last_block = tl.max(tl.where(unique, sorted_blocks, -1))
    tl.store(sparse_indices + row * K + compact_column, sorted_blocks, unique)
    # Prefix and padding stores target disjoint addresses; no second kernel.
    tl.store(
        sparse_indices + row * K + column,
        tl.maximum(last_block, 0),
        column >= valid_blocks,
    )
    tail = tl.minimum(tl.maximum(length - last_block * BLOCK, 0), BLOCK)
    count = tl.where(valid_blocks > 0, (valid_blocks - 1) * BLOCK + tail, 0)
    tl.store(sparse_end + row, count)


def prepare_plan(
    candidates: torch.Tensor,
    visible: torch.Tensor,
    key_count: int,
    block_size: int = 8,
) -> SparsePrefillPlan | None:
    """Expand/sort/deduplicate one request-local chunk and build its DG schedule.

    Negative and out-of-causal-range candidate IDs are discarded. Repeated
    IDs represent one block, matching candidate-mask set membership. This
    operation stays on the device, including visible's int64-to-int32 clamp.
    A stream must run eager preparation once before CUDA graph capture,
    because the DG schedule builder initializes a per-stream workspace.
    """
    if not _plan_supported(candidates, visible, key_count, block_size):
        return None
    stream = torch.cuda.current_stream(candidates.device)
    stream_key = (stream.device.index, stream.cuda_stream)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and stream_key not in _WARMED_STREAMS:
        return None
    rows, count = candidates.shape
    bounds = torch.empty((3, rows), dtype=torch.int32, device=candidates.device)
    row_ks, row_ke, sparse_end = bounds.unbind(0)
    indices = torch.empty((rows, count), dtype=torch.int32, device=candidates.device)
    _prepare_sparse_prefill_plan_kernel[(rows,)](
        candidates,
        visible,
        row_ks,
        row_ke,
        indices,
        sparse_end,
        candidates.stride(0),
        visible.stride(0),
        count,
        key_count,
        block_size,
        num_warps=16 if count >= 1024 else 4,
    )
    # The pinned DeepGEMM non-paged metadata ABI uses (ke - ks) ONLY to
    # derive ceil_div(compact_length, BLOCK). Actual K addresses come from
    # sparse_indices, and the scoring kernel is bounded by key_count.
    # Give it compact length, not logical visibility: repeated padding
    # violates its unique-prefix merge-path contract and can assert.
    # See scheduler/sm100_sparse_mqa_logits_metadata.cuh:get_num_kv_blocks.
    # row_ke remains the logical causal boundary for selection/remapping.
    metadata = _get_deep_gemm().get_sparse_mqa_logits_metadata(
        row_ks,
        sparse_end,
        key_count,
        indices,
        torch.int8,
        block_size,
        use_unaligned_ks=True,
    )
    if not capturing:
        _WARMED_STREAMS.add(stream_key)
    return SparsePrefillPlan(
        row_ks, row_ke, indices, sparse_end, metadata, key_count, block_size
    )


def score(
    q_payload: torch.Tensor,
    q_sf: torch.Tensor,
    keys: PrefillIndexerKeys,
    weights: torch.Tensor,
    plan: SparsePrefillPlan,
) -> torch.Tensor | None:
    """Compute BF16 sparse logits; padding outside plan.end is not a valid score.

    FP4 payload and packed group-32 UE8M0 scales are consumed unchanged.
    Head weights are converted to BF16 (not multiplied by Q scales). This
    precision change follows vLLM's sparse contract and is intentional.
    """
    if not (
        isinstance(plan, SparsePrefillPlan)
        and q_payload.is_cuda
        and _score_supported(
            q_payload,
            q_sf,
            keys,
            weights,
            plan.rows,
            plan.key_count,
            plan.sparse_indices.device,
        )
    ):
        return None
    return _get_deep_gemm().fp8_fp4_sparse_mqa_logits(
        (q_payload, q_sf),
        (keys.quant, keys.scale),
        weights.to(torch.bfloat16),
        plan.metadata,
        plan.sparse_indices.shape[1],
        plan.block_size,
        use_unaligned_ks=True,
    )


@triton.jit
def _remap_sparse_prefill_topk_kernel(
    columns,
    indices,
    sparse_end,
    visible,
    logits,
    output,
    COLUMN_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    LOGITS_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SPARSE_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LOGITS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, TILE)
    column = tl.load(columns + row * COLUMN_STRIDE + slot, slot < TOPK, other=-1)
    count = tl.load(sparse_end + row)
    length = tl.load(visible + row)
    valid = (slot < TOPK) & (column >= 0) & (column < count) & (column < SPARSE_COLUMNS)
    block = tl.load(indices + row * INDEX_STRIDE + column // BLOCK, valid, other=0)
    position = block * BLOCK + column % BLOCK
    valid = valid & (position >= 0) & (position < length)
    if HAS_LOGITS:
        value = tl.load(
            logits + row * LOGITS_STRIDE + column, valid, other=float("nan")
        ).to(tl.float32)
        valid = valid & (tl.abs(value) < float("inf"))
    tl.store(
        output + row * OUTPUT_STRIDE + slot, tl.where(valid, position, -1), slot < TOPK
    )


def remap(
    columns: torch.Tensor,
    plan: SparsePrefillPlan,
    *,
    logits: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Map selected sparse columns back to request-local compressed positions.

    -1, out-of-prefix, and causal-tail positions remain -1. Passing logits
    also preserves RTP's filtering of selected NaN and +/-inf scores. The
    caller can supply a chunk slice of its int32 final output buffer.
    """
    if not (
        isinstance(plan, SparsePrefillPlan)
        and columns.is_cuda
        and columns.device == plan.sparse_indices.device
        and columns.dtype == torch.int32
        and columns.ndim == 2
        and columns.shape[0] == plan.rows
        and 0 < columns.shape[1] <= 4096
        and columns.stride(1) == 1
        and columns.stride(0) >= columns.shape[1]
    ):
        return None
    if logits is not None and not (
        logits.device == columns.device
        and logits.dtype == torch.bfloat16
        and logits.shape == (plan.rows, plan.sparse_columns)
        and logits.stride(1) == 1
        and logits.stride(0) >= plan.sparse_columns
    ):
        return None
    if out is not None and not (
        out.device == columns.device
        and out.dtype == torch.int32
        and out.shape == columns.shape
        and out.stride(1) == 1
        and out.stride(0) >= out.shape[1]
    ):
        return None
    if out is None:
        out = torch.empty(columns.shape, dtype=torch.int32, device=columns.device)
    _remap_sparse_prefill_topk_kernel[(plan.rows,)](
        columns,
        plan.sparse_indices,
        plan.end,
        plan.row_ke,
        logits,
        out,
        columns.stride(0),
        plan.sparse_indices.stride(0),
        logits.stride(0) if logits is not None else 0,
        out.stride(0),
        columns.shape[1],
        plan.sparse_columns,
        plan.block_size,
        logits is not None,
        triton.next_power_of_2(columns.shape[1]),
    )
    return out

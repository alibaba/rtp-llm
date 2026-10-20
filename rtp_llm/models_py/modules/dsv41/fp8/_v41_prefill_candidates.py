# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 prefill candidate pooling and reusable packed candidate masks.

Adapted from vLLM's model_executor/kernels/attention/dsa/candidate_blocks.py
and RTP's _v41_decode_topk.py. Bitmap publication retains torch.topk(sorted=True).
Candidate-only publication skips the final score sort: sparse plans sort and
deduplicate block IDs before scoring, so they only consume the selected set.
The caller has already applied causal masking to logits, as in the old
select_candidate_blocks path. Only newest-block overwrite is done here.

The reusable flags are int32 bitmaps, with 32 candidate blocks per word.
Rows may be slices of a larger cache, and that cache may have extra words
for another request with a larger logical key count. Each row is initialized
before atomic bit updates in the same CTA. Masking never inspects those extra
words. No host tensor construction or host synchronization is performed.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

DEFAULT_MAX_BITMAP_BYTES = 256 * 1024 * 1024


def bitmap_words(width: int, block_size: int) -> int:
    return triton.cdiv(triton.cdiv(width, block_size), 32)


def bitmap_size_bytes(rows: int, width: int, block_size: int) -> int:
    return rows * bitmap_words(width, block_size) * 4


def bitmap_is_bounded(rows: int, width: int, block_size: int) -> bool:
    """Check the complete cross-layer cache size, not just one scoring chunk."""
    if rows < 0 or width <= 0 or block_size <= 0:
        return False
    limit = int(
        os.environ.get(
            "DSV41_PREFILL_CANDIDATE_FLAGS_MAX_BYTES", DEFAULT_MAX_BITMAP_BYTES
        )
    )
    return bitmap_size_bytes(rows, width, block_size) <= max(0, limit)


def is_supported(
    logits: torch.Tensor,
    visible: torch.Tensor,
    block_size: int,
    topk_blocks: int,
) -> bool:
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] > 0
        and 0 < logits.shape[1] <= 1 << 20
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and visible.device == logits.device
        and visible.ndim == 1
        and visible.numel() == logits.shape[0]
        and visible.dtype in (torch.int32, torch.int64)
        and visible.stride(0) > 0
        and block_size in (1, 2, 4, 8, 16, 32)
        and 0 < topk_blocks <= 4096
        and not (torch.is_grad_enabled() and logits.requires_grad)
    )


@triton.jit
def _max_with_nan(a, b):
    # Torch MaxNanFunctor takes the right finite tie and the left NaN.
    bits = tl.where(
        (a != a) | (a > b),
        a.to(tl.int32, bitcast=True),
        b.to(tl.int32, bitcast=True),
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _torch_amax(values, BLOCK_SIZE: tl.constexpr):
    if BLOCK_SIZE <= 4:
        # TILE=128 uses one lane per reduced value for B<=4 in the qualified
        # CUDA13 toolchain, matching Torch's descending-offset shuffle tree.
        return tl.reduce(values, 1, _max_with_nan)
    else:
        # Larger groups may span registers: force the Torch tree explicitly.
        for log_width in tl.static_range(5, 0, -1):
            if BLOCK_SIZE >= (1 << log_width):
                pairs = tl.permute(
                    tl.reshape(values, (values.shape[0], 2, values.shape[1] // 2)),
                    (0, 2, 1),
                )
                left, right = tl.split(pairs)
                values = _max_with_nan(left, right)
        return tl.reshape(values, (values.shape[0],))


@triton.jit(do_not_specialize=["WIDTH", "STRIDE", "NBLOCKS", "OUT_K", "OUT_STRIDE"])
def _prefill_candidate_pool_kernel(
    logits,
    visible,
    scores,
    WIDTH,
    STRIDE,
    VISIBLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NBLOCKS,
    TILE: tl.constexpr,
    RETURN_IDS: tl.constexpr = False,
    OUT_K=0,
    OUT_STRIDE=0,
    MASK_TAIL: tl.constexpr = False,
    token_indices=None,
    token_ends=None,
    FILTER_TOKENS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, BLOCK_SIZE)
    columns = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    mask = (blocks[:, None] < NBLOCKS) & (columns < WIDTH)
    if MASK_TAIL:
        length = tl.load(visible + row * VISIBLE_STRIDE)
        mask &= columns < length
    values = tl.load(
        logits + row * STRIDE + columns,
        mask,
        other=-float("inf"),
    )
    pooled = _torch_amax(values, BLOCK_SIZE)
    if not MASK_TAIL:
        length = tl.load(visible + row * VISIBLE_STRIDE)
    newest = tl.maximum(length - 1, 0) // BLOCK_SIZE
    pooled = tl.where(
        blocks == newest,
        tl.where(length > 0, float("inf"), -float("inf")),
        pooled,
    )
    if RETURN_IDS:
        valid = (blocks < NBLOCKS) & (pooled > -float("inf"))
        tl.store(
            scores + row * OUT_STRIDE + blocks,
            tl.where(valid, blocks, -1),
            blocks < OUT_K,
        )
    else:
        tl.store(scores + row * NBLOCKS + blocks, pooled, blocks < NBLOCKS)
    if FILTER_TOKENS:
        if tl.program_id(1) == 0:
            slot = tl.arange(0, 512)
            end = tl.load(token_ends + row)
            index = tl.load(token_indices + row * 512 + slot)
            in_range = (index >= 0) & (index < end)
            value = tl.load(
                logits + row * STRIDE + index, in_range, other=-float("inf")
            )
            keep = in_range & (tl.abs(value) < float("inf"))
            tl.store(token_indices + row * 512 + slot, tl.where(keep, index, -1))


@triton.jit(do_not_specialize=["K", "FLAG_WORDS", "FLAG_STRIDE"])
def _prefill_candidate_store_bitmap_kernel(
    values,
    indices,
    candidates,
    flags,
    K,
    OUT_K: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    FLAG_WORDS,
    FLAG_STRIDE,
    WRITE_FLAGS: tl.constexpr,
    FLAG_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if WRITE_FLAGS:
        word = tl.arange(0, FLAG_TILE)
        tl.store(flags + row * FLAG_STRIDE + word, 0, word < FLAG_WORDS)
    columns = tl.arange(0, triton.next_power_of_2(OUT_K))
    score = tl.load(values + row * K + columns, columns < K, other=-float("inf"))
    index = tl.load(indices + row * K + columns, columns < K, other=-1).to(tl.int32)
    valid = (columns < K) & (score > -float("inf"))
    # Keep +inf newest-block pins; filter NaN and -inf exactly like torch.
    tl.store(
        candidates + row * OUT_STRIDE + columns,
        tl.where(valid, index, -1),
        columns < OUT_K,
    )
    if WRITE_FLAGS:
        tl.debug_barrier()
        bit = (1 << (index & 31)).to(tl.int32)
        tl.atomic_or(
            flags + row * FLAG_STRIDE + (index >> 5), bit, valid, sem="relaxed"
        )


@triton.jit(do_not_specialize=["WIDTH", "STRIDE", "FLAG_STRIDE"])
def _prefill_candidate_mask_kernel(
    logits,
    flags,
    WIDTH,
    STRIDE,
    FLAG_STRIDE,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = columns // BLOCK_SIZE
    word = tl.load(
        flags + row * FLAG_STRIDE + (block >> 5), columns < WIDTH, other=0
    ).to(tl.uint32)
    keep = ((word >> (block & 31)) & 1) != 0
    tl.store(
        logits + row * STRIDE + columns,
        -float("inf"),
        (columns < WIDTH) & ~keep,
    )


@triton.jit(do_not_specialize=["NBLOCKS", "FLAG_WORDS", "FLAG_STRIDE"])
def _prefill_candidate_build_bitmap_kernel(
    candidates,
    flags,
    K: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    NBLOCKS,
    FLAG_WORDS,
    FLAG_STRIDE,
    FLAG_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    word = tl.arange(0, FLAG_TILE)
    tl.store(flags + row * FLAG_STRIDE + word, 0, word < FLAG_WORDS)
    columns = tl.arange(0, triton.next_power_of_2(K))
    index = tl.load(candidates + row * IN_STRIDE + columns, columns < K, other=-1)
    valid = (columns < K) & (index >= 0) & (index < NBLOCKS)
    tl.debug_barrier()
    tl.atomic_or(
        flags + row * FLAG_STRIDE + (index >> 5),
        (1 << (index & 31)).to(tl.int32),
        valid,
        sem="relaxed",
    )


def _valid_output(tensor, rows, columns, device):
    return (
        tensor.device == device
        and tensor.dtype == torch.int32
        and tensor.ndim == 2
        and tensor.shape[0] == rows
        and tensor.shape[1] >= columns
        and tensor.stride(1) == 1
        and tensor.stride(0) >= tensor.shape[1]
    )


def can_select_candidates(
    logits: torch.Tensor,
    visible: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    *,
    out: torch.Tensor | None = None,
    flags: torch.Tensor | None = None,
    build_bitmap: bool = True,
    token_indices: torch.Tensor | None = None,
    token_ends: torch.Tensor | None = None,
) -> bool:
    """Metadata-only preflight shared by selection and its fused caller."""
    if not is_supported(logits, visible, block_size, topk_blocks):
        return False
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    count = min(topk_blocks, nblocks)
    words = bitmap_words(width, block_size)
    if out is not None and (
        not _valid_output(out, rows, count, logits.device) or out.shape[1] > 4096
    ):
        return False
    if flags is not None and not _valid_output(flags, rows, words, logits.device):
        return False
    if (
        build_bitmap
        and flags is None
        and not bitmap_is_bounded(rows, width, block_size)
    ):
        return False
    if token_indices is not None or token_ends is not None:
        if not (
            isinstance(token_indices, torch.Tensor)
            and isinstance(token_ends, torch.Tensor)
            and token_indices.device == token_ends.device == logits.device
            and token_indices.dtype == token_ends.dtype == torch.int32
            and token_indices.shape == (rows, 512)
            and token_ends.shape == (rows,)
            and token_indices.is_contiguous()
            and token_ends.is_contiguous()
        ):
            return False
        storage = token_indices.untyped_storage().data_ptr()
        if any(
            t is not None and t.untyped_storage().data_ptr() == storage
            for t in (logits, visible, token_ends, out, flags)
        ):
            return False
    return True


def select_candidates(
    logits: torch.Tensor,
    visible: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    *,
    out: torch.Tensor | None = None,
    flags: torch.Tensor | None = None,
    build_bitmap: bool = True,
    mask_tail: bool = False,
    token_indices: torch.Tensor | None = None,
    token_ends: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Return candidate IDs and bitmap, or None before launching on fallback.

    ``visible`` contains request-local lengths in [0, logits.shape[1]]. The
    default requires masked logits; ``mask_tail=True`` masks at the pool load.
    Optional contiguous raw K512 IDs are finite-filtered in each row's tile0;
    their ends must describe the same visible range. Candidate +inf pins and
    token finite filtering deliberately use different predicates.
    Flags may be bounded cache views; extra words are cleared. Output padding
    is -1. IDs are score-ordered by default; ``build_bitmap=False`` consumers
    use an unordered set and can rebuild their own bounded bitmap.
    """
    if not can_select_candidates(
        logits,
        visible,
        block_size,
        topk_blocks,
        out=out,
        flags=flags,
        build_bitmap=build_bitmap,
        token_indices=token_indices,
        token_ends=token_ends,
    ):
        return None
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    count = min(topk_blocks, nblocks)
    words = bitmap_words(width, block_size)
    if out is None:
        out = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    if build_bitmap and flags is None:
        flags = torch.empty((rows, words), dtype=torch.int32, device=logits.device)
    if not build_bitmap:
        flags = None
        if nblocks <= topk_blocks:
            # Every block is selected, so only pooling/pinning/filtering remain.
            _prefill_candidate_pool_kernel[(rows, triton.cdiv(out.shape[1], 128))](
                logits,
                visible,
                out,
                width,
                logits.stride(0),
                visible.stride(0),
                block_size,
                nblocks,
                128,
                True,
                out.shape[1],
                out.stride(0),
                mask_tail,
                token_indices,
                token_ends,
                token_indices is not None,
            )
            return out, None
    scores = torch.empty((rows, nblocks), dtype=logits.dtype, device=logits.device)
    _prefill_candidate_pool_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        visible,
        scores,
        width,
        logits.stride(0),
        visible.stride(0),
        block_size,
        nblocks,
        128,
        MASK_TAIL=mask_tail,
        token_indices=token_indices,
        token_ends=token_ends,
        FILTER_TOKENS=token_indices is not None,
    )
    if not build_bitmap:
        from ._v41_candidate_topk import try_select

        selected = try_select(scores, out, k=count)
        if selected is not None:
            return selected, None
    values, indices = scores.topk(count, dim=-1, largest=True, sorted=build_bitmap)
    _prefill_candidate_store_bitmap_kernel[(rows,)](
        values,
        indices,
        out,
        flags,
        count,
        out.shape[1],
        out.stride(0),
        flags.shape[1] if flags is not None else 1,
        flags.stride(0) if flags is not None else 1,
        build_bitmap,
        triton.next_power_of_2(flags.shape[1]) if flags is not None else 1,
    )
    return out, flags


def build_flags(
    candidates: torch.Tensor,
    width: int,
    block_size: int,
    *,
    flags: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a bounded chunk bitmap when caching all query rows is too large.

    For example, CP4 512K prefill needs 1 GiB for the whole-query bitmap, but
    a 1024-row consumer chunk needs only 8 MiB. Rebuild that chunk from the
    cached int32 candidate IDs rather than falling back to a dense bool mask.
    """
    if not (
        candidates.is_cuda
        and candidates.dtype == torch.int32
        and candidates.ndim == 2
        and candidates.shape[0] > 0
        and 0 < candidates.shape[1] <= 4096
        and candidates.stride(1) == 1
        and candidates.stride(0) >= candidates.shape[1]
        and 0 < width <= 1 << 20
        and block_size in (1, 2, 4, 8, 16, 32)
    ):
        return None
    rows = candidates.shape[0]
    words = bitmap_words(width, block_size)
    if flags is not None and not _valid_output(flags, rows, words, candidates.device):
        return None
    if flags is None:
        if not bitmap_is_bounded(rows, width, block_size):
            return None
        flags = torch.empty((rows, words), dtype=torch.int32, device=candidates.device)
    _prefill_candidate_build_bitmap_kernel[(rows,)](
        candidates,
        flags,
        candidates.shape[1],
        candidates.stride(0),
        triton.cdiv(width, block_size),
        flags.shape[1],
        flags.stride(0),
        triton.next_power_of_2(flags.shape[1]),
    )
    return flags


def mask_candidates(logits: torch.Tensor, flags: torch.Tensor, block_size: int) -> bool:
    """Apply a previously built bitmap in place; False means no work launched."""
    if not (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] > 0
        and 0 < logits.shape[1] <= 1 << 20
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and block_size in (1, 2, 4, 8, 16, 32)
        and _valid_output(
            flags,
            logits.shape[0],
            bitmap_words(logits.shape[1], block_size),
            logits.device,
        )
        and not (torch.is_grad_enabled() and logits.requires_grad)
    ):
        return False
    rows, width = logits.shape
    _prefill_candidate_mask_kernel[(rows, triton.cdiv(width, 1024))](
        logits, flags, width, logits.stride(0), flags.stride(0), block_size, 1024
    )
    return True

"""Exact sparse TopK and logical-ID mapping for PPU Decode/Verify.

One CTA owns a row, including prefix/tentative masking, radix selection,
finite filtering and remapping. Fixed 1024-column tiles bound register use;
row widths and strides are runtime values, not shape-specialized JIT keys.
There is no packed-score/rank matrix or inter-kernel selection workspace.

Ordinary Decode preserves score-descending, compact-column-ascending order.
Verify's native selector promises neither order nor a particular tied subset;
this implementation chooses the same deterministic order as Decode. NaNs and
infinities consume selection slots before finite filtering, with no refill.
Short rows preserve the existing ordinal output, independently of scores.

BF16 provenance permits two exact radix bytes. Arbitrary FP32 Verify scores
use all four bytes, with no conversion to BF16. The rollout flag defaults off;
this implementation is not a claim of measured numerical/performance parity.
"""

from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import SPARSE_TOPK, enabled


def _raw(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.requires_grad
        and not value.is_conj()
        and not value.is_neg()
    )


def _vector(value, rows, device, dtypes):
    return (
        _raw(value)
        and value.device == device
        and value.dtype in dtypes
        and value.shape == (rows,)
        and value.stride(0) > 0
    )


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def score_key(score, BF16_KEYS: tl.constexpr):
        # Same ordering as graph_indexer.descending_score_key: all NaNs first,
        # signed zero equal, otherwise the exact IEEE float ordering.
        bits = score.to(tl.uint32, bitcast=True)
        magnitude = bits & 0x7FFFFFFF
        bits = tl.where(magnitude == 0, 0, bits).to(tl.uint32)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        key = tl.where(magnitude > 0x7F800000, 0xFFFFFFFF, ordered).to(tl.uint32)
        if BF16_KEYS:
            key = key >> 16
        return key

    @triton.jit
    def load_scores(
        scores, logical_ids, tentative, allowed,
        row, column, score_stride, block_count, tile, capacity,
        compact_length, length, split, active,
        tail_width, tail_row, tail_col, allowed_stride,
        VERIFY: tl.constexpr,
    ):
        prefix_width = block_count * tile
        prefix_live = (column >= 0) & (column < prefix_width) & (column < compact_length)
        block = tl.load(
            logical_ids + row * block_count + column // tile,
            prefix_live, other=-1,
        ).to(tl.int64)
        logical = block * tile + column % tile
        if VERIFY:
            prefix_live = prefix_live & (logical >= 0) & (logical < split)
        prefix_live = prefix_live & (logical < capacity) & (logical < length)
        prefix_score = tl.load(
            scores + row * score_stride + column, prefix_live, other=-float("inf")
        ).to(tl.float32)
        if VERIFY:
            tail = column - prefix_width
            tail_logical = split + tail
            tail_live = (tail >= 0) & (tail < tail_width)
            tail_live = tail_live & (tail_logical >= 0)
            tail_live = tail_live & (tail_logical < capacity) & (tail_logical < length)
            permitted = tl.load(
                allowed + row * allowed_stride + tail_logical // tile,
                tail_live, other=0,
            ) != 0
            tail_live = tail_live & (~active | permitted)
            tail_score = tl.load(
                tentative + row * tail_row + tail * tail_col,
                tail_live, other=-float("inf"),
            ).to(tl.float32)
            value = tl.where(prefix_live, prefix_score, tail_score)
            logical = tl.where(
                prefix_live, logical, tl.where(tail_live, tail_logical, -1)
            )
        else:
            value = prefix_score
        return value, logical

    @triton.jit(do_not_specialize=[
        "score_stride", "block_count", "capacity", "length_stride",
        "visible_stride", "valid_stride", "first_stride", "active_stride",
        "tail_width", "tail_row", "tail_col", "allowed_stride",
    ])
    def select_sparse(
        scores, logical_ids, lengths, visible, valid, first, tentative, allowed,
        candidate_active, output,
        score_stride, block_count, capacity, length_stride,
        visible_stride, valid_stride, first_stride, active_stride,
        tail_width, tail_row, tail_col, allowed_stride,
        TILE: tl.constexpr, VERIFY: tl.constexpr, BF16_KEYS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        lane = tl.arange(0, 512)
        length = tl.maximum(tl.load(visible + row * visible_stride), 0)
        live = tl.load(valid + row * valid_stride)
        length = tl.where(live, length, 0)
        if length <= 512:
            tl.store(output + row * 512 + lane, tl.where(lane < length, lane, -1))
        else:
            compact_length = tl.load(lengths + row * length_stride)
            split = length
            active = False
            if VERIFY:
                split = tl.load(first + row * first_stride)
                active = tl.load(candidate_active + row * active_stride)
            width = block_count * TILE + tail_width
            offset = tl.arange(0, BLOCK)
            bins = tl.arange(0, 256)
            radix_bytes: tl.constexpr = 2 if BF16_KEYS else 4
            threshold = tl.full((), 0, tl.uint32)
            prefix_mask = tl.full((), 0, tl.uint32)
            remaining = tl.full((), 512, tl.int32)

            # Refine the exact kth key a byte at a time. Inactive sparse slots
            # retain their -Inf scores and still participate, as in the native
            # padded score domain. No finite-only prefilter changes underfill.
            for digit in tl.static_range(radix_bytes):
                shift: tl.constexpr = (radix_bytes - 1 - digit) * 8
                histogram = tl.full((256,), 0, tl.int32)
                for start in range(0, tl.cdiv(width, BLOCK)):
                    column = start * BLOCK + offset
                    value, _ = load_scores(
                        scores, logical_ids, tentative, allowed,
                        row, column, score_stride, block_count, TILE, capacity,
                        compact_length, length, split, active,
                        tail_width, tail_row, tail_col, allowed_stride, VERIFY,
                    )
                    key = score_key(value, BF16_KEYS)
                    eligible = (column < width) & ((key & prefix_mask) == threshold)
                    histogram += tl.histogram(
                        ((key >> shift) & 255).to(tl.int32), 256, mask=eligible
                    )
                cumulative = tl.cumsum(histogram, reverse=True)
                pivot = tl.max(tl.where(cumulative >= remaining, bins, 0), 0)
                remaining -= tl.sum(tl.where(bins > pivot, histogram, 0), 0)
                threshold = threshold | (pivot.to(tl.uint32) << shift)
                prefix_mask = prefix_mask | tl.full((), 255 << shift, tl.uint32)

            # Exactly 512 - remaining scores exceed the threshold. Equal keys
            # fill the remainder in compact-column order, without a tie buffer.
            greater_total = 512 - remaining
            greater_prefix = tl.full((), 0, tl.int32)
            equal_prefix = tl.full((), 0, tl.int32)
            for start in range(0, tl.cdiv(width, BLOCK)):
                column = start * BLOCK + offset
                value, _ = load_scores(
                    scores, logical_ids, tentative, allowed,
                    row, column, score_stride, block_count, TILE, capacity,
                    compact_length, length, split, active,
                    tail_width, tail_row, tail_col, allowed_stride, VERIFY,
                )
                key = score_key(value, BF16_KEYS)
                greater = (column < width) & (key > threshold)
                equal = (column < width) & (key == threshold)
                greater_rank = greater_prefix + tl.cumsum(greater.to(tl.int32)) - 1
                equal_rank = greater_total + equal_prefix + tl.cumsum(equal.to(tl.int32)) - 1
                rank = tl.where(greater, greater_rank, equal_rank)
                tl.store(
                    output + row * 512 + rank, column,
                    (greater | equal) & (rank >= 0) & (rank < 512),
                )
                greater_prefix += tl.sum(greater.to(tl.int32), 0)
                equal_prefix += tl.sum(equal.to(tl.int32), 0)
            # One CTA owns the whole row. The output itself is the 512-index
            # scratch; synchronize its scatter before the final ordered loads.
            tl.debug_barrier()
            selected = tl.load(output + row * 512 + lane)
            values, _ = load_scores(
                scores, logical_ids, tentative, allowed,
                row, selected, score_stride, block_count, TILE, capacity,
                compact_length, length, split, active,
                tail_width, tail_row, tail_col, allowed_stride, VERIFY,
            )
            key = score_key(values, BF16_KEYS)
            ranked = (key.to(tl.uint64) << 32) | (
                0xFFFFFFFF - selected.to(tl.uint32)
            ).to(tl.uint64)
            ranked = tl.sort(ranked, descending=True)
            selected = (0xFFFFFFFF - (ranked & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
            values, logical = load_scores(
                scores, logical_ids, tentative, allowed,
                row, selected, score_stride, block_count, TILE, capacity,
                compact_length, length, split, active,
                tail_width, tail_row, tail_col, allowed_stride, VERIFY,
            )
            keep = tl.abs(values) < float("inf")
            tl.store(output + row * 512 + lane, tl.where(keep, logical, -1).to(tl.int32))

    return select_sparse


def try_select_sparse(
    scored, visible, valid_mask, topk=512, *, first=None, tentative=None,
    allowed=None, candidate_active=None, bf16_tail_provenance=False,
):
    """Return [rows,512] logical IDs, or None before unsupported dispatch.

    ``scored`` is the existing _sparse_candidate_scores result: scores,
    logical_ids, lengths, blocks, tile and capacity. Omit all four Verify
    tensors for ordinary Decode. For Verify provide all four, with the same
    prefix/journal/candidate semantics as _pack_verify_compact_logits.

    Metadata admission does not inspect device values. The scorer guarantees
    0 <= lengths <= blocks*tile and live logical IDs reference its cache.
    ``bf16_tail_provenance`` asserts every tentative FP32 score is an exact
    BF16 expansion; it never requests a cast/rounding. Without that proof,
    Verify uses full FP32 ranking. Provider/JIT errors after dispatch propagate.
    """
    if not enabled(SPARSE_TOPK):
        return None
    scores = getattr(scored, "scores", None)
    ids = getattr(scored, "logical_ids", None)
    lengths = getattr(scored, "lengths", None)
    blocks = getattr(scored, "blocks", None)
    tile = getattr(scored, "tile", None)
    capacity = getattr(scored, "capacity", None)
    if not (
        topk == 512
        and type(bf16_tail_provenance) is bool
        and _raw(scores)
        and scores.is_cuda
        and scores.dtype in (torch.bfloat16, torch.float32)
        and scores.ndim == 2
        and scores.shape[0] > 0
        and scores.stride(1) == 1
        and scores.stride(0) >= scores.shape[1]
        and type(blocks) is int and 0 < blocks <= 4096
        and type(tile) is int and tile in (8, 16)
        and type(capacity) is int and 0 < capacity < 2**31
        and 512 <= blocks * tile <= scores.shape[1]
        and _raw(ids)
        and ids.device == scores.device
        and ids.dtype == torch.int32
        and ids.shape == (scores.shape[0], blocks)
        and ids.is_contiguous()
        and is_ppu()
    ):
        return None
    rows, device = scores.shape[0], scores.device
    integer = (torch.int32, torch.int64)
    if not (
        _vector(lengths, rows, device, (torch.int32,))
        and _vector(visible, rows, device, integer)
        and _vector(valid_mask, rows, device, (torch.bool,))
    ):
        return None
    verify = first is not None
    tail_width = tail_row = tail_col = allowed_stride = first_stride = active_stride = 0
    if verify:
        if not (
            _vector(first, rows, device, integer)
            and _vector(candidate_active, rows, device, (torch.bool,))
            and _raw(tentative)
            and tentative.device == device
            and tentative.dtype in (torch.bfloat16, torch.float32)
            and tentative.ndim == 2
            and tentative.shape[0] == rows
            and 1 <= tentative.shape[1] <= 1024
            and tentative.stride(1) > 0
            and tentative.stride(0) >= tentative.shape[1] * tentative.stride(1)
            and _raw(allowed)
            and allowed.device == device
            and allowed.dtype == torch.bool
            and allowed.ndim == 2
            and allowed.shape[0] == rows
            and allowed.shape[1] >= (capacity + tile - 1) // tile
            and allowed.stride(1) == 1
            and allowed.stride(0) >= allowed.shape[1]
        ):
            return None
        tail_width = tentative.shape[1]
        tail_row, tail_col = tentative.stride()
        allowed_stride = allowed.stride(0)
        first_stride, active_stride = first.stride(0), candidate_active.stride(0)
        bf16_keys = scores.dtype == torch.bfloat16 and (
            tentative.dtype == torch.bfloat16 or bf16_tail_provenance
        )
    else:
        if (
            any(value is not None for value in (tentative, allowed, candidate_active))
            or scores.dtype != torch.bfloat16
            or blocks * tile > 32768
        ):
            return None
        bf16_keys = True
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    _kernel()[(rows,)](
        scores, ids, lengths, visible, valid_mask, first, tentative, allowed,
        candidate_active, output,
        scores.stride(0), blocks, capacity, lengths.stride(0),
        visible.stride(0), valid_mask.stride(0), first_stride, active_stride,
        tail_width, tail_row, tail_col, allowed_stride,
        tile, verify, bf16_keys, 1024,
        num_warps=8, enable_fp_fusion=False,
    )
    return output

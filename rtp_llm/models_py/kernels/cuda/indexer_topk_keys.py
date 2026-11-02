"""Direct TopK keys: finite FP32 score descending, global position ascending."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sanitize(S, P, START, END, ROWS: tl.constexpr, COLS: tl.constexpr,
              S_ROW: tl.constexpr, P_ROW: tl.constexpr,
              BOUNDED: tl.constexpr, HAS_START: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // COLS, offset % COLS
    mask = row < ROWS
    score = tl.load(S + row * S_ROW + col, mask, other=-float("inf"))
    position = tl.load(P + row * P_ROW + col, mask, other=-1)
    invalid = (tl.abs(score) >= float("inf")) | (score != score) | (position < 0)
    if BOUNDED:
        end = tl.load(END + row, mask, other=0)
        invalid |= col >= end
        if HAS_START:
            start = tl.load(START + row, mask, other=0)
            invalid |= col < start
    tl.store(S + row * S_ROW + col, -float("inf"), mask & invalid)


def sanitize_indexer_scores(score, positions, *, starts=None, ends=None):
    """Mask phase-owned scores, optionally with per-query causal/request bounds.

    Native FP8 MQA can leave columns outside [ks,ke) unwritten. Apply those
    bounds in the existing sanitizer without a history-sized boolean tensor.
    The unbounded BF16 caller retains its original invalid-value policy.
    """
    if starts is not None and ends is None:
        raise ValueError("Indexer score starts require ends")
    for bounds in (starts, ends):
        if bounds is not None and (
            bounds.dtype != torch.int32
            or bounds.shape != (score.shape[0],)
            or bounds.device != score.device
            or not bounds.is_contiguous()
        ):
            raise ValueError("Indexer score bounds must be contiguous per-row int32")
    if score.numel():
        _sanitize[(triton.cdiv(score.numel(), 256),)](
            score, positions, starts if starts is not None else positions,
            ends if ends is not None else positions,
            score.shape[0], score.shape[1], score.stride(0),
            positions.stride(0) if positions.ndim == 2 else 0,
            ends is not None, starts is not None, 256,
        )


@triton.jit
def _pack(
    S,
    P,
    K,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    P_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // COLS, offset % COLS
    mask = row < ROWS
    score = tl.load(S + row * S_ROW + col, mask, other=-float("inf"))
    position = tl.load(P + row * P_ROW + col, mask, other=-1).to(tl.uint32)
    bits = tl.where(score == 0, 0, score.to(tl.uint32, bitcast=True))
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    upper = (ordered ^ 0x80000000).to(tl.int32).to(tl.int64)
    key = (upper << 32) | (0xFFFFFFFF - position).to(tl.int64)
    valid = (tl.abs(score) < float("inf")) & (position < 0x80000000)
    tl.store(K + offset, tl.where(valid, key, -9223372036854775808), mask)


@triton.jit
def _unpack(
    K,
    S,
    P,
    ROWS: tl.constexpr,
    SELECTED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // TOPK, offset % TOPK
    mask = row < ROWS
    key = tl.load(
        K + row * SELECTED + col, mask & (col < SELECTED), other=-9223372036854775808
    )
    valid = key != -9223372036854775808
    ordered = (key >> 32).to(tl.uint32) ^ 0x80000000
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
    score = bits.to(tl.float32, bitcast=True)
    position = (0xFFFFFFFF - key.to(tl.uint32)).to(tl.int32)
    tl.store(S + offset, tl.where(valid, score, -float("inf")), mask)
    tl.store(P + offset, tl.where(valid, position, -1), mask)


def pack_indexer_topk_keys(score, positions, keys):
    if (
        score.dtype != torch.float32
        or positions.dtype != torch.int32
        or keys.dtype != torch.int64
        or keys.shape != score.shape
        or not score.is_cuda
        or not keys.is_contiguous()
        or score.stride(-1) != 1
        or positions.stride(-1) != 1
        or positions.shape not in (score.shape, (score.shape[1],))
    ):
        raise ValueError("invalid direct Indexer TopK key buffers")
    if score.numel():
        _pack[(triton.cdiv(score.numel(), 256),)](
            score,
            positions,
            keys,
            score.shape[0],
            score.shape[1],
            score.stride(0),
            positions.stride(0) if positions.ndim == 2 else 0,
            256,
        )


def unpack_indexer_topk_keys(keys, scores, positions):
    if (
        keys.dtype != torch.int64
        or scores.dtype != torch.float32
        or positions.dtype != torch.int32
        or positions.shape != scores.shape
        or keys.shape[0] != scores.shape[0]
        or keys.shape[1] > scores.shape[1]
        or any(not t.is_contiguous() for t in (keys, scores, positions))
    ):
        raise ValueError("invalid bounded Indexer candidate buffers")
    if scores.numel():
        _unpack[(triton.cdiv(scores.numel(), 256),)](
            keys,
            scores,
            positions,
            scores.shape[0],
            keys.shape[1],
            scores.shape[1],
            256,
        )

"""Qwen4 two/three-token hashes with exact int64 and EOS reset semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hash_ngram_ids(
    HISTORY,
    VOCAB,
    OFFSETS,
    MULTIPLIERS,
    OUT,
    BATCH: tl.constexpr,
    SEQ: tl.constexpr,
    TOTAL: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.arange(0, 16)
    batch = row // SEQ
    position = TOTAL - SEQ + row % SEQ
    current = tl.load(
        HISTORY + batch * HISTORY_STRIDE + position, row < BATCH * SEQ, other=EOS
    )
    previous = tl.load(
        HISTORY + batch * HISTORY_STRIDE + position - 1,
        (row < BATCH * SEQ) & (position >= 1),
        other=EOS,
    )
    before_previous = tl.load(
        HISTORY + batch * HISTORY_STRIDE + position - 2,
        (row < BATCH * SEQ) & (position >= 2),
        other=EOS,
    )
    # A preceding EOS restarts both histories. If the second preceding token
    # is EOS itself, its value already equals the required reset padding.
    before_previous = tl.where(previous == EOS, EOS, before_previous)
    two = (current * tl.load(MULTIPLIERS)) ^ (previous * tl.load(MULTIPLIERS + 1))
    three = two ^ (before_previous * tl.load(MULTIPLIERS + 2))
    mixed = tl.where(head[None, :] < 8, two[:, None], three[:, None])
    vocab = tl.load(VOCAB + head)
    remainder = mixed % vocab[None, :]
    remainder = tl.where(remainder < 0, remainder + vocab[None, :], remainder)
    values = remainder + tl.load(OFFSETS + head)[None, :]
    tl.store(
        OUT + row[:, None] * 16 + head[None, :], values, row[:, None] < BATCH * SEQ
    )


def try_hashed_ngram_ids(
    history,
    seq_len,
    vocab_sizes,
    offsets,
    multipliers,
    *,
    ngram_size,
    heads_per_ngram,
    eos_token_id
):
    if not (
        history.is_cuda
        and torch.version.hip is None
        and history.dtype == torch.int64
        and history.ndim == 2
        and history.shape[0] > 0
        and history.stride(0) > 0
        and history.stride(1) == 1
        and isinstance(seq_len, int)
        and 0 < seq_len <= history.shape[1]
        and ngram_size == 3
        and heads_per_ngram == 8
        and vocab_sizes.shape == offsets.shape == (16,)
        and multipliers.shape == (3,)
        and all(
            t.device == history.device and t.dtype == torch.int64 and t.is_contiguous()
            for t in (vocab_sizes, offsets, multipliers)
        )
        and torch.cuda.get_device_capability(history.device) == (12, 0)
    ):
        return None
    out = torch.empty(
        (history.shape[0], seq_len, 16), device=history.device, dtype=torch.int64
    )
    _hash_ngram_ids[(triton.cdiv(history.shape[0] * seq_len, 64),)](
        history,
        vocab_sizes,
        offsets,
        multipliers,
        out,
        BATCH=history.shape[0],
        SEQ=seq_len,
        TOTAL=history.shape[1],
        HISTORY_STRIDE=history.stride(0),
        EOS=eos_token_id,
        BLOCK=64,
        num_warps=4,
    )
    return out

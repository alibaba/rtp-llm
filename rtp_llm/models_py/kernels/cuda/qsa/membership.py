"""Inline QSA membership construction shared by prefill kernels."""

import triton
import triton.language as tl


@triton.jit
def _find_request(Cu, row, B):
    low = 0
    high = B
    while low < high:
        mid = (low + high) // 2
        if row >= tl.load(Cu + mid + 1):
            low = mid + 1
        else:
            high = mid
    return low


@triton.jit
def _write_membership(Ids, Mask, row, S, WORDS, MAX_WORDS: tl.constexpr):
    word = tl.arange(0, MAX_WORDS)
    tl.store(Mask + row * WORDS + word, 0, word < WORDS)
    # Every row belongs to one CTA; finish initialization before any atomic OR.
    tl.debug_barrier()
    block = tl.arange(0, 512)
    start = tl.load(Ids + row * 2051 + block * 4)
    bits = tl.full((512,), 15, tl.uint32) << (start % 32)
    tl.atomic_or(
        Mask + row * WORDS + start // 32,
        bits,
        (start >= 0) & (start + 3 < S),
        sem="relaxed",
    )
    tail = tl.arange(0, 4)
    index = tl.load(Ids + row * 2051 + 2048 + tail, tail < 3, other=-1)
    tail_bits = tl.full((4,), 1, tl.uint32) << (index % 32)
    tl.atomic_or(
        Mask + row * WORDS + index // 32,
        tail_bits,
        (tail < 3) & (index >= 0) & (index < S),
        sem="relaxed",
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 causal subset of vLLM v0.29.0's 2D unified attention.

Retains grouped-query Tensor Core QK/PV, FP32 online-softmax accumulation,
and a single unsplit KV traversal. RTP supplies HND cache strides and absolute
positions. The packed path shares KV loads across queries in one request; the
indexed path handles arbitrary row order using the same arithmetic tiles.
Neither tile sizes nor the softmax partition depend on batch or execution phase.
See UPSTREAM.md for source provenance and the intentionally unsupported features.
"""

import triton
import triton.language as tl


@triton.jit
def _find_sequence(query_start, target, num_seqs, BLOCK_Q: tl.constexpr):
    # vLLM's find_seq_idx in query-block mode. An extra slot at each sequence
    # boundary avoids realizing ragged query lengths on the CPU.
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start + mid) // BLOCK_Q + mid
        if mid_val <= target:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _softmax_step(scores, maximum, total):
    # vLLM's softmax_step: keep the row-local reduction and recurrence together.
    next_max = tl.maximum(maximum, tl.max(scores, axis=1))
    next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
    probs = tl.exp(scores - next_max[:, None])
    tile_sum = tl.sum(probs, axis=1)
    correction = tl.exp(maximum - next_max)
    return next_max, total * correction + tile_sum, probs, correction


@triton.jit
def unified_attention_kernel(
    Q,
    CACHE,
    TABLE,
    REQUEST,
    POSITION,
    QUERY_START,
    OUT,
    NUM_SEQS,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    CS3: tl.constexpr,
    CS4: tl.constexpr,
    TS: tl.constexpr,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_SIZE: tl.constexpr = 32,
    PACKED: tl.constexpr = False,
):
    block, kv_head = tl.program_id(0), tl.program_id(1)
    m = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    t = tl.arange(0, TILE_SIZE)
    if PACKED:
        req = _find_sequence(QUERY_START, block, NUM_SEQS, BLOCK_Q)
        start = tl.load(QUERY_START + req)
        stop = tl.load(QUERY_START + req + 1)
        local_block = block - (start // BLOCK_Q + req)
        first = start + local_block * BLOCK_Q
        if first >= stop:
            return
        rows = first + m // GROUP
        row_mask = (rows < stop) & (m < BLOCK_Q * GROUP)
    else:
        req = tl.load(REQUEST + block)
        rows = tl.full((BLOCK_M,), block, tl.int32)
        row_mask = m < GROUP

    heads = kv_head * GROUP + m % GROUP
    positions = tl.load(POSITION + rows, row_mask, -1)
    q = tl.load(
        Q + rows[:, None] * QS0 + heads[:, None] * QS1 + d[None, :],
        row_mask[:, None],
        0.0,
    )
    maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    total = tl.full((BLOCK_M,), 1.0, tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    # A query tile may straddle a causal boundary. Extra iterations for an
    # earlier row contain zero probabilities and preserve its accumulator.
    end = tl.max(positions, 0) + 1
    for j in range(tl.cdiv(end, TILE_SIZE)):
        key_pos = j * TILE_SIZE + t
        valid = key_pos < end
        # Unlike upstream's padded block-table assumption, RTP only guarantees
        # the allocated width. Mask both the table and last-page KV loads.
        page = tl.load(TABLE + req * TS + key_pos // PAGE, valid, 0).to(tl.int64)
        k_offset = (
            page[None, :] * CS0
            + kv_head * CS2
            + (key_pos % PAGE)[None, :] * CS3
            + d[:, None] * CS4
        )
        v_offset = (
            page[:, None] * CS0
            + CS1
            + kv_head * CS2
            + (key_pos % PAGE)[:, None] * CS3
            + d[None, :] * CS4
        )
        k = tl.load(CACHE + k_offset, valid[None, :], 0.0)
        v = tl.load(CACHE + v_offset, valid[:, None], 0.0)
        scores = SCALE * tl.dot(q, k)
        scores = tl.where(
            row_mask[:, None] & (key_pos[None, :] <= positions[:, None]),
            scores,
            float("-inf"),
        )
        maximum, total, probs, correction = _softmax_step(scores, maximum, total)
        acc = acc * correction[:, None]
        acc += tl.dot(probs.to(v.dtype), v)
    acc = acc / total[:, None]
    tl.store(
        OUT + (rows[:, None] * H + heads[:, None]) * D + d[None, :],
        acc,
        row_mask[:, None],
    )

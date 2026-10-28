"""Experimental short-tail V vectorization with production merge mathematics.

The fast arm retains the original independent tail softmax and prefix merge.
Only shuffled V's load/reduction layout changes. Runtime short-tail guards and
fallback geometry match the separate common+V candidate. This file adds no
production dispatch and requires independent numerical/performance validation.
"""

import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.attention.beam_search_cascade import (
    _private_tail_merge,
)


@triton.jit
def _private_tail_merge_layout_only(
    Q,
    KV,
    TABLE,
    LENGTHS,
    CUTS,
    PART_OUT,
    PART_LSE,
    OUT,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    KV_PLANE: tl.constexpr,
    KV_HEAD: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    LINEAR_V: tl.constexpr,
    SPLITS: tl.constexpr,
    BEAMS: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    INDEX64: tl.constexpr,
    NATIVE_PREFIX: tl.constexpr = False,
    ROW_TILE: tl.constexpr = 1,
):
    SUPPORTED: tl.constexpr = (
        DIM == 128
        and PAGE == 16
        and G == 2
        and HEADS == KV_HEADS * G
        and ROW_TILE == 1
        and N == 8
        and not LINEAR_V
        and (not NATIVE_PREFIX or SPLITS == 1)
        and Q.dtype.element_ty == tl.bfloat16
        and KV.dtype.element_ty == tl.bfloat16
        and OUT.dtype.element_ty == tl.bfloat16
    )
    if SUPPORTED:
        row_id = tl.program_id(0)
        cut = tl.load(CUTS + row_id, row_id < ROWS, other=0)
        row_length = tl.load(LENGTHS + row_id, row_id < ROWS, other=0)
        use_short = (
            (row_id < ROWS)
            & (cut > 0)
            & (cut % PAGE == 0)
            & (row_length > cut)
            & (row_length - cut <= PAGE)
        )
    else:
        use_short = False

    if use_short:
        tile, head = tl.program_id(0), tl.program_id(1)
        X: tl.constexpr = 8
        M: tl.constexpr = ROW_TILE * G
        m = tl.arange(0, M)
        row = tile * ROW_TILE + m // G
        d = tl.arange(0, DIM)
        n = tl.arange(0, N)
        qheads = head * G + m % G
        q = tl.load(
            Q + row[:, None] * Q_ROW + qheads[:, None] * Q_HEAD + d[None, :],
            row[:, None] < ROWS,
            other=0,
        ).to(tl.float32)
        length = tl.load(LENGTHS + row, row < ROWS, other=0)
        begin = tl.load(CUTS + tile * ROW_TILE)
        maximum = tl.full((M,), -float("inf"), tl.float32)
        denom = tl.full((M,), 0, tl.float32)
        acc = tl.full((M, DIM), 0, tl.float32)
        for start in range(begin, tl.max(length, 0), N):
            tokens = start + n
            valid = (row[:, None] < ROWS) & (tokens[None, :] < length[:, None])
            block = tl.load(
                TABLE + row[:, None] * TABLE_ROW + tokens[None, :] // PAGE,
                valid,
                other=0,
            )
            if INDEX64:
                block = block.to(tl.int64)
            t = tokens % PAGE
            koff = (d[None, :] // X) * PAGE * X + t[:, None] * X + d[None, :] % X
            k = tl.load(
                KV + block[:, :, None] * KV_BLOCK + head * KV_HEAD + koff[None, :, :],
                valid[:, :, None],
                other=0,
            ).to(tl.float32)
            logits = tl.sum(q[:, None, :] * k, 2) * SCALE
            logits = tl.where(valid, logits, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(logits, 1))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe_maximum)
            p = tl.exp2(logits - safe_maximum[:, None])
            # The short-tail guard proves all eight lanes stay in one allocated page.
            # Keep the vector load unmasked by token; zero inactive values before PV.
            vblock = tl.load(
                TABLE + row * TABLE_ROW + begin // PAGE,
                row < ROWS,
                other=0,
            )
            if INDEX64:
                vblock = vblock.to(tl.int64)
            voff = ((start - begin) // X) * DIM * X + d[:, None] * X + n[None, :]
            v = tl.load(
                KV
                + vblock[:, None, None] * KV_BLOCK
                + KV_PLANE
                + head * KV_HEAD
                + voff[None, :, :],
                row[:, None, None] < ROWS,
                other=0,
            ).to(tl.float32)
            v = tl.where(valid[:, None, :], v, 0.0)
            acc = acc * alpha[:, None] + tl.sum(p[:, None, :] * v, 2)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = new_maximum
        safe_denom = tl.where(denom > 0, denom, 1.0)
        tail_lse = tl.where(denom > 0, maximum + tl.log2(safe_denom), -float("inf"))
        splits = tl.arange(0, SPLITS)
        positions = (splits[:, None] * ROWS + row[None, :]) * HEADS + qheads[None, :]
        if NATIVE_PREFIX:
            prefix_lse = (
                tl.load(
                    PART_LSE + qheads[None, :] * ROWS + row[None, :],
                    row[None, :] < ROWS,
                    other=-float("inf"),
                )
                * 1.4426950408889634
            )
            prefix_lse = tl.where(begin > 0, prefix_lse, -float("inf"))
        else:
            prefix_lse = tl.load(
                PART_LSE + positions, row[None, :] < ROWS, other=-float("inf")
            )
        merge_max = tl.maximum(tail_lse, tl.max(prefix_lse, 0))
        merge_max = tl.where(merge_max == -float("inf"), 0.0, merge_max)
        prefix_weight = tl.exp2(prefix_lse - merge_max[None, :])
        tail_weight = tl.exp2(tail_lse - merge_max)
        prefix_out = tl.load(
            PART_OUT + positions[:, :, None] * DIM + d[None, None, :],
            row[None, :, None] < ROWS,
            other=0,
        )
        prefix_out = tl.where(begin > 0, prefix_out, 0.0)
        merged = tl.sum(prefix_out * prefix_weight[:, :, None], 0)
        merged += acc / safe_denom[:, None] * tail_weight[:, None]
        merged /= tl.maximum(tl.sum(prefix_weight, 0) + tail_weight, 1e-30)[:, None]
        tl.store(
            OUT + (row[:, None] * HEADS + qheads[:, None]) * DIM + d[None, :],
            merged,
            row[:, None] < ROWS,
        )
    else:
        _private_tail_merge(
            Q,
            KV,
            TABLE,
            LENGTHS,
            CUTS,
            PART_OUT,
            PART_LSE,
            OUT,
            Q_ROW,
            Q_HEAD,
            KV_BLOCK,
            KV_PLANE,
            KV_HEAD,
            TABLE_ROW,
            ROWS,
            HEADS,
            KV_HEADS,
            DIM,
            PAGE,
            LINEAR_V,
            SPLITS,
            BEAMS,
            G,
            N,
            SCALE,
            INDEX64,
            NATIVE_PREFIX,
            ROW_TILE,
        )

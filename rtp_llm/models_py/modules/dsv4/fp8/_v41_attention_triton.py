"""SM120 joint sparse attention with planar FP8/FP4 cache gathering.

Uses the checkpoint reference's BF16 probability contraction and FP32 online
softmax. Codec formulas come from _v41_cache_triton (see its license file).
"""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import _unpack_fp4


@triton.jit
def _pool(
    q,
    acc,
    maximum,
    total,
    pool,
    indices,
    row,
    stride,
    pages,
    WIDTH: tl.constexpr,
    ENTRIES: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    columns = tl.arange(0, 512)
    for tile in range(tl.cdiv(WIDTH, 32)):
        offset = tile * 32 + tl.arange(0, 32)
        slot = tl.load(indices + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        tl.device_assert(
            (slot >= -1) & (slot < pages * ENTRIES), "V4.1 attention invalid slot"
        )
        valid = (slot >= 0) & (slot < pages * ENTRIES)
        base = (slot // ENTRIES) * stride.to(tl.int64)
        entry = slot % ENTRIES
        if GLOBAL:
            packed = tl.load(
                pool + base[:, None] + entry[:, None] * 256 + columns[None, :] // 2,
                valid[:, None],
                other=0,
            )
            codes = (packed.to(tl.int32) >> ((columns[None, :] % 2) * 4)) & 15
            values = _unpack_fp4(codes)
            scales = (
                tl.load(
                    pool
                    + base[:, None]
                    + ENTRIES * 256
                    + entry[:, None] * 32
                    + columns[None, :] // 16,
                    valid[:, None],
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
        else:
            values = (
                tl.load(
                    pool + base[:, None] + entry[:, None] * 512 + columns[None, :],
                    valid[:, None],
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            scale_bytes = tl.load(
                pool
                + base[:, None]
                + ENTRIES * 512
                + entry[:, None] * 16
                + columns[None, :] // 32,
                valid[:, None],
                other=127,
            )
            bits = scale_bytes.to(tl.uint32) << 23
            bits = tl.where(scale_bytes == 0, 0x00400000, bits)
            bits = tl.where(scale_bytes == 255, 0x7FC00000, bits)
            scales = bits.to(tl.uint32).to(tl.float32, bitcast=True)
        kv = (values * scales).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(kv)) * (512**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - finite_max)
        probabilities = tl.exp(scores - finite_max[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(probabilities.to(tl.bfloat16), kv, acc)
        total = total * alpha + tl.sum(probabilities, 1)
        maximum = next_max
    return acc, maximum, total


@triton.jit(
    do_not_specialize=["swa_stride", "swa_pages", "global_stride", "global_pages"]
)
def _attention(
    query,
    sinks,
    swa,
    swa_slots,
    valid_rows,
    global_pool,
    global_slots,
    out,
    swa_stride,
    swa_pages,
    global_stride,
    global_pages,
    HEADS: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    SWA_ENTRIES: tl.constexpr,
    GLOBAL_WIDTH: tl.constexpr,
    GLOBAL_ENTRIES: tl.constexpr,
    DIRECT_SINK_DIV_RN: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    addresses = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    if not tl.load(valid_rows + row):
        tl.store(out + addresses, 0, heads[:, None] < HEADS)
        return
    q = tl.load(query + addresses, heads[:, None] < HEADS, other=0)
    acc = tl.full((16, 512), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    acc, maximum, total = _pool(
        q,
        acc,
        maximum,
        total,
        swa,
        swa_slots,
        row,
        swa_stride,
        swa_pages,
        SWA_WIDTH,
        SWA_ENTRIES,
        False,
    )
    if GLOBAL_WIDTH:
        acc, maximum, total = _pool(
            q,
            acc,
            maximum,
            total,
            global_pool,
            global_slots,
            row,
            global_stride,
            global_pages,
            GLOBAL_WIDTH,
            GLOBAL_ENTRIES,
            True,
        )
    sink = tl.load(sinks + heads, heads < HEADS, other=0)
    if DIRECT_SINK_DIV_RN:
        has_keys = total > 0
        finite_max = tl.where(has_keys, maximum, 0.0)
        denominator = total + tl.exp(sink - finite_max)
        result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    else:
        next_max = tl.maximum(maximum, sink)
        alpha = tl.exp(maximum - next_max)
        denominator = total * alpha + tl.exp(sink - next_max)
        result = acc * (alpha / denominator)[:, None]
    tl.store(out + addresses, result, heads[:, None] < HEADS)


def run(query, sinks, swa, swa_slots, valid_rows, global_cache, global_slots, out):
    _attention[(query.shape[0], triton.cdiv(query.shape[1], 16))](
        query,
        sinks,
        swa.raw,
        swa_slots,
        valid_rows,
        global_cache.raw if global_cache is not None else swa.raw,
        global_slots if global_slots is not None else swa_slots,
        out,
        swa.raw.stride(0),
        swa.raw.shape[0],
        global_cache.raw.stride(0) if global_cache is not None else 0,
        global_cache.raw.shape[0] if global_cache is not None else 0,
        HEADS=query.shape[1],
        SWA_WIDTH=swa_slots.shape[1],
        SWA_ENTRIES=swa.layout.entries,
        GLOBAL_WIDTH=global_slots.shape[1] if global_slots is not None else 0,
        GLOBAL_ENTRIES=global_cache.layout.entries if global_cache is not None else 0,
        num_warps=4,
        num_stages=1,
    )

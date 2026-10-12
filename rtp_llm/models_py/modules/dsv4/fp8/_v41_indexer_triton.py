"""SM120 group-32 FP4 index queries and planar-key scoring."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import _pack_fp4, _unpack_fp4


@triton.jit
def _query_quant(query, valid_rows, out):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, 128)
    valid = tl.load(valid_rows + row)
    x = (
        tl.load(query + row * 4096 + head * 128 + cols, valid, other=0)
        .to(tl.float32)
        .reshape(4, 32)
    )
    maximum = tl.max(tl.abs(x), 1)
    scaled = tl.maximum(maximum, 6.0 * (2.0**-126)) * (1.0 / 6.0)
    bits = scaled.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    normalized = tl.clamp(tl.div_rn(x, scale[:, None]), -6.0, 6.0)
    packed = _pack_fp4(normalized, 128)
    codes = (tl.gather(packed, cols // 2, 0).to(tl.int32) >> ((cols % 2) * 4)) & 15
    decoded = _unpack_fp4(codes).reshape(4, 32) * scale[:, None]
    tl.store(out + row * 4096 + head * 128 + cols, decoded.reshape(128))


@triton.jit(do_not_specialize=["stride", "pages"])
def _score(
    query,
    weights,
    pool,
    page_table,
    positions,
    valid_rows,
    out,
    stride,
    pages,
    WIDTH: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    RATIO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.program_id(1) * 64 + tl.arange(0, 64)
    if not tl.load(valid_rows + row):
        tl.store(out + row * WIDTH + index, -float("inf"), index < WIDTH)
        return
    position = tl.load(positions + row).to(tl.int64)
    tl.device_assert(
        (position >= 0) & (position < WIDTH * RATIO),
        "V4.1 index position outside capacity",
    )
    visible = (position + 1) // RATIO
    active = (index < WIDTH) & (index < visible)
    page = tl.load(
        page_table + row * TABLE_COLS + index // (256 // RATIO), active, other=-1
    ).to(tl.int64)
    tl.device_assert(
        ~active | ((page >= 0) & (page < pages)), "V4.1 index page is not allocated"
    )
    active = active & (page >= 0) & (page < pages)
    entry = index % (256 // RATIO)
    cols = tl.arange(0, 128)
    base = page * stride.to(tl.int64)
    packed = tl.load(
        pool + base[:, None] + entry[:, None] * 64 + cols[None, :] // 2,
        active[:, None],
        other=0,
    )
    codes = (packed.to(tl.int32) >> ((cols[None, :] % 2) * 4)) & 15
    scale_bytes = tl.load(
        pool + base[:, None] + 256 * 64 + entry[:, None] * 4 + cols[None, :] // 32,
        active[:, None],
        other=127,
    )
    bits = scale_bytes.to(tl.uint32) << 23
    bits = tl.where(scale_bytes == 0, 0x00400000, bits)
    bits = tl.where(scale_bytes == 255, 0x7FC00000, bits)
    key = (_unpack_fp4(codes) * bits.to(tl.uint32).to(tl.float32, bitcast=True)).to(
        tl.bfloat16
    )
    heads = tl.arange(0, 32)
    q = tl.load(query + row * 4096 + heads[:, None] * 128 + cols[None, :])
    dot = tl.dot(q, tl.trans(key)).to(tl.bfloat16).to(tl.float32)
    weight = (
        (tl.load(weights + row * 32 + heads).to(tl.float32) * (1.0 / 64))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    weighted = (tl.maximum(dot, 0.0) * weight[:, None]).to(tl.bfloat16).to(tl.float32)
    score = tl.sum(weighted, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + row * WIDTH + index, tl.where(active, score, -float("inf")), index < WIDTH
    )


def score(query, weights, cache, page_table, positions, valid_rows, ratio, out):
    import torch

    quantized = torch.empty_like(query)
    _query_quant[(query.shape[0], 32)](query, valid_rows, quantized, num_warps=4)
    _score[(query.shape[0], triton.cdiv(out.shape[1], 64))](
        quantized,
        weights,
        cache.raw,
        page_table,
        positions,
        valid_rows,
        out,
        cache.raw.stride(0),
        cache.raw.shape[0],
        WIDTH=out.shape[1],
        TABLE_COLS=page_table.shape[1],
        RATIO=ratio,
        num_warps=4,
        num_stages=1,
    )

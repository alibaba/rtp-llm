"""V4.1 unsharded planar cache codecs and lossless row transport.

FP4 formulas adapted from DeepSeek-V4.1-Flash inference/kernel.py at
2bc89ac599031fa673cab993f1df02fc4a98c673 (Copyright (c) 2023 DeepSeek, MIT),
through RTP-LLM donor 88993fcb36838def206525a079145a823e879e8a.
SWA formulas adapted from vLLM deepseek_v41/common/ops/cache_utils.py
(Copyright contributors to the vLLM project, Apache-2.0), through that donor.
See _v41_cache_triton.LICENSE. No donor CP byte sharding is enabled here.
"""

import triton
import triton.language as tl


@triton.jit
def _pack_fp4(values, DIM: tl.constexpr):
    magnitude = tl.abs(values)
    codes = (magnitude > 0.25).to(tl.uint8)
    codes += (magnitude >= 0.75).to(tl.uint8)
    codes += (magnitude > 1.25).to(tl.uint8)
    codes += (magnitude >= 1.75).to(tl.uint8)
    codes += (magnitude > 2.5).to(tl.uint8)
    codes += (magnitude >= 3.5).to(tl.uint8)
    codes += (magnitude > 5.0).to(tl.uint8)
    codes |= ((values.to(tl.uint32, bitcast=True) >> 28) & 8).to(tl.uint8)
    even, odd = tl.split(codes.reshape((DIM // 2, 2)))
    return even | (odd << 4)


@triton.jit
def _unpack_fp4(codes):
    magnitude = codes & 7
    normal = tl.exp2((magnitude >> 1).to(tl.float32) - 1.0)
    normal *= 1.0 + (magnitude & 1).to(tl.float32) * 0.5
    value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5, normal)
    # Triton's unary minus is 0 - value, which loses the sign of negative zero.
    bits = value.to(tl.uint32, bitcast=True) | ((codes & 8).to(tl.uint32) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["block_stride", "pages"])
def _encode_kernel(
    values,
    slots,
    pool,
    block_stride,
    pages,
    ENTRIES: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    SWA: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    tl.device_assert((slot >= -1) & (slot < pages * ENTRIES), "V4.1 invalid cache slot")
    if slot < 0 or slot >= pages * ENTRIES:
        return
    x = tl.load(values + row * DIM + tl.arange(0, DIM)).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 0) == DIM,
        "V4.1 nonfinite codec input",
    )
    x = x.reshape((DIM // GROUP, GROUP))
    maximum = tl.max(tl.abs(x), 1)
    if SWA:
        exponent = tl.ceil(tl.log2(tl.maximum(maximum, 1e-4) * (1.0 / 448.0)))
        normalized = tl.clamp(x * tl.exp2(-exponent)[:, None], -448.0, 448.0)
        payload = (
            normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            .to(tl.uint8, bitcast=True)
            .reshape((DIM,))
        )
        scale_bytes = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
        PAYLOAD: tl.constexpr = DIM
    else:
        if GLOBAL:
            scale_fp8 = tl.minimum(
                tl.div_rn(tl.maximum(maximum, 6.0 * (2.0**-9)), 6.0), 448.0
            ).to(tl.float8e4nv, fp_downcast_rounding="rtne")
            scale = scale_fp8.to(tl.float32)
            scale_bytes = scale_fp8.to(tl.uint8, bitcast=True)
        else:
            scaled = tl.maximum(maximum, 6.0 * (2.0**-126)) * (1.0 / 6.0)
            bits = scaled.to(tl.uint32, bitcast=True)
            exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            scale_bytes = exponent.to(tl.uint8)
        normalized = tl.clamp(tl.div_rn(x, scale[:, None]), -6.0, 6.0)
        payload = _pack_fp4(normalized, DIM)
        PAYLOAD: tl.constexpr = DIM // 2
    page, offset = slot // ENTRIES, slot % ENTRIES
    base = pool + page * block_stride.to(tl.int64)
    tl.store(base + offset * PAYLOAD + tl.arange(0, PAYLOAD), payload)
    tl.store(
        base + ENTRIES * PAYLOAD + offset * (DIM // GROUP) + tl.arange(0, DIM // GROUP),
        scale_bytes,
    )


@triton.jit(do_not_specialize=["block_stride", "pages"])
def _decode_kernel(
    pool,
    slots,
    out,
    block_stride,
    pages,
    ENTRIES: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    SWA: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    columns = tl.arange(0, DIM)
    output = out + row * DIM + columns
    tl.device_assert((slot >= -1) & (slot < pages * ENTRIES), "V4.1 invalid cache slot")
    if slot < 0 or slot >= pages * ENTRIES:
        tl.store(output, tl.full((DIM,), 0, tl.float32))
        return
    page, offset = slot // ENTRIES, slot % ENTRIES
    base = pool + page * block_stride.to(tl.int64)
    if SWA:
        PAYLOAD: tl.constexpr = DIM
        values = (
            tl.load(base + offset * PAYLOAD + columns)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
    else:
        PAYLOAD: tl.constexpr = DIM // 2
        packed = tl.load(base + offset * PAYLOAD + columns // 2)
        codes = (packed.to(tl.int32) >> ((columns % 2) * 4)) & 15
        values = _unpack_fp4(codes)
    scale_bytes = tl.load(
        base + ENTRIES * PAYLOAD + offset * (DIM // GROUP) + columns // GROUP
    )
    if GLOBAL:
        scales = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        bits = scale_bytes.to(tl.uint32) << 23
        bits = tl.where(scale_bytes == 0, 0x00400000, bits)
        bits = tl.where(scale_bytes == 255, 0x7FC00000, bits)
        scales = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    tl.store(output, values * scales)


@triton.jit(do_not_specialize=["block_stride", "pages"])
def _transfer_kernel(
    pool,
    slots,
    rows,
    block_stride,
    pages,
    ENTRIES: tl.constexpr,
    PAYLOAD: tl.constexpr,
    SCALES: tl.constexpr,
    WIDTH: tl.constexpr,
    SCATTER: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    cols = tl.arange(0, WIDTH)
    row_ptr = rows + row * (PAYLOAD + SCALES) + cols
    mask = cols < PAYLOAD + SCALES
    tl.device_assert((slot >= -1) & (slot < pages * ENTRIES), "V4.1 invalid cache slot")
    if slot < 0 or slot >= pages * ENTRIES:
        if not SCATTER:
            tl.store(row_ptr, tl.full((WIDTH,), 0, tl.uint8), mask)
        return
    page, offset = slot // ENTRIES, slot % ENTRIES
    addresses = page * block_stride.to(tl.int64) + tl.where(
        cols < PAYLOAD,
        offset * PAYLOAD + cols,
        ENTRIES * PAYLOAD + offset * SCALES + cols - PAYLOAD,
    )
    if SCATTER:
        tl.store(pool + addresses, tl.load(row_ptr, mask, other=0), mask)
    else:
        tl.store(row_ptr, tl.load(pool + addresses, mask, other=0), mask)


def _geometry(cache):
    return dict(
        ENTRIES=cache.layout.entries,
        DIM=128 if cache.tag == "indexer_kv" else 512,
        GROUP=16 if cache.tag in ("csa_kv", "hca_kv") else 32,
        SWA=cache.tag == "swa_kv",
        GLOBAL=cache.tag in ("csa_kv", "hca_kv"),
    )


def encode_rows(cache, slots, values):
    _encode_kernel[(slots.numel(),)](
        values,
        slots,
        cache.raw,
        cache.raw.stride(0),
        cache.raw.shape[0],
        **_geometry(cache),
        num_warps=4,
    )


def decode_rows(cache, slots, out):
    _decode_kernel[(slots.numel(),)](
        cache.raw,
        slots,
        out,
        cache.raw.stride(0),
        cache.raw.shape[0],
        **_geometry(cache),
        num_warps=4,
    )


def transfer_rows(cache, slots, rows, *, scatter):
    layout = cache.layout
    _transfer_kernel[(slots.numel(),)](
        cache.raw,
        slots,
        rows,
        cache.raw.stride(0),
        cache.raw.shape[0],
        ENTRIES=layout.entries,
        PAYLOAD=layout.payload_bytes_per_entry,
        SCALES=layout.scale_bytes_per_entry,
        WIDTH=triton.next_power_of_2(
            layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
        ),
        SCATTER=scatter,
        num_warps=4,
    )

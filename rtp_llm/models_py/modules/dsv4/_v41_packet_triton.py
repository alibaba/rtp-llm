"""Byte-preserving V4.1 dispatch and MXFP8 return packet writers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(x):
    maximum = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-4)
    bits = (maximum * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (encoded << 23).to(tl.float32, bitcast=True)
    normalized = tl.clamp(tl.div_rn(x, scale[:, None]), -448.0, 448.0)
    payload = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(
        tl.uint8, bitcast=True
    )
    return payload, encoded


@triton.jit
def _dispatch(
    X,
    W,
    I,
    A,
    Out,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    k = tl.arange(0, BLOCK_K)
    active = tl.load(A + row * K + k, (row < ROWS) & (k < K), 0)
    live = tl.sum(active.to(tl.int32), 0) != 0
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * D + col, (row < ROWS) & live & (col < D), 0).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 1) == 32,
        "V4.1 activation quantization requires finite inputs",
    )
    values, encoded = _quantize(x)
    tl.store(Out + row * ROW_BYTES + col, values, col < D)
    tl.store(
        Out + row * ROW_BYTES + D + group,
        tl.where(live & (group < D // 32), encoded, 0).to(tl.uint8),
        group < SCALE_BYTES,
    )
    if tl.program_id(1) == 0:
        weights = tl.load(W + row * K + k, (row < ROWS) & (k < K), 0)
        ids = tl.load(I + row * K + k, (row < ROWS) & (k < K), -1)
        w_ptr = (Out + row * ROW_BYTES + D + SCALE_BYTES).to(
            tl.pointer_type(tl.float32)
        )
        i_ptr = (Out + row * ROW_BYTES + D + SCALE_BYTES + K * 4).to(
            tl.pointer_type(tl.int32)
        )
        tl.store(w_ptr + k, tl.where(active, weights, 0), k < K)
        tl.store(i_ptr + k, tl.where(active, ids, -1).to(tl.int32), k < K)


@triton.jit
def _return(
    X,
    Out,
    D: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * XS0 + col * XS1, col < D, 0).to(tl.bfloat16).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 1) == 32,
        "V4.1 activation quantization requires finite inputs",
    )
    values, encoded = _quantize(x)
    tl.store(Out + row * ROW_BYTES + col, values, col < D)
    tl.store(Out + row * ROW_BYTES + D + group, encoded.to(tl.uint8), group < D // 32)


def is_supported(x):
    return x.is_cuda and torch.cuda.get_device_capability(x.device) == (12, 0)


def write_dispatch(x, weights, indices, active, layout, output):
    """Inputs and capacity must already have passed pack_dispatch validation."""
    if output.shape[0]:
        _dispatch[(output.shape[0], triton.cdiv(layout.hidden // 32, 8))](
            x,
            weights,
            indices,
            active,
            output,
            x.shape[0],
            layout.hidden,
            layout.topk,
            layout.packed_scales * 4,
            layout.payload_bytes,
            triton.next_power_of_2(layout.topk),
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def write_return(partial, layout):
    output = torch.empty(
        (partial.shape[0], layout.payload_bytes),
        dtype=torch.uint8,
        device=partial.device,
    )
    if partial.shape[0]:
        _return[(partial.shape[0], triton.cdiv(layout.hidden // 32, 8))](
            partial,
            output,
            layout.hidden,
            partial.stride(0),
            partial.stride(1),
            layout.payload_bytes,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output

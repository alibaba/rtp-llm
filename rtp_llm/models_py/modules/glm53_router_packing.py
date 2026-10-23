"""Fused BF16-to-FP32 router packing and result scatter for zigzag rows."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=256)
def row_maps(device, stream, thread, rows, fragments):
    mapping = [-1] * rows
    for local, offset, count in fragments:
        mapping[offset:offset + count] = range(local, local + count)
    return torch.tensor(mapping, dtype=torch.int32, device=device)


@triton.jit
def _pack(X, MAP, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr,
          ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
          BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = x < ROWS * WIDTH
    row = tl.load(MAP + x // WIDTH, valid, -1).to(tl.int64)
    value = tl.load(X + row * ROW_STRIDE + x % WIDTH * COL_STRIDE,
                    valid & (row >= 0), 0)
    tl.store(OUT + x, value.to(tl.float32), valid)


@triton.jit
def _scatter(X, MAP, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr,
             BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = x < ROWS * WIDTH
    row = tl.load(MAP + x // WIDTH, valid, -1).to(tl.int64)
    value = tl.load(X + x, valid & (row >= 0), 0)
    tl.store(OUT + row * WIDTH + x % WIDTH, value, valid & (row >= 0))


def pack_rows(x, mapping, out):
    _pack[(triton.cdiv(out.numel(), 1024),)](
        x, mapping, out, ROWS=out.shape[0], WIDTH=out.shape[1],
        ROW_STRIDE=x.stride(0), COL_STRIDE=x.stride(1), BLOCK=1024)


def scatter_rows(x, mapping, out):
    _scatter[(triton.cdiv(x.numel(), 512),)](
        x, mapping, out, ROWS=x.shape[0], WIDTH=x.shape[1], BLOCK=512)

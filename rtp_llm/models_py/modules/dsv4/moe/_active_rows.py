"""Device-only capacity adapter using trusted current-call route metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def active_row_capacity(n: int, topk: int, experts: int) -> tuple[int, int]:
    """A=sum(ceil4(count_i)) <= N*K+3E; bound int32 prefix/scale arithmetic."""
    if any(type(v) is not int for v in (n, topk, experts)):
        raise TypeError("active rows requires concrete Python dimensions")
    if n < 0 or not 1 <= topk <= 1024 or not 1 <= experts <= 1024:
        raise ValueError("invalid active-row dimensions / histogram launch bounds")
    capacity = ((n * topk + 3 * experts + 3) // 4) * 4
    if capacity + 127 * experts > 2**31 - 1:
        raise ValueError("active-row prefix/scale capacity exceeds int32")
    return capacity, ((capacity + 127 * experts) // 128) * 128


def _tensor_meta(t, shape, dtypes, device, name):
    if tuple(t.shape) != tuple(shape) or t.dtype not in dtypes:
        raise ValueError(name + " shape/dtype mismatch")
    if t.device != device or t.device.type != "cuda" or not t.is_contiguous():
        raise ValueError(name + " must be contiguous on the input CUDA device")


def validate_active_inputs(owner, x, weights, indices, input_scale):
    """Host metadata only. Deliberately narrow to the existing eager FI ABI."""
    if x.dim() != 2 or indices.dim() != 2:
        raise ValueError("active rows expects 2-D input/routes")
    n, d = x.shape
    e, inter = owner.cfg.n_routed_experts, owner.cfg.moe_inter_dim
    k = indices.shape[1]
    capacity, sf_rows = active_row_capacity(n, k, e)
    if any(type(v) is not int or v <= 0 or v % 128 for v in (d, inter)):
        raise ValueError(
            "FI input/intermediate dimensions must be positive multiples of 128"
        )
    if max(d, 2 * inter) > 2**31 - 1:
        raise ValueError("FI problem dimensions exceed int32")
    # Byte-address bounds (not an allocation availability claim).
    if (
        max(
            capacity * max(d, 2 * inter) * 2,
            sf_rows * max(d, inter) // 32,
            e * 2 * inter * d,
            n * k * 8,
        )
        > 2**63 - 1
    ):
        raise ValueError("active-row storage offsets exceed int64")
    dev = x.device
    _tensor_meta(
        x,
        (n, d),
        (torch.bfloat16,) if input_scale is None else (torch.float8_e4m3fn,),
        dev,
        "input",
    )
    _tensor_meta(weights, (n, k), (torch.float32,), dev, "router weights")
    _tensor_meta(indices, (n, k), (torch.int32, torch.int64), dev, "routes")
    if input_scale is not None:
        _tensor_meta(input_scale, (n, d // 32), (torch.uint8,), dev, "input scale")
    _tensor_meta(
        owner._w13, (e, 2 * inter, d // 2), (torch.int8, torch.uint8), dev, "w13"
    )
    _tensor_meta(owner._w2, (e, d, inter // 2), (torch.int8, torch.uint8), dev, "w2")
    _tensor_meta(owner._s13_sm120, (e, 2 * inter, d // 32), (torch.uint8,), dev, "s13")
    _tensor_meta(owner._s2_sm120, (e, d, inter // 32), (torch.uint8,), dev, "s2")
    return capacity, sf_rows


@triton.jit
def _pack_active_scale_kernel(
    linear,
    indptr,
    m_indices,
    padded,
    C: tl.constexpr,
    E: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Do not read capacity-only IDs or linear scale bytes, even speculatively
    # via an unmasked tl.load. Empty groups add no scale-copy work.
    active = (row < C) & (row < tl.load(indptr + E))
    expert = tl.load(m_indices + row, mask=active, other=0).to(tl.int64)
    start = tl.load(indptr + expert, mask=active, other=0).to(tl.int64)
    sf_start = ((start + expert * 127) // 128) * 128
    dest = sf_start + row - start
    mask = active & (cols < COLS)
    values = tl.load(linear + row * COLS + cols, mask=mask, other=0)
    tl.store(padded + dest * COLS + cols, values, mask=mask)


def pack_active_scale(linear, indptr, m_indices, sf_rows):
    """Pack zero-padded scales using producer-validated aligned prefixes and row indices."""
    if linear.dim() != 2 or indptr.dim() != 1:
        raise ValueError("scale/prefix ranks")
    c, cols = linear.shape
    e = indptr.shape[0] - 1
    if (
        type(sf_rows) is not int
        or type(c) is not int
        or c <= 0
        or c % 4
        or not 1 <= e <= 1024
        or cols <= 0
        or cols % 4
        or c + 127 * e > 2**31 - 1
        or sf_rows != ((c + 127 * e) // 128) * 128
    ):
        raise ValueError("scale capacity/layout bounds")
    _tensor_meta(linear, (c, cols), (torch.uint8,), linear.device, "linear scale")
    _tensor_meta(indptr, (e + 1,), (torch.int32,), linear.device, "indptr")
    _tensor_meta(
        m_indices,
        (((c + 127) // 128) * 128,),
        (torch.int32,),
        linear.device,
        "m_indices",
    )
    padded = torch.zeros((sf_rows, cols), dtype=torch.uint8, device=linear.device)
    _pack_active_scale_kernel[(c, triton.cdiv(cols, 128))](
        linear,
        indptr,
        m_indices,
        padded,
        C=c,
        E=e,
        COLS=cols,
        BLOCK=128,
    )
    return padded


@triton.jit
def _pack_active_scale_interleaved_kernel(
    linear,
    indptr,
    m_indices,
    swizzled,
    C: tl.constexpr,
    E: tl.constexpr,
    COLS: tl.constexpr,
    NUM_SF_TILES_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse active-row scatter with FI's 128x4 scale interleave."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = (row < C) & (row < tl.load(indptr + E))
    expert = tl.load(m_indices + row, mask=active, other=0).to(tl.int64)
    start = tl.load(indptr + expert, mask=active, other=0).to(tl.int64)
    sf_start = ((start + expert * 127) // 128) * 128
    dest = sf_start + row - start
    mask = active & (cols < COLS)
    values = tl.load(linear + row * COLS + cols, mask=mask, other=0)
    # TensorRT-LLM/FlashInfer get_sf_out_offset_128x4 for one batch:
    # tile=((m//128)*ceil(K/4)+k//4)*512, then 32x4 row subgroup.
    offset = (
        ((dest // 128) * NUM_SF_TILES_K + cols // 4) * 512
        + (dest % 32) * 16
        + ((dest % 128) // 32) * 4
        + cols % 4
    )
    tl.store(swizzled + offset, values, mask=mask)


def pack_active_scale_interleaved(linear, indptr, m_indices, sf_rows):
    """Fuse active-scale packing and FI 128x4 interleave; zero all inactive/padded bytes."""
    if linear.dim() != 2 or indptr.dim() != 1:
        raise ValueError("scale/prefix ranks")
    c, cols = linear.shape
    e = indptr.shape[0] - 1
    if (
        type(sf_rows) is not int
        or type(c) is not int
        or c <= 0
        or c % 4
        or not 1 <= e <= 1024
        or cols <= 0
        or cols % 4
        or c + 127 * e > 2**31 - 1
        or sf_rows != ((c + 127 * e) // 128) * 128
    ):
        raise ValueError("interleaved scale capacity/layout bounds")
    _tensor_meta(linear, (c, cols), (torch.uint8,), linear.device, "linear scale")
    _tensor_meta(indptr, (e + 1,), (torch.int32,), linear.device, "indptr")
    _tensor_meta(
        m_indices,
        (((c + 127) // 128) * 128,),
        (torch.int32,),
        linear.device,
        "m_indices",
    )
    swizzled = torch.zeros((sf_rows, cols), dtype=torch.uint8, device=linear.device)
    _pack_active_scale_interleaved_kernel[(c, triton.cdiv(cols, 128))](
        linear,
        indptr,
        m_indices,
        swizzled,
        C=c,
        E=e,
        COLS=cols,
        NUM_SF_TILES_K=(cols + 3) // 4,
        BLOCK=128,
    )
    return swizzled

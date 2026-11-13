# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Adapted from vLLM 52373885168d70886d0a047477a33f65f06b5dc0,
# third_party/flash_linear_attention/ops/fused_recurrent.py.
# The original FLA implementation is MIT licensed.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.gdn_gating import _gdn_gate_values
from rtp_llm.models_py.triton_kernels.fla.op import exp
from rtp_llm.models_py.utils.device_capability import cached_device_capability


@triton.jit
def _packed_decode(
    qkv,
    a,
    b,
    alog,
    bias,
    states,
    block_map,
    lengths,
    output,
    qkv_stride: tl.constexpr,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    state_stride: tl.constexpr,
    map_stride: tl.constexpr,
    HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    VALUE_TILE: tl.constexpr,
):
    value_tile, row_head = tl.program_id(0), tl.program_id(1)
    row, value_head = row_head // VALUE_HEADS, row_head % VALUE_HEADS
    head = value_head // (VALUE_HEADS // HEADS)
    feature = tl.arange(0, DIM)
    value_row = value_tile * VALUE_TILE + tl.arange(0, VALUE_TILE)
    output_ptr = output + (row * VALUE_HEADS + value_head) * DIM + value_row
    length = tl.load(lengths + row).to(tl.int64)
    if length <= 0:
        tl.store(output_ptr, 0.0)
        return

    # A page boundary may require separate source and destination state pages.
    source_page = tl.load(block_map + row * map_stride + (length - 2) // PAGE_SIZE)
    if source_page <= 0:
        tl.store(output_ptr, 0.0)
        return
    state_offsets = value_head * DIM * DIM + value_row[:, None] * DIM + feature[None, :]
    state = tl.load(
        states + source_page.to(tl.int64) * state_stride + state_offsets
    ).to(tl.float32)
    packed = qkv + row * qkv_stride
    query = tl.load(packed + head * DIM + feature).to(tl.float32)
    key = tl.load(packed + (HEADS + head) * DIM + feature).to(tl.float32)
    value = tl.load(packed + 2 * HEADS * DIM + value_head * DIM + value_row).to(
        tl.float32
    )
    query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
    key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query = query * (DIM**-0.5)

    gate, beta = _gdn_gate_values(
        tl.load(alog + value_head),
        tl.load(a + row * a_stride + value_head),
        tl.load(b + row * b_stride + value_head),
        tl.load(bias + value_head),
        1.0,
        20.0,
    )
    # Match the existing CUDA gate's BF16 materialization before recurrence.
    beta = beta.to(b.dtype.element_ty).to(tl.float32)
    state *= exp(gate)
    value -= tl.sum(state * key[None, :], 1)
    value *= beta
    state += value[:, None] * key[None, :]
    result = tl.sum(state * query[None, :], 1)
    tl.store(output_ptr, result.to(output.dtype.element_ty))
    destination_page = tl.load(block_map + row * map_stride + (length - 1) // PAGE_SIZE)
    tl.store(
        states + destination_page.to(tl.int64) * state_stride + state_offsets,
        state.to(states.dtype.element_ty),
    )


def _run_packed_decode(
    qkv,
    a,
    b,
    alog,
    bias,
    states,
    block_map,
    lengths,
    page_size,
    value_tile=8,
    num_warps=1,
):
    batch, features = qkv.shape
    value_heads, dimension = states.shape[1], states.shape[-1]
    heads = (features - value_heads * dimension) // (2 * dimension)
    output = torch.empty(
        (batch, 1, value_heads, dimension), device=qkv.device, dtype=qkv.dtype
    )
    _packed_decode[(triton.cdiv(dimension, value_tile), batch * value_heads)](
        qkv,
        a,
        b,
        alog,
        bias,
        states,
        block_map,
        lengths,
        output,
        qkv.stride(0),
        a.stride(0),
        b.stride(0),
        states.stride(0),
        block_map.stride(0),
        heads,
        value_heads,
        dimension,
        page_size,
        value_tile,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


def try_packed_decode(qkv, a, b, alog, bias, states, block_map, lengths, page_size):
    """Single-token TP2 decode. Lengths follow the original 0-padding/2+ contract."""
    if qkv.ndim != 2 or block_map is None or lengths is None:
        return None
    batch = qkv.shape[0]
    if not (
        qkv.is_cuda
        and cached_device_capability(qkv.device) == (9, 0)
        and qkv.shape[1] == 5120
        and 0 < batch <= 128
        and page_size in (64, 128)
        and qkv.dtype == a.dtype == b.dtype == torch.bfloat16
        and states.dtype == torch.float32
        and states.shape[1:] == (24, 128, 128)
        and states.stride()[1:] == (16384, 128, 1)
        and qkv.is_contiguous()
        and a.shape == b.shape == (batch, 24)
        and a.stride(1) == b.stride(1) == 1
        and alog.shape == bias.shape == (24,)
        and alog.is_contiguous()
        and bias.is_contiguous()
        and block_map.ndim == 2
        and block_map.shape[0] == batch
        and block_map.stride(1) == 1
        and block_map.dtype in (torch.int32, torch.int64)
        and lengths.shape == (batch,)
        and lengths.is_contiguous()
        and lengths.dtype in (torch.int32, torch.int64)
        and all(
            tensor.device == qkv.device
            for tensor in (a, b, alog, bias, states, block_map, lengths)
        )
        and not any(tensor.requires_grad for tensor in (qkv, a, b, alog, bias))
    ):
        return None
    return _run_packed_decode(
        qkv,
        a,
        b,
        alog,
        bias,
        states,
        block_map,
        lengths,
        page_size,
        value_tile=4 if batch in (16, 32) else 8,
    )

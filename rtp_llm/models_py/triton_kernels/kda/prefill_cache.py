"""Gather and publish paged KDA prefill recurrent states."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "cache_blocks",
        "stride_ids",
        "stride_c_b",
        "stride_c_h",
        "stride_c_k",
        "stride_c_v",
        "stride_s_b",
        "stride_s_h",
        "stride_s_k",
        "stride_s_v",
    ]
)
def _copy_kda_prefill_states_kernel(
    cache,
    block_ids,
    states,
    cache_blocks,
    stride_ids,
    stride_c_b,
    stride_c_h,
    stride_c_k,
    stride_c_v,
    stride_s_b,
    stride_s_h,
    stride_s_k,
    stride_s_v,
    HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid_element = offsets < HEADS * KEY_DIM * VALUE_DIM
    head = offsets // (KEY_DIM * VALUE_DIM)
    key = offsets // VALUE_DIM % KEY_DIM
    value = offsets % VALUE_DIM
    block_id = tl.load(block_ids + row * stride_ids).to(tl.int64)
    valid_block = (block_id > 0) & (block_id < cache_blocks)
    cache_block = tl.where(valid_block, block_id, 0)
    cache_offsets = (
        cache_block * stride_c_b
        + head * stride_c_h
        + key * stride_c_k
        + value * stride_c_v
    )
    state_offsets = (
        row * stride_s_b + head * stride_s_h + key * stride_s_k + value * stride_s_v
    )
    if GATHER:
        values = tl.load(
            cache + cache_offsets, mask=valid_element & valid_block, other=0
        )
        tl.store(states + state_offsets, values, mask=valid_element)
    else:
        values = tl.load(states + state_offsets, mask=valid_element, other=0)
        tl.store(cache + cache_offsets, values, mask=valid_element & valid_block)


def _copy_kda_prefill_states(cache, block_ids, states, *, gather):
    if cache.ndim != 4 or states.ndim != 4 or block_ids.ndim != 1:
        raise ValueError("KDA prefill cache/states must be 4D and block IDs must be 1D")
    if tuple(cache.shape[1:]) != tuple(states.shape[1:]) or (
        states.shape[0] != block_ids.numel()
    ):
        raise ValueError("KDA prefill state shapes must match cache and block IDs")
    if cache.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("KDA prefill cache must be FP32, BF16 or FP16")
    if states.dtype != torch.float32 or block_ids.dtype != torch.int64:
        raise ValueError("KDA prefill states must be FP32 and block IDs must be INT64")
    if not cache.is_cuda or any(
        tensor.device != cache.device for tensor in (block_ids, states)
    ):
        raise ValueError(
            "KDA prefill cache, block IDs and states must share a CUDA device"
        )
    heads, key_dim, value_dim = cache.shape[1:]
    if min(heads, key_dim, value_dim) <= 0:
        raise ValueError("KDA prefill state dimensions must be positive")
    if block_ids.numel() == 0:
        return
    block = 256
    _copy_kda_prefill_states_kernel[
        (block_ids.numel(), triton.cdiv(heads * key_dim * value_dim, block))
    ](
        cache,
        block_ids,
        states,
        cache.shape[0],
        block_ids.stride(0),
        *cache.stride(),
        *states.stride(),
        HEADS=heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        GATHER=gather,
        BLOCK=block,
        num_warps=4,
    )


def load_kda_prefill_states(cache, block_ids, output):
    """Gather cache rows into FP32 ``output[rows,H,K,V]``; invalid IDs become zero."""
    _copy_kda_prefill_states(cache, block_ids, output, gather=True)


def store_kda_prefill_states(checkpoints, block_ids, cache):
    """Publish FP32 ``checkpoints[rows,H,K,V]`` into positive, in-range cache rows."""
    _copy_kda_prefill_states(cache, block_ids, checkpoints, gather=False)

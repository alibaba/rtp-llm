"""Kimi K3 linear-cache state storage kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_cache_store_kernel(
    recurrent,
    q_state,
    k_state,
    v_state,
    ssm_cache,
    conv_cache,
    stride_r_h,
    stride_r_v,
    stride_r_k,
    stride_q_c,
    stride_q_w,
    stride_k_c,
    stride_k_w,
    stride_v_c,
    stride_v_w,
    stride_ssm_h,
    stride_ssm_v,
    stride_ssm_k,
    stride_conv_w,
    stride_conv_c,
    heads: tl.constexpr,
    state_dim: tl.constexpr,
    conv_channels: tl.constexpr,
    history_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)

    ssm_elements = heads * state_dim * state_dim
    ssm_mask = offsets < ssm_elements
    state_k = offsets % state_dim
    state_v = (offsets // state_dim) % state_dim
    state_h = offsets // (state_dim * state_dim)
    state_value = tl.load(
        recurrent + state_h * stride_r_h + state_v * stride_r_v + state_k * stride_r_k,
        mask=ssm_mask,
        other=0.0,
    )
    tl.store(
        ssm_cache
        + state_h * stride_ssm_h
        + state_v * stride_ssm_v
        + state_k * stride_ssm_k,
        state_value,
        mask=ssm_mask,
    )

    packed_channels = 3 * conv_channels
    conv_elements = history_size * packed_channels
    conv_mask = offsets < conv_elements
    history = offsets // packed_channels
    packed_channel = offsets % packed_channels
    channel = packed_channel % conv_channels
    q_mask = conv_mask & (packed_channel < conv_channels)
    k_mask = (
        conv_mask
        & (packed_channel >= conv_channels)
        & (packed_channel < 2 * conv_channels)
    )
    v_mask = conv_mask & (packed_channel >= 2 * conv_channels)
    conv_value = tl.load(
        q_state + channel * stride_q_c + history * stride_q_w,
        mask=q_mask,
        other=0.0,
    )
    conv_value += tl.load(
        k_state + channel * stride_k_c + history * stride_k_w,
        mask=k_mask,
        other=0.0,
    )
    conv_value += tl.load(
        v_state + channel * stride_v_c + history * stride_v_w,
        mask=v_mask,
        other=0.0,
    )
    tl.store(
        conv_cache + history * stride_conv_w + packed_channel * stride_conv_c,
        conv_value,
        mask=conv_mask,
    )


@triton.jit
def _linear_cache_store_batched_kernel(
    recurrent,
    q_state,
    k_state,
    v_state,
    block_ids,
    ssm_cache,
    conv_cache,
    stride_r_n,
    stride_r_h,
    stride_r_v,
    stride_r_k,
    stride_q_n,
    stride_q_c,
    stride_q_w,
    stride_k_n,
    stride_k_c,
    stride_k_w,
    stride_v_n,
    stride_v_c,
    stride_v_w,
    stride_ssm_b,
    stride_ssm_h,
    stride_ssm_v,
    stride_ssm_k,
    stride_conv_b,
    stride_conv_w,
    stride_conv_c,
    heads: tl.constexpr,
    state_dim: tl.constexpr,
    conv_channels: tl.constexpr,
    history_size: tl.constexpr,
    block: tl.constexpr,
):
    state_index = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    cache_block = tl.load(block_ids + state_index)
    valid_block = cache_block >= 0

    ssm_elements = heads * state_dim * state_dim
    ssm_mask = valid_block & (offsets < ssm_elements)
    state_k = offsets % state_dim
    state_v = (offsets // state_dim) % state_dim
    state_h = offsets // (state_dim * state_dim)
    state_value = tl.load(
        recurrent
        + state_index * stride_r_n
        + state_h * stride_r_h
        + state_v * stride_r_v
        + state_k * stride_r_k,
        mask=ssm_mask,
        other=0.0,
    )
    tl.store(
        ssm_cache
        + cache_block * stride_ssm_b
        + state_h * stride_ssm_h
        + state_v * stride_ssm_v
        + state_k * stride_ssm_k,
        state_value,
        mask=ssm_mask,
    )

    packed_channels = 3 * conv_channels
    conv_elements = history_size * packed_channels
    conv_mask = valid_block & (offsets < conv_elements)
    history = offsets // packed_channels
    packed_channel = offsets % packed_channels
    channel = packed_channel % conv_channels
    q_mask = conv_mask & (packed_channel < conv_channels)
    k_mask = (
        conv_mask
        & (packed_channel >= conv_channels)
        & (packed_channel < 2 * conv_channels)
    )
    v_mask = conv_mask & (packed_channel >= 2 * conv_channels)
    conv_value = tl.load(
        q_state
        + state_index * stride_q_n
        + channel * stride_q_c
        + history * stride_q_w,
        mask=q_mask,
        other=0.0,
    )
    conv_value += tl.load(
        k_state
        + state_index * stride_k_n
        + channel * stride_k_c
        + history * stride_k_w,
        mask=k_mask,
        other=0.0,
    )
    conv_value += tl.load(
        v_state
        + state_index * stride_v_n
        + channel * stride_v_c
        + history * stride_v_w,
        mask=v_mask,
        other=0.0,
    )
    tl.store(
        conv_cache
        + cache_block * stride_conv_b
        + history * stride_conv_w
        + packed_channel * stride_conv_c,
        conv_value,
        mask=conv_mask,
    )


@torch.compiler.disable
def kimi_k3_store_linear_cache_state(
    recurrent: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    ssm_cache: torch.Tensor,
    conv_cache: torch.Tensor,
) -> None:
    """Store one backend-native KDA state without eager cat/transpose/copy ops."""

    if (
        recurrent.ndim != 3
        or q_state.ndim != 2
        or k_state.shape != q_state.shape
        or v_state.shape != q_state.shape
        or ssm_cache.shape != recurrent.shape
    ):
        raise ValueError("invalid K3 linear-cache state shapes")
    if conv_cache.shape != (q_state.shape[1], 3 * q_state.shape[0]):
        raise ValueError(
            "invalid K3 packed conv-cache shape: "
            f"cache={tuple(conv_cache.shape)} state={tuple(q_state.shape)}"
        )
    if not recurrent.is_cuda or not q_state.is_cuda:
        raise ValueError("K3 fused linear-cache store requires CUDA tensors")
    total = max(recurrent.numel(), conv_cache.numel())
    block = 256
    _linear_cache_store_kernel[(triton.cdiv(total, block),)](
        recurrent,
        q_state,
        k_state,
        v_state,
        ssm_cache,
        conv_cache,
        recurrent.stride(0),
        recurrent.stride(1),
        recurrent.stride(2),
        q_state.stride(0),
        q_state.stride(1),
        k_state.stride(0),
        k_state.stride(1),
        v_state.stride(0),
        v_state.stride(1),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        conv_cache.stride(0),
        conv_cache.stride(1),
        heads=recurrent.shape[0],
        state_dim=recurrent.shape[1],
        conv_channels=q_state.shape[0],
        history_size=q_state.shape[1],
        block=block,
        num_warps=4,
    )


@torch.compiler.disable
def kimi_k3_store_linear_cache_states(
    recurrent: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    block_ids: torch.Tensor,
    ssm_cache: torch.Tensor,
    conv_cache: torch.Tensor,
) -> None:
    """Store packed KDA checkpoints into arbitrary physical cache blocks."""

    if (
        recurrent.ndim != 4
        or q_state.ndim != 3
        or k_state.shape != q_state.shape
        or v_state.shape != q_state.shape
        or recurrent.shape[0] != q_state.shape[0]
        or block_ids.ndim != 1
        or block_ids.numel() != recurrent.shape[0]
        or ssm_cache.ndim != 4
        or tuple(ssm_cache.shape[1:]) != tuple(recurrent.shape[1:])
    ):
        raise ValueError("invalid packed K3 linear-cache state shapes")
    expected_conv_shape = (q_state.shape[2], 3 * q_state.shape[1])
    if conv_cache.ndim != 3 or tuple(conv_cache.shape[1:]) != expected_conv_shape:
        raise ValueError(
            "invalid packed K3 conv-cache shape: "
            f"cache={tuple(conv_cache.shape)} state={tuple(q_state.shape)}"
        )
    if block_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"K3 cache block IDs must be int32/int64, got {block_ids.dtype}"
        )
    tensors = (recurrent, q_state, k_state, v_state, block_ids, ssm_cache, conv_cache)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("K3 packed linear-cache store requires CUDA tensors")
    if recurrent.shape[0] == 0:
        return
    total = max(recurrent[0].numel(), conv_cache[0].numel())
    block = 256
    _linear_cache_store_batched_kernel[
        (recurrent.shape[0], triton.cdiv(total, block))
    ](
        recurrent,
        q_state,
        k_state,
        v_state,
        block_ids,
        ssm_cache,
        conv_cache,
        recurrent.stride(0),
        recurrent.stride(1),
        recurrent.stride(2),
        recurrent.stride(3),
        q_state.stride(0),
        q_state.stride(1),
        q_state.stride(2),
        k_state.stride(0),
        k_state.stride(1),
        k_state.stride(2),
        v_state.stride(0),
        v_state.stride(1),
        v_state.stride(2),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        ssm_cache.stride(3),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        heads=recurrent.shape[1],
        state_dim=recurrent.shape[2],
        conv_channels=q_state.shape[1],
        history_size=q_state.shape[2],
        block=block,
        num_warps=4,
    )


__all__ = [
    "kimi_k3_store_linear_cache_state",
    "kimi_k3_store_linear_cache_states",
]

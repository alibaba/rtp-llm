# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# Licensed under the Apache License, Version 2.0
"""Expert scatter and ordered gather for DSv4 on SM120."""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import _fwd_kernel_ep_scatter_1


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    output_capacity,
    topk_num: tl.constexpr,
    num_experts: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    index_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = index_in_s < SCALE_HIDDEN_SIZE
    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(
            recv_x + token_id * recv_x_stride0 + offset_in * recv_x_stride1,
            mask=mask,
        )
        to_copy_s = tl.load(
            recv_x_scale
            + token_id * recv_x_scale_stride0
            + index_in_s * recv_x_scale_stride1,
            mask=mask_s,
        )
        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk
                + token_id * recv_topk_stride0
                + topk_index * recv_topk_stride1
            )
            if expert_id >= 0 and expert_id < num_experts:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                dest_is_valid = dest_token_index < output_capacity
                tl.store(
                    output_index
                    + token_id * output_index_stride0
                    + topk_index * output_index_stride1,
                    tl.where(dest_is_valid, dest_token_index_int32, -1),
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(
                    output_tensor_ptr + offset_in * output_tensor_stride1,
                    to_copy,
                    mask=mask & dest_is_valid,
                )
                tl.store(
                    output_tensor_scale_ptr + index_in_s * output_tensor_scale_stride1,
                    to_copy_s,
                    mask=mask_s & dest_is_valid,
                )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    align_m: int = 128,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    grid = num_experts
    scale_hidden_size = recv_x_scale.shape[1]

    assert m_indices.shape[0] % align_m == 0
    assert recv_x_scale.dtype == output_tensor_scale.dtype
    assert recv_x_scale.shape[1] == output_tensor_scale.shape[1] == scale_hidden_size
    dev_ctx = torch.cuda.device(recv_x.device) if recv_x.is_cuda else nullcontext()
    with dev_ctx:
        _fwd_kernel_ep_scatter_1[(grid,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            m_indices.shape[0],
            num_experts=num_experts,
            num_warps=num_warps,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            ALIGN_M=align_m,
        )
        grid = min(recv_topk.shape[0], 1024 * 8)
        _fwd_kernel_ep_scatter_2[(grid,)](
            recv_topk.shape[0],
            expert_start_loc,
            recv_x,
            recv_x.stride(0),
            recv_x.stride(1),
            recv_x_scale,
            recv_x_scale.stride(0),
            recv_x_scale.stride(1),
            recv_topk,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor_scale,
            output_tensor_scale.stride(0),
            output_tensor_scale.stride(1),
            output_index,
            output_index.stride(0),
            output_index.stride(1),
            output_tensor.shape[0],
            topk_num=recv_topk.shape[1],
            num_experts=num_experts,
            num_warps=num_warps,
            HIDDEN_SIZE=hidden_size,
            HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
            SCALE_HIDDEN_SIZE=scale_hidden_size,
            SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
        )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    total_input_tokens,
    hidden_size,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RANK_BITS: tl.constexpr = 0,
):
    cur_block = tl.program_id(0).to(tl.int64)
    start_cur_token_int32 = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        col = cur_block * BLOCK_D + off_d
        col_mask = col < hidden_size
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk_ids
                + cur_token * recv_topk_ids_stride0
                + topk_index * recv_topk_ids_stride1
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index
                    + cur_token * input_index_stride0
                    + topk_index * input_index_stride1
                )
                source_token_index = source_token_index_int32.to(tl.int64)
                if source_token_index >= 0 and source_token_index < total_input_tokens:
                    acc_weight = tl.load(
                        recv_topk_weight
                        + cur_token * recv_topk_weight_stride0
                        + topk_index * recv_topk_weight_stride1
                    )
                    tmp = tl.load(
                        input_tensor
                        + source_token_index * input_tensor_stride0
                        + col * input_tensor_stride1,
                        mask=col_mask,
                        other=0.0,
                    )
                    accumulator += tmp.to(tl.float32) * acc_weight
        output_token = cur_token
        if RANK_BITS:
            shard_rows = total_token_num // (1 << RANK_BITS)
            shard = cur_token // shard_rows
            destination = tl.full((), 0, tl.int64)
            for bit in tl.static_range(RANK_BITS):
                destination |= ((shard >> bit) & 1) << (RANK_BITS - 1 - bit)
            output_token = destination * shard_rows + cur_token % shard_rows
        tl.store(
            output_tensor
            + output_token * output_tensor_stride0
            + col * output_tensor_stride1,
            accumulator.to(output_tensor.dtype.element_ty),
            mask=col_mask,
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    output_rank_count: int = 1,
):
    """Combine local experts; optionally bit-reverse equal destination shards.

    ``output_rank_count > 1`` is the producer layout required by
    ``ordered_reduce_scatter``. Ordinary callers retain token order.
    """
    BLOCK_D = 512  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    if (
        output_rank_count < 1
        or output_rank_count & (output_rank_count - 1)
        or num_tokens % output_rank_count
    ):
        raise ValueError("ep_gather output requires equal power-of-two rank shards")
    hidden_size = input_tensor.shape[1]
    if num_tokens == 0:
        return
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    dev_ctx = (
        torch.cuda.device(output_tensor.device)
        if output_tensor.is_cuda
        else nullcontext()
    )
    with dev_ctx:
        _fwd_kernel_ep_gather[grid](
            num_tokens,
            input_tensor.shape[0],
            hidden_size,
            input_tensor,
            input_tensor.stride(0),
            input_tensor.stride(1),
            recv_topk_ids,
            recv_topk_ids.stride(0),
            recv_topk_ids.stride(1),
            recv_topk_weight,
            recv_topk_weight.stride(0),
            recv_topk_weight.stride(1),
            input_index,
            input_index.stride(0),
            input_index.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            topk_num=recv_topk_ids.shape[1],
            num_warps=num_warps,
            BLOCK_D=BLOCK_D,
            RANK_BITS=output_rank_count.bit_length() - 1,
        )
    return

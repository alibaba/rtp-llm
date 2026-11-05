"""PPU scatter kernels for the DeepEP Normal MoE path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _init_contiguous_layout(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offsets,
        mask=offsets < num_experts,
        other=0,
    )
    starts = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offsets, starts, mask=offsets < num_experts)

    expert_mask = offsets == cur_expert
    expert_start = tl.sum(tl.where(expert_mask, starts, tl.zeros_like(starts)))
    expert_tokens = tl.sum(
        tl.where(expert_mask, tokens_per_expert, tl.zeros_like(tokens_per_expert))
    )
    block_offsets = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, expert_tokens, BLOCK_E, num_stages=4):
        tl.store(
            m_indices + expert_start + start_m + block_offsets,
            cur_expert,
            mask=(start_m + block_offsets) < expert_tokens,
        )


@triton.jit
def _init_masked_layout(
    alignment,
    expert_start_loc,
    num_experts: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_EXPERT_NUM)
    tl.store(
        expert_start_loc + offsets,
        offsets * alignment,
        mask=offsets < num_experts,
    )


@triton.jit
def _scatter_rows(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    num_experts: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    WITH_SCALE: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    hidden_offsets = tl.arange(0, HIDDEN_SIZE_PAD)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        row = tl.load(
            recv_x + token_id * recv_x_stride0 + hidden_offsets * recv_x_stride1,
            mask=hidden_mask,
        )
        if WITH_SCALE:
            row_scale = tl.load(recv_x_scale + token_id * recv_x_scale_stride0)
        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_idx = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk + token_id * recv_topk_stride0 + topk_idx * recv_topk_stride1
            )
            if expert_id >= 0 and expert_id < num_experts:
                dest_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest = dest_int32.to(tl.int64)
                tl.store(
                    output_index
                    + token_id * output_index_stride0
                    + topk_idx * output_index_stride1,
                    dest_int32,
                )
                tl.store(
                    output_tensor + dest * output_tensor_stride0 + hidden_offsets,
                    row,
                    mask=hidden_mask,
                )
                if WITH_SCALE:
                    tl.store(
                        output_tensor_scale + dest * output_tensor_scale_stride0,
                        row_scale,
                    )


@torch.no_grad()
def ep_scatter_bf16(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    block_e = 128
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    _init_contiguous_layout[(num_experts,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _scatter_rows[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x,
        0,
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor,
        0,
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=8,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        WITH_SCALE=False,
    )


@torch.no_grad()
def ep_scatter_v2_bf16(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    alignment: int,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    num_experts = expert_start_loc.shape[0]
    hidden_size = recv_x.shape[1]
    _init_masked_layout[(1,)](
        alignment,
        expert_start_loc,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _scatter_rows[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x,
        0,
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor,
        0,
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=8,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        WITH_SCALE=False,
    )


def _validate_int8_scatter_tensors(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
) -> None:
    if recv_x.dtype != torch.int8 or output_tensor.dtype != torch.int8:
        raise ValueError("INT8 scatter expects INT8 input and output tensors")
    if (
        recv_x_scale.dtype != torch.float32
        or output_tensor_scale.dtype != torch.float32
    ):
        raise ValueError("INT8 scatter expects FP32 input and output scales")
    if recv_x_scale.shape != (recv_x.shape[0], 1):
        raise ValueError(
            f"input scale shape {recv_x_scale.shape} does not match ({recv_x.shape[0]}, 1)"
        )
    if output_tensor_scale.shape != (output_tensor.shape[0], 1):
        raise ValueError(
            "output scale shape "
            f"{output_tensor_scale.shape} does not match ({output_tensor.shape[0]}, 1)"
        )
    if recv_x.shape[1] != output_tensor.shape[1]:
        raise ValueError(
            f"input/output hidden sizes differ: {recv_x.shape[1]}/{output_tensor.shape[1]}"
        )


@torch.no_grad()
def ep_scatter_int8(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    _validate_int8_scatter_tensors(
        recv_x, recv_x_scale, output_tensor, output_tensor_scale
    )
    block_e = 128
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    _init_contiguous_layout[(num_experts,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _scatter_rows[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=8,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        WITH_SCALE=True,
    )


@torch.no_grad()
def ep_scatter_v2_int8(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    alignment: int,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    _validate_int8_scatter_tensors(
        recv_x, recv_x_scale, output_tensor, output_tensor_scale
    )
    num_experts = expert_start_loc.shape[0]
    hidden_size = recv_x.shape[1]
    _init_masked_layout[(1,)](
        alignment,
        expert_start_loc,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _scatter_rows[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=8,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        WITH_SCALE=True,
    )

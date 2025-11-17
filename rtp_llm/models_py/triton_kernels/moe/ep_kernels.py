# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# Licensed under the Apache License, Version 2.0

import logging

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.utils import ceil_div

logger = logging.getLogger(__name__)


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


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
    topk_num: tl.constexpr,
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
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale
            + token_id * recv_x_scale_stride0
            + index_in_s * recv_x_scale_stride1,
            mask=mask_s,
        )
        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(
                    output_tensor_scale_ptr + index_in_s * output_tensor_scale_stride1,
                    to_copy_s,
                    mask=mask_s,
                )


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py
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
    scale_ue8m0: bool = False,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts
    scale_hidden_size = hidden_size // BLOCK_D
    if scale_ue8m0:
        # ue8m0 scales are packed here (4 scales per int32),
        # hence the effective size of this dimension is divided by 4.
        scale_hidden_size = ceil_div(scale_hidden_size, 4)

    assert m_indices.shape[0] % BLOCK_E == 0
    assert recv_x_scale.dtype == output_tensor_scale.dtype
    assert recv_x_scale.shape[1] == output_tensor_scale.shape[1] == scale_hidden_size
    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
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
        topk_num=recv_topk.shape[1],
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
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)
    start_cur_token_int32 = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index = source_token_index_int32.to(tl.int64)
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 512  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
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
    )
    return


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.
    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.
    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    m,
    k_div_block_size,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    for m_base in range(pid_m, m, grid_m):
        input_offset = (
            input_scale_ptr
            + m_base * input_scale_stride_m
            + k_offsets * input_scale_stride_k
        )
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)
        output_offset = (
            output_ptr + k_offsets * output_stride_k + m_base * output_stride_m
        )
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/quantization/triton_quant/fp8/fp8act_quant_kernel.py
def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() == 2
    m, k_div_block_size = input_scale.shape
    padd_m = get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty(
        (k_div_block_size, padd_m), dtype=input_scale.dtype, device=input_scale.device
    )
    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)
    _tma_align_input_scale_kernel[(grid_m,)](
        input_scale_ptr=input_scale,
        output_ptr=output,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_m=input_scale.stride(0),
        input_scale_stride_k=input_scale.stride(1),
        output_stride_m=output.stride(1),  # Note: these are swapped
        output_stride_k=output.stride(0),  # for column-major
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output.t()[:m]


@triton.jit
def recompute_topk_ids_triton_kernel(
    topk_ids_ptr,
    adjusted_topk_ids_ptr,
    expert_count_ptr,
    current_expert_start_id,
    num_local_experts,
    num_total,
    BLOCK_SIZE: tl.constexpr,
):
    token_indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_indices < num_total  # Mask out-of-bounds threads

    # 1. Load
    expert_id = tl.load(topk_ids_ptr + token_indices, mask=mask, other=-1)

    # 2. Adjust expert index
    adjusted = expert_id - current_expert_start_id
    valid = mask & (adjusted >= 0) & (adjusted < num_local_experts)

    # 3. Store
    out = tl.where(valid, adjusted, -1)
    tl.store(adjusted_topk_ids_ptr + token_indices, out, mask=mask)

    # 4. Atomic add - use scalar value for efficiency
    tl.atomic_add(expert_count_ptr + adjusted, 1, mask=valid)


def recompute_topk_ids_sum_expert_count(
    topk_ids: torch.Tensor, current_expert_start_id: int, num_local_experts: int
):
    """
    Recompute topk_ids by subtracting current_expert_start_id and count expert tokens.

    Args:
        topk_ids: Tensor of shape [num_tokens, topk] containing expert IDs
        current_expert_start_id: Starting expert ID to subtract
        num_local_experts: Number of local experts

    Returns:
        tuple: (adjusted_topk_ids, expert_count)
    """
    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    num_total = num_tokens * topk

    # Create output tensors
    adjusted_topk_ids = torch.empty_like(topk_ids)
    expert_count = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

    # Configure triton kernel parameters
    # Use smaller block size for better vectorization when topk is large
    # Ensure BLOCK_SIZE is a power of 2 for triton compatibility
    base_block_size = min(256, 1024 // max(topk, 1))
    BLOCK_SIZE = triton.next_power_of_2(base_block_size)

    # Launch recompute kernel
    grid_recompute = (triton.cdiv(num_total, BLOCK_SIZE),)
    recompute_topk_ids_triton_kernel[grid_recompute](
        topk_ids,
        adjusted_topk_ids,
        expert_count,
        current_expert_start_id,
        num_local_experts,
        num_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return adjusted_topk_ids, expert_count

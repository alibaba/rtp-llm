# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1
BLOCK_M = 8
BLOCK_N = 256

from rtp_llm.models_py.triton_kernels.causal_conv1d.op import cal_block_idx


@dataclass
class CausalConv1dMetadata:
    """Metadata for causal_conv1d_fn kernel launch.

    This class holds precomputed metadata that can be reused across multiple
    calls to causal_conv1d_fn with the same sequence configuration.
    """

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total: int


def prepare_causal_conv1d_metadata(
    query_start_loc: torch.Tensor,
    device: torch.device,
) -> CausalConv1dMetadata:
    """Prepare metadata for causal_conv1d_fn.

    This function precomputes the batch_ptr and token_chunk_offset_ptr tensors
    that are needed for the kernel launch. By calling this function ahead of time,
    you can avoid the overhead of computing these values during each kernel call.

    Args:
        query_start_loc: (batch + 1) int32 tensor containing cumulative sequence lengths.
            For example, if sequences have lengths [5, 1, 1, 1], then
            query_start_loc = [0, 5, 6, 7, 8].
        device: The device to place the output tensors on.
        block_m: Block size for BLOCK_M parameter (default: 8).

    Returns:
        CausalConv1dMetadata containing precomputed batch_ptr, token_chunk_offset_ptr,
        total number of programs, and block_m value.

    Example:
        >>> query_start_loc = torch.tensor([0, 5, 6, 7, 8], dtype=torch.int32)
        >>> metadata = prepare_causal_conv1d_metadata(query_start_loc, device='cuda')
        >>> output = causal_conv1d_fn(x, weight, bias, ..., metadata=metadata)
    """
    seqlens = np.diff(query_start_loc.cpu().numpy())

    # Calculate number of chunks per sequence
    nums = -(-seqlens // BLOCK_M)  # Ceiling division
    total = int(nums.sum())

    # Build mlist: which sequence each program handles
    mlist = np.repeat(np.arange(len(nums)), nums)

    # Build offsetlist: which chunk within the sequence each program handles
    offsetlist = []
    for num in nums:
        offsetlist.extend(range(num))

    # Create tensors
    batch_ptr = torch.full((total + 1,), PAD_SLOT_ID, dtype=torch.int32, device=device)
    token_chunk_offset_ptr = torch.full(
        (total + 1,), PAD_SLOT_ID, dtype=torch.int32, device=device
    )

    batch_ptr[: len(mlist)].copy_(torch.from_numpy(mlist.astype(np.int32)))
    token_chunk_offset_ptr[: len(offsetlist)].copy_(
        torch.from_numpy(np.array(offsetlist, dtype=np.int32))
    )

    return CausalConv1dMetadata(
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
        total=total,
    )


@triton.jit(do_not_specialize=["max_block_size"])
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    block_map_ptr,  # conv_state_indices_ptr
    prefix_lengths_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    batch: tl.int32,  # actually padded_batch
    dim: tl.constexpr,
    max_block_size: tl.int32,
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0)).to(tl.int64)
    prefix_length = tl.load(prefix_lengths_ptr + idx_seq).to(tl.int32)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset.to(tl.int64)
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        if HAS_CACHE and prefix_length > 0:
            init_state_block_pos = (prefix_length - 1) // SEQ_SIZE_PER_BLOCK
            init_state_block_idx = tl.load(
                block_map_ptr + idx_seq * max_block_size + init_state_block_pos
            ).to(tl.int64)
            conv_states_base = (
                conv_states_ptr
                + (init_state_block_idx * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )  # [BLOCK_N,]
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        dest_idx = prefix_length + idx_token + token_offset
        # when token is the last token of the block or the last token of the sequence, write back
        # currently, prefix_legth % seq_size_per_block should be zero, otherwise there maybe conflict between block read/write
        write_to_block = HAS_CACHE and (
            (dest_idx + 1) % SEQ_SIZE_PER_BLOCK == 0
            or idx_token + token_offset + 1 == seqlen
        )

        write_page_idx = tl.load(
            block_map_ptr + idx_seq * max_block_size + dest_idx // SEQ_SIZE_PER_BLOCK
        )

        if write_to_block and write_page_idx >= 0:
            # tl.device_print("idx_seq:", idx_seq)
            # tl.device_print("max_block_size:", max_block_size)
            # tl.device_print("dest_idx:", dest_idx)
            # tl.device_print("SEQ_SIZE_PER_BLOCK:", SEQ_SIZE_PER_BLOCK)

            if KERNEL_WIDTH == 2:
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim,
                    col0,
                    mask_1d,
                )
            elif KERNEL_WIDTH == 3:
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim,
                    col0,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 1 * stride_conv_state_tok,
                    col1,
                    mask_1d,
                )
            elif KERNEL_WIDTH == 4:
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim,
                    col0,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 1 * stride_conv_state_tok,
                    col1,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 2 * stride_conv_state_tok,
                    col2,
                    mask_1d,
                )
            elif KERNEL_WIDTH == 5:
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim,
                    col0,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 1 * stride_conv_state_tok,
                    col1,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 2 * stride_conv_state_tok,
                    col2,
                    mask_1d,
                )
                tl.store(
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + 3 * stride_conv_state_tok,
                    col3,
                    mask_1d,
                )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata: Optional[CausalConv1dMetadata] = None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    block_map: (batch, max_num_block)  int32
        indicates the corresponding state index, padding with -1
    prefix_lengths: (batch) int32
        if prefix_lenths = 0, means not inital_state, otherwise should read initial_state from conv_states
    seq_size_per_block: int
        seq step to record conv state in each block
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    metadata: Optional[CausalConv1dMetadata]
        Precomputed metadata from prepare_causal_conv1d_metadata().
        If None, metadata will be computed on-the-fly.
        Providing precomputed metadata can improve performance when
        calling this function multiple times with the same sequence configuration.

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(weight.dtype)
    out = torch.empty_like(x).contiguous()

    # Prepare metadata if not provided
    if metadata is None:
        metadata = prepare_causal_conv1d_metadata(
            query_start_loc=query_start_loc,
            device=x.device,
        )

    batch_ptr = metadata.batch_ptr
    token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    grid_x = metadata.total

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    max_block_size = block_map.size(1) if block_map is not None else 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        assert dim == conv_states.shape[1] and width - 1 <= conv_states.shape[2]
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if block_map is not None:
            assert block_map.dim() == 2
            assert block_map.size(1) == padded_batch
        assert prefix_lengths.dim() == 1
        assert prefix_lengths.dim() == padded_batch
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    grid = (grid_x, triton.cdiv(dim, BLOCK_N))
    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        (
            torch.empty([0], device=x.device, dtype=x.dtype)
            if conv_states is None
            else conv_states
        ),
        (
            torch.empty([0], device=x.device, dtype=torch.int32)
            if block_map is None
            else block_map
        ),
        prefix_lengths,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        out,
        # Matrix dimensions
        padded_batch,
        dim,
        max_block_size,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_CACHE=block_map is not None and conv_states is not None,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        IS_CONTINUOUS_BATCHING=True,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out.to(original_x_dtype)


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    block_map_ptr,
    stride_block_map: tl.int32,
    sequence_lengths_ptr,
    query_start_loc_ptr,  # (batch + 1)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_STATELEN_TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # if IS_VARLEN:
    #     query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    #     query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
    #     # revise state_len and seqlen
    #     state_len = state_len - (seqlen - (query_end_index - query_start_index))
    #     seqlen = query_end_index - query_start_index
    #     x_offset = query_start_index * stride_x_token
    #     o_offset = query_start_index * stride_o_token
    # else:
    # query_start_index = idx_seq * seqlen
    # query_end_index = query_start_index + seqlen
    x_offset = idx_seq * stride_x_seq
    o_offset = idx_seq * stride_o_seq

    # if query_start_index >= query_end_index:
    #     return

    sequence_length = tl.load(sequence_lengths_ptr + idx_seq).to(tl.int32)
    read_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
    read_block_id = tl.load(
        block_map_ptr + idx_seq * stride_block_map + read_block_offset
    ).to(tl.int64)
    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN_TOTAL)  # [BLOCK_M]

    # the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = ((idx_tokens + 1) < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    # without debug barrier, the final conv_state and o is not correct

    VAL = state_len - 1
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)
    # for seqLen = n, we need to write n block in sequential manner
    for idx in tl.range(seqlen):
        write_block_offset = (cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK)) + idx
        write_block_id = tl.load(
            block_map_ptr + idx_seq * stride_block_map + write_block_offset
        ).to(tl.int32)

        # print write_block_id (only once per sequence, first feature block)
        if tl.program_id(1) == 0 and tl.min(idx_feats) == 0:
            tl.device_print("write_block_id:", write_block_id)
            tl.device_print("write_block_offset:", write_block_offset)
            tl.device_print("sequence_length:", sequence_length)

        if write_block_id != -1:
            conv_state_base = (
                conv_state_ptr
                + (write_block_id * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )  # [BLOCK_N,]

            # base offset
            idx_tokens_offset = idx_tokens - idx

            conv_state_ptrs_target = (
                conv_state_base + (idx_tokens_offset * stride_conv_state_tok)[:, None]
            )  # [BLOCK_M, BLOCK_N]
            mask = (
                (idx_tokens_offset >= 0)[:, None]
                & (idx_tokens_offset < state_len)[:, None]
                & (idx_feats < dim)[None, :]
            )
            tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert (
            cache_seqlens is None
        ), "cache_seqlens is not supported for speculative decoding"
        assert (
            pad_slot_id is not None
        ), "pad_slot_id is required for speculative decoding"
        assert (
            x.stride(1) == 1
        ), "x is expected to be contiguous along the feature-dimension"
        assert block_map is not None, "block_map is required for speculative decoding"
        assert (
            query_start_loc is None
        ), "query_start_loc is not supported for speculative decoding"

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = torch.empty([batch, seqlen, dim], device=x.device, dtype=x.dtype).transpose(
        1, 2
    )
    stride_w_dim, stride_w_width = weight.stride()

    # X (batch, dim, seqlen)
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    # when speculative, we load (width - 2) token from conv_state and (seqlen) token from x, then store them in different block
    np2_statelen_total = triton.next_power_of_2(state_len - 1 + seqlen)

    stride_block_map = block_map.size(1) if block_map is not None else 0

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        block_map,
        stride_block_map,
        sequence_lengths,
        query_start_loc,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        NP2_STATELEN=np2_statelen,
        NP2_STATELEN_TOTAL=np2_statelen_total,
        BLOCK_N=256,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)

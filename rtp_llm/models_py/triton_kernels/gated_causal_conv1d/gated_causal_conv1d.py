# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Gated variant of causal_conv1d for LFM2-style short conv layers.
#
# Derived from rtp_llm/models_py/triton_kernels/causal_conv1d/causal_conv1d.py
# (itself adapted from Dao-AILab/causal-conv1d and vLLM). The kernel and
# grid/tiling structure are copied from that file; only the load/store sites
# change so the two torch elementwise gate multiplies of the decomposed path
#
#     bx = B * x                      # torch bf16 mul
#     conv_out = causal_conv1d(bx, w) # Triton depthwise causal conv
#     y = C * conv_out                # torch bf16 mul
#
# are fused in-register: the kernel reads the 3H-wide ``in_proj`` output
# ``projected`` (chunks [0:H)=B, [H:2H)=C, [2H:3H)=X) and writes y directly.
# The original causal_conv1d.py is shared with kimi_linear and stays
# untouched; this file is an independent copy.
#
# Numerics follow the decomposed reference bit-for-bit:
#   * B*X is rounded to the input dtype before entering the conv taps
#     (a materialized ``b * x`` tensor would carry exactly these bits);
#   * conv accumulation stays fp32 with the same tap order;
#   * the conv result is rounded to the weight dtype and then the output
#     dtype (mirroring ``out = empty_like(x.to(weight.dtype))`` followed by
#     ``out.to(x.dtype)``) before the C gate multiply, which itself is
#     rounded once to the output dtype;
#   * conv states still store the *gated* B*X values of the last
#     ``kernel - 1`` tokens, so state layout and continuation semantics are
#     unchanged.

from typing import Optional, Union

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1
# must match causal_conv1d.BLOCK_M: metadata chunk offsets are counted in
# BLOCK_M units by prepare_causal_conv1d_metadata (imported below).
BLOCK_M = 8
BLOCK_N = 256

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.op import cal_block_idx


@triton.jit
def _load_gated_bx(b_ptrs, x_ptrs, mask, ref_ptr):
    """Load the B and X gate chunks and return B * X.

    The product is rounded to the projected (input) dtype exactly like a
    materialized torch elementwise ``b * x`` (fp32 opmath, one bf16
    rounding), then cast to ``ref_ptr``'s dtype like the ``x.to(dtype)``
    conversions the ungated wrappers perform on the materialized tensor.
    Masked lanes yield 0.0 on both operands, so the product is 0.0 as well.
    """
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    bx = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)
    return bx.to(ref_ptr.dtype.element_ty)


@triton.jit
def _apply_c_gate(acc, c_ptrs, mask, state_ptr, o_ptr):
    """Multiply the conv accumulator by the C gate chunk and round to the
    output dtype.

    ``acc`` is first rounded to the state/weight dtype and then the output
    dtype, reproducing the reference chain ``store(acc) -> out.to(x.dtype)``
    -> ``c * conv_out`` (fp32 opmath, single rounding to the output dtype).
    """
    c = tl.load(c_ptrs, mask=mask, other=0.0)
    acc_conv = acc.to(state_ptr.dtype.element_ty).to(o_ptr.dtype.element_ty)
    return (c.to(tl.float32) * acc_conv.to(tl.float32)).to(o_ptr.dtype.element_ty)


@triton.jit(do_not_specialize=["stride_block_map"])
def _gated_causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    projected_ptr,  # (cu_seq_len, 3 * dim): B/C/X gate chunks along the last dim
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    block_map_ptr,  # conv_state_indices_ptr
    prefix_lengths_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (cu_seq_len, dim)
    # Matrix dimensions
    batch: tl.int32,  # actually padded_batch
    dim: tl.constexpr,
    stride_block_map: tl.int64,
    # Strides
    stride_p_seq: tl.constexpr,  # unused, kept for parity with causal_conv1d
    stride_p_dim: tl.constexpr,  # stride between feature-values of one gate
    stride_p_token: tl.constexpr,  # stride to get to next token (same gate)
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

    # base of the sequence, split into the three gate chunks:
    # B at [0, dim), C at [dim, 2 * dim), X at [2 * dim, 3 * dim)
    proj_base = (
        projected_ptr + sequence_start_index * stride_p_token + idx_feats * stride_p_dim
    )  # [BLOCK_N,]
    b_base = proj_base
    c_base = proj_base + dim * stride_p_dim
    x_base = proj_base + 2 * dim * stride_p_dim

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        if HAS_CACHE and prefix_length > 0:
            init_state_block_pos = (prefix_length - 1) // SEQ_SIZE_PER_BLOCK
            init_state_block_idx = tl.load(
                block_map_ptr + idx_seq * stride_block_map + init_state_block_pos
            ).to(tl.int64)
            tl.device_assert(
                init_state_block_idx >= 0,
                "gated_causal_conv1d: init_state_block_idx must be non-negative",
            )
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
                # (w dtype: the ungated kernel reads the weight-converted x)
                col0 = tl.zeros((BLOCK_N,), dtype=w_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=w_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=w_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=w_ptr.dtype.element_ty)

    else:  # chunk_offset > 0
        # read prior-token data from `projected` (gated on the fly)
        prior_b = b_base + (token_offset - 1) * stride_p_token
        prior_x = x_base + (token_offset - 1) * stride_p_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = _load_gated_bx(prior_b, prior_x, mask_w, w_ptr)  # [BLOCK_N]
        if KERNEL_WIDTH == 3:
            col1 = _load_gated_bx(prior_b, prior_x, mask_w, w_ptr)  # [BLOCK_N]
            col0 = _load_gated_bx(
                prior_b - 1 * stride_p_token,
                prior_x - 1 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]
        if KERNEL_WIDTH == 4:
            col2 = _load_gated_bx(prior_b, prior_x, mask_w, w_ptr)  # [BLOCK_N]
            col1 = _load_gated_bx(
                prior_b - 1 * stride_p_token,
                prior_x - 1 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]
            col0 = _load_gated_bx(
                prior_b - 2 * stride_p_token,
                prior_x - 2 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]
        if KERNEL_WIDTH == 5:
            col3 = _load_gated_bx(prior_b, prior_x, mask_w, w_ptr)  # [BLOCK_N]
            col2 = _load_gated_bx(
                prior_b - 1 * stride_p_token,
                prior_x - 1 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]
            col1 = _load_gated_bx(
                prior_b - 2 * stride_p_token,
                prior_x - 2 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]
            col0 = _load_gated_bx(
                prior_b - 3 * stride_p_token,
                prior_x - 3 * stride_p_token,
                mask_w,
                w_ptr,
            )  # [BLOCK_N]

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    b_base_1d = b_base + token_offset * stride_p_token  # starting of chunk
    c_base_1d = c_base + token_offset * stride_p_token
    x_base_1d = x_base + token_offset * stride_p_token

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
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(b_ptrs_1d, x_ptrs_1d, mask_x_1d, w_ptr)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(b_ptrs_1d, x_ptrs_1d, mask_x_1d, w_ptr)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(b_ptrs_1d, x_ptrs_1d, mask_x_1d, w_ptr)

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

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        out_val = _apply_c_gate(
            acc,
            c_base_1d + idx_token * stride_p_token,
            mask_x_1d,
            w_ptr,
            o_ptr,
        )
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, out_val, mask=mask_1d)

        dest_idx = prefix_length + idx_token + token_offset
        # when token is the last token of the block or the last token of the sequence, write back
        # currently, prefix_length % seq_size_per_block should be zero,
        # otherwise there may be conflict between block read/write
        write_to_block = HAS_CACHE and (
            (dest_idx + 1) % SEQ_SIZE_PER_BLOCK == 0
            or idx_token + token_offset + 1 == seqlen
        )

        write_page_idx = tl.full([], -1, tl.int64)
        if write_to_block:
            write_page_idx = tl.load(
                block_map_ptr
                + idx_seq * stride_block_map
                + dest_idx // SEQ_SIZE_PER_BLOCK
            ).to(tl.int64)

        if write_to_block and write_page_idx >= 0:
            # note: col0/col1/... already hold the gated B*X values, so the
            # conv-state write-back semantics are identical to the ungated
            # kernel running on a materialized bx tensor.
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


def gated_causal_conv1d_fn(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    activation: Optional[str] = None,
    metadata: Optional[CausalConv1dMetadata] = None,
):
    """Gated varlen prefill: fuse ``C * causal_conv1d(B * X)`` into one kernel.

    Same contract as ``causal_conv1d_fn`` (continuous batching, block-mapped
    conv states, prefix continuation) except that the input is the unsplit
    ``in_proj`` output instead of the pre-gated ``B * X`` tensor:

    projected: (cu_seq_len, 3 * dim) bf16
        rows are tokens; the last dim holds the three gate chunks
         [0, dim) = B, [dim, 2*dim) = C, [2*dim, 3*dim) = X.
    weight: (dim, width) depthwise taps
    conv_states: (..., dim, width - 1) holding gated B*X values
        updated inplace if provided, indexed through block_map
    query_start_loc: (batch + 1) int32 cumulative sequence lengths
    block_map: (batch, max_num_block) int32 state block indices, padded -1
    prefix_lengths: (batch) int32; 0 = no initial state, otherwise the
        initial state is read from conv_states
    seq_size_per_block: tokens per state block
    activation: None (LFM2 keeps both gates outside) or "silu"/"swish"
        applied to the conv accumulator before the C gate multiply
    metadata: Optional[CausalConv1dMetadata] from
        prepare_causal_conv1d_metadata(); computed on the fly if None

    out: (cu_seq_len, dim), same dtype as ``projected``
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    # Prepare metadata if not provided
    if metadata is None:
        metadata = prepare_causal_conv1d_metadata(
            query_start_loc=query_start_loc,
            device=projected.device,
        )

    batch_ptr = metadata.batch_ptr
    token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    grid_x = metadata.total

    tokens, total_dim = projected.shape
    dim = total_dim // 3
    _, width = weight.shape
    assert (
        total_dim == 3 * weight.shape[0]
    ), f"projected last dim {total_dim} != 3 * conv dim {weight.shape[0]}"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    out = torch.empty([tokens, dim], device=projected.device, dtype=projected.dtype)

    padded_batch = query_start_loc.size(0) - 1
    stride_p_seq = 0
    stride_p_dim = projected.stride(1)
    stride_p_token = projected.stride(0)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    stride_block_map = block_map.stride(0) if block_map is not None else 0
    if conv_states is not None:
        # same cache contract as causal_conv1d_fn: conv_states is indexed
        # through block_map and can hold more cache lines than batch entries.
        assert dim == conv_states.shape[1] and width - 1 <= conv_states.shape[2]
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    # out is (cu_seq_len, dim) and contiguous along the feature dim
    stride_o_seq = 0
    stride_o_dim = out.stride(1)
    stride_o_token = out.stride(0)

    if batch_ptr.device != projected.device:
        batch_ptr = batch_ptr.to(projected.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(projected.device)

    grid = (grid_x, triton.cdiv(dim, BLOCK_N))
    _gated_causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        projected,
        weight,
        None,
        (
            torch.empty([0], device=projected.device, dtype=projected.dtype)
            if conv_states is None
            else conv_states
        ),
        (
            torch.empty([0], device=projected.device, dtype=torch.int32)
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
        stride_block_map,
        # stride
        stride_p_seq,
        stride_p_dim,
        stride_p_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        PAD_SLOT_ID,
        # META
        HAS_BIAS=False,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_CACHE=block_map is not None and conv_states is not None,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        IS_CONTINUOUS_BATCHING=True,
        USE_PAD_SLOT=PAD_SLOT_ID is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit()
def _gated_causal_conv1d_update_kernel(
    # Pointers to matrices
    projected_ptr,  # (batch, 3 * dim, seqlen): B/C/X gate chunks on dim-axis
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    block_map_ptr,
    stride_block_map: tl.int64,
    sequence_lengths_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    # Strides
    stride_p_seq: tl.constexpr,
    stride_p_dim: tl.constexpr,
    stride_p_token: tl.constexpr,
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
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    x_offset = idx_seq * stride_p_seq
    o_offset = idx_seq * stride_o_seq

    sequence_length = tl.load(sequence_lengths_ptr + idx_seq).to(tl.int32)
    # CUDA graph replay rounds a live batch up to the next captured bucket and
    # marks the extra rows with sequence_length == 0.  Do not derive a page
    # offset for those rows: cal_block_idx(sequence_length - 1, ...) would be
    # negative and read before the row's block-map storage.
    if sequence_length <= 0:
        return
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
    # gate chunk bases: B at [0, dim), C at [dim, 2*dim), X at [2*dim, 3*dim)
    proj_base = projected_ptr + x_offset + (idx_feats * stride_p_dim)  # [BLOCK_N]
    b_base = proj_base
    c_base = proj_base + dim * stride_p_dim
    x_base = proj_base + 2 * dim * stride_p_dim

    b_ptrs = (
        b_base[None, :] + ((idx_tokens - VAL) * stride_p_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_p_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    # the state is updated with the gated B*X values, exactly like the
    # ungated kernel running on a materialized bx tensor.
    loaded_x = _load_gated_bx(b_ptrs, x_ptrs, mask_x, conv_state_ptr)

    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)
    # for seqLen = n, we need to write n block in sequential manner
    for idx in tl.range(seqlen):
        write_block_offset = (cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK)) + idx
        write_block_id = tl.load(
            block_map_ptr + idx_seq * stride_block_map + write_block_offset
        ).to(tl.int64)

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

    b_base_1d = b_base  # starting of chunk [BLOCK_N]
    c_base_1d = c_base
    x_base_1d = x_base
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
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(
                        b_ptrs_1d, x_ptrs_1d, mask_x_1d, conv_state_ptr
                    )
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(
                        b_ptrs_1d, x_ptrs_1d, mask_x_1d, conv_state_ptr
                    )
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(
                        b_ptrs_1d, x_ptrs_1d, mask_x_1d, conv_state_ptr
                    )
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
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(
                        b_ptrs_1d, x_ptrs_1d, mask_x_1d, conv_state_ptr
                    )
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
                    b_ptrs_1d = b_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    x_ptrs_1d = x_base_1d + idx_token * stride_p_token  # [BLOCK_N]
                    matrix_x = _load_gated_bx(
                        b_ptrs_1d, x_ptrs_1d, mask_x_1d, conv_state_ptr
                    )

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
        out_val = _apply_c_gate(
            acc,
            c_base_1d + idx_token * stride_p_token,
            mask_x_1d,
            conv_state_ptr,
            o_ptr,
        )
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, out_val, mask=mask_1d)


def gated_causal_conv1d_update(
    projected: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
):
    """Gated decode step: fuse ``C * causal_conv1d_update(B * X)``.

    Same contract as ``causal_conv1d_update`` (sliding-window state update
    through block_map, CUDA-graph padding rows skipped via
    ``sequence_lengths == 0``) except that the input is the unsplit
    ``in_proj`` output instead of the pre-gated ``B * X`` tensor:

    projected: Input tensor which can take the following shapes:

    - ``[batch, 3 * dim]`` - single token per sequence
    - ``[batch, 3 * dim, seqlen]`` - single or multiple tokens per sequence
      (B/C/X gate chunks live along the dim axis)

    conv_state: (..., dim, state_len), where state_len >= width - 1; the
        stored values are the gated B*X inputs, updated in place
    weight: (dim, width)
    bias: (dim,)
    activation: None (LFM2 applies both gates through this kernel) or
        "silu"/"swish" applied before the C gate multiply
    cache_seqlens: unused, kept for signature parity with causal_conv1d_update
    block_map: (batch, max_num_block) int32 state block indices
    seq_size_per_block: tokens per state block
    sequence_lengths: (batch) int32; padded CUDA-graph rows use 0

    out: (batch, dim) or (batch, dim, seqlen), matching the input layout
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = projected.dtype
    unsqueeze = projected.dim() == 2
    if unsqueeze:
        # make it (batch, 3 * dim, seqlen) with seqlen == 1
        projected = projected.unsqueeze(-1)
    batch, total_dim, seqlen = projected.shape
    dim = total_dim // 3
    _, width = weight.shape
    assert (
        total_dim == 3 * weight.shape[0]
    ), f"projected dim-axis {total_dim} != 3 * conv dim {weight.shape[0]}"
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    assert dim == weight.size(0)
    assert dim == conv_state.size(1)
    assert conv_state.stride(-2) == 1, (
        f"ERROR: expect contiguous along feat-dim of conv_state "
        f"(currently stride={conv_state.stride()})"
    )
    assert state_len >= width - 1
    assert num_cache_lines >= batch

    # write the gated result directly in the projected (original) dtype;
    # the kernel reproduces the reference rounding chain
    # store(acc)->out.to(x.dtype)->c*conv_out internally.
    out = torch.empty(
        [batch, seqlen, dim], device=projected.device, dtype=original_x_dtype
    ).transpose(1, 2)
    stride_w_dim, stride_w_width = weight.stride()

    # projected (batch, 3 * dim, seqlen)
    stride_p_seq, stride_p_dim, stride_p_token = projected.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    # when speculative, we load (width - 2) token from conv_state and (seqlen)
    # token from x, then store them in different block
    np2_statelen_total = triton.next_power_of_2(state_len - 1 + seqlen)

    stride_block_map = block_map.stride(0) if block_map is not None else 0

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _gated_causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        projected,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        block_map,
        stride_block_map,
        sequence_lengths,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        # stride
        stride_p_seq,
        stride_p_dim,
        stride_p_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        PAD_SLOT_ID,
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
    return out

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Extracted from ROCm/aiter 2570b35f95ff012b2abac5f5d4f4158ee03d77a3.
# Diagnostic-only copy. See provenance.json and LICENSE; no default routing.
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def define_layout(
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
) -> gl.constexpr:
    # Register allocation configuration based on group size and compute block size
    if QUERY_GROUP_SIZE_POW2 >= 16:
        if QUERY_GROUP_SIZE_POW2 == 16:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [0, 128]]
        elif QUERY_GROUP_SIZE_POW2 == 32:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [16, 0]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                ]
        elif QUERY_GROUP_SIZE_POW2 == 64:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [16, 0],
                    [32, 0],
                ]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                    [32, 0],
                ]

        # Distributed layout for QK linear operations
        qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=register_bases,
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        )
    else:
        VGPRS0: gl.constexpr = QUERY_SEQ_LEN_POW2
        THREADS0: gl.constexpr = triton.cdiv(QUERY_GROUP_SIZE_POW2, (4 * VGPRS0))
        THREADS1: gl.constexpr = 64 // THREADS0

        qk_linear_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[VGPRS0, CONTEXT_PARTITION_SIZE // THREADS1],
            threads_per_warp=[THREADS0, THREADS1],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
    return qk_linear_layout


define_layout.__triton_builtin__ = True


def store_temporary_result(
    max_logits,
    exp_sums,
    attention_accumulator,
    max_logits_ptr,
    exp_sums_ptr,
    output_ptr,
    max_logits_offsets,
    output_offsets,
    qk_row_mask,
    output_mask,
    _semantic=None,
) -> None:
    gl.amd.cdna3.buffer_store(
        stored_value=max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=exp_sums,
        ptr=exp_sums_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_accumulator,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
        _semantic=_semantic,
    )


store_temporary_result.__triton_builtin__ = True


@gluon.jit
def paged_attention_decode_v2_gluon_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    stride_output_seq: int,
    stride_output_head: int,
    stride_output_part: int,
    stride_output_group: int,
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    head_size: int,
    num_seqs: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    actual_partial_fp32_ptr,  # FP32; same shape/element strides as output_ptr
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN: gl.constexpr,
    ONE_QUERY_GROUP_SIZE: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    CDNA_VERSION: gl.constexpr = 3,
    SLIDING_WINDOW: gl.constexpr = 0,
):
    """
    Paged Attention Decode Kernel with FP8/BF16 support for AMD GPUs.

    This kernel implements the attention mechanism for decoding in transformer models
    with support for paged KV caches and FP8 quantization. It handles causal masking,
    ALiBi biases, and various quantization schemes.

    Args:
        exp_sums_ptr: Pointer to exponential sums output tensor
        max_logits_ptr: Pointer to maximum logits output tensor
        output_ptr: Pointer to attention output tensor
        query_ptr: Pointer to query tensor
        key_cache_ptr: Pointer to key cache in block layout
        value_cache_ptr: Pointer to value cache in block layout
        block_tables_ptr: Pointer to block tables mapping sequences to physical blocks
        context_lengths_ptr: Pointer to sequence lengths for each sequence
        softmax_scale: Scaling factor for softmax
        query_scale: Query quantization scales
        key_scale: Key quantization scales
        value_scale: Value quantization scales
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration

    Note:
        This kernel uses AMD CDNA3 MFMA instructions for efficient matrix operations
        and supports both FP8 and BF16 data types with various quantization modes.
    """
    # ==================== VALIDATION CHECKS ====================
    gl.static_assert(
        KV_BLOCK_SIZE == 16 or KV_BLOCK_SIZE == 64,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE in [16, 64]",
    )
    # Data type validation
    gl.static_assert(
        query_ptr.dtype.is_fp8()
        or query_ptr.dtype.element_ty == gl.bfloat16
        or query_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        key_cache_ptr.dtype.is_fp8()
        or key_cache_ptr.dtype.element_ty == gl.bfloat16
        or key_cache_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        value_cache_ptr.dtype.is_fp8()
        or value_cache_ptr.dtype.element_ty == gl.bfloat16
        or value_cache_ptr.dtype.element_ty == gl.float16
    )

    if QUERY_QUANT_MODE >= 0:
        gl.static_assert(query_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(key_scale.dtype.element_ty == gl.float32)
        gl.static_assert(value_scale.dtype.element_ty == gl.float32)

    # ==================== CONSTANTS AND CONFIGURATION ====================
    if COMPUTE_TYPE.is_fp8() or CDNA_VERSION == 4:
        MFMA_INSTR_K: gl.constexpr = 32
    else:
        MFMA_INSTR_K: gl.constexpr = 16
    QK_PV_MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, MFMA_INSTR_K]

    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8

    if COMPUTE_TYPE.is_fp8():
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE
    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential conversion

    # Calculate MTP (Multi-Token Prefill) layout parameters
    QUERY_SEQ_LEN_POW2: gl.constexpr = triton.next_power_of_2(QUERY_SEQ_LEN)
    if ONE_QUERY_GROUP_SIZE <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = triton.next_power_of_2(
            ONE_QUERY_GROUP_SIZE
        )
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2

    K_HEAD_SIZE_SPLITS: gl.constexpr = HEAD_SIZE_POW2 // KV_16B_ELEMENT_COUNT
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE

    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # MTP Query tensor layout - 3D [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        # ONE_QUERY_GROUP_SIZE_POW2 may be 4, 8, 16
        # corresponding Q_WARPS_PER_CTA_DIM1 should be 1, 2, 4
        # corresponding Q_WARPS_PER_CTA_DIM0 should be 4, 2, 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 4 // Q_WARPS_PER_CTA_DIM1
    else:
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = 4
    mtp_blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 8],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1, 1],
        order=[2, 1, 0],
    )
    # [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        KV_16B_ELEMENT_COUNT, 1, 16, order=[1, 0]
    )

    # Key cache layout - optimized for block-wise access patterns
    blocked_key_layout_fp8: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )
    key_warps_per_cta_f16: gl.constexpr = (
        [4, 1, 1, 1] if KV_BLOCK_SIZE == 16 else [1, 1, 4, 1]
    )
    blocked_key_layout_f16: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=key_warps_per_cta_f16,
        order=[3, 2, 1, 0],
    )
    blocked_key_layout: gl.constexpr = (
        blocked_key_layout_fp8 if KV_16B_ELEMENT_COUNT == 16 else blocked_key_layout_f16
    )

    DOT_QK_K_WIDTH: gl.constexpr = KV_16B_ELEMENT_COUNT
    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=DOT_QK_K_WIDTH
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=DOT_QK_K_WIDTH
    )

    # Register allocation configuration based on group size and compute block size
    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
    )

    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )
        blocked_value_layout_f16: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, 8],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )
        blocked_value_layout_fp8: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, 16],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )
        blocked_value_layout: gl.constexpr = (
            blocked_value_layout_fp8
            if KV_16B_ELEMENT_COUNT == 16
            else blocked_value_layout_f16
        )
        value_dim1_offsets = gl.arange(
            0,
            KV_BLOCK_SIZE // KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim2_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim3_offsets = gl.arange(
            0,
            KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout))
            ),
        )
    else:
        # Standard value layout
        value_threads_per_warp: gl.constexpr = (
            [4, 16, 1] if KV_BLOCK_SIZE == 16 else [1, 16, 4]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 16],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 4, 1],
            order=[2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_value_layout)),
        )
        value_dim2_offsets = gl.arange(
            0,
            KV_BLOCK_SIZE,
            layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_value_layout)),
        )

    # PV Matrix multiplication layout using AMD MFMA instructions
    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    pv_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # ==================== LAYOUT SLICE DEFINITIONS ====================

    # MTP Query layout slices (for 3D layout)
    mtp_query_len_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_query_group_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_head_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, mtp_blocked_query_layout)
    )

    # Key layout slices
    block_id_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
    )
    head_size_split_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
    )
    block_element_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_key_layout))
    )
    contiguous_kv_elements_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_key_layout))
    )

    # Coordinate offsets for various dimensions
    # MTP offsets (for 3D layout)
    mtp_query_len_offsets = gl.arange(
        0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
    )
    mtp_query_group_size_offsets = gl.arange(
        0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
    )
    mtp_head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=mtp_head_size_layout)

    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(0, KV_BLOCK_SIZE, layout=block_element_layout)
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    query_row_mask_3d = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
        mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
    )
    query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    output_partition_idx = gl.program_id(2)
    context_length = gl.load(context_lengths_ptr + sequence_idx)
    if SLIDING_WINDOW > 0:
        sequence_start_idx = context_length - SLIDING_WINDOW
        partition_offset = gl.maximum(0, sequence_start_idx // CONTEXT_PARTITION_SIZE)
        sequence_partition_idx = partition_offset + output_partition_idx
    else:
        sequence_start_idx = 0
        sequence_partition_idx = output_partition_idx

    # Load query tensor with 3D MTP layout
    # Query shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    mtp_query_offsets = (
        sequence_idx * stride_query_bs
        + mtp_query_len_offsets[:, None, None] * stride_query_qlen
        + kv_head_idx * stride_query_kv_head
        + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
        + mtp_head_size_offsets[None, None, :]
    )
    mtp_query_mask = (
        (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN)
        & (mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE)
        & (mtp_head_size_offsets[None, None, :] < head_size)
    )
    # [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    mtp_query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
    )
    mtp_query_tensor = gl.reshape(
        mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    )
    query_tensor = gl.convert_layout(mtp_query_tensor, layout=blocked_query_layout)
    query_shared = gl.allocate_shared_memory(
        query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
    )

    # Load query quantization scales if needed
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = tl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * stride_query_scale_bs
            + mtp_query_len_offsets[:, None, None] * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + mtp_query_group_size_offsets[None, :, None]
        )
        query_scale_mask = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
            mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=query_scale_mask,
        )
        query_scale_value = gl.reshape(query_scale_value, [QUERY_GROUP_SIZE_POW2, 1])
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # Initialize output pointers and accumulators
    max_logits_base_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    # Convert MTP layout indices to continuous indices for exp_sums/max_logits
    max_logits_query_len_idx = max_logits_base_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    max_logits_group_idx_in_len = (
        max_logits_base_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    )
    max_logits_base_offsets = (
        max_logits_query_len_idx * ONE_QUERY_GROUP_SIZE + max_logits_group_idx_in_len
    )
    max_logits_offsets = (
        sequence_idx * stride_max_logits_seq
        + kv_head_idx * stride_max_logits_head
        + output_partition_idx * stride_max_logits_part
        + max_logits_base_offsets
    )

    output_group_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    # Convert MTP layout indices to continuous indices for temporary_output
    # MTP layout: [QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2] with valid indices at non-contiguous positions
    # Continuous layout: [QUERY_SEQ_LEN * ONE_QUERY_GROUP_SIZE] with valid indices at contiguous positions
    output_query_len_idx = output_group_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    output_group_idx_in_len = output_group_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    output_group_offsets = (
        output_query_len_idx * ONE_QUERY_GROUP_SIZE + output_group_idx_in_len
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = pv_row_mask[:, None] & (output_head_size_offsets[None, :] < head_size)

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        output_partition_idx * stride_output_part
        + output_group_offsets[:, None] * stride_output_group
        + output_head_size_offsets[None, :]
    )

    # Initialize attention state variables
    max_logits = max_logits_base_offsets.to(gl.float32) * float(0.0) - float("inf")
    exp_sums = max_logits_base_offsets.to(gl.float32) * float(0.0)
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== SEQUENCE PROCESSING ====================

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    SEQUENCE_PARTITION_KV_BLOCKS: gl.constexpr = CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE

    if SLIDING_WINDOW > 0:
        # Only program 0 processes the full sliding window in a single FP32
        # accumulation pass, avoiding BF16 intermediate quantization loss from
        # multi-partition PS combine. All other programs early-return with defaults.
        if output_partition_idx > 0:
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator.to(OUTPUT_DTYPE),
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
            return
        window_start_partition = gl.maximum(
            0, sequence_start_idx // CONTEXT_PARTITION_SIZE
        )
        window_end_partition = gl.cdiv(context_length, CONTEXT_PARTITION_SIZE)
        MAX_WINDOW_BLOCKS: gl.constexpr = (
            SLIDING_WINDOW // CONTEXT_PARTITION_SIZE + 2
        ) * KV_COMPUTE_BLOCK_COUNT
    else:
        kv_sequence_start_idx = sequence_partition_idx * CONTEXT_PARTITION_SIZE
        if kv_sequence_start_idx >= context_length:
            return
        MAX_WINDOW_BLOCKS: gl.constexpr = KV_COMPUTE_BLOCK_COUNT

    # Process KV sequence in compute blocks
    for block_idx in gl.static_range(MAX_WINDOW_BLOCKS):
        if SLIDING_WINDOW > 0:
            partition_in_window = block_idx // KV_COMPUTE_BLOCK_COUNT
            kv_compute_idx = block_idx % KV_COMPUTE_BLOCK_COUNT
            raw_partition = window_start_partition + partition_in_window
            is_valid_partition = raw_partition < window_end_partition
            current_partition = gl.minimum(raw_partition, window_end_partition - 1)
            kv_sequence_start_idx = current_partition * CONTEXT_PARTITION_SIZE
        else:
            kv_compute_idx = block_idx
            current_partition = sequence_partition_idx
            is_valid_partition = True

        kv_subsequence_start_idx = (
            kv_sequence_start_idx + kv_compute_idx * KV_COMPUTE_BLOCK_SIZE
        )
        kv_subsequence_end_idx = gl.minimum(
            kv_subsequence_start_idx + KV_COMPUTE_BLOCK_SIZE, context_length
        )

        num_kv_blocks = gl.cdiv(
            kv_subsequence_end_idx - kv_subsequence_start_idx, KV_BLOCK_SIZE
        )
        kv_block_start_idx = (
            current_partition * SEQUENCE_PARTITION_KV_BLOCKS
            + kv_compute_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
        )
        qk_column_offsets = kv_block_start_idx * KV_BLOCK_SIZE + gl.arange(
            0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )

        # Load KV block indices from block table
        block_indices = gl.arange(
            0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout
        )
        # Create mask for valid blocks
        valid_block_mask = block_indices < num_kv_blocks
        masked_block_indices = gl.where(valid_block_mask, block_indices, 0)
        block_table_start_ptr = block_tables_ptr + sequence_idx * stride_block_table_seq
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_table_start_ptr + kv_block_start_idx, offsets=masked_block_indices
        )
        kv_block_numbers = kv_block_numbers.to(gl.int64)

        # ==================== KEY LOADING AND PROCESSING ====================
        # Calculate key cache offsets and load keys
        key_block_offsets = (
            kv_block_numbers[:, None, None, None] * stride_key_block
            + kv_head_idx * stride_key_head
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            + block_element_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
            + contiguous_kv_element_offsets[None, None, None, :]
        )
        key_tensor = gl.load(key_cache_ptr + key_block_offsets)

        # Load key quantization scales if needed
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 0:
                # Per-tensor quantization
                key_scale_value = tl.load(key_scale)
                value_scale_value = tl.load(value_scale)
            elif KV_QUANT_MODE == 1:
                # Per-token quantization
                key_scale_offsets = (
                    kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                    + kv_head_idx * kv_scale_stride_1
                    + block_element_offsets[None, None, :, None]
                )
                key_scale_offsets = gl.reshape(
                    key_scale_offsets, [KV_COMPUTE_BLOCK_SIZE]
                )
                key_scale_offsets = gl.convert_layout(
                    key_scale_offsets, layout=gl.SliceLayout(0, qk_linear_layout)
                )
                key_scale_value = gl.load(key_scale + key_scale_offsets)
                value_scale_value = gl.load(value_scale + key_scale_offsets)

        # Reshape key tensor for matrix multiplication
        key_tensor = gl.permute(key_tensor, [1, 3, 0, 2])
        key_tensor = gl.reshape(key_tensor, [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE])

        # ==================== ATTENTION SCORE COMPUTATION ====================
        # Initialize QK accumulator
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )

        # Convert layouts for MFMA operation
        query_converted = query_shared.load(qk_lhs_operand_layout)
        key_converted = gl.convert_layout(key_tensor, layout=qk_rhs_operand_layout)

        query_converted = query_converted.to(COMPUTE_TYPE)
        key_converted = key_converted.to(COMPUTE_TYPE)

        # Compute QK attention scores using MFMA
        attention_scores = gl.amd.cdna3.mfma(
            query_converted, key_converted, qk_accumulator
        )
        attention_scores = gl.reshape(
            attention_scores, [QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        )

        # ==================== VALUE LOADING AND PROCESSING ====================
        if VALUE_TRANSPOSED:
            # Load values from transposed cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(
                    1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                ),
            )
            value_block_offsets = (
                kv_block_numbers_reshaped[:, None, None, None] * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim1_offsets[None, :, None, None] * stride_value_head_size
                + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                + value_dim3_offsets[None, None, None, :]
            )
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
            value_tensor = gl.reshape(
                value_tensor, [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            )
        else:
            # Load values from standard cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
            )
            value_block_offsets = (
                kv_block_numbers_reshaped[:, None, None] * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim1_offsets[None, :, None] * stride_value_head_size
                + value_dim2_offsets[None, None, :]
            )
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])
            value_tensor = gl.reshape(
                value_tensor, [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            )

        # Apply quantization scaling to attention scores
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Expand scale for broadcasting
                key_scale_value = key_scale_value[None, :]
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value * key_scale_value
            else:
                qk_scale_value = softmax_scale * key_scale_value
        else:
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value
            else:
                qk_scale_value = softmax_scale

        # ==================== ATTENTION MASKING ====================
        # Create boundary mask for valid sequence positions
        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = (
                QUERY_SEQ_LEN - 1 - qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
            )
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
            if SLIDING_WINDOW > 0:
                causal_mask = causal_mask & (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    >= sequence_start_idx
                )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length
            if SLIDING_WINDOW > 0:
                query_token_idx = qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
                causal_mask = causal_mask & (
                    qk_column_offsets[None, :]
                    >= sequence_start_idx + query_token_idx[:, None]
                )

        boundary_mask = qk_row_mask[:, None] & causal_mask
        if SLIDING_WINDOW > 0:
            boundary_mask = boundary_mask & is_valid_partition

        attention_scores = gl.convert_layout(attention_scores, layout=qk_linear_layout)
        attention_scores = qk_scale_value * attention_scores

        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        attention_scores = gl.where(boundary_mask, attention_scores, float(-3.4e38))

        # ==================== SOFTMAX COMPUTATION ====================
        # Update running maximum for numerical stability
        current_max_logits = gl.max(attention_scores, axis=1)
        new_max_logits = gl.maximum(max_logits, current_max_logits)

        # Compute scaling factor for previous accumulator
        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)

        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_logits[:, None]) * LOG2_E
        )
        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)

        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # Create mask for valid tokens
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = tl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / (value_scale_max + 1e-8)
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max / float(FP8_MAX_VALUE)
            elif KV_QUANT_MODE == 0:
                # Per-tensor quantization scaling
                attention_probs *= float(FP8_MAX_VALUE)
                probability_scale = value_scale_value / float(FP8_MAX_VALUE)
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)

        # Convert layouts for PV MFMA operation
        probs_converted = gl.convert_layout(
            attention_probs, layout=pv_lhs_operand_layout
        )
        values_converted = gl.convert_layout(value_tensor, layout=pv_rhs_operand_layout)
        values_converted = values_converted.to(COMPUTE_TYPE)

        # Scale previous accumulator and compute new attention output
        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_accumulator *= accumulator_scale_expanded

        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        attention_output = gl.amd.cdna3.mfma(
            probs_converted, values_converted, pv_accumulator
        )
        if KV_QUANT_MODE >= 0:
            attention_accumulator += probability_scale * attention_output
        else:
            attention_accumulator += attention_output

        # Update running maximum for next iteration
        max_logits = new_max_logits

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
    gl.store(
        actual_partial_fp32_ptr + output_offsets,
        attention_accumulator,
        mask=output_mask,
    )
    attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)

    # Store results to global memory
    store_temporary_result(
        max_logits,
        exp_sums,
        attention_accumulator,
        max_logits_ptr,
        exp_sums_ptr,
        output_ptr,
        max_logits_offsets,
        output_offsets,
        qk_row_mask,
        output_mask,
    )

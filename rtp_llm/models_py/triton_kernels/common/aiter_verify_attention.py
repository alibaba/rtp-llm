# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Single-launch BF16 paged verify attention for CDNA3.

The dot schedule derives from AITER pa_decode_gluon. Fixed workers loop over
live KV partitions, then the last producer merges their partial results without
spinning. The 4096 tuning point sets worker count, never a KV length limit.
"""
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def define_layout(
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
) -> gl.constexpr:
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
        qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=register_bases,
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        )
    else:
        VGPRS0: gl.constexpr = QUERY_SEQ_LEN_POW2
        THREADS0: gl.constexpr = triton.cdiv(QUERY_GROUP_SIZE_POW2, 4 * VGPRS0)
        THREADS1: gl.constexpr = 64 // THREADS0
        qk_linear_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[VGPRS0, CONTEXT_PARTITION_SIZE // THREADS1],
            threads_per_warp=[THREADS0, THREADS1],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
    return qk_linear_layout


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


define_layout.__triton_builtin__ = True
store_temporary_result.__triton_builtin__ = True


@gluon.jit
def _partition_attention(
    exp_sums_ptr,
    max_logits_ptr,
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    context_lengths_ptr,
    stride_block_table_seq: gl.constexpr,
    QUERY_SEQ_LEN: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: gl.constexpr,
    PARTS: gl.constexpr,
    Q_LEN: gl.constexpr,
):
    MFMA_INSTR_K: gl.constexpr = 16
    QK_PV_MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, MFMA_INSTR_K]
    KV_16B_ELEMENT_COUNT: gl.constexpr = 8
    OUTPUT_DTYPE: gl.constexpr = gl.bfloat16
    LOG2_E: gl.constexpr = 1.4426950408889634
    QUERY_SEQ_LEN_POW2: gl.constexpr = triton.next_power_of_2(QUERY_SEQ_LEN)
    if 6 <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = triton.next_power_of_2(6)
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    K_HEAD_SIZE_SPLITS: gl.constexpr = 256 // KV_16B_ELEMENT_COUNT
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = KV_COMPUTE_BLOCK_SIZE // 16
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
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
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        KV_16B_ELEMENT_COUNT, 1, 16, order=[1, 0]
    )
    blocked_key_layout_fp8: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )
    key_warps_per_cta_f16: gl.constexpr = [4, 1, 1, 1] if 16 == 16 else [1, 1, 4, 1]
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
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
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
    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE, QUERY_SEQ_LEN_POW2
    )
    value_threads_per_warp: gl.constexpr = [4, 1, 16, 1] if 16 == 16 else [1, 4, 16, 1]
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
        16 // KV_16B_ELEMENT_COUNT,
        layout=gl.SliceLayout(
            0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
        ),
    )
    value_dim2_offsets = gl.arange(
        0,
        256,
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
    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
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
    mtp_query_len_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_query_group_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_head_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, mtp_blocked_query_layout)
    )
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
    mtp_query_len_offsets = gl.arange(
        0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
    )
    mtp_query_group_size_offsets = gl.arange(
        0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
    )
    mtp_head_size_offsets = gl.arange(0, 256, layout=mtp_head_size_layout)
    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(0, 16, layout=block_element_layout)
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    query_row_mask_3d = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
        mtp_query_group_size_offsets[None, :, None] < 6
    )
    query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    output_partition_idx = gl.program_id(2)
    tiles_per_sequence: gl.constexpr = triton.cdiv(Q_LEN, QUERY_SEQ_LEN)
    batch_idx = sequence_idx // tiles_per_sequence
    query_tile_idx = sequence_idx % tiles_per_sequence
    live_kv = gl.load(context_lengths_ptr + batch_idx)
    # Each tile ends at a different causal position. Only the memory-read bound
    # is clipped for the last partial Q tile; its causal positions stay intact.
    tile_context_length = live_kv - Q_LEN + (query_tile_idx + 1) * QUERY_SEQ_LEN
    context_length = gl.maximum(0, tile_context_length)
    if Q_LEN % QUERY_SEQ_LEN != 0:
        context_length = gl.minimum(live_kv, context_length)
    sequence_start_idx = 0
    sequence_partition_idx = output_partition_idx
    mtp_query_offsets = (
        (batch_idx * Q_LEN + query_tile_idx * QUERY_SEQ_LEN) * 3072
        + mtp_query_len_offsets[:, None, None] * 3072
        + kv_head_idx * 1536
        + mtp_query_group_size_offsets[None, :, None] * 256
        + mtp_head_size_offsets[None, None, :]
    )
    mtp_query_mask = (
        (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN)
        & (mtp_query_group_size_offsets[None, :, None] < 6)
        & (mtp_head_size_offsets[None, None, :] < 256)
    )
    if Q_LEN % QUERY_SEQ_LEN != 0:
        mtp_query_mask = mtp_query_mask & (
            query_tile_idx * QUERY_SEQ_LEN + mtp_query_len_offsets[:, None, None]
            < Q_LEN
        )
    mtp_query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
    )
    mtp_query_tensor = gl.reshape(mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, 256])
    query_tensor = gl.convert_layout(mtp_query_tensor, layout=blocked_query_layout)
    query_shared = gl.allocate_shared_memory(
        query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
    )
    max_logits_base_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    max_logits_query_len_idx = max_logits_base_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    max_logits_group_idx_in_len = (
        max_logits_base_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    )
    max_logits_base_offsets = max_logits_query_len_idx * 6 + max_logits_group_idx_in_len
    max_logits_offsets = (
        sequence_idx * (2 * PARTS * QUERY_SEQ_LEN * 6)
        + kv_head_idx * (PARTS * QUERY_SEQ_LEN * 6)
        + output_partition_idx * (QUERY_SEQ_LEN * 6)
        + max_logits_base_offsets
    )
    output_group_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    output_query_len_idx = output_group_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    output_group_idx_in_len = output_group_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    output_group_offsets = output_query_len_idx * 6 + output_group_idx_in_len
    output_head_size_offsets = gl.arange(
        0, 256, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = pv_row_mask[:, None] & (output_head_size_offsets[None, :] < 256)
    output_offsets = sequence_idx * (2 * PARTS * QUERY_SEQ_LEN * 6 * 256)
    output_offsets += kv_head_idx * (PARTS * QUERY_SEQ_LEN * 6 * 256)
    output_offsets += (
        output_partition_idx * (QUERY_SEQ_LEN * 6 * 256)
        + output_group_offsets[:, None] * 256
        + output_head_size_offsets[None, :]
    )
    max_logits = max_logits_base_offsets.to(gl.float32) * 0.0 - 3.4e38
    exp_sums = max_logits_base_offsets.to(gl.float32) * 0.0
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, 256), dtype=gl.float32, layout=pv_mfma_layout
    )
    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    SEQUENCE_PARTITION_KV_BLOCKS: gl.constexpr = CONTEXT_PARTITION_SIZE // 16
    # Fixed workers cover arbitrarily many logical partitions using online
    # softmax. Empty workers still store neutral partials and publish completion.
    for current_partition in range(
        output_partition_idx,
        gl.cdiv(context_length, CONTEXT_PARTITION_SIZE),
        gl.num_programs(2),
    ):
        kv_sequence_start_idx = current_partition * CONTEXT_PARTITION_SIZE
        is_valid_partition = True
        for kv_compute_idx in gl.static_range(KV_COMPUTE_BLOCK_COUNT):
            if (
                kv_sequence_start_idx + kv_compute_idx * KV_COMPUTE_BLOCK_SIZE
                < context_length
            ):
                kv_subsequence_start_idx = (
                    kv_sequence_start_idx + kv_compute_idx * KV_COMPUTE_BLOCK_SIZE
                )
                kv_subsequence_end_idx = gl.minimum(
                    kv_subsequence_start_idx + KV_COMPUTE_BLOCK_SIZE, context_length
                )
                num_kv_blocks = gl.cdiv(
                    kv_subsequence_end_idx - kv_subsequence_start_idx, 16
                )
                kv_block_start_idx = (
                    current_partition * SEQUENCE_PARTITION_KV_BLOCKS
                    + kv_compute_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
                )
                qk_column_offsets = kv_block_start_idx * 16 + gl.arange(
                    0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
                )
                block_indices = gl.arange(
                    0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout
                )
                valid_block_mask = block_indices < num_kv_blocks
                masked_block_indices = gl.where(valid_block_mask, block_indices, 0)
                block_table_start_ptr = (
                    block_tables_ptr + batch_idx * stride_block_table_seq
                )
                kv_block_numbers = gl.amd.cdna3.buffer_load(
                    ptr=block_table_start_ptr + kv_block_start_idx,
                    offsets=masked_block_indices,
                )
                kv_block_numbers = kv_block_numbers.to(gl.int64)
                key_block_offsets = (
                    kv_block_numbers[:, None, None, None] * 16384
                    + kv_head_idx * 4096
                    + head_size_split_offsets[None, :, None, None] * 128
                    + block_element_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + contiguous_kv_element_offsets[None, None, None, :]
                )
                key_tensor = gl.load(key_cache_ptr + key_block_offsets)
                key_tensor = gl.permute(key_tensor, [1, 3, 0, 2])
                key_tensor = gl.reshape(key_tensor, [256, KV_COMPUTE_BLOCK_SIZE])
                qk_accumulator = gl.zeros(
                    (QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE),
                    dtype=gl.float32,
                    layout=qk_mfma_layout,
                )
                query_converted = query_shared.load(qk_lhs_operand_layout)
                key_converted = gl.convert_layout(
                    key_tensor, layout=qk_rhs_operand_layout
                )
                query_converted = query_converted.to(gl.bfloat16)
                key_converted = key_converted.to(gl.bfloat16)
                attention_scores = gl.amd.cdna3.mfma(
                    query_converted, key_converted, qk_accumulator
                )
                attention_scores = gl.reshape(
                    attention_scores, [QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
                )
                kv_block_numbers_reshaped = gl.convert_layout(
                    kv_block_numbers,
                    layout=gl.SliceLayout(
                        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                    ),
                )
                value_block_offsets = (
                    kv_block_numbers_reshaped[:, None, None, None] * 16384
                    + kv_head_idx * 4096
                    + value_dim1_offsets[None, :, None, None] * 2048
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )
                value_tensor = gl.load(value_cache_ptr + value_block_offsets)
                value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
                value_tensor = gl.reshape(value_tensor, [KV_COMPUTE_BLOCK_SIZE, 256])
                qk_scale_value = 0.0625
                sequence_position_extension = (
                    QUERY_SEQ_LEN - 1 - qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
                )
                causal_mask = (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    < tile_context_length
                )
                boundary_mask = qk_row_mask[:, None] & causal_mask
                attention_scores = gl.convert_layout(
                    attention_scores, layout=qk_linear_layout
                )
                attention_scores = qk_scale_value * attention_scores
                attention_scores = gl.where(boundary_mask, attention_scores, -3.4e38)
                current_max_logits = gl.max(attention_scores, axis=1)
                new_max_logits = gl.maximum(max_logits, current_max_logits)
                accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)
                attention_probs = tl.math.exp2(
                    (attention_scores - new_max_logits[:, None]) * LOG2_E
                )
                attention_probs = gl.where(boundary_mask, attention_probs, 0.0)
                exp_sums = accumulator_scale * exp_sums + gl.sum(
                    attention_probs, axis=1
                )
                attention_probs = attention_probs.to(gl.bfloat16)
                probs_converted = gl.convert_layout(
                    attention_probs, layout=pv_lhs_operand_layout
                )
                values_converted = gl.convert_layout(
                    value_tensor, layout=pv_rhs_operand_layout
                )
                values_converted = values_converted.to(gl.bfloat16)
                accumulator_scale_expanded = gl.convert_layout(
                    accumulator_scale[:, None], layout=pv_mfma_layout
                )
                attention_accumulator *= accumulator_scale_expanded
                pv_accumulator = gl.zeros(
                    (QUERY_GROUP_SIZE_POW2, 256),
                    dtype=gl.float32,
                    layout=pv_mfma_layout,
                )
                attention_output = gl.amd.cdna3.mfma(
                    probs_converted, values_converted, pv_accumulator
                )
                attention_accumulator += attention_output
                max_logits = new_max_logits
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
    attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)
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


@gluon.jit
def _gluon_reduce(
    exp_ptr,
    max_ptr,
    partial_ptr,
    out_ptr,
    gid,
    kh,
    PARTS: gl.constexpr,
    ROWS: gl.constexpr,
    ROWS_PAD: gl.constexpr,
    QTILE: gl.constexpr,
    Q_LEN: gl.constexpr,
):
    D: gl.constexpr = 256
    stats_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    vec_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    part = gl.arange(0, PARTS, layout=gl.SliceLayout(1, stats_layout))
    stat_row = gl.arange(0, ROWS_PAD, layout=gl.SliceLayout(0, stats_layout))
    stat_mask = stat_row[None, :] < ROWS
    stat_base = (gid * 2 + kh) * PARTS * ROWS
    part_offsets = stat_base + part[:, None] * ROWS + stat_row[None, :]
    pmax = gl.load(max_ptr + part_offsets, mask=stat_mask, other=-3.4e38)
    psum = gl.load(exp_ptr + part_offsets, mask=stat_mask, other=0.0)
    gmax = gl.max(pmax, axis=0)
    weights = gl.exp(pmax - gmax[None, :])
    gsum = gl.sum(weights * psum, axis=0)
    out_row = gl.arange(0, ROWS_PAD, layout=gl.SliceLayout(1, vec_layout))
    d = gl.arange(0, D, layout=gl.SliceLayout(0, vec_layout))
    row_mask = out_row < ROWS
    gmax = gl.convert_layout(gmax, gl.SliceLayout(1, vec_layout))
    gsum = gl.convert_layout(gsum, gl.SliceLayout(1, vec_layout))
    acc = gl.zeros((ROWS_PAD, D), gl.float32, layout=vec_layout)
    for p in range(PARTS):
        pm = gl.load(
            max_ptr + stat_base + p * ROWS + out_row, mask=row_mask, other=-3.4e38
        )
        pe = gl.load(exp_ptr + stat_base + p * ROWS + out_row, mask=row_mask, other=0.0)
        pm = gl.convert_layout(pm, gl.SliceLayout(1, vec_layout))
        pe = gl.convert_layout(pe, gl.SliceLayout(1, vec_layout))
        weight = gl.exp(pm - gmax) * pe / gl.maximum(gsum, 1e-20)
        poff = (
            ((gid * 2 + kh) * PARTS + p) * ROWS * D + out_row[:, None] * D + d[None, :]
        )
        value = gl.load(partial_ptr + poff, mask=row_mask[:, None], other=0.0)
        acc += value.to(gl.float32) * weight[:, None]
    qpos = out_row // 6
    gqa = out_row % 6
    out_off = (
        (
            gid // triton.cdiv(Q_LEN, QTILE) * Q_LEN
            + gid % triton.cdiv(Q_LEN, QTILE) * QTILE
        )
        * 12
        * D
        + qpos[:, None] * 12 * D
        + (kh * 6 + gqa[:, None]) * D
        + d[None, :]
    )
    if Q_LEN % QTILE != 0:
        row_mask = row_mask & (gid % triton.cdiv(Q_LEN, QTILE) * QTILE + qpos < Q_LEN)
    gl.store(out_ptr + out_off, acc.to(gl.bfloat16), mask=row_mask[:, None])


@gluon.jit
def _verify_attention(
    query,
    kv,
    block_table,
    lengths,
    exp_sums,
    max_logits,
    partial,
    counters,
    output,
    BT_STRIDE: gl.constexpr,
    Q_LEN: gl.constexpr,
    QTILE: gl.constexpr,
    PARTITION: gl.constexpr,
    COMPUTE: gl.constexpr,
    PARTS: gl.constexpr,
    ROWS: gl.constexpr,
    ROWS_PAD: gl.constexpr,
):
    gid = gl.program_id(0)
    head = gl.program_id(1)
    _partition_attention(
        exp_sums,
        max_logits,
        partial,
        query,
        kv,
        kv + 8192,
        block_table,
        lengths,
        BT_STRIDE,
        QTILE,
        PARTITION,
        COMPUTE,
        PARTS,
        Q_LEN,
    )
    # Publish all lanes' scratch stores before the agent-scope RMW. The last
    # producer acquires preceding publications; no CTA waits for another CTA.
    tl.debug_barrier()
    old = tl.atomic_add(counters + gid * 2 + head, 1, sem="acq_rel", scope="gpu")
    if old == PARTS - 1:
        _gluon_reduce(
            exp_sums,
            max_logits,
            partial,
            output,
            gid,
            head,
            PARTS,
            ROWS,
            ROWS_PAD,
            QTILE,
            Q_LEN,
        )
        tl.debug_barrier()
        tl.atomic_xchg(counters + gid * 2 + head, 0, sem="release", scope="gpu")


# (maximum batch, query tile, KV partition, compute tile). Q5-7 reuse Q8 values.
_CONFIGS = (
    (1, 2, 512, 128),
    (2, 4, 512, 256),
    (4, 4, 1024, 256),
    (16, 8, 1024, 128),
    (32, 8, 512, 256),
)


def supports_shape(
    batch_size: int,
    query_length: int,
    head_num: int,
    head_num_kv: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> bool:
    return (
        1 <= batch_size <= 32
        and 5 <= query_length <= 8
        and (head_num, head_num_kv, head_dim, page_size) == (12, 2, 256, 16)
        and dtype == torch.bfloat16
    )


@lru_cache(maxsize=None)
def is_supported_device(device: torch.device) -> bool:
    device = torch.device(device)
    if not torch.version.hip or device.type != "cuda":
        return False
    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx942"
    )


class VerifyAttentionWorkspace:
    """Workspace for one fixed B/Q, owned by one serial execution stream.

    Allocate before capture. Inputs may change contents, but graph replay must
    retain their addresses and shapes. Live lengths include Q and must be zero
    (padding) or >=Q, within the allocated block-table/KV capacity. The caller
    validates device metadata before replay; this wrapper never synchronizes it.
    """

    def __init__(self, batch_size: int, query_length: int, device: torch.device):
        if not supports_shape(batch_size, query_length, 12, 2, 256, 16, torch.bfloat16):
            raise ValueError("verify attention requires B=1..32 and Q=5..8")
        self.batch_size = batch_size
        self.query_length = query_length
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("verify attention requires a CUDA/HIP device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        _, self.q_tile, self.partition, self.compute = next(
            c for c in _CONFIGS if batch_size <= c[0]
        )
        self.parts = 4096 // self.partition
        self.rows = self.q_tile * 6
        self.groups = batch_size * triton.cdiv(query_length, self.q_tile)
        stats_shape = (self.groups, 2, self.parts, self.rows)
        self.exp_sums = torch.empty(
            stats_shape, device=self.device, dtype=torch.float32
        )
        self.max_logits = torch.empty_like(self.exp_sums)
        self.partial = torch.empty(
            (*stats_shape, 256), device=self.device, dtype=torch.bfloat16
        )
        self.counters = torch.zeros(
            (self.groups, 2), device=self.device, dtype=torch.int32
        )
        self.output = torch.empty(
            (batch_size * query_length, 12, 256),
            device=self.device,
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        block_table: torch.Tensor,
        kv_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query.shape != self.output.shape
            or query.dtype != torch.bfloat16
            or not query.is_contiguous()
            or query.device != self.device
        ):
            raise ValueError(
                "query must match the prepared contiguous BF16 B/Q geometry"
            )
        if (
            kv.ndim != 5
            or tuple(kv.shape[1:]) != (2, 2, 16, 256)
            or kv.dtype != torch.bfloat16
            or not kv.is_contiguous()
            or kv.device != self.device
        ):
            raise ValueError(
                "KV must use contiguous BF16 vectorized AITER page storage"
            )
        if (
            block_table.ndim != 2
            or block_table.shape[0] != self.batch_size
            or block_table.shape[1] == 0
            or block_table.stride(1) != 1
            or block_table.dtype != torch.int32
            or block_table.device != self.device
        ):
            raise ValueError(
                "block table must be row-major int32 with one row per stream"
            )
        if (
            kv_lengths.shape != (self.batch_size,)
            or kv_lengths.dtype != torch.int32
            or not kv_lengths.is_contiguous()
            or kv_lengths.device != self.device
        ):
            raise ValueError(
                "live lengths must be contiguous int32 with one entry per stream"
            )
        self.compiled_kernel = _verify_attention[(self.groups, 2, self.parts)](
            query,
            kv,
            block_table,
            kv_lengths,
            self.exp_sums,
            self.max_logits,
            self.partial,
            self.counters,
            self.output,
            block_table.stride(0),
            self.query_length,
            self.q_tile,
            self.partition,
            self.compute,
            self.parts,
            self.rows,
            triton.next_power_of_2(self.rows),
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
        )
        return self.output
